"""ONNX Runtime computer vision pipeline: SCRFD-10G detection, 5-point alignment, ArcFace 512-D embeddings, yaw/blur filtering, and torso outfit descriptors."""

from __future__ import annotations

import base64
import io
import math
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from .config import Settings, settings
from .db import FACE_DIM, OUTFIT_DIM
from .matcher import MultiAngleQueryTemplate, ReferencePose


# Canonical 5-point facial landmarks for 112x112 ArcFace / AdaFace alignment
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)


@dataclass
class DetectedFace:
    bbox_xyxy: list[float]       # pixel coordinates in original EXIF-corrected image
    bbox_norm: list[float]       # normalized [x1/W, y1/H, x2/W, y2/H] in [0, 1]
    kps: np.ndarray              # shape (5, 2) in detection image coordinates
    det_score: float
    yaw_deg: float
    pose_label: str              # "left" | "front" | "right"
    blur_score: float
    quality_norm: float          # pre-normalization L2 norm (quality proxy)
    embedding: np.ndarray        # (512,) L2-normalized float32
    outfit_embedding: np.ndarray # (96,) L2-normalized float32 torso descriptor
    aligned_bgr: np.ndarray      # (112, 112, 3) aligned crop


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def _nms(dets: np.ndarray, thresh: float = 0.4) -> list[int]:
    if len(dets) == 0:
        return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def estimate_yaw(kps: np.ndarray) -> tuple[float, str]:
    """
    Estimate horizontal head yaw angle from SCRFD 5-point landmarks.
    Negative yaw: nose closer to image-left eye -> 'left'
    Near-zero yaw: centered nose -> 'front'
    Positive yaw: nose closer to image-right eye -> 'right'
    """
    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]

    eye_vec = right_eye - left_eye
    eye_dist = float(math.hypot(float(eye_vec[0]), float(eye_vec[1])))
    if eye_dist < 1e-4:
        return 0.0, "front"

    eye_unit = eye_vec / eye_dist
    eye_mid = 0.5 * (left_eye + right_eye)
    signed_offset = float(np.dot(nose - eye_mid, eye_unit)) / eye_dist
    yaw_deg = float(np.clip(signed_offset * 95.0, -75.0, 75.0))

    if yaw_deg <= -14.0:
        label = "left"
    elif yaw_deg >= 14.0:
        label = "right"
    else:
        label = "front"
    return yaw_deg, label


def compute_blur_score(aligned_bgr: np.ndarray) -> float:
    """Laplacian variance sharpness metric on aligned 112x112 face crop."""
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_torso_descriptor(img_bgr: np.ndarray, bbox_xyxy: list[float]) -> np.ndarray:
    """
    Compute a 96-D Hellinger-normalized HSV+LAB color descriptor of the torso
    region directly below a detected face for Pass-2 in-event outfit matching.
    Returns a zero vector if insufficient torso area is visible.
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    fw = max(1.0, x2 - x1)
    fh = max(1.0, y2 - y1)

    tx1 = max(0, int(x1 - 0.50 * fw))
    tx2 = min(w, int(x2 + 0.50 * fw))
    ty1 = min(h, int(y2 + 0.25 * fh))
    ty2 = min(h, int(y2 + 2.25 * fh))

    if (tx2 - tx1) < 20 or (ty2 - ty1) < 24:
        return np.zeros(OUTFIT_DIM, dtype=np.float32)

    torso = img_bgr[ty1:ty2, tx1:tx2]
    torso = cv2.resize(torso, (64, 64), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)

    h_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

    l_hist = cv2.calcHist([lab], [0], None, [12], [0, 256]).flatten()
    a_hist = cv2.calcHist([lab], [1], None, [14], [0, 256]).flatten()
    b_hist = cv2.calcHist([lab], [2], None, [14], [0, 256]).flatten()

    parts = [h_hist, s_hist, v_hist, l_hist, a_hist, b_hist]
    norm_parts = []
    for p in parts:
        s = float(np.sum(p))
        p_prob = p / s if s > 1e-6 else p
        # Square-root (Hellinger/Bhattacharyya) transform makes cosine dot-product sharp
        norm_parts.append(np.sqrt(np.maximum(p_prob, 0.0)))

    desc = np.concatenate(norm_parts).astype(np.float32)
    norm = float(np.linalg.norm(desc))
    if norm > 1e-6:
        desc /= norm
    return desc


def encode_preview_b64(img_bgr: np.ndarray, quality: int = 86) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def decode_image_bytes(raw_bytes: bytes) -> tuple[np.ndarray, int, int]:
    """Decode raw image bytes with EXIF auto-rotation into an OpenCV BGR array."""
    with Image.open(io.BytesIO(raw_bytes)) as pil_img:
        pil_img = ImageOps.exif_transpose(pil_img)
        rgb = pil_img.convert("RGB")
        width, height = rgb.size
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    return bgr, width, height


def save_original_and_thumb(
    raw_bytes: bytes,
    original_path: Path,
    thumb_path: Path,
    thumb_max_dim: int = 480,
) -> tuple[np.ndarray, int, int]:
    """
    Decode EXIF-corrected image, persist normalized JPEG original and WebP thumbnail,
    and return (bgr_image, width, height).
    """
    with Image.open(io.BytesIO(raw_bytes)) as pil_img:
        pil_img = ImageOps.exif_transpose(pil_img)
        rgb = pil_img.convert("RGB")
        width, height = rgb.size

        # Save EXIF-normalized JPEG original so browser coordinates match bounding boxes 1:1
        rgb.save(original_path, format="JPEG", quality=92, optimize=True)

        # Generate fast WebP thumbnail for responsive gallery loading
        thumb = rgb.copy()
        thumb.thumbnail((thumb_max_dim, thumb_max_dim), Image.Resampling.LANCZOS)
        thumb.save(thumb_path, format="WEBP", quality=82)

        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    return bgr, width, height


class VisionPipeline:
    """Self-contained SCRFD-10G + ArcFace (w600k_r50) ONNX Runtime pipeline."""

    def __init__(self, cfg: Settings = settings) -> None:
        self.cfg = cfg
        self.det_session: ort.InferenceSession | None = None
        self.rec_session: ort.InferenceSession | None = None
        self.active_provider: str = "uninitialized"
        self._anchor_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def ensure_models_downloaded(self) -> None:
        """Download official InsightFace buffalo_l ONNX models if not already cached."""
        self.cfg.ensure_dirs()
        if self.cfg.det_model_path.exists() and self.cfg.rec_model_path.exists():
            return

        zip_path = self.cfg.models_dir.parent / "buffalo_l.zip"
        urllib.request.urlretrieve(self.cfg.buffalo_l_url, zip_path)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    fname = Path(member).name
                    if fname in {"det_10g.onnx", "w600k_r50.onnx"}:
                        target = self.cfg.models_dir / fname
                        with zf.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
        finally:
            if zip_path.exists():
                zip_path.unlink()

    def load(self) -> None:
        """Initialize ONNX Runtime inference sessions."""
        if self.det_session is not None and self.rec_session is not None:
            return

        self.ensure_models_downloaded()
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.det_session = ort.InferenceSession(
            str(self.cfg.det_model_path), sess_options=sess_opts, providers=providers
        )
        self.rec_session = ort.InferenceSession(
            str(self.cfg.rec_model_path), sess_options=sess_opts, providers=providers
        )
        self.active_provider = self.det_session.get_providers()[0]

    def _get_anchors(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached
        anchor_centers = np.stack(
            np.mgrid[:height, :width][::-1], axis=-1
        ).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        # SCRFD-10G uses 2 anchors per spatial location
        anchor_centers = np.stack([anchor_centers, anchor_centers], axis=1).reshape((-1, 2))
        self._anchor_cache[key] = anchor_centers
        return anchor_centers

    def detect_raw(
        self,
        img_bgr: np.ndarray,
        *,
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float | None = None,
        nms_thresh: float = 0.4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run SCRFD-10G detection and return (bboxes_xyxys, kpss_5x2) in img_bgr pixels."""
        self.load()
        assert self.det_session is not None

        thresh = self.cfg.min_det_score if det_thresh is None else det_thresh
        h, w = img_bgr.shape[:2]
        input_w, input_h = det_size

        im_ratio = float(h) / max(w, 1)
        model_ratio = float(input_h) / input_w
        if im_ratio > model_ratio:
            new_h = input_h
            new_w = max(1, int(new_h / im_ratio))
        else:
            new_w = input_w
            new_h = max(1, int(new_w * im_ratio))

        det_scale = float(new_h) / max(h, 1)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w, :] = resized

        blob = cv2.dnn.blobFromImage(
            canvas,
            1.0 / 128.0,
            (input_w, input_h),
            (127.5, 127.5, 127.5),
            swapRB=True,
        )

        input_name = self.det_session.get_inputs()[0].name
        output_names = [o.name for o in self.det_session.get_outputs()]
        net_outs = self.det_session.run(output_names, {input_name: blob})

        strides = (8, 16, 32)
        fmc = 3
        scores_list: list[np.ndarray] = []
        bboxes_list: list[np.ndarray] = []
        kpss_list: list[np.ndarray] = []

        for idx, stride in enumerate(strides):
            scores = net_outs[idx].reshape(-1)
            bbox_preds = net_outs[idx + fmc].reshape((-1, 4)) * stride
            kps_preds = net_outs[idx + fmc * 2].reshape((-1, 10)) * stride

            pos_inds = np.where(scores >= thresh)[0]
            if len(pos_inds) == 0:
                continue

            fm_h = input_h // stride
            fm_w = input_w // stride
            anchors = self._get_anchors(fm_h, fm_w, stride)

            bboxes = _distance2bbox(anchors[pos_inds], bbox_preds[pos_inds])
            kpss = _distance2kps(anchors[pos_inds], kps_preds[pos_inds]).reshape((-1, 5, 2))

            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes)
            kpss_list.append(kpss)

        if not scores_list:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)

        all_scores = np.concatenate(scores_list, axis=0)
        all_bboxes = np.vstack(bboxes_list) / det_scale
        all_kpss = np.vstack(kpss_list) / det_scale

        # Clip boxes to image dimensions
        all_bboxes[:, 0] = np.clip(all_bboxes[:, 0], 0, w)
        all_bboxes[:, 1] = np.clip(all_bboxes[:, 1], 0, h)
        all_bboxes[:, 2] = np.clip(all_bboxes[:, 2], 0, w)
        all_bboxes[:, 3] = np.clip(all_bboxes[:, 3], 0, h)

        pre_det = np.hstack([all_bboxes, all_scores[:, None]]).astype(np.float32)
        keep = _nms(pre_det, thresh=nms_thresh)
        if not keep:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)

        return pre_det[keep], all_kpss[keep]

    @staticmethod
    def align_face(img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """Apply 5-point similarity affine transform to align face to 112x112."""
        M, _ = cv2.estimateAffinePartial2D(
            kps.astype(np.float32), ARCFACE_DST, method=cv2.LMEDS
        )
        if M is None:
            # Fallback crop if degenerate landmarks
            xs, ys = kps[:, 0], kps[:, 1]
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            half = max(32.0, float(max(np.ptp(xs), np.ptp(ys))) * 1.4)
            h, w = img_bgr.shape[:2]
            x1, y1 = max(0, int(cx - half)), max(0, int(cy - half))
            x2, y2 = min(w, int(cx + half)), min(h, int(cy + half))
            crop = img_bgr[y1:y2, x1:x2]
            return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
        return cv2.warpAffine(
            img_bgr, M, (112, 112), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
        )

    def embed_aligned(self, aligned_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Run ArcFace w600k_r50 ONNX on 112x112 aligned face -> (L2-norm 512-D, raw_norm)."""
        self.load()
        assert self.rec_session is not None

        blob = cv2.dnn.blobFromImage(
            aligned_bgr,
            1.0 / 127.5,
            (112, 112),
            (127.5, 127.5, 127.5),
            swapRB=True,
        )
        input_name = self.rec_session.get_inputs()[0].name
        output_name = self.rec_session.get_outputs()[0].name
        raw = self.rec_session.run([output_name], {input_name: blob})[0][0].astype(np.float32)

        raw_norm = float(np.linalg.norm(raw))
        if raw_norm > 1e-8:
            emb = (raw / raw_norm).astype(np.float32)
        else:
            emb = np.zeros(FACE_DIM, dtype=np.float32)
        return emb, raw_norm

    def analyze_image(
        self,
        img_bgr: np.ndarray,
        *,
        det_thresh: float | None = None,
        extract_outfit: bool = True,
    ) -> list[DetectedFace]:
        """
        Full single-image analysis:
        1. In-memory downscale to max_detection_dim (e.g. 1600px) for fast detection
        2. SCRFD-10G detection + 5-point landmarks
        3. 112x112 affine alignment + ArcFace 512-D embedding
        4. Yaw estimation, Laplacian blur score, and torso outfit descriptor
        """
        orig_h, orig_w = img_bgr.shape[:2]
        max_dim = max(orig_h, orig_w)
        if max_dim > self.cfg.max_detection_dim:
            scale = self.cfg.max_detection_dim / float(max_dim)
            work_w = max(1, int(round(orig_w * scale)))
            work_h = max(1, int(round(orig_h * scale)))
            work_img = cv2.resize(img_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            work_img = img_bgr
            work_w, work_h = orig_w, orig_h

        dets, kpss = self.detect_raw(work_img, det_thresh=det_thresh)
        results: list[DetectedFace] = []

        for det, kps in zip(dets, kpss):
            wx1, wy1, wx2, wy2, score = [float(v) for v in det]
            if (wx2 - wx1) < self.cfg.min_face_size_px or (wy2 - wy1) < self.cfg.min_face_size_px:
                continue

            aligned = self.align_face(work_img, kps)
            emb, quality_norm = self.embed_aligned(aligned)
            blur_score = compute_blur_score(aligned)
            yaw_deg, pose_label = estimate_yaw(kps)

            if extract_outfit:
                outfit_vec = compute_torso_descriptor(work_img, [wx1, wy1, wx2, wy2])
            else:
                outfit_vec = np.zeros(OUTFIT_DIM, dtype=np.float32)

            ox1 = max(0.0, min(float(orig_w), wx1 / scale))
            oy1 = max(0.0, min(float(orig_h), wy1 / scale))
            ox2 = max(0.0, min(float(orig_w), wx2 / scale))
            oy2 = max(0.0, min(float(orig_h), wy2 / scale))

            results.append(
                DetectedFace(
                    bbox_xyxy=[ox1, oy1, ox2, oy2],
                    bbox_norm=[
                        ox1 / max(orig_w, 1),
                        oy1 / max(orig_h, 1),
                        ox2 / max(orig_w, 1),
                        oy2 / max(orig_h, 1),
                    ],
                    kps=kps,
                    det_score=score,
                    yaw_deg=yaw_deg,
                    pose_label=pose_label,
                    blur_score=blur_score,
                    quality_norm=quality_norm,
                    embedding=emb,
                    outfit_embedding=outfit_vec,
                    aligned_bgr=aligned,
                )
            )

        # Sort faces by bounding box area descending (primary/foreground face first)
        results.sort(
            key=lambda f: (f.bbox_xyxy[2] - f.bbox_xyxy[0]) * (f.bbox_xyxy[3] - f.bbox_xyxy[1]),
            reverse=True,
        )
        return results

    def build_reference_from_images(
        self,
        images: list[tuple[str, bytes, str | None]],
    ) -> tuple[MultiAngleQueryTemplate, list[dict[str, Any]]]:
        """
        Extract primary face embeddings from 1..N reference images (Left, Front, Right).
        Returns (MultiAngleQueryTemplate, skipped_or_warning_notes).
        """
        poses: list[ReferencePose] = []
        skipped: list[dict[str, Any]] = []

        for idx, (filename, raw_bytes, hint_label) in enumerate(images):
            try:
                bgr, _, _ = decode_image_bytes(raw_bytes)
            except Exception as exc:
                skipped.append({"filename": filename, "reason": f"Invalid image: {exc}"})
                continue

            faces = self.analyze_image(bgr, det_thresh=0.45, extract_outfit=False)
            if not faces:
                skipped.append({"filename": filename, "reason": "No face detected in photo."})
                continue

            primary = faces[0]  # largest foreground face
            warnings: list[str] = []
            if len(faces) > 1:
                warnings.append(
                    f"{len(faces)} faces detected; used largest foreground face."
                )
            if primary.blur_score < self.cfg.min_blur_variance:
                warnings.append("Photo appears slightly blurry; sharper input improves matching.")
            if primary.quality_norm < 10.5:
                warnings.append("Low face quality/lighting detected; check occlusion or shadow.")

            label = hint_label if hint_label in {"left", "front", "right"} else primary.pose_label
            # Disambiguate repeated pose labels cleanly if user uploads multiple frontal photos
            existing_labels = {p.label for p in poses}
            if label in existing_labels:
                label = f"{label}_{idx + 1}"

            poses.append(
                ReferencePose(
                    label=label,
                    yaw_deg=round(primary.yaw_deg, 1),
                    blur_score=round(primary.blur_score, 1),
                    quality_norm=round(primary.quality_norm, 2),
                    det_score=round(primary.det_score, 3),
                    embedding=primary.embedding,
                    preview_b64=encode_preview_b64(primary.aligned_bgr),
                    source_name=filename,
                    warnings=warnings,
                )
            )

        return MultiAngleQueryTemplate(poses=poses), skipped

    def build_reference_from_video(
        self,
        video_bytes: bytes,
        filename: str = "head_turn.webm",
    ) -> tuple[MultiAngleQueryTemplate, dict[str, Any]]:
        """
        Process a short head-turn video (Left <-> Front <-> Right):
        Samples frames, estimates head yaw + Laplacian sharpness, and automatically selects
        the sharpest crisp frame in each pose bucket (left, front, right).
        """
        suffix = Path(filename).suffix.lower()
        if suffix not in {".mp4", ".webm", ".mov", ".avi", ".mkv"}:
            suffix = ".webm"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = Path(tmp.name)

        try:
            cap = cv2.VideoCapture(str(tmp_path))
            if not cap.isOpened():
                raise ValueError("Could not decode uploaded video file.")

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            # Some WebM recordings omit FPS/frame_count metadata; fall back safely
            valid_fps = fps if 1.0 <= fps <= 120.0 else 30.0
            est_duration = (frame_count / valid_fps) if frame_count > 0 else 0.0

            if est_duration > self.cfg.max_video_seconds + 3.0:
                cap.release()
                raise ValueError(
                    f"Video is {est_duration:.1f}s long (limit is {self.cfg.max_video_seconds:.0f}s)."
                )

            # Sample ~30–40 frames evenly across the video
            stride = max(1, int(round(valid_fps / 5.0)))  # ~5 frames per second
            max_frames_to_scan = int(valid_fps * (self.cfg.max_video_seconds + 2.0))

            candidates: list[tuple[DetectedFace, int]] = []
            frame_idx = 0
            sampled = 0
            while frame_idx <= max_frames_to_scan and sampled < 48:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % stride == 0:
                    sampled += 1
                    faces = self.analyze_image(frame, det_thresh=0.50, extract_outfit=False)
                    if faces:
                        candidates.append((faces[0], frame_idx))
                frame_idx += 1

            cap.release()
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        if not candidates:
            return MultiAngleQueryTemplate(poses=[]), {
                "frames_sampled": sampled,
                "frames_with_face": 0,
                "buckets_filled": [],
            }

        # Bin candidates into left / front / right and select highest sharpness * quality_norm
        buckets: dict[str, list[tuple[DetectedFace, int]]] = {
            "left": [],
            "front": [],
            "right": [],
        }
        for face, fidx in candidates:
            buckets[face.pose_label].append((face, fidx))

        selected_poses: list[ReferencePose] = []
        for bucket_name in ("left", "front", "right"):
            items = buckets[bucket_name]
            if not items:
                continue
            # Prefer sharp, high-quality frames near ideal yaw for each bin
            ideal_yaw = {"left": -30.0, "front": 0.0, "right": 30.0}[bucket_name]

            def rank_key(pair: tuple[DetectedFace, int]) -> float:
                f = pair[0]
                sharp = min(f.blur_score, 300.0) / 300.0
                qual = min(f.quality_norm, 30.0) / 30.0
                yaw_penalty = abs(f.yaw_deg - ideal_yaw) / 45.0
                return 0.45 * sharp + 0.45 * qual - 0.10 * yaw_penalty

            best_face, best_fidx = max(items, key=rank_key)
            warnings: list[str] = []
            if best_face.blur_score < self.cfg.min_blur_variance:
                warnings.append("Selected video frame had slight motion blur.")

            selected_poses.append(
                ReferencePose(
                    label=bucket_name,
                    yaw_deg=round(best_face.yaw_deg, 1),
                    blur_score=round(best_face.blur_score, 1),
                    quality_norm=round(best_face.quality_norm, 2),
                    det_score=round(best_face.det_score, 3),
                    embedding=best_face.embedding,
                    preview_b64=encode_preview_b64(best_face.aligned_bgr),
                    source_name=f"{filename} (frame {best_fidx})",
                    warnings=warnings,
                )
            )

        # If user didn't turn enough to fill multiple buckets, keep up to 3 sharpest distinct frames
        if len(selected_poses) == 1 and len(candidates) >= 3:
            sorted_all = sorted(
                candidates,
                key=lambda p: (min(p[0].blur_score, 300.0) + p[0].quality_norm * 10.0),
                reverse=True,
            )
            chosen_fidxs = {selected_poses[0].source_name}
            for face, fidx in sorted_all:
                src_label = f"{filename} (frame {fidx})"
                if src_label in chosen_fidxs:
                    continue
                selected_poses.append(
                    ReferencePose(
                        label=f"angle_{len(selected_poses) + 1}",
                        yaw_deg=round(face.yaw_deg, 1),
                        blur_score=round(face.blur_score, 1),
                        quality_norm=round(face.quality_norm, 2),
                        det_score=round(face.det_score, 3),
                        embedding=face.embedding,
                        preview_b64=encode_preview_b64(face.aligned_bgr),
                        source_name=src_label,
                        warnings=["Limited head rotation detected; picked additional crisp frame."],
                    )
                )
                chosen_fidxs.add(src_label)
                if len(selected_poses) >= 3:
                    break

        meta = {
            "frames_sampled": sampled,
            "frames_with_face": len(candidates),
            "buckets_filled": [p.label for p in selected_poses],
        }
        return MultiAngleQueryTemplate(poses=selected_poses), meta
