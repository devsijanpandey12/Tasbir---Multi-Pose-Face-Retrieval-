"""FastAPI application for Tasbir — Multi-Angle Event & Group Photo Finder."""

from __future__ import annotations

import hashlib
import io
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings, settings
from .db import PhotoPoolStore, PhotoRecord
from .matcher import MatchExecutionReport, MultiAngleQueryTemplate, match_pool
from .vision import VisionPipeline, save_original_and_thumb


ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
ALLOWED_VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}


# --- Pydantic Response Schemas ---


class FaceBoxOut(BaseModel):
    id: str
    bbox_xyxy: list[float]
    bbox_norm: list[float]
    det_score: float
    quality_norm: float
    blur_score: float
    pose_label: str


class PhotoOut(BaseModel):
    id: str
    filename: str
    thumb_url: str
    original_url: str
    width: int
    height: int
    face_count: int
    uploaded_at: str
    faces: list[FaceBoxOut] = []


class PoolUploadItemResult(BaseModel):
    filename: str
    status: str  # "indexed" | "duplicate" | "error"
    photo_id: str | None = None
    face_count: int = 0
    message: str | None = None


class PoolUploadResponse(BaseModel):
    indexed_count: int
    duplicate_count: int
    error_count: int
    total_faces_added: int
    processing_ms: float
    items: list[PoolUploadItemResult]
    pool_stats: dict[str, int]


class ReferencePoseOut(BaseModel):
    label: str
    yaw_deg: float
    blur_score: float
    quality_norm: float
    det_score: float
    preview_b64: str
    source_name: str
    warnings: list[str] = []


class MatchedPhotoOut(BaseModel):
    photo: PhotoOut
    score: float
    face_score: float
    outfit_score: float | None = None
    matched_by: str
    matched_face_id: str
    bbox_xyxy: list[float]
    bbox_norm: list[float]
    pass_stage: int


class MatchResponse(BaseModel):
    threshold: float
    enable_pass2: bool
    search_ms: float
    extraction_ms: float
    total_faces_searched: int
    pass1_count: int
    pass2_recovered_count: int
    references: list[ReferencePoseOut]
    skipped_references: list[dict[str, Any]] = []
    video_meta: dict[str, Any] | None = None
    matches: list[MatchedPhotoOut]


class ZipDownloadRequest(BaseModel):
    photo_ids: list[str] = Field(..., min_length=1, max_length=500)


# --- App Factory ---


def _photo_to_out(rec: PhotoRecord) -> PhotoOut:
    thumb_name = Path(rec.thumb_path).name
    orig_name = Path(rec.file_path).name
    return PhotoOut(
        id=rec.id,
        filename=rec.filename,
        thumb_url=f"/media/thumbs/{thumb_name}",
        original_url=f"/media/originals/{orig_name}",
        width=rec.width,
        height=rec.height,
        face_count=rec.face_count,
        uploaded_at=rec.uploaded_at,
        faces=[
            FaceBoxOut(
                id=f.id,
                bbox_xyxy=f.bbox_xyxy,
                bbox_norm=f.bbox_norm,
                det_score=round(f.det_score, 3),
                quality_norm=round(f.quality_norm, 2),
                blur_score=round(f.blur_score, 1),
                pose_label=f.pose_label,
            )
            for f in rec.faces
        ],
    )


def _build_match_response(
    *,
    query: MultiAngleQueryTemplate,
    report: MatchExecutionReport,
    store: PhotoPoolStore,
    threshold: float,
    enable_pass2: bool,
    extraction_ms: float,
    skipped_references: list[dict[str, Any]] | None = None,
    video_meta: dict[str, Any] | None = None,
) -> MatchResponse:
    matched_ids = [m.photo_id for m in report.matches]
    photos_map = store.get_photos_by_ids(matched_ids)

    match_items: list[MatchedPhotoOut] = []
    for m in report.matches:
        photo = photos_map.get(m.photo_id)
        if not photo:
            continue
        match_items.append(
            MatchedPhotoOut(
                photo=_photo_to_out(photo),
                score=round(m.score, 4),
                face_score=round(m.face_score, 4),
                outfit_score=round(m.outfit_score, 4) if m.outfit_score is not None else None,
                matched_by=m.matched_by,
                matched_face_id=m.matched_face_id,
                bbox_xyxy=m.bbox_xyxy,
                bbox_norm=m.bbox_norm,
                pass_stage=m.pass_stage,
            )
        )

    return MatchResponse(
        threshold=threshold,
        enable_pass2=enable_pass2,
        search_ms=round(report.search_ms, 2),
        extraction_ms=round(extraction_ms, 1),
        total_faces_searched=report.total_faces_searched,
        pass1_count=report.pass1_count,
        pass2_recovered_count=report.pass2_recovered_count,
        references=[
            ReferencePoseOut(
                label=p.label,
                yaw_deg=p.yaw_deg,
                blur_score=p.blur_score,
                quality_norm=p.quality_norm,
                det_score=p.det_score,
                preview_b64=p.preview_b64,
                source_name=p.source_name,
                warnings=p.warnings,
            )
            for p in query.poses
        ],
        skipped_references=skipped_references or [],
        video_meta=video_meta,
        matches=match_items,
    )


def create_app(
    custom_store: PhotoPoolStore | None = None,
    custom_vision: VisionPipeline | None = None,
    custom_settings: Settings | None = None,
) -> FastAPI:
    cfg = custom_settings or settings
    cfg.ensure_dirs()
    store = custom_store or PhotoPoolStore(
        db_path=cfg.db_path,
        embeddings_path=cfg.embeddings_path,
        outfit_embeddings_path=cfg.outfit_embeddings_path,
        nolock=cfg.sqlite_nolock,
    )
    vision = custom_vision or VisionPipeline(cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        cfg.ensure_dirs()
        vision.load()
        yield

    app = FastAPI(
        title=cfg.app_name,
        version=cfg.app_version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve stored originals and thumbnails
    app.mount(
        "/media/originals",
        StaticFiles(directory=str(cfg.originals_dir)),
        name="originals",
    )
    app.mount(
        "/media/thumbs",
        StaticFiles(directory=str(cfg.thumbs_dir)),
        name="thumbs",
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        vision.load()
        return {
            "status": "ok",
            "app": cfg.app_name,
            "version": cfg.app_version,
            "models": {
                "detector": "SCRFD-10G (det_10g.onnx)",
                "embedder": "ArcFace ResNet-50 WebFace600K (w600k_r50.onnx)",
                "provider": vision.active_provider,
            },
            "limits": {
                "max_pool_batch": cfg.max_pool_batch,
                "max_photo_size_mb": cfg.max_photo_size_mb,
                "max_video_size_mb": cfg.max_video_size_mb,
                "max_video_seconds": cfg.max_video_seconds,
                "max_reference_photos": cfg.max_reference_photos,
                "default_threshold": cfg.default_threshold,
            },
            "stats": store.get_stats(),
        }

    @app.get("/api/pool")
    def list_pool() -> dict[str, Any]:
        photos = store.list_photos()
        return {
            "stats": store.get_stats(),
            "photos": [_photo_to_out(p) for p in photos],
        }

    @app.post("/api/pool/upload", response_model=PoolUploadResponse)
    async def upload_pool(files: list[UploadFile] = File(...)) -> PoolUploadResponse:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided.")
        if len(files) > cfg.max_pool_batch:
            raise HTTPException(
                status_code=400,
                detail=f"Batch exceeds limit of {cfg.max_pool_batch} photos per upload.",
            )

        t0 = time.perf_counter()
        max_bytes = cfg.max_photo_size_mb * 1024 * 1024
        results: list[PoolUploadItemResult] = []
        indexed_count = 0
        duplicate_count = 0
        error_count = 0
        total_faces_added = 0

        for upload in files:
            fname = upload.filename or "photo.jpg"
            ext = Path(fname).suffix.lower()
            if ext and ext not in ALLOWED_IMAGE_EXTS:
                error_count += 1
                results.append(
                    PoolUploadItemResult(
                        filename=fname,
                        status="error",
                        message=f"Unsupported image extension '{ext}'.",
                    )
                )
                continue

            raw_bytes = await upload.read()
            if len(raw_bytes) == 0:
                error_count += 1
                results.append(
                    PoolUploadItemResult(
                        filename=fname, status="error", message="Empty file."
                    )
                )
                continue

            if len(raw_bytes) > max_bytes:
                error_count += 1
                results.append(
                    PoolUploadItemResult(
                        filename=fname,
                        status="error",
                        message=f"File exceeds {cfg.max_photo_size_mb} MB limit.",
                    )
                )
                continue

            file_hash = hashlib.sha256(raw_bytes).hexdigest()
            existing_id = store.has_file_hash(file_hash)
            if existing_id is not None:
                duplicate_count += 1
                results.append(
                    PoolUploadItemResult(
                        filename=fname,
                        status="duplicate",
                        photo_id=existing_id,
                        message="Already indexed in photo pool.",
                    )
                )
                continue

            photo_id = str(uuid.uuid4())
            orig_path = cfg.originals_dir / f"{photo_id}.jpg"
            thumb_path = cfg.thumbs_dir / f"{photo_id}.webp"

            try:
                bgr, width, height = save_original_and_thumb(
                    raw_bytes,
                    orig_path,
                    thumb_path,
                    thumb_max_dim=cfg.thumb_max_dim,
                )
                detected = vision.analyze_image(bgr, extract_outfit=True)
                face_dicts = [
                    {
                        "bbox_xyxy": f.bbox_xyxy,
                        "bbox_norm": f.bbox_norm,
                        "det_score": f.det_score,
                        "quality_norm": f.quality_norm,
                        "blur_score": f.blur_score,
                        "pose_label": f.pose_label,
                        "embedding": f.embedding,
                        "outfit_embedding": f.outfit_embedding,
                    }
                    for f in detected
                ]
                store.add_photo(
                    photo_id=photo_id,
                    filename=fname,
                    file_hash=file_hash,
                    file_path=str(orig_path),
                    thumb_path=str(thumb_path),
                    width=width,
                    height=height,
                    faces=face_dicts,
                )
                indexed_count += 1
                total_faces_added += len(face_dicts)
                results.append(
                    PoolUploadItemResult(
                        filename=fname,
                        status="indexed",
                        photo_id=photo_id,
                        face_count=len(face_dicts),
                    )
                )
            except Exception as exc:
                orig_path.unlink(missing_ok=True)
                thumb_path.unlink(missing_ok=True)
                error_count += 1
                results.append(
                    PoolUploadItemResult(
                        filename=fname,
                        status="error",
                        message=f"Processing failed: {exc}",
                    )
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PoolUploadResponse(
            indexed_count=indexed_count,
            duplicate_count=duplicate_count,
            error_count=error_count,
            total_faces_added=total_faces_added,
            processing_ms=round(elapsed_ms, 1),
            items=results,
            pool_stats=store.get_stats(),
        )

    @app.delete("/api/pool/{photo_id}")
    def delete_pool_photo(photo_id: str) -> dict[str, Any]:
        removed = store.delete_photo(photo_id)
        if removed is None:
            raise HTTPException(status_code=404, detail="Photo not found.")
        orig_path, thumb_path = removed
        Path(orig_path).unlink(missing_ok=True)
        Path(thumb_path).unlink(missing_ok=True)
        return {"deleted": photo_id, "stats": store.get_stats()}

    @app.delete("/api/pool")
    def clear_pool() -> dict[str, Any]:
        removed = store.clear_all()
        for orig_path, thumb_path in removed:
            Path(orig_path).unlink(missing_ok=True)
            Path(thumb_path).unlink(missing_ok=True)
        return {"cleared_count": len(removed), "stats": store.get_stats()}

    @app.post("/api/match/photos", response_model=MatchResponse)
    async def match_by_photos(
        files: list[UploadFile] = File(...),
        angles: list[str] | None = Form(default=None),
        threshold: float = Form(default=cfg.default_threshold),
        enable_pass2: bool = Form(default=True),
    ) -> MatchResponse:
        if not files:
            raise HTTPException(status_code=400, detail="Upload at least 1 reference photo.")
        if len(files) > cfg.max_reference_photos:
            raise HTTPException(
                status_code=400,
                detail=f"Upload at most {cfg.max_reference_photos} reference photos.",
            )
        threshold = max(0.15, min(0.85, float(threshold)))
        max_bytes = cfg.max_photo_size_mb * 1024 * 1024

        t_ext = time.perf_counter()
        prepared: list[tuple[str, bytes, str | None]] = []
        for idx, upload in enumerate(files):
            raw = await upload.read()
            if len(raw) == 0 or len(raw) > max_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Reference image '{upload.filename}' must be between 1 B and {cfg.max_photo_size_mb} MB.",
                )
            hint = angles[idx].strip().lower() if angles and idx < len(angles) else None
            prepared.append((upload.filename or f"ref_{idx + 1}.jpg", raw, hint))

        query, skipped = vision.build_reference_from_images(prepared)
        extraction_ms = (time.perf_counter() - t_ext) * 1000.0

        if not query.poses:
            raise HTTPException(
                status_code=422,
                detail="No usable face detected in the uploaded reference photos. Please upload clear face photos.",
            )

        face_records, face_embeddings, outfit_embeddings = store.get_search_snapshot()
        report = match_pool(
            query,
            face_records,
            face_embeddings,
            outfit_embeddings,
            threshold=threshold,
            enable_pass2=enable_pass2,
            pass2_anchor_threshold=cfg.pass2_anchor_threshold,
            pass2_min_face_score=cfg.pass2_min_face_score,
            pass2_min_outfit_score=cfg.pass2_min_outfit_score,
            pass2_combined_threshold=cfg.pass2_combined_threshold,
        )
        return _build_match_response(
            query=query,
            report=report,
            store=store,
            threshold=threshold,
            enable_pass2=enable_pass2,
            extraction_ms=extraction_ms,
            skipped_references=skipped,
        )

    @app.post("/api/match/video", response_model=MatchResponse)
    async def match_by_video(
        video: UploadFile = File(...),
        threshold: float = Form(default=cfg.default_threshold),
        enable_pass2: bool = Form(default=True),
    ) -> MatchResponse:
        fname = video.filename or "head_turn.webm"
        ext = Path(fname).suffix.lower()
        if ext and ext not in ALLOWED_VIDEO_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported video format '{ext}'. Use MP4, WebM, or MOV.",
            )

        raw = await video.read()
        max_bytes = cfg.max_video_size_mb * 1024 * 1024
        if len(raw) == 0 or len(raw) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"Video must be between 1 B and {cfg.max_video_size_mb} MB.",
            )

        threshold = max(0.15, min(0.85, float(threshold)))
        t_ext = time.perf_counter()
        try:
            query, video_meta = vision.build_reference_from_video(raw, filename=fname)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        extraction_ms = (time.perf_counter() - t_ext) * 1000.0

        if not query.poses:
            raise HTTPException(
                status_code=422,
                detail="No usable face frames found in the uploaded video. Try better lighting and face the camera.",
            )

        face_records, face_embeddings, outfit_embeddings = store.get_search_snapshot()
        report = match_pool(
            query,
            face_records,
            face_embeddings,
            outfit_embeddings,
            threshold=threshold,
            enable_pass2=enable_pass2,
            pass2_anchor_threshold=cfg.pass2_anchor_threshold,
            pass2_min_face_score=cfg.pass2_min_face_score,
            pass2_min_outfit_score=cfg.pass2_min_outfit_score,
            pass2_combined_threshold=cfg.pass2_combined_threshold,
        )
        return _build_match_response(
            query=query,
            report=report,
            store=store,
            threshold=threshold,
            enable_pass2=enable_pass2,
            extraction_ms=extraction_ms,
            video_meta=video_meta,
        )

    @app.post("/api/download-zip")
    def download_zip(payload: ZipDownloadRequest) -> StreamingResponse:
        photos_map = store.get_photos_by_ids(payload.photo_ids)
        if not photos_map:
            raise HTTPException(status_code=404, detail="No matching photos found to download.")

        buf = io.BytesIO()
        used_names: dict[str, int] = {}
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for pid in payload.photo_ids:
                rec = photos_map.get(pid)
                if not rec:
                    continue
                fp = Path(rec.file_path)
                if not fp.exists():
                    continue
                base_name = Path(rec.filename).name
                if not base_name.lower().endswith(".jpg"):
                    base_name = f"{Path(base_name).stem}.jpg"
                count = used_names.get(base_name, 0)
                used_names[base_name] = count + 1
                arc_name = (
                    base_name
                    if count == 0
                    else f"{Path(base_name).stem}_{count}{Path(base_name).suffix}"
                )
                zf.write(fp, arcname=arc_name)

        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="tasbir_matched_photos.zip"'},
        )

    return app


app = create_app()
