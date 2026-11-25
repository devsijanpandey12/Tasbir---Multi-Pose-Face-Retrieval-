"""Automated end-to-end tests for Tasbir database, matcher, ONNX vision engine, and FastAPI routes."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.db import FACE_DIM, OUTFIT_DIM, FaceRecord, PhotoPoolStore, SessionStoreManager
from app.main import create_app
from app.matcher import MultiAngleQueryTemplate, ReferencePose, match_pool
from app.vision import VisionPipeline, compute_blur_score, compute_torso_descriptor, estimate_yaw


def _unit_vec(seed: int, dim: int = FACE_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _blend_vec(a: np.ndarray, b: np.ndarray, target_cos: float) -> np.ndarray:
    """Construct a unit vector having exact cosine similarity `target_cos` with `a`."""
    b_orth = b - float(np.dot(b, a)) * a
    b_orth = b_orth / np.linalg.norm(b_orth)
    v = target_cos * a + np.sqrt(max(0.0, 1.0 - target_cos**2)) * b_orth
    return (v / np.linalg.norm(v)).astype(np.float32)


def _make_jpeg_bytes(width: int = 320, height: int = 240, color=(120, 140, 160)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_store_crud_dedup_and_compaction(tmp_path: Path) -> None:
    store = PhotoPoolStore(
        db_path=tmp_path / "test.db",
        embeddings_path=tmp_path / "emb.npy",
        outfit_embeddings_path=tmp_path / "outfit.npy",
    )

    v1 = _unit_vec(1)
    v2 = _unit_vec(2)
    v3 = _unit_vec(3)
    o1 = _unit_vec(101, OUTFIT_DIM)

    store.add_photo(
        photo_id="p1",
        filename="wedding_1.jpg",
        file_hash="hash_1",
        file_path=str(tmp_path / "p1.jpg"),
        thumb_path=str(tmp_path / "p1.webp"),
        width=1000,
        height=800,
        faces=[
            {
                "bbox_xyxy": [10, 10, 100, 100],
                "bbox_norm": [0.01, 0.01, 0.1, 0.12],
                "det_score": 0.95,
                "quality_norm": 23.0,
                "blur_score": 120.0,
                "pose_label": "front",
                "embedding": v1,
                "outfit_embedding": o1,
            },
            {
                "bbox_xyxy": [200, 10, 300, 100],
                "bbox_norm": [0.2, 0.01, 0.3, 0.12],
                "det_score": 0.91,
                "quality_norm": 21.0,
                "blur_score": 95.0,
                "pose_label": "left",
                "embedding": v2,
                "outfit_embedding": o1,
            },
        ],
    )
    store.add_photo(
        photo_id="p2",
        filename="wedding_2.jpg",
        file_hash="hash_2",
        file_path=str(tmp_path / "p2.jpg"),
        thumb_path=str(tmp_path / "p2.webp"),
        width=1000,
        height=800,
        faces=[
            {
                "bbox_xyxy": [50, 50, 160, 160],
                "bbox_norm": [0.05, 0.06, 0.16, 0.2],
                "det_score": 0.93,
                "quality_norm": 24.5,
                "blur_score": 150.0,
                "pose_label": "right",
                "embedding": v3,
                "outfit_embedding": o1,
            }
        ],
    )

    assert store.has_file_hash("hash_1") == "p1"
    assert store.get_stats()["total_photos"] == 2
    assert store.get_stats()["total_faces"] == 3

    # Delete p1 and ensure p2's face vector compacted cleanly to index 0
    store.delete_photo("p1")
    recs, f_embs, _ = store.get_search_snapshot()
    assert len(recs) == 1
    assert recs[0].photo_id == "p2"
    assert recs[0].vector_index == 0
    assert np.allclose(f_embs[0], v3, atol=1e-5)

    store.clear_all()
    assert store.get_stats()["total_photos"] == 0
    assert store.get_stats()["total_faces"] == 0


def test_multi_pose_max_cosine_and_pass2_outfit_recovery() -> None:
    # Person A's 3 reference angles
    p_front = _unit_vec(10)
    p_left = _unit_vec(11)
    p_right = _unit_vec(12)
    stranger = _unit_vec(99)

    outfit_a = _unit_vec(200, OUTFIT_DIM)
    outfit_other = _unit_vec(201, OUTFIT_DIM)

    query = MultiAngleQueryTemplate(
        poses=[
            ReferencePose("front", 0.0, 150.0, 24.0, 0.96, p_front, "", "front.jpg"),
            ReferencePose("left", -30.0, 140.0, 22.0, 0.94, p_left, "", "left.jpg"),
            ReferencePose("right", 32.0, 145.0, 23.0, 0.95, p_right, "", "right.jpg"),
        ]
    )

    # Pool faces:
    # 0: clear frontal shot of Person A (Pass-1 anchor)
    # 1: candid side-profile shot matching p_left (Pass-1 via left template!)
    # 2: heavily occluded sunglasses shot of Person A (face sim ~0.34, same outfit -> Pass-2 recovery!)
    # 3: stranger wearing different outfit (should NOT match)
    f0 = _blend_vec(p_front, _unit_vec(30), 0.90)
    f1 = _blend_vec(p_left, _unit_vec(31), 0.78)
    f2 = _blend_vec(p_front, _unit_vec(32), 0.36)
    f3 = stranger

    records = [
        FaceRecord("f0", "photo_anchor", [0, 0, 10, 10], [0, 0, 0.1, 0.1], 0.95, 24.0, 120.0, "front", 0),
        FaceRecord("f1", "photo_side", [0, 0, 10, 10], [0, 0, 0.1, 0.1], 0.91, 21.0, 110.0, "left", 1),
        FaceRecord("f2", "photo_sunglasses", [0, 0, 10, 10], [0, 0, 0.1, 0.1], 0.85, 16.0, 90.0, "front", 2),
        FaceRecord("f3", "photo_stranger", [0, 0, 10, 10], [0, 0, 0.1, 0.1], 0.96, 25.0, 130.0, "front", 3),
    ]
    face_embs = np.vstack([f0, f1, f2, f3]).astype(np.float32)
    outfit_embs = np.vstack([outfit_a, outfit_a, outfit_a, outfit_other]).astype(np.float32)

    report = match_pool(
        query,
        records,
        face_embs,
        outfit_embs,
        threshold=0.42,
        enable_pass2=True,
    )
    matched_ids = [m.photo_id for m in report.matches]
    assert "photo_anchor" in matched_ids
    assert "photo_side" in matched_ids
    assert "photo_sunglasses" in matched_ids
    assert "photo_stranger" not in matched_ids
    assert report.pass1_count == 2
    assert report.pass2_recovered_count == 1

    # Verify disabling Pass-2 leaves out the occluded sunglasses photo
    report_p1_only = match_pool(
        query, records, face_embs, outfit_embs, threshold=0.42, enable_pass2=False
    )
    assert [m.photo_id for m in report_p1_only.matches] == ["photo_anchor", "photo_side"]


def test_yaw_blur_and_torso_helpers() -> None:
    # Centered nose -> front
    kps_front = np.array([[38, 50], [74, 50], [56, 70], [42, 92], [70, 92]], dtype=np.float32)
    yaw_f, label_f = estimate_yaw(kps_front)
    assert label_f == "front" and abs(yaw_f) < 5.0

    # Nose shifted left -> left
    kps_left = np.array([[38, 50], [74, 50], [44, 70], [40, 92], [66, 92]], dtype=np.float32)
    yaw_l, label_l = estimate_yaw(kps_left)
    assert label_l == "left" and yaw_l < -14.0

    # Nose shifted right -> right
    kps_right = np.array([[38, 50], [74, 50], [68, 70], [46, 92], [72, 92]], dtype=np.float32)
    yaw_r, label_r = estimate_yaw(kps_right)
    assert label_r == "right" and yaw_r > 14.0

    # Sharp checkerboard has much higher Laplacian variance than flat image
    flat = np.full((112, 112, 3), 128, dtype=np.uint8)
    checker = np.zeros((112, 112, 3), dtype=np.uint8)
    checker[::2, ::2] = 255
    assert compute_blur_score(checker) > compute_blur_score(flat) + 100.0

    # Torso descriptor returns L2-normalized 96-D vector when torso is visible
    scene = np.full((400, 300, 3), (40, 90, 200), dtype=np.uint8)
    desc = compute_torso_descriptor(scene, [100, 40, 180, 120])
    assert desc.shape == (OUTFIT_DIM,)
    assert abs(float(np.linalg.norm(desc)) - 1.0) < 1e-4


def test_onnx_models_load_and_api_endpoints(tmp_path: Path) -> None:
    cfg = Settings(data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    vision = VisionPipeline(cfg)
    vision.load()
    assert vision.det_session is not None
    assert vision.rec_session is not None

    # Verify ArcFace embedding session produces normalized 512-D output
    dummy_crop = np.full((112, 112, 3), 120, dtype=np.uint8)
    emb, raw_norm = vision.embed_aligned(dummy_crop)
    assert emb.shape == (512,)
    assert raw_norm > 0.0
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4

    # Verify FastAPI endpoints with isolated ephemeral session manager
    session_mgr = SessionStoreManager(
        sessions_root=cfg.sessions_dir,
        ttl_seconds=60,
    )
    app = create_app(
        custom_session_manager=session_mgr,
        custom_vision=vision,
        custom_settings=cfg,
    )

    sess_a = {"X-Session-Id": "session_alpha_12345"}
    sess_b = {"X-Session-Id": "session_beta_67890"}

    with TestClient(app) as client:
        r_health = client.get("/api/health", headers=sess_a)
        assert r_health.status_code == 200
        body = r_health.json()
        assert body["status"] == "ok"
        assert body["ephemeral"] is True
        assert body["stats"]["total_photos"] == 0

        # Upload a synthetic blank photo into session A -> isolated from session B
        img_bytes = _make_jpeg_bytes()
        r_up = client.post(
            "/api/pool/upload",
            headers=sess_a,
            files=[("files", ("blank.jpg", img_bytes, "image/jpeg"))],
        )
        assert r_up.status_code == 200
        up_data = r_up.json()
        assert up_data["indexed_count"] == 1
        assert up_data["items"][0]["status"] == "indexed"
        photo_id = up_data["items"][0]["photo_id"]

        # Session B must start completely empty (cannot see Session A's photos)
        r_pool_b = client.get("/api/pool", headers=sess_b)
        assert r_pool_b.status_code == 200
        assert r_pool_b.json()["stats"]["total_photos"] == 0
        assert r_pool_b.json()["photos"] == []

        # Duplicate upload in same session A is deduplicated
        r_dup = client.post(
            "/api/pool/upload",
            headers=sess_a,
            files=[("files", ("blank.jpg", img_bytes, "image/jpeg"))],
        )
        assert r_dup.status_code == 200
        assert r_dup.json()["duplicate_count"] == 1

        # ZIP download works for Session A and returns 404 for Session B
        r_zip = client.post(
            "/api/download-zip", headers=sess_a, json={"photo_ids": [photo_id]}
        )
        assert r_zip.status_code == 200
        assert r_zip.headers["content-type"] == "application/zip"

        r_zip_b = client.post(
            "/api/download-zip", headers=sess_b, json={"photo_ids": [photo_id]}
        )
        assert r_zip_b.status_code == 404

        # Matching with a blank reference returns 422 cleanly (no face detected)
        r_match = client.post(
            "/api/match/photos",
            headers=sess_a,
            files=[("files", ("no_face.jpg", img_bytes, "image/jpeg"))],
            data={"threshold": "0.42", "enable_pass2": "true"},
        )
        assert r_match.status_code == 422

        # Cleanup endpoint (used by sendBeacon on tab close) wipes session A completely
        r_clean = client.post(
            "/api/session/cleanup?session_id=session_alpha_12345"
        )
        assert r_clean.status_code == 200
        assert r_clean.json()["cleaned"] is True
        assert client.get("/api/pool", headers=sess_a).json()["stats"]["total_photos"] == 0
        assert not (cfg.sessions_dir / "session_alpha_12345").exists()
