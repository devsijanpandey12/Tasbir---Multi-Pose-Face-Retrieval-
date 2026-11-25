"""SQLite metadata store and atomic NumPy vector persistence for Tasbir."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


FACE_DIM = 512
OUTFIT_DIM = 96


@dataclass
class FaceRecord:
    id: str
    photo_id: str
    bbox_xyxy: list[float]
    bbox_norm: list[float]
    det_score: float
    quality_norm: float
    blur_score: float
    pose_label: str
    vector_index: int


@dataclass
class PhotoRecord:
    id: str
    filename: str
    file_hash: str
    file_path: str
    thumb_path: str
    width: int
    height: int
    face_count: int
    uploaded_at: str
    faces: list[FaceRecord]


class PhotoPoolStore:
    """Manages SQLite photo/face metadata synchronized with local .npy vector files."""

    def __init__(
        self,
        db_path: Path,
        embeddings_path: Path,
        outfit_embeddings_path: Path,
    ) -> None:
        self.db_path = db_path
        self.embeddings_path = embeddings_path
        self.outfit_embeddings_path = outfit_embeddings_path
        self._lock = threading.RLock()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._face_embeddings = self._load_matrix(self.embeddings_path, FACE_DIM)
        self._outfit_embeddings = self._load_matrix(
            self.outfit_embeddings_path, OUTFIT_DIM
        )
        self._verify_index_integrity()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS photos (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_hash TEXT UNIQUE NOT NULL,
                    file_path TEXT NOT NULL,
                    thumb_path TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    face_count INTEGER NOT NULL DEFAULT 0,
                    uploaded_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS detected_faces (
                    id TEXT PRIMARY KEY,
                    photo_id TEXT NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
                    bbox_xyxy TEXT NOT NULL,
                    bbox_norm TEXT NOT NULL,
                    det_score REAL NOT NULL,
                    quality_norm REAL NOT NULL,
                    blur_score REAL NOT NULL,
                    pose_label TEXT NOT NULL,
                    vector_index INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_detected_faces_photo_id
                    ON detected_faces(photo_id);
                CREATE INDEX IF NOT EXISTS idx_detected_faces_vector_index
                    ON detected_faces(vector_index);
                """
            )

    @staticmethod
    def _load_matrix(path: Path, dim: int) -> np.ndarray:
        if path.exists():
            arr = np.load(path)
            if arr.ndim == 2 and arr.shape[1] == dim:
                return arr.astype(np.float32, copy=False)
        return np.empty((0, dim), dtype=np.float32)

    @staticmethod
    def _save_matrix(path: Path, matrix: np.ndarray) -> None:
        tmp_path = path.with_suffix(".tmp.npy")
        np.save(tmp_path, matrix.astype(np.float32, copy=False))
        tmp_path.replace(path)

    def _verify_index_integrity(self) -> None:
        """Ensure vector array lengths match detected_faces rows."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM detected_faces").fetchone()
            db_count = int(row["cnt"]) if row else 0
            if (
                db_count == 0
                and (len(self._face_embeddings) > 0 or len(self._outfit_embeddings) > 0)
            ):
                self._face_embeddings = np.empty((0, FACE_DIM), dtype=np.float32)
                self._outfit_embeddings = np.empty((0, OUTFIT_DIM), dtype=np.float32)
                self._save_matrix(self.embeddings_path, self._face_embeddings)
                self._save_matrix(self.outfit_embeddings_path, self._outfit_embeddings)

    def has_file_hash(self, file_hash: str) -> str | None:
        """Return existing photo_id if a photo with this SHA-256 hash already exists."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM photos WHERE file_hash = ?", (file_hash,)
            ).fetchone()
            return str(row["id"]) if row else None

    def add_photo(
        self,
        *,
        photo_id: str,
        filename: str,
        file_hash: str,
        file_path: str,
        thumb_path: str,
        width: int,
        height: int,
        faces: list[dict[str, Any]],
    ) -> PhotoRecord:
        """Insert a photo and its detected faces + embeddings atomically."""
        uploaded_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            base_idx = len(self._face_embeddings)
            new_face_vecs: list[np.ndarray] = []
            new_outfit_vecs: list[np.ndarray] = []
            face_records: list[FaceRecord] = []

            for offset, face in enumerate(faces):
                vec_idx = base_idx + offset
                face_id = str(uuid.uuid4())
                face_vec = np.asarray(face["embedding"], dtype=np.float32).reshape(FACE_DIM)
                outfit_vec = np.asarray(
                    face.get("outfit_embedding", np.zeros(OUTFIT_DIM, dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(OUTFIT_DIM)
                new_face_vecs.append(face_vec)
                new_outfit_vecs.append(outfit_vec)
                face_records.append(
                    FaceRecord(
                        id=face_id,
                        photo_id=photo_id,
                        bbox_xyxy=[float(v) for v in face["bbox_xyxy"]],
                        bbox_norm=[float(v) for v in face["bbox_norm"]],
                        det_score=float(face["det_score"]),
                        quality_norm=float(face["quality_norm"]),
                        blur_score=float(face["blur_score"]),
                        pose_label=str(face["pose_label"]),
                        vector_index=vec_idx,
                    )
                )

            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO photos (
                        id, filename, file_hash, file_path, thumb_path,
                        width, height, face_count, uploaded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        photo_id,
                        filename,
                        file_hash,
                        file_path,
                        thumb_path,
                        width,
                        height,
                        len(face_records),
                        uploaded_at,
                    ),
                )
                for fr in face_records:
                    conn.execute(
                        """
                        INSERT INTO detected_faces (
                            id, photo_id, bbox_xyxy, bbox_norm, det_score,
                            quality_norm, blur_score, pose_label, vector_index
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            fr.id,
                            fr.photo_id,
                            json.dumps(fr.bbox_xyxy),
                            json.dumps(fr.bbox_norm),
                            fr.det_score,
                            fr.quality_norm,
                            fr.blur_score,
                            fr.pose_label,
                            fr.vector_index,
                        ),
                    )

            if new_face_vecs:
                face_block = np.vstack(new_face_vecs).astype(np.float32)
                outfit_block = np.vstack(new_outfit_vecs).astype(np.float32)
                self._face_embeddings = (
                    np.vstack([self._face_embeddings, face_block])
                    if len(self._face_embeddings) > 0
                    else face_block
                )
                self._outfit_embeddings = (
                    np.vstack([self._outfit_embeddings, outfit_block])
                    if len(self._outfit_embeddings) > 0
                    else outfit_block
                )
                self._save_matrix(self.embeddings_path, self._face_embeddings)
                self._save_matrix(self.outfit_embeddings_path, self._outfit_embeddings)

            return PhotoRecord(
                id=photo_id,
                filename=filename,
                file_hash=file_hash,
                file_path=file_path,
                thumb_path=thumb_path,
                width=width,
                height=height,
                face_count=len(face_records),
                uploaded_at=uploaded_at,
                faces=face_records,
            )

    def list_photos(self) -> list[PhotoRecord]:
        """Return all indexed photos (newest first) along with their detected faces."""
        with self._lock, self._connect() as conn:
            photo_rows = conn.execute(
                "SELECT * FROM photos ORDER BY uploaded_at DESC"
            ).fetchall()
            face_rows = conn.execute(
                "SELECT * FROM detected_faces ORDER BY vector_index ASC"
            ).fetchall()

            faces_by_photo: dict[str, list[FaceRecord]] = {}
            for row in face_rows:
                fr = self._row_to_face(row)
                faces_by_photo.setdefault(fr.photo_id, []).append(fr)

            return [
                PhotoRecord(
                    id=str(p["id"]),
                    filename=str(p["filename"]),
                    file_hash=str(p["file_hash"]),
                    file_path=str(p["file_path"]),
                    thumb_path=str(p["thumb_path"]),
                    width=int(p["width"]),
                    height=int(p["height"]),
                    face_count=int(p["face_count"]),
                    uploaded_at=str(p["uploaded_at"]),
                    faces=faces_by_photo.get(str(p["id"]), []),
                )
                for p in photo_rows
            ]

    def get_photos_by_ids(self, photo_ids: list[str]) -> dict[str, PhotoRecord]:
        if not photo_ids:
            return {}
        placeholders = ",".join("?" for _ in photo_ids)
        with self._lock, self._connect() as conn:
            photo_rows = conn.execute(
                f"SELECT * FROM photos WHERE id IN ({placeholders})", photo_ids
            ).fetchall()
            face_rows = conn.execute(
                f"SELECT * FROM detected_faces WHERE photo_id IN ({placeholders}) ORDER BY vector_index ASC",
                photo_ids,
            ).fetchall()

            faces_by_photo: dict[str, list[FaceRecord]] = {}
            for row in face_rows:
                fr = self._row_to_face(row)
                faces_by_photo.setdefault(fr.photo_id, []).append(fr)

            return {
                str(p["id"]): PhotoRecord(
                    id=str(p["id"]),
                    filename=str(p["filename"]),
                    file_hash=str(p["file_hash"]),
                    file_path=str(p["file_path"]),
                    thumb_path=str(p["thumb_path"]),
                    width=int(p["width"]),
                    height=int(p["height"]),
                    face_count=int(p["face_count"]),
                    uploaded_at=str(p["uploaded_at"]),
                    faces=faces_by_photo.get(str(p["id"]), []),
                )
                for p in photo_rows
            }

    def get_search_snapshot(
        self,
    ) -> tuple[list[FaceRecord], np.ndarray, np.ndarray]:
        """Return aligned (face_records, face_embeddings, outfit_embeddings) ordered by vector_index."""
        with self._lock, self._connect() as conn:
            face_rows = conn.execute(
                "SELECT * FROM detected_faces ORDER BY vector_index ASC"
            ).fetchall()
            records = [self._row_to_face(row) for row in face_rows]
            return (
                records,
                self._face_embeddings.copy(),
                self._outfit_embeddings.copy(),
            )

    def delete_photo(self, photo_id: str) -> tuple[str, str] | None:
        """Delete a photo and compact vector arrays so vector_index stays contiguous."""
        with self._lock:
            with self._connect() as conn:
                photo = conn.execute(
                    "SELECT file_path, thumb_path FROM photos WHERE id = ?",
                    (photo_id,),
                ).fetchone()
                if not photo:
                    return None

                file_paths = (str(photo["file_path"]), str(photo["thumb_path"]))
                conn.execute("DELETE FROM photos WHERE id = ?", (photo_id,))

                remaining = conn.execute(
                    "SELECT id, vector_index FROM detected_faces ORDER BY vector_index ASC"
                ).fetchall()

                if remaining:
                    keep_indices = [int(r["vector_index"]) for r in remaining]
                    self._face_embeddings = self._face_embeddings[keep_indices]
                    self._outfit_embeddings = self._outfit_embeddings[keep_indices]
                    for new_idx, row in enumerate(remaining):
                        if int(row["vector_index"]) != new_idx:
                            conn.execute(
                                "UPDATE detected_faces SET vector_index = ? WHERE id = ?",
                                (new_idx, str(row["id"])),
                            )
                else:
                    self._face_embeddings = np.empty((0, FACE_DIM), dtype=np.float32)
                    self._outfit_embeddings = np.empty((0, OUTFIT_DIM), dtype=np.float32)

                self._save_matrix(self.embeddings_path, self._face_embeddings)
                self._save_matrix(self.outfit_embeddings_path, self._outfit_embeddings)
                return file_paths

    def clear_all(self) -> list[tuple[str, str]]:
        """Delete all photos and reset vector arrays. Returns deleted file paths."""
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute("SELECT file_path, thumb_path FROM photos").fetchall()
                paths = [(str(r["file_path"]), str(r["thumb_path"])) for r in rows]
                conn.execute("DELETE FROM detected_faces")
                conn.execute("DELETE FROM photos")

            self._face_embeddings = np.empty((0, FACE_DIM), dtype=np.float32)
            self._outfit_embeddings = np.empty((0, OUTFIT_DIM), dtype=np.float32)
            self._save_matrix(self.embeddings_path, self._face_embeddings)
            self._save_matrix(self.outfit_embeddings_path, self._outfit_embeddings)
            return paths

    def get_stats(self) -> dict[str, int]:
        with self._lock, self._connect() as conn:
            p_cnt = int(conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0])
            f_cnt = int(conn.execute("SELECT COUNT(*) FROM detected_faces").fetchone()[0])
            return {
                "total_photos": p_cnt,
                "total_faces": f_cnt,
                "embedding_dim": FACE_DIM,
            }

    @staticmethod
    def _row_to_face(row: sqlite3.Row) -> FaceRecord:
        return FaceRecord(
            id=str(row["id"]),
            photo_id=str(row["photo_id"]),
            bbox_xyxy=json.loads(row["bbox_xyxy"]),
            bbox_norm=json.loads(row["bbox_norm"]),
            det_score=float(row["det_score"]),
            quality_norm=float(row["quality_norm"]),
            blur_score=float(row["blur_score"]),
            pose_label=str(row["pose_label"]),
            vector_index=int(row["vector_index"]),
        )
