"""Multi-angle template matching and Two-Pass Outfit Re-ID recovery engine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .db import FaceRecord


@dataclass
class ReferencePose:
    label: str
    yaw_deg: float
    blur_score: float
    quality_norm: float
    det_score: float
    embedding: np.ndarray  # (512,) L2-normalized float32
    preview_b64: str
    source_name: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class MultiAngleQueryTemplate:
    poses: list[ReferencePose]

    @property
    def mean_embedding(self) -> np.ndarray:
        stacked = np.stack([p.embedding for p in self.poses], axis=0)
        mean_vec = np.mean(stacked, axis=0)
        norm = float(np.linalg.norm(mean_vec))
        return (mean_vec / norm).astype(np.float32) if norm > 1e-8 else stacked[0]

    @property
    def labels(self) -> list[str]:
        base = [p.label for p in self.poses]
        if len(self.poses) > 1:
            base.append("mean")
        return base

    @property
    def matrix(self) -> np.ndarray:
        """Shape (K, 512) matrix of all pose vectors plus centroid for one-shot dot product."""
        vecs = [p.embedding for p in self.poses]
        if len(self.poses) > 1:
            vecs.append(self.mean_embedding)
        return np.stack(vecs, axis=0).astype(np.float32)


@dataclass
class MatchedPhotoResult:
    photo_id: str
    score: float
    face_score: float
    outfit_score: float | None
    matched_by: str  # "left", "front", "right", "mean", "pass2_outfit", etc.
    matched_face_id: str
    bbox_xyxy: list[float]
    bbox_norm: list[float]
    pass_stage: int  # 1 = primary face match, 2 = outfit/anchor recovery


@dataclass
class MatchExecutionReport:
    matches: list[MatchedPhotoResult]
    search_ms: float
    total_faces_searched: int
    pass1_count: int
    pass2_recovered_count: int


def match_pool(
    query: MultiAngleQueryTemplate,
    face_records: list[FaceRecord],
    face_embeddings: np.ndarray,
    outfit_embeddings: np.ndarray,
    *,
    threshold: float = 0.42,
    enable_pass2: bool = True,
    pass2_anchor_threshold: float = 0.50,
    pass2_min_face_score: float = 0.25,
    pass2_min_outfit_score: float = 0.55,
    pass2_combined_threshold: float = 0.44,
) -> MatchExecutionReport:
    """
    Match a MultiAngleQueryTemplate against all indexed pool faces.

    Pass 1: Max-Over-Templates cosine similarity across all reference poses + centroid.
    Pass 2 (optional): Uses high-confidence Pass 1 matches from the same event/pool to
    extract an in-event face anchor + torso outfit descriptor, rescuing occluded or
    extreme-angle shots where face similarity falls into [pass2_min_face_score, threshold).
    """
    t0 = time.perf_counter()

    n_faces = len(face_records)
    if n_faces == 0 or len(face_embeddings) == 0 or len(query.poses) == 0:
        return MatchExecutionReport(
            matches=[],
            search_ms=(time.perf_counter() - t0) * 1000.0,
            total_faces_searched=n_faces,
            pass1_count=0,
            pass2_recovered_count=0,
        )

    # Ensure alignment in case matrix length differs slightly
    count = min(n_faces, len(face_embeddings))
    face_records = face_records[:count]
    face_embeddings = face_embeddings[:count]
    outfit_embeddings = outfit_embeddings[:count] if len(outfit_embeddings) >= count else None

    # Pass 1: One-shot matrix dot product -> (N_faces, K_templates)
    query_matrix = query.matrix
    template_labels = query.labels
    sim_matrix = face_embeddings @ query_matrix.T
    best_template_idx = np.argmax(sim_matrix, axis=1)
    best_scores = sim_matrix[np.arange(count), best_template_idx]

    best_by_photo: dict[str, MatchedPhotoResult] = {}
    anchor_indices: list[int] = []
    effective_anchor_threshold = max(threshold + 0.08, pass2_anchor_threshold)

    for idx in range(count):
        score = float(best_scores[idx])
        if score >= threshold:
            rec = face_records[idx]
            matched_label = template_labels[int(best_template_idx[idx])]
            candidate = MatchedPhotoResult(
                photo_id=rec.photo_id,
                score=score,
                face_score=score,
                outfit_score=None,
                matched_by=matched_label,
                matched_face_id=rec.id,
                bbox_xyxy=rec.bbox_xyxy,
                bbox_norm=rec.bbox_norm,
                pass_stage=1,
            )
            prev = best_by_photo.get(rec.photo_id)
            if prev is None or candidate.score > prev.score:
                best_by_photo[rec.photo_id] = candidate

            if score >= effective_anchor_threshold:
                anchor_indices.append(idx)

    pass1_count = len(best_by_photo)
    pass2_recovered = 0

    # Pass 2: In-Event Query Expansion + Outfit Re-ID Occlusion Recovery
    if (
        enable_pass2
        and anchor_indices
        and outfit_embeddings is not None
        and len(outfit_embeddings) == count
    ):
        # Compute normalized in-event face anchor
        event_face_anchor = np.mean(face_embeddings[anchor_indices], axis=0)
        ef_norm = float(np.linalg.norm(event_face_anchor))
        if ef_norm > 1e-8:
            event_face_anchor = (event_face_anchor / ef_norm).astype(np.float32)

        # Compute normalized in-event outfit anchor from non-zero torso descriptors
        anchor_outfits = outfit_embeddings[anchor_indices]
        outfit_norms = np.linalg.norm(anchor_outfits, axis=1)
        valid_outfits = anchor_outfits[outfit_norms > 1e-6]

        if len(valid_outfits) > 0:
            event_outfit_anchor = np.mean(valid_outfits, axis=0)
            eo_norm = float(np.linalg.norm(event_outfit_anchor))
            if eo_norm > 1e-8:
                event_outfit_anchor = (event_outfit_anchor / eo_norm).astype(np.float32)

                event_face_sims = face_embeddings @ event_face_anchor
                event_outfit_sims = outfit_embeddings @ event_outfit_anchor
                effective_combined_gate = max(threshold + 0.02, pass2_combined_threshold)

                for idx in range(count):
                    rec = face_records[idx]
                    if rec.photo_id in best_by_photo:
                        continue

                    base_face_score = float(best_scores[idx])
                    if base_face_score < pass2_min_face_score or base_face_score >= threshold:
                        continue

                    outfit_score = float(event_outfit_sims[idx])
                    if outfit_score < pass2_min_outfit_score:
                        continue

                    boosted_face = max(base_face_score, float(event_face_sims[idx]))
                    combined_score = 0.65 * boosted_face + 0.35 * outfit_score

                    if combined_score >= effective_combined_gate:
                        candidate = MatchedPhotoResult(
                            photo_id=rec.photo_id,
                            score=combined_score,
                            face_score=boosted_face,
                            outfit_score=outfit_score,
                            matched_by="pass2_outfit",
                            matched_face_id=rec.id,
                            bbox_xyxy=rec.bbox_xyxy,
                            bbox_norm=rec.bbox_norm,
                            pass_stage=2,
                        )
                        prev = best_by_photo.get(rec.photo_id)
                        if prev is None or candidate.score > prev.score:
                            if prev is None:
                                pass2_recovered += 1
                            best_by_photo[rec.photo_id] = candidate

    sorted_matches = sorted(
        best_by_photo.values(), key=lambda m: m.score, reverse=True
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return MatchExecutionReport(
        matches=sorted_matches,
        search_ms=elapsed_ms,
        total_faces_searched=count,
        pass1_count=pass1_count,
        pass2_recovered_count=pass2_recovered,
    )
