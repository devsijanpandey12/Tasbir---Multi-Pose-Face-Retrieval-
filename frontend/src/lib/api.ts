export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function mediaUrl(path: string): string {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://") || path.startsWith("data:")) {
    return path;
  }
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

export interface FaceBoxOut {
  id: string;
  bbox_xyxy: number[];
  bbox_norm: number[];
  det_score: number;
  quality_norm: number;
  blur_score: number;
  pose_label: string;
}

export interface PhotoOut {
  id: string;
  filename: string;
  thumb_url: string;
  original_url: string;
  width: number;
  height: number;
  face_count: number;
  uploaded_at: string;
  faces: FaceBoxOut[];
}

export interface PoolStats {
  total_photos: number;
  total_faces: number;
  embedding_dim: number;
}

export interface HealthResponse {
  status: string;
  app: string;
  version: string;
  models: {
    detector: string;
    embedder: string;
    provider: string;
  };
  limits: {
    max_pool_batch: number;
    max_photo_size_mb: number;
    max_video_size_mb: number;
    max_video_seconds: number;
    max_reference_photos: number;
    default_threshold: number;
  };
  stats: PoolStats;
}

export interface PoolListResponse {
  stats: PoolStats;
  photos: PhotoOut[];
}

export interface PoolUploadItemResult {
  filename: string;
  status: "indexed" | "duplicate" | "error";
  photo_id?: string | null;
  face_count: number;
  message?: string | null;
}

export interface PoolUploadResponse {
  indexed_count: number;
  duplicate_count: number;
  error_count: number;
  total_faces_added: number;
  processing_ms: number;
  items: PoolUploadItemResult[];
  pool_stats: PoolStats;
}

export interface ReferencePoseOut {
  label: string;
  yaw_deg: number;
  blur_score: number;
  quality_norm: number;
  det_score: number;
  preview_b64: string;
  source_name: string;
  warnings: string[];
}

export interface MatchedPhotoOut {
  photo: PhotoOut;
  score: number;
  face_score: number;
  outfit_score: number | null;
  matched_by: string;
  matched_face_id: string;
  bbox_xyxy: number[];
  bbox_norm: number[];
  pass_stage: number;
}

export interface MatchResponse {
  threshold: number;
  enable_pass2: boolean;
  search_ms: number;
  extraction_ms: number;
  total_faces_searched: number;
  pass1_count: number;
  pass2_recovered_count: number;
  references: ReferencePoseOut[];
  skipped_references: Array<{ filename: string; reason: string }>;
  video_meta?: {
    frames_sampled: number;
    frames_with_face: number;
    buckets_filled: string[];
  } | null;
  matches: MatchedPhotoOut[];
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      // ignore non-JSON body
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
  return handleResponse<HealthResponse>(res);
}

export async function fetchPool(): Promise<PoolListResponse> {
  const res = await fetch(`${API_BASE}/api/pool`, { cache: "no-store" });
  return handleResponse<PoolListResponse>(res);
}

export async function uploadPoolPhotos(files: File[]): Promise<PoolUploadResponse> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  const res = await fetch(`${API_BASE}/api/pool/upload`, {
    method: "POST",
    body: form,
  });
  return handleResponse<PoolUploadResponse>(res);
}

export async function deletePoolPhoto(photoId: string): Promise<{ deleted: string; stats: PoolStats }> {
  const res = await fetch(`${API_BASE}/api/pool/${encodeURIComponent(photoId)}`, {
    method: "DELETE",
  });
  return handleResponse(res);
}

export async function clearPool(): Promise<{ cleared_count: number; stats: PoolStats }> {
  const res = await fetch(`${API_BASE}/api/pool`, {
    method: "DELETE",
  });
  return handleResponse(res);
}

export interface ReferenceFileInput {
  file: File;
  angle?: "left" | "front" | "right" | "";
}

export async function matchByPhotos(
  inputs: ReferenceFileInput[],
  threshold: number,
  enablePass2: boolean
): Promise<MatchResponse> {
  const form = new FormData();
  for (const item of inputs) {
    form.append("files", item.file);
    form.append("angles", item.angle || "");
  }
  form.append("threshold", String(threshold));
  form.append("enable_pass2", String(enablePass2));

  const res = await fetch(`${API_BASE}/api/match/photos`, {
    method: "POST",
    body: form,
  });
  return handleResponse<MatchResponse>(res);
}

export async function matchByVideo(
  videoFile: File,
  threshold: number,
  enablePass2: boolean
): Promise<MatchResponse> {
  const form = new FormData();
  form.append("video", videoFile);
  form.append("threshold", String(threshold));
  form.append("enable_pass2", String(enablePass2));

  const res = await fetch(`${API_BASE}/api/match/video`, {
    method: "POST",
    body: form,
  });
  return handleResponse<MatchResponse>(res);
}

export async function downloadMatchedZip(photoIds: string[]): Promise<void> {
  const res = await fetch(`${API_BASE}/api/download-zip`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ photo_ids: photoIds }),
  });
  if (!res.ok) {
    throw new Error(`Could not create ZIP (${res.status})`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "tasbir_matched_photos.zip";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
