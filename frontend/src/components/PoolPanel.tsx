"use client";

import React, { useRef, useState } from "react";
import {
  CheckCircle2,
  Images,
  Loader2,
  Trash2,
  UploadCloud,
  Users,
  AlertTriangle,
} from "lucide-react";
import {
  PhotoOut,
  PoolUploadResponse,
  clearPool,
  deletePoolPhoto,
  mediaUrl,
  uploadPoolPhotos,
} from "@/lib/api";
import PhotoLightbox from "./PhotoLightbox";

interface PoolPanelProps {
  photos: PhotoOut[];
  maxBatch: number;
  maxPhotoMb: number;
  onPoolChanged: () => Promise<void>;
}

export default function PoolPanel({
  photos,
  maxBatch,
  maxPhotoMb,
  onPoolChanged,
}: PoolPanelProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpload, setLastUpload] = useState<PoolUploadResponse | null>(null);
  const [lightboxIdx, setLightboxIdx] = useState<number | null>(null);

  const handleFiles = async (fileList: FileList | File[]) => {
    const incoming = Array.from(fileList).filter((f) =>
      f.type.startsWith("image/") || /\.(jpe?g|png|webp|bmp)$/i.test(f.name)
    );
    if (incoming.length === 0) {
      setError("Please drop or select image files (JPG, PNG, WebP).");
      return;
    }
    if (incoming.length > maxBatch) {
      setError(`Batch limit is ${maxBatch} photos per upload (selected ${incoming.length}).`);
      return;
    }

    setError(null);
    setUploading(true);
    try {
      const res = await uploadPoolPhotos(incoming);
      setLastUpload(res);
      await onPoolChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    setDeletingId(id);
    try {
      await deletePoolPhoto(id);
      await onPoolChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete photo.");
    } finally {
      setDeletingId(null);
    }
  };

  const handleClearAll = async () => {
    setClearing(true);
    setError(null);
    try {
      await clearPool();
      setLastUpload(null);
      setConfirmClear(false);
      await onPoolChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to clear pool.");
    } finally {
      setClearing(false);
    }
  };

  const totalFaces = photos.reduce((acc, p) => acc + p.face_count, 0);

  return (
    <div className="space-y-6">
      {/* Bulk Upload Dropzone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (e.dataTransfer.files) void handleFiles(e.dataTransfer.files);
        }}
        className={`relative rounded-2xl border-2 border-dashed p-8 text-center transition ${
          dragging
            ? "border-amber-400 bg-amber-400/10"
            : "border-zinc-800 bg-[#11141c]/90 hover:border-zinc-700"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => {
            if (e.target.files) void handleFiles(e.target.files);
          }}
        />

        <div className="mx-auto flex max-w-lg flex-col items-center">
          <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-amber-400/10 text-amber-300 ring-1 ring-amber-400/30">
            {uploading ? (
              <Loader2 className="h-6 w-6 animate-spin" />
            ) : (
              <UploadCloud className="h-6 w-6" />
            )}
          </div>
          <h2 className="text-base font-semibold text-zinc-100">
            {uploading
              ? "Detecting faces & building 512-D embeddings…"
              : "Step 1 · Upload Event or Group Photo Pool"}
          </h2>
          <p className="mt-1.5 text-xs leading-relaxed text-zinc-400">
            Drag and drop event or group photos here. Each image is auto-oriented,
            scanned by <span className="font-mono text-zinc-200">SCRFD-10G</span>, and
            indexed with <span className="font-mono text-zinc-200">ArcFace 512-D</span>{" "}
            vectors locally.
          </p>

          <div className="mt-4 flex flex-wrap items-center justify-center gap-3">
            <button
              type="button"
              disabled={uploading}
              onClick={() => inputRef.current?.click()}
              className="inline-flex items-center gap-2 rounded-xl bg-amber-400 px-5 py-2.5 text-xs font-semibold text-zinc-950 transition hover:bg-amber-300 disabled:opacity-50"
            >
              <Images className="h-4 w-4" />
              Select Group Photos
            </button>
            <span className="font-mono text-[11px] text-zinc-500">
              Up to {maxBatch} photos/batch · ≤ {maxPhotoMb} MB each · Auto-deduplicated
            </span>
          </div>
        </div>
      </div>

      {/* Feedback messages */}
      {error && (
        <div className="flex items-start gap-2.5 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-xs text-rose-200">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {lastUpload && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-xs text-emerald-200">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-400" />
            <span>
              Indexed <strong>{lastUpload.indexed_count}</strong> new photo
              {lastUpload.indexed_count === 1 ? "" : "s"} (
              <strong>{lastUpload.total_faces_added}</strong> face
              {lastUpload.total_faces_added === 1 ? "" : "s"} extracted)
              {lastUpload.duplicate_count > 0 &&
                ` · ${lastUpload.duplicate_count} duplicate skipped`}
              {lastUpload.error_count > 0 && ` · ${lastUpload.error_count} failed`}
            </span>
          </div>
          <span className="font-mono text-[11px] text-emerald-300/80">
            {(lastUpload.processing_ms / 1000).toFixed(2)}s total
          </span>
        </div>
      )}

      {/* Pool header & actions */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-zinc-100">Indexed Photo Pool</h3>
          <span className="rounded-full bg-zinc-800 px-2.5 py-0.5 font-mono text-xs text-zinc-300">
            {photos.length} photo{photos.length === 1 ? "" : "s"} · {totalFaces} face
            {totalFaces === 1 ? "" : "s"}
          </span>
        </div>

        {photos.length > 0 && (
          <div className="flex items-center gap-2">
            {!confirmClear ? (
              <button
                type="button"
                onClick={() => setConfirmClear(true)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-400 transition hover:border-rose-500/40 hover:text-rose-300"
              >
                <Trash2 className="h-3.5 w-3.5" />
                Clear Pool
              </button>
            ) : (
              <div className="flex items-center gap-2 rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-1.5 text-xs">
                <span className="text-rose-200">Delete all {photos.length} photos?</span>
                <button
                  type="button"
                  disabled={clearing}
                  onClick={handleClearAll}
                  className="rounded bg-rose-500 px-2 py-0.5 font-semibold text-white hover:bg-rose-400"
                >
                  {clearing ? "Clearing…" : "Confirm"}
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmClear(false)}
                  className="text-zinc-400 hover:text-zinc-200"
                >
                  Cancel
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Indexed gallery grid */}
      {photos.length === 0 ? (
        <div className="rounded-2xl border border-zinc-800/80 bg-[#10131b] px-6 py-14 text-center">
          <Images className="mx-auto mb-3 h-8 w-8 text-zinc-600" />
          <p className="text-sm font-medium text-zinc-300">No photos indexed yet</p>
          <p className="mt-1 text-xs text-zinc-500">
            Upload a batch of group or event photos above, then switch to{" "}
            <strong className="text-amber-300">Find My Photos</strong> to filter by face.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
          {photos.map((photo, idx) => (
            <div
              key={photo.id}
              onClick={() => setLightboxIdx(idx)}
              className="group relative cursor-pointer overflow-hidden rounded-xl border border-zinc-800/90 bg-[#12151f] transition hover:border-amber-400/50"
            >
              <div className="relative aspect-[4/3] overflow-hidden bg-zinc-950">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={mediaUrl(photo.thumb_url)}
                  alt={photo.filename}
                  loading="lazy"
                  className="h-full w-full object-cover transition duration-300 group-hover:scale-105"
                />
                <div className="absolute top-2 left-2 inline-flex items-center gap-1 rounded-full bg-black/75 px-2 py-0.5 font-mono text-[10px] font-medium text-zinc-100 backdrop-blur">
                  <Users className="h-3 w-3 text-amber-300" />
                  {photo.face_count}
                </div>
                <button
                  type="button"
                  disabled={deletingId === photo.id}
                  onClick={(e) => handleDelete(e, photo.id)}
                  aria-label={`Delete ${photo.filename}`}
                  className="absolute top-2 right-2 rounded-lg bg-black/70 p-1.5 text-zinc-300 opacity-0 backdrop-blur transition group-hover:opacity-100 hover:bg-rose-500 hover:text-white"
                >
                  {deletingId === photo.id ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Trash2 className="h-3.5 w-3.5" />
                  )}
                </button>
              </div>
              <div className="flex items-center justify-between gap-2 px-3 py-2">
                <span className="truncate text-xs text-zinc-300">{photo.filename}</span>
                <span className="shrink-0 font-mono text-[10px] text-zinc-500">
                  {photo.width}×{photo.height}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {lightboxIdx !== null && (
        <PhotoLightbox
          items={photos.map((p) => ({ photo: p }))}
          index={lightboxIdx}
          onClose={() => setLightboxIdx(null)}
          onNavigate={setLightboxIdx}
        />
      )}
    </div>
  );
}
