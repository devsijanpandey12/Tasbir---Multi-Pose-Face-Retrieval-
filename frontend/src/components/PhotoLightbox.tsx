"use client";

import React, { useEffect } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Download,
  Eye,
  EyeOff,
  Sparkles,
  UserCheck,
  X,
} from "lucide-react";
import { FaceBoxOut, MatchedPhotoOut, PhotoOut, mediaUrl } from "@/lib/api";

export interface LightboxItem {
  photo: PhotoOut;
  match?: MatchedPhotoOut;
}

interface PhotoLightboxProps {
  items: LightboxItem[];
  index: number;
  onClose: () => void;
  onNavigate: (nextIndex: number) => void;
}

export default function PhotoLightbox({
  items,
  index,
  onClose,
  onNavigate,
}: PhotoLightboxProps) {
  const [showBoxes, setShowBoxes] = React.useState(true);
  const current = items[index];

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft" && index > 0) onNavigate(index - 1);
      if (e.key === "ArrowRight" && index < items.length - 1) onNavigate(index + 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [index, items.length, onClose, onNavigate]);

  if (!current) return null;

  const { photo, match } = current;
  const matchedFaceId = match?.matched_face_id;

  const renderBox = (face: FaceBoxOut) => {
    const [x1, y1, x2, y2] = face.bbox_norm;
    const isMatched = matchedFaceId === face.id;
    const isPass2 = isMatched && match?.pass_stage === 2;

    const borderClass = isMatched
      ? isPass2
        ? "border-sky-400 shadow-[0_0_20px_rgba(56,189,248,0.55)]"
        : "border-emerald-400 shadow-[0_0_20px_rgba(52,211,153,0.55)]"
      : "border-white/40";

    return (
      <div
        key={face.id}
        style={{
          left: `${x1 * 100}%`,
          top: `${y1 * 100}%`,
          width: `${(x2 - x1) * 100}%`,
          height: `${(y2 - y1) * 100}%`,
        }}
        className={`pointer-events-none absolute rounded-md border-2 transition-all ${borderClass}`}
      >
        {(isMatched || !match) && (
          <div
            className={`absolute -top-6 left-0 whitespace-nowrap rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold ${
              isMatched
                ? isPass2
                  ? "bg-sky-500 text-slate-950"
                  : "bg-emerald-400 text-slate-950"
                : "bg-black/70 text-zinc-200"
            }`}
          >
            {isMatched && match
              ? `${Math.round(match.score * 100)}% · ${match.matched_by}`
              : `${face.pose_label} · ${Math.round(face.det_score * 100)}%`}
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 p-4 backdrop-blur-md"
      onClick={onClose}
    >
      <div
        className="relative flex max-h-[92vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-zinc-800 bg-[#10131a] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Top toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800/80 px-5 py-3.5">
          <div className="flex items-center gap-3 min-w-0">
            <span className="rounded-md bg-zinc-800 px-2 py-0.5 font-mono text-xs text-zinc-300">
              {index + 1} / {items.length}
            </span>
            <h3 className="truncate text-sm font-medium text-zinc-100">
              {photo.filename}
            </h3>
            <span className="hidden font-mono text-xs text-zinc-500 sm:inline">
              {photo.width}×{photo.height} · {photo.face_count} face
              {photo.face_count === 1 ? "" : "s"}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setShowBoxes((v) => !v)}
              className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs font-medium text-zinc-200 transition hover:border-zinc-500 hover:bg-zinc-800"
            >
              {showBoxes ? <Eye className="h-3.5 w-3.5 text-amber-400" /> : <EyeOff className="h-3.5 w-3.5 text-zinc-400" />}
              {showBoxes ? "Boxes On" : "Boxes Off"}
            </button>

            <a
              href={mediaUrl(photo.original_url)}
              download={photo.filename}
              className="inline-flex items-center gap-1.5 rounded-lg bg-amber-400 px-3 py-1.5 text-xs font-semibold text-zinc-950 transition hover:bg-amber-300"
            >
              <Download className="h-3.5 w-3.5" />
              Original
            </a>

            <button
              type="button"
              onClick={onClose}
              aria-label="Close lightbox"
              className="rounded-lg border border-zinc-800 bg-zinc-900 p-1.5 text-zinc-400 transition hover:text-white"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Main stage */}
        <div className="relative flex flex-1 items-center justify-center overflow-auto bg-[#080a0e] p-6">
          {index > 0 && (
            <button
              type="button"
              onClick={() => onNavigate(index - 1)}
              aria-label="Previous photo"
              className="absolute left-4 z-10 rounded-full border border-zinc-700/80 bg-zinc-950/80 p-2.5 text-zinc-200 backdrop-blur transition hover:border-amber-400/60 hover:text-white"
            >
              <ChevronLeft className="h-5 w-5" />
            </button>
          )}

          <div className="relative inline-block max-h-[70vh] max-w-full">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={mediaUrl(photo.original_url)}
              alt={photo.filename}
              className="block max-h-[70vh] max-w-full rounded-lg object-contain select-none"
            />
            {showBoxes && (
              <div className="pointer-events-none absolute inset-0">
                {photo.faces.map(renderBox)}
              </div>
            )}
          </div>

          {index < items.length - 1 && (
            <button
              type="button"
              onClick={() => onNavigate(index + 1)}
              aria-label="Next photo"
              className="absolute right-4 z-10 rounded-full border border-zinc-700/80 bg-zinc-950/80 p-2.5 text-zinc-200 backdrop-blur transition hover:border-amber-400/60 hover:text-white"
            >
              <ChevronRight className="h-5 w-5" />
            </button>
          )}
        </div>

        {/* Bottom match metadata bar */}
        {match && (
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-zinc-800/80 bg-[#121620] px-5 py-3 text-xs">
            <div className="flex flex-wrap items-center gap-3">
              <span
                className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 font-medium ${
                  match.pass_stage === 2
                    ? "bg-sky-500/15 text-sky-300 ring-1 ring-sky-500/30"
                    : "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30"
                }`}
              >
                {match.pass_stage === 2 ? (
                  <Sparkles className="h-3.5 w-3.5" />
                ) : (
                  <UserCheck className="h-3.5 w-3.5" />
                )}
                {match.pass_stage === 2
                  ? "Recovered via Pass-2 Outfit Re-ID"
                  : `Matched by ${match.matched_by.toUpperCase()} template`}
              </span>
              <span className="font-mono text-zinc-300">
                Overall: <strong className="text-white">{(match.score * 100).toFixed(1)}%</strong>
              </span>
              <span className="font-mono text-zinc-400">
                Face Cosine: <strong className="text-zinc-200">{match.face_score.toFixed(3)}</strong>
              </span>
              {match.outfit_score !== null && (
                <span className="font-mono text-sky-300">
                  Outfit Sim: <strong>{match.outfit_score.toFixed(3)}</strong>
                </span>
              )}
            </div>
            <span className="text-zinc-500">Use ← / → keys to browse</span>
          </div>
        )}
      </div>
    </div>
  );
}
