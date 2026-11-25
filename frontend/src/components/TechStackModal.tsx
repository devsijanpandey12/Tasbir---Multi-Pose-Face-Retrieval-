"use client";

import React, { useEffect } from "react";
import { Cpu, Database, ExternalLink, Layers, Monitor, X } from "lucide-react";
import { HealthResponse } from "@/lib/api";

const REPO_URL =
  "https://github.com/devsijanpandey12/Tasbir---Multi-Pose-Face-Retrieval-";

interface TechStackModalProps {
  open: boolean;
  health: HealthResponse | null;
  onClose: () => void;
}

const STACK_SECTIONS = [
  {
    title: "AI & Computer Vision",
    icon: Cpu,
    accent: "text-amber-300",
    ring: "border-amber-400/20 bg-amber-400/5",
    items: [
      {
        name: "SCRFD-10G (det_10g.onnx)",
        role: "Joint multi-scale face detection + 5-point landmarks for 112×112 affine alignment",
      },
      {
        name: "ArcFace ResNet-50 (w600k_r50.onnx)",
        role: "Zero-shot 512-D L2-normalized face embeddings trained on WebFace600K",
      },
      {
        name: "ONNX Runtime",
        role: "Fast native inference without PyTorch or C++ build toolchains",
      },
      {
        name: "Multi-Pose Max-Cosine + Pass-2 Outfit Re-ID",
        role: "Left/Front/Right/Mean template bank with HSV+LAB torso occlusion recovery",
      },
      {
        name: "OpenCV & NumPy",
        role: "5-point yaw estimation, Laplacian blur filtering, and vectorized matrix search",
      },
    ],
  },
  {
    title: "Backend & Storage",
    icon: Database,
    accent: "text-emerald-300",
    ring: "border-emerald-400/20 bg-emerald-400/5",
    items: [
      {
        name: "FastAPI + Uvicorn (Python 3.11)",
        role: "Async REST API for bulk pool indexing, photo/video matching, and ZIP export",
      },
      {
        name: "SQLite + Atomic .npy Vectors",
        role: "Relational photo/face metadata synced with contiguous float32 embedding matrices",
      },
      {
        name: "Pillow + SHA-256 Deduplication",
        role: "EXIF orientation fix, 480px WebP thumbnails, and duplicate upload protection",
      },
    ],
  },
  {
    title: "Frontend Web App",
    icon: Monitor,
    accent: "text-sky-300",
    ring: "border-sky-400/20 bg-sky-400/5",
    items: [
      {
        name: "Next.js 15 + React 19 + TypeScript",
        role: "Responsive two-stage studio interface with strongly typed REST client",
      },
      {
        name: "Tailwind CSS v4",
        role: "Custom dark gallery UI with normalized face bounding-box Lightbox overlays",
      },
      {
        name: "WebRTC (getUserMedia + MediaRecorder)",
        role: "In-browser guided 3-angle webcam capture and 4.5s head-turn video recording",
      },
    ],
  },
];

export default function TechStackModal({
  open,
  health,
  onClose,
}: TechStackModalProps) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-md"
      onClick={onClose}
    >
      <div
        className="relative flex max-h-[88vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-zinc-800 bg-[#10131b] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800/80 px-6 py-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber-400/15 text-amber-300 ring-1 ring-amber-400/30">
              <Layers className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-white">
                Tech Stack & Architecture
              </h2>
              <p className="text-[11px] text-zinc-400">
                Tasbir — Multi-Pose Face Retrieval Pipeline
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="Close tech stack modal"
            className="rounded-lg border border-zinc-800 bg-zinc-900 p-1.5 text-zinc-400 transition hover:text-white"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body */}
        <div className="space-y-4 overflow-y-auto px-6 py-5">
          {health && (
            <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-emerald-500/20 bg-emerald-500/5 px-4 py-2.5 font-mono text-[11px] text-emerald-200">
              <span>
                Runtime: <strong>{health.models.provider}</strong>
              </span>
              <span>
                Index: <strong>{health.stats.total_photos}</strong> photos ·{" "}
                <strong>{health.stats.total_faces}</strong> faces (
                {health.stats.embedding_dim}-D)
              </span>
            </div>
          )}

          {STACK_SECTIONS.map((section) => {
            const Icon = section.icon;
            return (
              <div
                key={section.title}
                className={`rounded-xl border p-4 ${section.ring}`}
              >
                <div className="mb-3 flex items-center gap-2">
                  <Icon className={`h-4 w-4 ${section.accent}`} />
                  <h3
                    className={`font-mono text-xs font-semibold uppercase tracking-wider ${section.accent}`}
                  >
                    {section.title}
                  </h3>
                </div>
                <div className="space-y-2.5">
                  {section.items.map((item) => (
                    <div
                      key={item.name}
                      className="flex flex-col gap-0.5 rounded-lg border border-zinc-800/80 bg-[#0c0f16]/90 px-3.5 py-2.5 sm:flex-row sm:items-baseline sm:justify-between sm:gap-4"
                    >
                      <span className="shrink-0 font-mono text-xs font-semibold text-zinc-100">
                        {item.name}
                      </span>
                      <span className="text-[11px] text-zinc-400 sm:text-right">
                        {item.role}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        {/* Modal footer */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-zinc-800/80 bg-[#0b0e14] px-6 py-3.5 text-xs text-zinc-400">
          <span>
            Made by <strong className="text-zinc-100">Sijan Pandey</strong>
          </span>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 font-medium text-amber-300 transition hover:text-amber-200"
          >
            View Source on GitHub
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
        </div>
      </div>
    </div>
  );
}
