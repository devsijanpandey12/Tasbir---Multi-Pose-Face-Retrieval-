"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  Aperture,
  Cpu,
  ExternalLink,
  Images,
  Layers,
  RefreshCw,
  Search,
  WifiOff,
} from "lucide-react";
import {
  HealthResponse,
  PhotoOut,
  fetchHealth,
  fetchPool,
} from "@/lib/api";
import FinderPanel from "@/components/FinderPanel";
import PoolPanel from "@/components/PoolPanel";
import TechStackModal from "@/components/TechStackModal";

const REPO_URL =
  "https://github.com/devsijanpandey12/Tasbir---Multi-Pose-Face-Retrieval-";

type ActiveTab = "find" | "pool";

export default function HomePage() {
  const [tab, setTab] = useState<ActiveTab>("pool");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [photos, setPhotos] = useState<PhotoOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [backendOffline, setBackendOffline] = useState(false);
  const [stackOpen, setStackOpen] = useState(false);

  const loadData = useCallback(async (autoSwitchTab = false) => {
    try {
      const [h, p] = await Promise.all([fetchHealth(), fetchPool()]);
      setHealth(h);
      setPhotos(p.photos);
      setBackendOffline(false);
      if (autoSwitchTab) {
        setTab(p.photos.length > 0 ? "find" : "pool");
      }
    } catch {
      setBackendOffline(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadData(true);
  }, [loadData]);

  const totalPhotos = health?.stats.total_photos ?? photos.length;
  const totalFaces =
    health?.stats.total_faces ?? photos.reduce((acc, p) => acc + p.face_count, 0);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Studio Top Bar */}
      <header className="sticky top-0 z-30 border-b border-zinc-800/80 bg-[#0a0c10]/90 backdrop-blur-xl">
        {/* Top Utility Strip: Engine Status (Left) + Author & GitHub (Top Right) */}
        <div className="border-b border-zinc-900/90 bg-[#07090d]/90">
          <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-2 px-5 py-1.5 text-[11px]">
            <div className="flex items-center gap-2">
              {backendOffline ? (
                <span className="inline-flex items-center gap-1.5 text-rose-300">
                  <WifiOff className="h-3 w-3" />
                  Backend Offline
                </span>
              ) : (
                <span className="inline-flex items-center gap-2 font-mono text-zinc-400">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                  <Cpu className="h-3 w-3 text-zinc-500" />
                  <span className="text-zinc-300">SCRFD-10G · ArcFace 512-D</span>
                  <span className="text-zinc-700">|</span>
                  <span className="text-amber-300">{totalFaces} faces indexed</span>
                </span>
              )}
            </div>

            <div className="flex flex-wrap items-center gap-2.5 text-zinc-400">
              <span>
                Made by <strong className="font-semibold text-zinc-100">Sijan Pandey</strong>
                <span className="mx-1.5 text-zinc-600">·</span>
                <span className="hidden font-mono text-[10px] text-zinc-400 sm:inline">
                  Tasbir Multi-Pose Face Retrieval
                </span>
              </span>
              <a
                href={REPO_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 rounded-md border border-zinc-800 bg-zinc-900/90 px-2.5 py-0.5 font-medium text-zinc-200 transition hover:border-amber-400/50 hover:text-amber-300"
              >
                <svg
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                  className="h-3.5 w-3.5 fill-current"
                >
                  <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56 0-.28-.01-1.02-.02-2-3.2.7-3.88-1.54-3.88-1.54-.52-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.71 1.26 3.37.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.18 1.18a11.04 11.04 0 0 1 5.79 0c2.21-1.49 3.18-1.18 3.18-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.42-2.69 5.39-5.26 5.68.41.36.78 1.07.78 2.15 0 1.55-.01 2.8-.01 3.18 0 .31.21.68.8.56A11.51 11.51 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z" />
                </svg>
                <span>GitHub Repository</span>
                <ExternalLink className="h-2.5 w-2.5 text-zinc-500" />
              </a>
            </div>
          </div>
        </div>

        {/* Main Navigation Row */}
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-5 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-amber-300 to-amber-500 text-zinc-950 shadow-lg shadow-amber-500/20">
              <Aperture className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold tracking-tight text-white">Tasbir</h1>
                <span className="rounded bg-zinc-800/90 px-1.5 py-0.5 font-mono text-[10px] text-amber-300">
                  2026 SOTA
                </span>
              </div>
              <p className="text-[11px] text-zinc-400">
                Multi-Angle Event & Group Photo Finder
              </p>
            </div>
          </div>

          {/* Primary Navigation Tabs */}
          <nav className="flex items-center rounded-xl border border-zinc-800 bg-zinc-950 p-1 text-xs">
            <button
              type="button"
              onClick={() => setTab("pool")}
              className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 font-medium transition ${
                tab === "pool"
                  ? "bg-zinc-800 text-white shadow"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <Images className="h-3.5 w-3.5 text-amber-400" />
              1. Photo Pool
              <span className="rounded-full bg-zinc-900 px-1.5 py-0.5 font-mono text-[10px] text-zinc-300">
                {totalPhotos}
              </span>
            </button>
            <button
              type="button"
              onClick={() => setTab("find")}
              className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 font-medium transition ${
                tab === "find"
                  ? "bg-zinc-800 text-white shadow"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <Search className="h-3.5 w-3.5 text-emerald-400" />
              2. Find My Photos
            </button>
          </nav>

          {/* Tech Stack Modal Button */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setStackOpen(true)}
              className="inline-flex items-center gap-2 rounded-xl border border-zinc-800 bg-zinc-950 px-3.5 py-2 text-xs font-medium text-zinc-200 transition hover:border-amber-400/50 hover:bg-zinc-900 hover:text-amber-300"
            >
              <Layers className="h-3.5 w-3.5 text-amber-400" />
              <span>Tech Stack</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto w-full max-w-7xl flex-1 px-5 py-7">
        {backendOffline && (
          <div className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-200">
            <div className="flex items-center gap-2.5">
              <WifiOff className="h-4 w-4 shrink-0 text-rose-400" />
              <span>
                Cannot reach FastAPI backend at{" "}
                <code className="rounded bg-black/40 px-1.5 py-0.5 font-mono">
                  http://localhost:8000
                </code>
                . Make sure the backend server is running.
              </span>
            </div>
            <button
              type="button"
              onClick={() => void loadData(false)}
              className="inline-flex items-center gap-1.5 rounded-lg bg-rose-500/20 px-3 py-1.5 font-medium text-rose-100 hover:bg-rose-500/30"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Retry
            </button>
          </div>
        )}

        {loading ? (
          <div className="flex min-h-[360px] items-center justify-center text-xs text-zinc-400">
            Loading Tasbir vision engine…
          </div>
        ) : tab === "pool" ? (
          <PoolPanel
            photos={photos}
            maxBatch={health?.limits.max_pool_batch ?? 200}
            maxPhotoMb={health?.limits.max_photo_size_mb ?? 15}
            onPoolChanged={() => loadData(false)}
          />
        ) : (
          <FinderPanel
            poolPhotoCount={totalPhotos}
            poolFaceCount={totalFaces}
            maxVideoMb={health?.limits.max_video_size_mb ?? 25}
            maxVideoSeconds={health?.limits.max_video_seconds ?? 12}
            onGoToPool={() => setTab("pool")}
          />
        )}
      </main>

      <footer className="border-t border-zinc-900/90 py-3.5 text-center font-mono text-[11px] text-zinc-600">
        Tasbir Local Engine · SCRFD-10G + ArcFace 512-D ONNX · Multi-Pose Max-Cosine &
        Two-Pass Outfit Re-ID
      </footer>

      <TechStackModal
        open={stackOpen}
        health={health}
        onClose={() => setStackOpen(false)}
      />
    </div>
  );
}
