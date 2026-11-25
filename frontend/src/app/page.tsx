"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  Aperture,
  Cpu,
  Images,
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

type ActiveTab = "find" | "pool";

export default function HomePage() {
  const [tab, setTab] = useState<ActiveTab>("pool");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [photos, setPhotos] = useState<PhotoOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [backendOffline, setBackendOffline] = useState(false);

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
      <header className="sticky top-0 z-30 border-b border-zinc-800/80 bg-[#0a0c10]/85 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-5 py-3.5">
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

          {/* Engine Telemetry Pill */}
          <div className="hidden items-center gap-2.5 lg:flex">
            {backendOffline ? (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-rose-500/30 bg-rose-500/10 px-3 py-1 text-xs text-rose-300">
                <WifiOff className="h-3.5 w-3.5" />
                Backend Offline
              </span>
            ) : (
              <div className="flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-950/80 px-3 py-1.5 text-[11px]">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                <Cpu className="h-3.5 w-3.5 text-zinc-400" />
                <span className="font-mono text-zinc-300">
                  SCRFD-10G · ArcFace 512-D
                </span>
                <span className="text-zinc-600">|</span>
                <span className="font-mono text-amber-300">{totalFaces} faces</span>
              </div>
            )}
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

      <footer className="border-t border-zinc-900 py-4 text-center font-mono text-[11px] text-zinc-600">
        Tasbir Local Engine · SCRFD-10G + ArcFace WebFace600K ONNX + Multi-Pose Max-Cosine &
        Two-Pass Outfit Re-ID
      </footer>
    </div>
  );
}
