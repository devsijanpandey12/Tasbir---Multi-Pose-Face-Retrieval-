"use client";

import React, { useRef, useState } from "react";
import {
  AlertCircle,
  Camera,
  Download,
  FileVideo,
  Images,
  Loader2,
  Search,
  SlidersHorizontal,
  Sparkles,
  UserCheck,
  X,
} from "lucide-react";
import {
  MatchResponse,
  ReferenceFileInput,
  downloadMatchedZip,
  matchByPhotos,
  matchByVideo,
  mediaUrl,
} from "@/lib/api";
import PhotoLightbox from "./PhotoLightbox";
import WebcamCapture from "./WebcamCapture";

interface FinderPanelProps {
  poolPhotoCount: number;
  poolFaceCount: number;
  maxVideoMb: number;
  maxVideoSeconds: number;
  onGoToPool: () => void;
}

type InputTab = "photos" | "webcam" | "video";
type AngleSlot = "front" | "left" | "right";

const ANGLE_SLOTS: Array<{ key: AngleSlot; label: string; hint: string }> = [
  { key: "front", label: "Front View", hint: "Facing camera (0°)" },
  { key: "left", label: "Left Profile", hint: "Turned ~30° left" },
  { key: "right", label: "Right Profile", hint: "Turned ~30° right" },
];

export default function FinderPanel({
  poolPhotoCount,
  poolFaceCount,
  maxVideoMb,
  maxVideoSeconds,
  onGoToPool,
}: FinderPanelProps) {
  const [inputTab, setInputTab] = useState<InputTab>("photos");
  const [slots, setSlots] = useState<
    Partial<Record<AngleSlot, { file: File; previewUrl: string }>>
  >({});
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [threshold, setThreshold] = useState(0.42);
  const [enablePass2, setEnablePass2] = useState(true);

  const [searching, setSearching] = useState(false);
  const [zipping, setZipping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<MatchResponse | null>(null);
  const [lightboxIdx, setLightboxIdx] = useState<number | null>(null);

  const multiInputRef = useRef<HTMLInputElement | null>(null);
  const videoInputRef = useRef<HTMLInputElement | null>(null);

  const setSlotFile = (slot: AngleSlot, file: File) => {
    const previewUrl = URL.createObjectURL(file);
    setSlots((prev) => {
      const old = prev[slot];
      if (old?.previewUrl) URL.revokeObjectURL(old.previewUrl);
      return { ...prev, [slot]: { file, previewUrl } };
    });
  };

  const clearSlot = (slot: AngleSlot) => {
    setSlots((prev) => {
      const old = prev[slot];
      if (old?.previewUrl) URL.revokeObjectURL(old.previewUrl);
      const next = { ...prev };
      delete next[slot];
      return next;
    });
  };

  const handleMultiDrop = (files: FileList) => {
    const imgs = Array.from(files)
      .filter((f) => f.type.startsWith("image/") || /\.(jpe?g|png|webp)$/i.test(f.name))
      .slice(0, 3);
    const order: AngleSlot[] = ["front", "left", "right"];
    imgs.forEach((f, idx) => setSlotFile(order[idx], f));
  };

  const runPhotoMatch = async (inputs: ReferenceFileInput[]) => {
    if (inputs.length === 0) {
      setError("Add at least one clear reference photo first.");
      return;
    }
    setError(null);
    setSearching(true);
    try {
      const res = await matchByPhotos(inputs, threshold, enablePass2);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Matching failed.");
    } finally {
      setSearching(false);
    }
  };

  const runVideoMatch = async (file: File) => {
    setError(null);
    setSearching(true);
    try {
      const res = await matchByVideo(file, threshold, enablePass2);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Video matching failed.");
    } finally {
      setSearching(false);
    }
  };

  const handleSubmitSlotPhotos = () => {
    const inputs: ReferenceFileInput[] = [];
    for (const key of ["front", "left", "right"] as AngleSlot[]) {
      const item = slots[key];
      if (item) inputs.push({ file: item.file, angle: key });
    }
    void runPhotoMatch(inputs);
  };

  const handleDownloadZip = async () => {
    if (!result || result.matches.length === 0) return;
    setZipping(true);
    try {
      await downloadMatchedZip(result.matches.map((m) => m.photo.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not download ZIP.");
    } finally {
      setZipping(false);
    }
  };

  const selectedSlotCount = Object.keys(slots).length;

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
      {/* Left column: Reference Enrollment + Controls */}
      <div className="space-y-5 lg:col-span-5">
        {poolPhotoCount === 0 && (
          <div className="flex items-start justify-between gap-3 rounded-2xl border border-amber-400/30 bg-amber-400/10 p-4 text-xs text-amber-200">
            <div>
              <p className="font-semibold text-amber-300">Photo pool is empty</p>
              <p className="mt-0.5 text-amber-200/80">
                Upload event/group photos first so Tasbir has images to search.
              </p>
            </div>
            <button
              type="button"
              onClick={onGoToPool}
              className="shrink-0 rounded-lg bg-amber-400 px-3 py-1.5 font-semibold text-zinc-950 hover:bg-amber-300"
            >
              Upload Pool
            </button>
          </div>
        )}

        <div className="rounded-2xl border border-zinc-800/90 bg-[#11141c] p-5 shadow-xl">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-zinc-100">
                Step 2 · Provide Your Reference Angles
              </h2>
              <p className="mt-0.5 text-xs text-zinc-400">
                Front + Left + Right boosts accuracy on candid side-profile shots.
              </p>
            </div>
          </div>

          {/* Segmented Input Switcher */}
          <div className="mb-4 grid grid-cols-3 gap-1 rounded-xl border border-zinc-800 bg-zinc-950 p-1 text-xs">
            <button
              type="button"
              onClick={() => setInputTab("photos")}
              className={`inline-flex items-center justify-center gap-1.5 rounded-lg py-2 font-medium transition ${
                inputTab === "photos"
                  ? "bg-zinc-800 text-amber-300 shadow"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <Images className="h-3.5 w-3.5" />
              3 Photos
            </button>
            <button
              type="button"
              onClick={() => setInputTab("webcam")}
              className={`inline-flex items-center justify-center gap-1.5 rounded-lg py-2 font-medium transition ${
                inputTab === "webcam"
                  ? "bg-zinc-800 text-amber-300 shadow"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <Camera className="h-3.5 w-3.5" />
              Live Cam
            </button>
            <button
              type="button"
              onClick={() => setInputTab("video")}
              className={`inline-flex items-center justify-center gap-1.5 rounded-lg py-2 font-medium transition ${
                inputTab === "video"
                  ? "bg-zinc-800 text-amber-300 shadow"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <FileVideo className="h-3.5 w-3.5" />
              Video
            </button>
          </div>

          {/* Mode 1: 3 Reference Photos */}
          {inputTab === "photos" && (
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-3">
                {ANGLE_SLOTS.map((slot) => {
                  const item = slots[slot.key];
                  return (
                    <label
                      key={slot.key}
                      className={`group relative flex cursor-pointer flex-col overflow-hidden rounded-xl border transition ${
                        item
                          ? "border-amber-400/50 bg-zinc-900"
                          : "border-dashed border-zinc-700 bg-zinc-950/70 hover:border-amber-400/50"
                      }`}
                    >
                      <input
                        type="file"
                        accept="image/*"
                        className="hidden"
                        onChange={(e) => {
                          const f = e.target.files?.[0];
                          if (f) setSlotFile(slot.key, f);
                          e.target.value = "";
                        }}
                      />
                      <div className="relative flex aspect-square items-center justify-center overflow-hidden bg-zinc-950">
                        {item ? (
                          <>
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img
                              src={item.previewUrl}
                              alt={slot.label}
                              className="h-full w-full object-cover"
                            />
                            <button
                              type="button"
                              onClick={(e) => {
                                e.preventDefault();
                                clearSlot(slot.key);
                              }}
                              aria-label={`Remove ${slot.label}`}
                              className="absolute top-1.5 right-1.5 rounded-full bg-black/80 p-1 text-zinc-200 hover:bg-rose-500"
                            >
                              <X className="h-3 w-3" />
                            </button>
                          </>
                        ) : (
                          <div className="px-2 text-center">
                            <Images className="mx-auto mb-1 h-5 w-5 text-zinc-600 transition group-hover:text-amber-300" />
                            <span className="block text-[11px] font-medium text-zinc-300">
                              {slot.label}
                            </span>
                            <span className="mt-0.5 block text-[10px] text-zinc-500">
                              {slot.hint}
                            </span>
                          </div>
                        )}
                      </div>
                      {item && (
                        <div className="truncate bg-zinc-900/90 px-2 py-1 text-center font-mono text-[10px] text-amber-300">
                          {slot.label}
                        </div>
                      )}
                    </label>
                  );
                })}
              </div>

              <div className="flex items-center justify-between text-[11px] text-zinc-500">
                <button
                  type="button"
                  onClick={() => multiInputRef.current?.click()}
                  className="text-zinc-400 underline underline-offset-4 hover:text-amber-300"
                >
                  Or pick 1–3 photos at once
                </button>
                <input
                  ref={multiInputRef}
                  type="file"
                  accept="image/*"
                  multiple
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files) handleMultiDrop(e.target.files);
                    e.target.value = "";
                  }}
                />
                <span>{selectedSlotCount}/3 angles loaded</span>
              </div>

              <button
                type="button"
                disabled={searching || selectedSlotCount === 0}
                onClick={handleSubmitSlotPhotos}
                className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-amber-400 px-4 py-3 text-sm font-semibold text-zinc-950 transition hover:bg-amber-300 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {searching ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Matching against {poolFaceCount} indexed faces…
                  </>
                ) : (
                  <>
                    <Search className="h-4 w-4" />
                    Find My Photos
                  </>
                )}
              </button>
            </div>
          )}

          {/* Mode 2: Live Webcam */}
          {inputTab === "webcam" && (
            <WebcamCapture
              disabled={searching}
              onSubmitPhotos={(inputs) => void runPhotoMatch(inputs)}
              onSubmitVideo={(vid) => void runVideoMatch(vid)}
            />
          )}

          {/* Mode 3: Video Upload */}
          {inputTab === "video" && (
            <div className="space-y-4">
              <div
                onClick={() => videoInputRef.current?.click()}
                className="cursor-pointer rounded-xl border-2 border-dashed border-zinc-700 bg-zinc-950/70 p-6 text-center transition hover:border-amber-400/60"
              >
                <input
                  ref={videoInputRef}
                  type="file"
                  accept="video/mp4,video/webm,video/quicktime,.mp4,.webm,.mov"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) setVideoFile(f);
                    e.target.value = "";
                  }}
                />
                <FileVideo className="mx-auto mb-2 h-7 w-7 text-amber-300" />
                <p className="text-xs font-medium text-zinc-200">
                  {videoFile ? videoFile.name : "Upload 3–5s head-turn video (MP4 / WebM / MOV)"}
                </p>
                <p className="mt-1 font-mono text-[11px] text-zinc-500">
                  Auto-selects sharpest Left, Front & Right frames · ≤ {maxVideoSeconds}s · ≤{" "}
                  {maxVideoMb} MB
                </p>
              </div>

              <button
                type="button"
                disabled={searching || !videoFile}
                onClick={() => videoFile && void runVideoMatch(videoFile)}
                className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-amber-400 px-4 py-3 text-sm font-semibold text-zinc-950 transition hover:bg-amber-300 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {searching ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Extracting angles & matching…
                  </>
                ) : (
                  <>
                    <Search className="h-4 w-4" />
                    Extract Pose Angles & Match
                  </>
                )}
              </button>
            </div>
          )}
        </div>

        {/* Threshold & Pass-2 Controls */}
        <div className="rounded-2xl border border-zinc-800/90 bg-[#11141c] p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-semibold text-zinc-200">
              <SlidersHorizontal className="h-3.5 w-3.5 text-amber-400" />
              Matching Sensitivity
            </div>
            <span className="rounded bg-zinc-900 px-2 py-0.5 font-mono text-xs text-amber-300">
              τ = {threshold.toFixed(2)}
            </span>
          </div>

          <input
            type="range"
            min={0.25}
            max={0.68}
            step={0.01}
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full accent-amber-400"
          />

          <div className="flex items-center justify-between gap-2 text-[11px]">
            {[
              { label: "Broad (0.35)", val: 0.35 },
              { label: "Balanced (0.42)", val: 0.42 },
              { label: "Strict (0.50)", val: 0.5 },
            ].map((preset) => (
              <button
                key={preset.val}
                type="button"
                onClick={() => setThreshold(preset.val)}
                className={`flex-1 rounded-lg border py-1 font-mono transition ${
                  Math.abs(threshold - preset.val) < 0.005
                    ? "border-amber-400/60 bg-amber-400/15 text-amber-200"
                    : "border-zinc-800 bg-zinc-950 text-zinc-400 hover:text-zinc-200"
                }`}
              >
                {preset.label}
              </button>
            ))}
          </div>

          <label className="flex cursor-pointer items-start justify-between gap-3 rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-3">
            <div>
              <div className="flex items-center gap-1.5 text-xs font-medium text-zinc-200">
                <Sparkles className="h-3.5 w-3.5 text-sky-400" />
                Two-Pass Outfit & Occlusion Recovery
              </div>
              <p className="mt-0.5 text-[11px] leading-relaxed text-zinc-500">
                Uses high-confidence event anchors + torso outfit similarity to recover
                sunglasses or partial side-profile shots.
              </p>
            </div>
            <input
              type="checkbox"
              checked={enablePass2}
              onChange={(e) => setEnablePass2(e.target.checked)}
              className="mt-1 h-4 w-4 accent-sky-400"
            />
          </label>
        </div>

        {error && (
          <div className="flex items-start gap-2.5 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-xs text-rose-200">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Right column: Reference Diagnostics + Matched Gallery */}
      <div className="space-y-5 lg:col-span-7">
        {!result ? (
          <div className="flex h-full min-h-[420px] flex-col items-center justify-center rounded-2xl border border-zinc-800/80 bg-[#10131b] p-10 text-center">
            <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-zinc-900 text-amber-300 ring-1 ring-zinc-800">
              <UserCheck className="h-6 w-6" />
            </div>
            <h3 className="text-base font-semibold text-zinc-100">
              Ready to find your photos
            </h3>
            <p className="mt-1.5 max-w-md text-xs leading-relaxed text-zinc-400">
              Upload 1 to 3 angles (Front, Left, Right) or record a brief head-turn video on
              the left. Tasbir computes your{" "}
              <span className="font-mono text-zinc-200">Multi-Pose Template Bank</span> and
              runs Max-Cosine matching across all{" "}
              <strong className="text-zinc-200">{poolFaceCount}</strong> indexed faces in
              milliseconds.
            </p>
          </div>
        ) : (
          <>
            {/* Enrolled 112x112 Reference Crops Diagnostics */}
            <div className="rounded-2xl border border-zinc-800/90 bg-[#11141c] p-4">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <span className="text-xs font-semibold text-zinc-200">
                  Extracted Reference Templates ({result.references.length} pose
                  {result.references.length === 1 ? "" : "s"}
                  {result.references.length > 1 ? " + Centroid" : ""})
                </span>
                <span className="font-mono text-[11px] text-zinc-500">
                  Extract {result.extraction_ms.toFixed(0)} ms · Vector search{" "}
                  <strong className="text-emerald-400">{result.search_ms.toFixed(2)} ms</strong>
                </span>
              </div>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                {result.references.map((ref, i) => (
                  <div
                    key={`${ref.label}-${i}`}
                    className="flex items-center gap-3 rounded-xl border border-zinc-800 bg-zinc-950/70 p-2.5"
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={ref.preview_b64}
                      alt={ref.label}
                      className="h-14 w-14 shrink-0 rounded-lg border border-zinc-700 object-cover"
                    />
                    <div className="min-w-0 flex-1 text-[11px]">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold uppercase text-amber-300">
                          {ref.label}
                        </span>
                        <span className="font-mono text-zinc-400">
                          {ref.yaw_deg > 0 ? `+${ref.yaw_deg}°` : `${ref.yaw_deg}°`}
                        </span>
                      </div>
                      <div className="mt-1 font-mono text-[10px] text-zinc-500">
                        Sharp: {ref.blur_score.toFixed(0)} · Qual: {ref.quality_norm.toFixed(1)}
                      </div>
                      {ref.warnings.length > 0 && (
                        <p className="mt-1 truncate text-[10px] text-amber-400/90" title={ref.warnings.join(" ")}>
                          ⚠ {ref.warnings[0]}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Matched Results Header */}
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="text-sm font-semibold text-zinc-100">
                  Matched Photos ({result.matches.length})
                </h3>
                <span className="rounded-full bg-emerald-500/15 px-2.5 py-0.5 font-mono text-[11px] text-emerald-300">
                  Pass-1: {result.pass1_count}
                </span>
                {result.pass2_recovered_count > 0 && (
                  <span className="rounded-full bg-sky-500/15 px-2.5 py-0.5 font-mono text-[11px] text-sky-300">
                    Pass-2 Outfit: +{result.pass2_recovered_count}
                  </span>
                )}
              </div>

              {result.matches.length > 0 && (
                <button
                  type="button"
                  disabled={zipping}
                  onClick={handleDownloadZip}
                  className="inline-flex items-center gap-2 rounded-xl bg-emerald-400 px-4 py-2 text-xs font-semibold text-zinc-950 transition hover:bg-emerald-300 disabled:opacity-50"
                >
                  {zipping ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Download className="h-3.5 w-3.5" />
                  )}
                  Download All ({result.matches.length}) .zip
                </button>
              )}
            </div>

            {/* Matched Grid */}
            {result.matches.length === 0 ? (
              <div className="rounded-2xl border border-zinc-800/80 bg-[#10131b] px-6 py-12 text-center">
                <p className="text-sm font-medium text-zinc-200">
                  No photos matched at threshold τ = {result.threshold.toFixed(2)}
                </p>
                <p className="mt-1 text-xs text-zinc-500">
                  Try lowering the similarity slider slightly (e.g., 0.35), adding Left/Right
                  profile angles, or verifying photos exist in the Pool tab.
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                {result.matches.map((item, idx) => {
                  const isPass2 = item.pass_stage === 2;
                  return (
                    <div
                      key={item.photo.id}
                      onClick={() => setLightboxIdx(idx)}
                      className="group relative cursor-pointer overflow-hidden rounded-xl border border-zinc-800 bg-[#12151f] transition hover:border-amber-400/60"
                    >
                      <div className="relative aspect-[4/3] overflow-hidden bg-zinc-950">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={mediaUrl(item.photo.thumb_url)}
                          alt={item.photo.filename}
                          loading="lazy"
                          className="h-full w-full object-cover transition duration-300 group-hover:scale-105"
                        />
                        <div
                          className={`absolute top-2 left-2 rounded-full px-2 py-0.5 font-mono text-[10px] font-semibold backdrop-blur ${
                            isPass2
                              ? "bg-sky-400/90 text-slate-950"
                              : "bg-emerald-400/90 text-slate-950"
                          }`}
                        >
                          {Math.round(item.score * 100)}%
                        </div>
                        <div className="absolute top-2 right-2 rounded-full bg-black/75 px-2 py-0.5 font-mono text-[10px] uppercase text-zinc-200 backdrop-blur">
                          {isPass2 ? "Pass-2 Outfit" : item.matched_by}
                        </div>
                      </div>
                      <div className="flex items-center justify-between gap-2 px-3 py-2">
                        <span className="truncate text-xs text-zinc-300">
                          {item.photo.filename}
                        </span>
                        <span className="shrink-0 font-mono text-[10px] text-zinc-500">
                          cos {item.face_score.toFixed(2)}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </>
        )}
      </div>

      {result && lightboxIdx !== null && (
        <PhotoLightbox
          items={result.matches.map((m) => ({ photo: m.photo, match: m }))}
          index={lightboxIdx}
          onClose={() => setLightboxIdx(null)}
          onNavigate={setLightboxIdx}
        />
      )}
    </div>
  );
}
