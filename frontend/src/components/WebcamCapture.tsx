"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Camera,
  CheckCircle2,
  CircleDot,
  RefreshCw,
  Square,
  Video,
} from "lucide-react";
import { ReferenceFileInput } from "@/lib/api";

interface WebcamCaptureProps {
  disabled?: boolean;
  onSubmitPhotos: (inputs: ReferenceFileInput[]) => void;
  onSubmitVideo: (videoFile: File) => void;
}

type PoseSlot = "front" | "left" | "right";

const STEPS: Array<{ key: PoseSlot; title: string; cue: string }> = [
  { key: "front", title: "1. Front View", cue: "Look straight into the camera" },
  { key: "left", title: "2. Left Profile", cue: "Turn your head ~30° left" },
  { key: "right", title: "3. Right Profile", cue: "Turn your head ~30° right" },
];

export default function WebcamCapture({
  disabled,
  onSubmitPhotos,
  onSubmitVideo,
}: WebcamCaptureProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [captureMode, setCaptureMode] = useState<"shots" | "video">("shots");
  const [activeSlot, setActiveSlot] = useState<PoseSlot>("front");
  const [captured, setCaptured] = useState<
    Partial<Record<PoseSlot, { file: File; previewUrl: string }>>
  >({});

  const [recording, setRecording] = useState(false);
  const [recordProgress, setRecordProgress] = useState(0);

  const stopCamera = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
    setRecording(false);
  }, []);

  useEffect(() => {
    return () => {
      stopCamera();
      Object.values(captured).forEach((item) => {
        if (item?.previewUrl) URL.revokeObjectURL(item.previewUrl);
      });
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      setCameraActive(true);
      requestAnimationFrame(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {});
        }
      });
    } catch (err) {
      setCameraError(
        err instanceof Error
          ? `Camera unavailable: ${err.message}`
          : "Could not access webcam. Check browser permissions."
      );
    }
  };

  const captureSnapshot = () => {
    const vid = videoRef.current;
    if (!vid || vid.videoWidth === 0) return;

    const canvas = document.createElement("canvas");
    canvas.width = vid.videoWidth;
    canvas.height = vid.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(vid, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const file = new File([blob], `webcam_${activeSlot}.jpg`, {
          type: "image/jpeg",
        });
        const previewUrl = URL.createObjectURL(blob);
        setCaptured((prev) => {
          const old = prev[activeSlot];
          if (old?.previewUrl) URL.revokeObjectURL(old.previewUrl);
          return { ...prev, [activeSlot]: { file, previewUrl } };
        });

        // Advance smoothly to next missing angle
        const order: PoseSlot[] = ["front", "left", "right"];
        const currentIdx = order.indexOf(activeSlot);
        const nextSlot = order[(currentIdx + 1) % order.length];
        setActiveSlot(nextSlot);
      },
      "image/jpeg",
      0.92
    );
  };

  const startVideoRecording = () => {
    if (!streamRef.current) return;
    chunksRef.current = [];
    setRecordProgress(0);

    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp8")
      ? "video/webm;codecs=vp8"
      : "video/webm";
    const recorder = new MediaRecorder(streamRef.current, { mimeType });
    recorderRef.current = recorder;

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
    };

    recorder.onstop = () => {
      setRecording(false);
      const blob = new Blob(chunksRef.current, { type: "video/webm" });
      if (blob.size > 0) {
        const videoFile = new File([blob], "head_turn_capture.webm", {
          type: "video/webm",
        });
        onSubmitVideo(videoFile);
      }
    };

    recorder.start(250);
    setRecording(true);

    const durationMs = 4500;
    const startTime = Date.now();
    const timer = window.setInterval(() => {
      const elapsed = Date.now() - startTime;
      const pct = Math.min(100, Math.round((elapsed / durationMs) * 100));
      setRecordProgress(pct);
      if (elapsed >= durationMs) {
        window.clearInterval(timer);
        if (recorder.state !== "inactive") {
          recorder.stop();
        }
      }
    }, 100);
  };

  const stopVideoEarly = () => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
  };

  const capturedEntries = (["front", "left", "right"] as PoseSlot[])
    .filter((k) => Boolean(captured[k]))
    .map((k) => ({ file: captured[k]!.file, angle: k }));

  const activeStepMeta = STEPS.find((s) => s.key === activeSlot) ?? STEPS[0];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="inline-flex rounded-lg border border-zinc-800 bg-zinc-950 p-1 text-xs">
          <button
            type="button"
            onClick={() => setCaptureMode("shots")}
            className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 font-medium transition ${
              captureMode === "shots"
                ? "bg-zinc-800 text-amber-300"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <Camera className="h-3.5 w-3.5" />
            Guided 3-Angle Snapshots
          </button>
          <button
            type="button"
            onClick={() => setCaptureMode("video")}
            className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 font-medium transition ${
              captureMode === "video"
                ? "bg-zinc-800 text-amber-300"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <Video className="h-3.5 w-3.5" />
            4.5s Head-Turn Video
          </button>
        </div>

        {cameraActive && (
          <button
            type="button"
            onClick={stopCamera}
            className="text-xs text-zinc-400 underline-offset-4 hover:text-zinc-200 hover:underline"
          >
            Turn off camera
          </button>
        )}
      </div>

      {cameraError && (
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-xs text-rose-200">
          {cameraError}
        </div>
      )}

      {/* Camera stage */}
      <div className="relative overflow-hidden rounded-2xl border border-zinc-800 bg-[#080a0f]">
        {!cameraActive ? (
          <div className="flex flex-col items-center justify-center px-6 py-12 text-center">
            <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-amber-400/10 text-amber-300 ring-1 ring-amber-400/30">
              <Camera className="h-5 w-5" />
            </div>
            <p className="text-sm font-medium text-zinc-100">
              Enable your webcam for guided capture
            </p>
            <p className="mt-1 max-w-sm text-xs text-zinc-400">
              Capture Front, Left, and Right profiles directly in the browser or record a quick 4.5-second head turn.
            </p>
            <button
              type="button"
              onClick={startCamera}
              className="mt-4 inline-flex items-center gap-2 rounded-xl bg-amber-400 px-4 py-2 text-xs font-semibold text-zinc-950 transition hover:bg-amber-300"
            >
              <Camera className="h-4 w-4" />
              Start Camera
            </button>
          </div>
        ) : (
          <div className="relative aspect-video w-full bg-black">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="h-full w-full scale-x-[-1] object-cover"
            />
            {/* Oval framing guide */}
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <div className="h-52 w-40 rounded-full border-2 border-dashed border-amber-300/60 shadow-[0_0_0_9999px_rgba(0,0,0,0.32)] sm:h-60 sm:w-48" />
            </div>

            {/* Live pose instruction pill */}
            <div className="absolute top-3 left-1/2 -translate-x-1/2 rounded-full border border-zinc-700/80 bg-zinc-950/85 px-4 py-1.5 text-xs font-medium text-amber-200 backdrop-blur">
              {captureMode === "shots"
                ? `${activeStepMeta.title} — ${activeStepMeta.cue}`
                : recording
                ? recordProgress < 33
                  ? "Look straight ahead…"
                  : recordProgress < 66
                  ? "Slowly turn LEFT…"
                  : "Now slowly turn RIGHT…"
                : "Click Record, then slowly look Front → Left → Right"}
            </div>

            {recording && (
              <div className="absolute right-4 bottom-4 left-4">
                <div className="h-1.5 overflow-hidden rounded-full bg-zinc-800">
                  <div
                    className="h-full bg-amber-400 transition-all duration-100"
                    style={{ width: `${recordProgress}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Controls below video */}
      {cameraActive && captureMode === "shots" && (
        <div className="space-y-3">
          <div className="grid grid-cols-3 gap-3">
            {STEPS.map((step) => {
              const item = captured[step.key];
              const isSelected = activeSlot === step.key;
              return (
                <button
                  key={step.key}
                  type="button"
                  onClick={() => setActiveSlot(step.key)}
                  className={`group relative overflow-hidden rounded-xl border p-2 text-left transition ${
                    isSelected
                      ? "border-amber-400 bg-amber-400/10"
                      : item
                      ? "border-emerald-500/40 bg-emerald-500/5"
                      : "border-zinc-800 bg-zinc-900/60 hover:border-zinc-700"
                  }`}
                >
                  <div className="flex items-center justify-between text-[11px] font-medium">
                    <span className={isSelected ? "text-amber-300" : "text-zinc-300"}>
                      {step.title}
                    </span>
                    {item && <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />}
                  </div>
                  <div className="mt-2 flex h-16 items-center justify-center overflow-hidden rounded-lg bg-zinc-950">
                    {item ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={item.previewUrl}
                        alt={step.title}
                        className="h-full w-full object-cover"
                      />
                    ) : (
                      <span className="text-[10px] text-zinc-500">Empty</span>
                    )}
                  </div>
                </button>
              );
            })}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={captureSnapshot}
              className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-zinc-100 px-4 py-2.5 text-xs font-semibold text-zinc-950 transition hover:bg-white"
            >
              <Camera className="h-4 w-4" />
              Capture {activeStepMeta.title}
            </button>

            <button
              type="button"
              disabled={disabled || capturedEntries.length === 0}
              onClick={() => onSubmitPhotos(capturedEntries)}
              className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-amber-400 px-4 py-2.5 text-xs font-semibold text-zinc-950 transition hover:bg-amber-300 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Search Pool ({capturedEntries.length}/3 Angles)
            </button>

            {capturedEntries.length > 0 && (
              <button
                type="button"
                onClick={() => {
                  Object.values(captured).forEach((c) => {
                    if (c?.previewUrl) URL.revokeObjectURL(c.previewUrl);
                  });
                  setCaptured({});
                  setActiveSlot("front");
                }}
                aria-label="Reset captures"
                className="rounded-xl border border-zinc-800 bg-zinc-900 p-2.5 text-zinc-400 hover:text-zinc-100"
              >
                <RefreshCw className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>
      )}

      {cameraActive && captureMode === "video" && (
        <div className="flex items-center gap-3">
          {!recording ? (
            <button
              type="button"
              disabled={disabled}
              onClick={startVideoRecording}
              className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-amber-400 px-4 py-2.5 text-xs font-semibold text-zinc-950 transition hover:bg-amber-300 disabled:opacity-40"
            >
              <CircleDot className="h-4 w-4" />
              Record 4.5s Head Turn & Auto-Match
            </button>
          ) : (
            <button
              type="button"
              onClick={stopVideoEarly}
              className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-rose-500 px-4 py-2.5 text-xs font-semibold text-white transition hover:bg-rose-400"
            >
              <Square className="h-3.5 w-3.5 fill-current" />
              Stop & Match Now ({recordProgress}%)
            </button>
          )}
        </div>
      )}
    </div>
  );
}
