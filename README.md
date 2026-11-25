# Tasbir — Multi-Pose Event & Group Photo Retrieval

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15-000000?logo=nextdotjs&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-SCRFD--10G_%2B_ArcFace-5C2D91)

**Tasbir** is a zero-training, full-stack Computer Vision web application that retrieves every photo containing a specific person from large event or wedding albums using **multi-angle reference selfies (Left, Front, Right)** or a **short head-turn video**.

Powered by a pure-ONNX **SCRFD-10G** detector, 5-point similarity alignment, **ArcFace ResNet-50 (WebFace600K)** 512-D embeddings, **Multi-Pose Max-Cosine matching**, and **Two-Pass In-Event Outfit Re-ID** for occlusion recovery.

---

## Key Capabilities

- **Zero-Training Open-Set Retrieval:** Works immediately on unseen people without collecting training datasets or retraining classifiers per user.
- **Multi-Angle Reference Enrollment (3 Modes):**
  - Upload **1–3 reference photos** (`Front`, `Left Profile`, `Right Profile`).
  - Capture live via **in-browser WebRTC webcam** (guided 3-angle snapshots or 4.5-second head-turn video).
  - Upload a **3–12s head-turn video** (`.mp4`, `.webm`, `.mov`) with automated 5-point yaw estimation and Laplacian sharpness frame selection.
- **Multi-Pose Template Bank (`Max-Cosine`):** Stores individual pose vectors plus their normalized centroid so candid side-profile event photos match strongly without frontal averaging washout.
- **Two-Pass Outfit & Occlusion Recovery:** Uses high-confidence Pass-1 matches from the same event to build an in-event face + torso outfit anchor, recovering partially occluded or sunglasses shots in Pass 2.
- **Production-Ready Local UX:** EXIF auto-orientation, SHA-256 duplicate protection, `1600px` in-memory detection scaling, `480px` WebP thumbnails, full-resolution lightbox with SCRFD bounding-box overlays, and one-click `.zip` export of matched originals.

---

## System Pipeline

```mermaid
flowchart TD
    subgraph Pool["1. Photo Pool Indexing (Run Once)"]
        direction TB
        P["Group / Event Photos"]
        Pre["EXIF Orient + 1600px Resize + WebP Thumb"]
        Det1["SCRFD-10G Detection + 5-Point Align (112x112)"]
        Emb1["ArcFace 512-D + 96-D Torso Descriptor"]
        DB[("SQLite Metadata + NumPy Vector Store")]

        P --> Pre --> Det1 --> Emb1 --> DB
    end

    subgraph Query["2. Multi-Angle User Query"]
        direction TB
        Ref["3 Reference Photos OR Head-Turn Video"]
        Det2["SCRFD-10G + Yaw Binning + Blur Filter"]
        Bank["Template Bank: Left, Front, Right, Mean"]

        Ref --> Det2 --> Bank
    end

    subgraph Match["3. Two-Pass Retrieval"]
        direction TB
        P1["Pass 1: Max-Cosine Similarity"]
        P2["Pass 2: In-Event Outfit Occlusion Recovery"]
        Out["Ranked Gallery + Face Boxes + ZIP Export"]

        P1 --> P2 --> Out
    end

    DB --> P1
    Bank --> P1
```

### Matching Formulation

Given $L_2$-normalized reference embeddings and an indexed face embedding $\mathbf{f} \in \mathbb{R}^{512}$:

```math
\mathcal{T} = \lbrace \mathbf{e}_{\text{left}},\; \mathbf{e}_{\text{front}},\; \mathbf{e}_{\text{right}},\; \mathbf{e}_{\text{mean}} \rbrace
```

**Pass 1 — Max-Over-Templates Cosine Similarity** (default $\tau = 0.42$):

```math
S_{\text{face}}(\mathbf{f}) = \max_{k \in \mathcal{T}} (\mathbf{f} \cdot \mathbf{e}_k) \ge \tau
```

**Pass 2 — In-Event Outfit Recovery (for borderline or occluded faces):**  
High-confidence Pass-1 anchors ($S_{\text{face}} \ge 0.50$) define an event-specific face anchor and torso descriptor. Borderline faces ($0.25 \le S_{\text{face}} < \tau$) are rescored via:

```math
S_{\text{combined}} = 0.65 \cdot S_{\text{event}} + 0.35 \cdot S_{\text{outfit}}
```

---

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **AI & Computer Vision** | `ONNX Runtime`, `SCRFD-10G` (`det_10g.onnx`), `ArcFace ResNet-50` (`w600k_r50.onnx`, 512-D), `OpenCV`, `NumPy`, `Pillow` |
| **Backend API** | `Python 3.11+`, `FastAPI`, `Uvicorn`, `Pydantic v2` |
| **Frontend Web App** | `Next.js 15` (App Router), `React 19`, `TypeScript`, `Tailwind CSS v4`, `WebRTC` (`getUserMedia` / `MediaRecorder`), `Lucide Icons` |
| **Storage & Testing** | `SQLite`, atomic `.npy` vector persistence, `Pytest`, `HTTPX` |

---

## Quickstart

### Prerequisites
- **Python 3.11+** (with [`uv`](https://docs.astral.sh/uv/) or `pip`)
- **Node.js 20+** and `npm`

### 1. Install Dependencies

**Backend (`backend/`):**
```powershell
cd backend
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```
*(Or with standard `pip`: `python -m venv .venv && .\.venv\Scripts\pip install -r requirements.txt`)*

**Frontend (`frontend/`):**
```powershell
cd frontend
npm install
```

### 2. Run Locally

**One-Command Launch (Windows PowerShell):**
```powershell
.\start.ps1
```

**Or run in two terminals:**
```powershell
# Terminal 1 — Backend API (auto-downloads SCRFD-10G + ArcFace ONNX on first start)
cd backend
.\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — Web Frontend
cd frontend
npm run dev
```

- **Web Application:** `http://localhost:3000`
- **Interactive API Docs (Swagger):** `http://localhost:8000/docs`

### 3. Run Automated Tests

```powershell
cd backend
$env:PYTHONPATH="."
.\.venv\Scripts\pytest.exe -v tests/test_pipeline.py
```

---

## REST API Reference

| Method & Endpoint | Description |
| :--- | :--- |
| `GET /api/health` | Returns model specs (`SCRFD-10G`, `ArcFace`), ONNX provider, limits, and pool stats |
| `GET /api/pool` | Lists all indexed photos with WebP thumbnail URLs and SCRFD face boxes |
| `POST /api/pool/upload` | Bulk indexes up to 200 photos/batch (`<= 15 MB` each), deduplicating by SHA-256 |
| `DELETE /api/pool/{id}` / `DELETE /api/pool` | Removes one photo (compacting vector indices) or clears the entire pool |
| `POST /api/match/photos` | Matches 1–5 reference photos with configurable `threshold` and `enable_pass2` |
| `POST /api/match/video` | Extracts sharpest Left/Front/Right frames from a head-turn video and runs matching |
| `POST /api/download-zip` | Streams a `.zip` archive of full-resolution originals for matched `photo_ids` |

---

## Repository Structure

```text
.
├── README.md                          # Project overview, architecture & quickstart
├── start.ps1                          # One-command local launcher (backend + frontend)
├── docs/
│   └── STUDY_GUIDE.md                 # CV concepts, interview notes & paper references
├── backend/
│   ├── app/
│   │   ├── config.py                  # Storage paths, upload guardrails & thresholds
│   │   ├── db.py                      # SQLite metadata + atomic .npy vector persistence
│   │   ├── vision.py                  # Pure-ONNX SCRFD-10G, 5-pt alignment, ArcFace, yaw & blur
│   │   ├── matcher.py                 # Multi-Pose Max-Cosine + Pass-2 Outfit Re-ID engine
│   │   └── main.py                    # FastAPI server & static media serving
│   ├── tests/
│   │   └── test_pipeline.py           # Pytest suite (DB, matcher, ONNX models, FastAPI)
│   └── requirements.txt
└── frontend/                          # Next.js 15 (TypeScript) + Tailwind CSS v4
    └── src/
        ├── lib/api.ts                 # Typed REST client
        ├── app/page.tsx               # Studio shell & live telemetry header
        └── components/
            ├── PoolPanel.tsx          # Bulk drag-and-drop uploader & indexed gallery
            ├── FinderPanel.tsx        # 3-angle photos, live webcam & video matching UI
            ├── WebcamCapture.tsx      # Guided 3-shot & 4.5s WebRTC head-turn recorder
            └── PhotoLightbox.tsx      # Full-res viewer with SCRFD bounding-box overlays
```

---

## Documentation & Cloud Scaling Roadmap

- **Deep-Dive Notes & Papers:** See [docs/STUDY_GUIDE.md](docs/STUDY_GUIDE.md) for mathematical intuition behind `SCRFD`, `ArcFace`, `AdaFace`, yaw binning, and occlusion handling.
- **Cloud Production Path (Phase 2):** Swap local SQLite + `.npy` for **PostgreSQL + `pgvector` (HNSW)**, replace local media directories with **Azure Blob Storage** (SAS URLs) + **Azure Storage Queues**, and deploy containerized FastAPI/ONNX workers on **Azure Container Apps** with the frontend on **Azure Static Web Apps**.

---

## Author

Made by **Sijan Pandey** — [GitHub Repository](https://github.com/devsijanpandey12/Tasbir---Multi-Pose-Face-Retrieval-)
