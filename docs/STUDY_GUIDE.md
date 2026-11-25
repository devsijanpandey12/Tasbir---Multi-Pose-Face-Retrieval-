# Tasbir — Computer Vision & Tech Stack Study Guide

> Reference notes covering the core Computer Vision concepts, design decisions, and papers behind **Tasbir**, useful for deep-dive reviews and technical interviews.

---

## 1. Five Core AI / Computer Vision Concepts

### 1.1 Joint Face Detection & 5-Point Affine Alignment (`SCRFD-10G`)
`SCRFD` (*Sample and Computation Redistribution for Efficient Face Detection*) predicts both face bounding boxes and **5 canonical facial landmarks** (left eye, right eye, nose tip, left mouth corner, right mouth corner) in a single forward pass.
- Those 5 points are mapped onto canonical ArcFace coordinates (`ARCFACE_DST`) using a 4-DOF similarity transform (`cv2.estimateAffinePartial2D`) to warp every face into a standardized **`112×112` aligned patch**.
- **Why alignment matters:** Face recognition backbones expect eyes, nose, and mouth corners at consistent pixel coordinates. Skipping alignment or using naive box crops significantly degrades recognition on tilted or candid faces.
- **Why SCRFD replaces YOLOv8 + 68-point Dlib:** Single-pass multi-scale detection runs fast on crowded event photos (detecting dozens of faces including small background guests) and remains reliable on 30°–45° side profiles where Dlib's 68-point regressor frequently fails.

### 1.2 Metric Learning & 512-D Hypersphere Embeddings (`ArcFace` & `AdaFace`)
Instead of training a closed-set classifier (like softmax or retraining KNN/SVM per user), metric-learning models map any human face onto a **512-dimensional unit hypersphere** ($\lVert \mathbf{e} \rVert_2 = 1$) where angular distance reflects identity similarity:
- **ArcFace (Additive Angular Margin Loss):** Enforces an additive angular margin between identities during training on massive web-scale face datasets (e.g., WebFace600K / Glint360K), generalizing **zero-shot** to brand-new people without retraining.
- **AdaFace (Quality-Adaptive Margin):** Uses the pre-normalization embedding $L_2$ norm as a proxy for image quality and adapts the angular margin accordingly, improving robustness under motion blur, low lighting, and unconstrained event conditions.

### 1.3 Cosine Similarity & Multi-Pose Template Matching
Because all 512-D face vectors are $L_2$-normalized, cosine similarity reduces to a dot product:

```math
\text{sim}(\mathbf{a}, \mathbf{b}) = \mathbf{a} \cdot \mathbf{b} \in [-1, 1]
```

A threshold around **τ ≈ 0.40–0.45** cleanly separates same-identity pairs from strangers in unconstrained event photos. Storing **Left, Front, Right, and Mean** separately and evaluating:

```math
S_{\text{face}}(\mathbf{f}) = \max_{k \in \lbrace \text{left}, \text{front}, \text{right}, \text{mean} \rbrace} (\mathbf{f} \cdot \mathbf{e}_k)
```

ensures candid side-profile event shots directly match the user's Left or Right reference embedding rather than getting diluted by an averaged frontal centroid.

### 1.4 Head Pose Yaw Estimation & Sharpness Filtering
When extracting 3 angles automatically from a short head-turn video:
- **Yaw Estimation:** Project the nose tip onto the unit axis connecting the two eyes (`left_eye`, `right_eye`) to estimate horizontal yaw invariant to slight head roll, binning frames into **Left (≤ -14°)**, **Front (-14° to +14°)**, and **Right (≥ +14°)**.
- **Sharpness Filtering:** Compute Laplacian variance (`cv2.Laplacian(gray, cv2.CV_64F).var()`) on each aligned crop to reject motion-blurred video frames and keep only crisp templates.

### 1.5 Handling Occlusions (Layered Strategy)
- **Enrollment Quality Check:** Warn users if a reference selfie is blurry or heavily shadowed so stored templates stay clean.
- **Two-Pass In-Event Outfit Re-ID (Implemented in `matcher.py`):**
  1. **Pass 1:** Discover high-confidence clean anchor matches of the guest inside the event (≥ 0.50).
  2. **Torso Descriptor:** Average the Hellinger-normalized HSV+LAB torso appearance vectors cropped below those confirmed faces (in production scale-up, swap in `OSNet` or `SigLIP`).
  3. **Pass 2:** Re-score borderline/occluded faces (`0.25 <= S_face < τ`) using the fused score below to recover sunglasses or partial side-profile shots from the same event:

```math
S_{\text{combined}} = 0.65 \cdot S_{\text{event}} + 0.35 \cdot S_{\text{outfit}}
```
- **Vision Transformer Extensions:** Spatial-Attention ViT models (`TransFace` / `SwinFace`) tokenize faces into patches and down-weight occluded patches (sunglasses, microphones, masks) via self-attention.

---

## 2. Tech Stack Learning Checklist

### Core Local Stack (Implemented Now)
- **`onnxruntime`:** Cross-platform `.onnx` inference without PyTorch/Cython compile overhead.
- **`OpenCV` (`cv2`) & `NumPy`:** Image preprocessing, affine alignment (`estimateAffinePartial2D`), Laplacian sharpness checks, HSV/LAB descriptors, and vectorized `@` cosine search.
- **`FastAPI` + `Uvicorn`:** Async multipart uploads, Pydantic schemas, static media mounting, and streaming `.zip` archives.
- **`Next.js 15` (React + TypeScript) + `Tailwind CSS`:** Responsive UI, WebRTC `getUserMedia` / `MediaRecorder` webcam capture, and normalized bounding-box overlays.

### Cloud Scale-Up Stack (Phase 2)
- **`PostgreSQL` + `pgvector` (HNSW):** Replace local SQLite + `.npy` matrices when scaling past ~50,000 faces across multi-tenant event rooms.
- **`Docker` & `Microsoft Azure`:** Deploy FastAPI + ONNX workers to **Azure Container Apps**, Next.js to **Azure Static Web Apps**, high-res images to **Azure Blob Storage** (emulated locally via **Azurite**), and bulk jobs to **Azure Storage Queues**.

---

## 3. Key Papers & Resources

- **InsightFace (`buffalo_l` Model Pack):** [github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)
- **SCRFD:** *Sample and Computation Redistribution for Efficient Face Detection* (ICLR 2022)
- **ArcFace:** *ArcFace: Additive Angular Margin Loss for Deep Face Recognition* (CVPR 2019)
- **AdaFace:** *AdaFace: Quality Adaptive Margin for Face Recognition* (CVPR 2022)
