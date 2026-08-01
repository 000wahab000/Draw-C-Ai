# Draw-C-AI — Master Task List

> **Ponytail rule**: stop at the first rung that holds. Ship the lazy version. No speculative abstractions.
> **Role of this doc**: living checklist — update `[ ]` → `[/]` → `[x]` as you go.
> **Target**: FastAPI on Hugging Face Spaces ← portfolio site (GitHub Pages, HTML/JS/CSS) POSTs to it.

---

## Phase 1 — Data Pipeline (the actual hard part)

### 1.1 Capture Tool (already ~80% done)
- [x] 32×32 draw grid (canvas, mousedown/move/up)
- [x] Flatten to 1024-element binary vector
- [x] Save as JSON with label + grid_size
- [ ] Add touch support (portfolio site will be viewed on mobile too)
- [ ] Add "erase" mode (hold Shift or toggle button) — currently all-or-nothing clear
- [ ] Export batch: keep a local queue, download all at once as a zip or NDJSON
- [ ] Visual feedback: show the 32×32 thumbnail next to the canvas so the user sees exactly what gets saved

### 1.2 Reference Dataset — Lucide Icons
- [ ] Run `fetch_lucide_icons.py` — confirm it creates `dataset_v2/raw/` (v1 is the manual SVG stash)
- [ ] Fix path issue: `dataset_manager.get_next_dataset_dir()` searches from cwd; the script is inside `src/scripts/` — pin the base path to the project root
- [ ] Verify output: 8 JSON files, each with 1024-length vector, label correct, values binary (0/1 only)
- [ ] Inspect visuals: write a 5-line debug script to render any JSON vector back to a PNG and eyeball it
- [ ] Decide threshold: currently `< 128` → binary; check if Lucide's thin strokes survive this (they might wash out at 32×32)

### 1.3 Programmatic Hand-Drawn Samples (replaces manual drawing)
- [ ] Write `generate_stylized.py` — for each class:
  - Render the SVG at 3 different stroke widths (cairosvg supports this)
  - Apply perspective skew (simulate hand-held angle)
  - Apply elastic distortion (simulates shaky pen)
  - Target: 15–20 distinct variants per class before augmentation
- [ ] Save each variant to `dataset_v2/raw/` with naming: `{class}_style_{n}.json`
- [ ] Visually spot-check at least 3 per class before proceeding

### 1.4 Augmentation Pipeline
- [x] Rotation (±15°)
- [x] Shift (±3px)
- [x] Noise (salt & pepper 1%)
- [x] Stroke width jitter (dilate/erode)
- [x] Fix the `__main__` block in `augment_data.py` — it looks for `dataset_v*` in cwd, not project root
- [ ] Add: horizontal flip (for asymmetric icons, label-preserve only where valid — e.g. "x" and "square" are flip-safe, "thumbs_up" is not)
- [ ] Add: brightness/contrast jitter (multiply pixel values, simulate faint vs bold strokes)
- [ ] Target multiplier: 10–20x per raw file → with 20 raw files/class × 15x = 300 samples/class = 2400 total
- [ ] Run augmentation, verify count, log a manifest (`manifest.json`: raw count, aug count, per-class breakdown)

### 1.5 Dataset Versioning
- [x] `get_next_dataset_dir()` — auto-increments, never overwrites
- [ ] Add `dataset_info.json` to each version root: timestamp, raw count, aug count, class list, script version
- [ ] Never delete a version directory — if re-running, always bump to next version
- [ ] Store the `dataset_info.json` in git (small, text) — the actual data files in `.gitignore`

---

## Phase 2 — V1: Raw Pixel Nearest Neighbor

### 2.1 Classifier Core
- [x] Write `classifiers/knn_pixel.py`:
  - `load_dataset(aug_dir)` → returns `(X: np.ndarray [N, 1024], y: list[str])`
  - `predict(query_vector, X, y, k=5)` → returns top-k `[(label, distance)]`
  - Distance: Euclidean first (simpler), expose cosine as a flag
  - No classes, no abstractions — just two functions
- [x] Write `test_knn_pixel.py` — load dataset, pick 5 random samples, predict, print results
- [ ] Log failures: for each wrong prediction, save `{query_label}_vs_{predicted_label}_{timestamp}.json` to `logs/failures/`

### 2.2 API Endpoint (Phase 2 version)
- [x] Write `api/app.py` (FastAPI):
  - `POST /predict` — accepts `{"vector": [1024 floats], "mode": "pixel_knn", "include_all_classes": false}` → returns `{"label": str, "confidence": float, "top_k": [...]}`
  - `GET /health` — returns `{"status": "ok"}`
  - Load dataset on startup (in-memory, it's tiny)
- [ ] Add `include_all_classes` flag to `/predict`: when `false` (default) returns top-3 only (fast). When `true` aggregates scores across all 8 classes and returns `all_classes: {label: percentage}`. **Never call with `true` automatically — only on explicit user request.**
- [ ] Test with `curl` or a tiny HTML test page before wiring to portfolio

### 2.3 End-to-End Test
- [ ] Draw → capture → POST to local API → display result in the capture tool UI
- [ ] Show "Show breakdown" button below the prediction result — clicking it fires a second POST with `include_all_classes: true` and renders the percentage bar list
- [ ] Document: which classes get confused and why (log to `logs/phase2_failures.md`)


---

## Phase 3 — V2: Embedding-Based Nearest Neighbor

### 3.1 Feature Extractor Setup
- [ ] Pick extractor: MobileNetV2 pretrained on ImageNet (frozen) — smallest, Colab-friendly
- [ ] Write `classifiers/knn_embedding.py`:
  - Load MobileNetV2 (remove classification head, freeze all weights)
  - `embed(vector_1024)` → resize 32×32 to 224×224 (3-channel), pass through net, return 1280-d embedding
  - Same `predict(query_vector, X_embedded, y, k=5)` interface as Phase 2
- [ ] Pre-compute embeddings for entire dataset, save to `embeddings_v1.npy` (reuse across runs)
- [ ] **No new dependencies if torchvision is already in env** — check requirements first

### 3.2 Comparison: V1 vs V2
- [ ] Run both on the same 20 test drawings (held out from Phase 2)
- [ ] Create `logs/v1_vs_v2_comparison.md`: per-class accuracy table, which classes improved, which regressed
- [ ] Write 1-paragraph explanation of WHY embeddings help (or don't) at 32×32 resolution

---

## Phase 4 — V3: Classification CNN

### 4.1 Dataset Prep
- [ ] Write `train/prepare_data.py`:
  - Load all augmented JSONs
  - Train/val/test split: 70/15/15, stratified by class
  - Save splits as `X_train.npy`, `y_train.npy`, etc. (not re-split each run)

### 4.2 CNN Architecture
- [ ] Write `train/cnn_model.py`:
  - Input: 1×32×32
  - Two conv blocks (Conv → BN → ReLU → MaxPool)
  - One fully connected layer → 8-class softmax
  - ~50K parameters max (anything bigger is overkill for 32×32 binary images)
- [ ] Train on Colab/Kaggle GPU, save checkpoint as `checkpoints/cnn_v1.pth`
- [ ] Log: train/val loss + accuracy per epoch (matplotlib plot saved to `logs/cnn_training.png`)

### 4.3 Evaluation
- [ ] Confusion matrix (sklearn, saved as `logs/confusion_matrix_cnn.png`)
- [ ] Per-class precision, recall, F1 (not just accuracy)
- [ ] Compare CNN confidence scores vs KNN distance scores on the same 20 test drawings
- [ ] Document where they agree and where they diverge — this is the learning, not the number

### 4.4 Export
- [ ] Export to ONNX: `torch.onnx.export(...)` → `models/cnn_v1.onnx`
- [ ] Verify ONNX inference matches PyTorch inference (same input, same output)
- [ ] Add CNN endpoint to `api/app.py`: `"mode": "cnn"`

---

## Phase 5 — V4: Transformer Experiment

### 5.1 ViT Fine-tune
- [ ] Use `timm` library: `vit_tiny_patch16_224` — smallest ViT, Colab-runnable
- [ ] Upsample 32×32 → 224×224 for input
- [ ] Fine-tune last 2 layers only (frozen backbone) for 20 epochs
- [ ] Save checkpoint: `checkpoints/vit_v1.pth`

### 5.2 ConvNeXt Comparison
- [ ] Use `timm`: `convnext_tiny` — same fine-tuning protocol as ViT
- [ ] Save checkpoint: `checkpoints/convnext_v1.pth`

### 5.3 Architecture Comparison Table
- [ ] Log all three (CNN, ViT, ConvNeXt) on identical test set
- [ ] Table: accuracy, F1, inference time (ms), model size (MB), params
- [ ] Write 1-paragraph verdict per architecture — what fits this problem and why

---

## Phase 6 — Integration & API Polish

### 6.1 Unified FastAPI Backend
- [ ] Single `api/app.py` serving all modes behind one endpoint
- [ ] Mode routing: `"mode": "pixel_knn" | "embedding_knn" | "cnn" | "vit" | "convnext"`
- [ ] Load all models on startup (or lazy-load with `@lru_cache`)
- [ ] Add `GET /modes` → returns list of available modes + descriptions

### 6.2 Feedback Loop
- [ ] `POST /feedback` — accepts `{"vector": [...], "predicted": str, "correct": str}`
- [ ] Append to `feedback_log.jsonl` (one JSON per line, never overwrite)
- [ ] Write a one-shot script `retrain_from_feedback.py`: reads feedback log, adds confirmed-correct vectors to next dataset version, triggers augmentation

### 6.3 Portfolio Integration
- [ ] Draw in capture tool (on portfolio page) → POST to HF Spaces API → display result
- [ ] Add mode selector dropdown in the portfolio UI
- [ ] Show top-3 predictions with confidence bars
- [ ] Add CORS config to FastAPI for GitHub Pages origin

### 6.4 Hugging Face Spaces Deployment
- [ ] Write `Dockerfile` for HF Spaces (or use `app.py` + `requirements.txt` directly)
- [ ] Write `README.md` for the Space (required by HF — describes the API)
- [ ] Test cold start time (HF free tier sleeps) — add a health ping from the portfolio JS on page load

---

## Phase 7 — Retrospective

- [ ] Write `docs/retrospective.md`:
  - What worked
  - What you'd do differently
  - Which architecture actually fit the problem and why (spoiler: probably the CNN, but prove it)
  - What the distance scores told you that accuracy didn't
  - What you'd try with more data
- [ ] Update `README.md` with final results table and a link to the live demo
- [ ] Archive all model checkpoints with a `models/README.md` noting which dataset version each was trained on

---

## Cross-Cutting (Do These Continuously)

- [ ] Keep `requirements.txt` updated — add libraries as you reach for them, not before
- [ ] Every script: `if __name__ == "__main__"` block that can be run standalone
- [ ] Every dataset write: check the version number, never silently overwrite
- [ ] Every model: log which dataset version it was trained on inside the checkpoint
- [ ] `logs/` directory: one markdown file per phase summarizing failures and learnings
