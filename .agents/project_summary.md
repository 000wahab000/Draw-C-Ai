# Draw-C-AI — Project Summary for Execution Models

> **Who this is for**: A smaller model or execution agent picking up a phase of this project.
> **Read this entire file before touching any code.**
> **Your planner has already made the big decisions. Your job is to execute cleanly inside those decisions.**

---

## What This Project Is

A **hand-drawn icon classifier** that lets users draw symbols on a 32×32 grid and have an ML model identify them. It is built for a portfolio website hosted on GitHub Pages (static HTML/JS/CSS). The classification API is deployed separately on **Hugging Face Spaces** (FastAPI, Python, free GPU tier).

The user draws → JS captures a 1024-element binary vector → POSTs to HF Spaces API → API returns prediction JSON → JS displays result.

---

## Classes (8, extensible)

| Internal Name | Lucide Icon Name |
|---|---|
| smile | smile |
| square | square |
| triangle | triangle |
| heart | heart |
| fire | flame |
| thumbs_up | thumbs-up |
| zap | zap |
| x | x |

These map to SVG icons downloaded from Lucide's GitHub. Keep this mapping in `CLASSES` dict in scripts.

---

## The One Data Contract — Never Break This

Every sample in this project is:
```json
{
  "label": "heart",
  "grid_size": 32,
  "vector": [0, 1, 0, 1, ...]
}
```
- `vector` is always 1024 integers, values 0 or 1 only
- `grid_size` is always 32
- Files are individual JSONs, one sample per file

The API accepts the raw vector from the browser. Internally, different models preprocess it differently (CNN upsamples to 32×32 tensor, embedding KNN upsamples to 224×224), but the **input to the API is always this 1024-element vector**.

---

## Directory Structure (Station Model)

> Each ML approach owns its station. Shared raw material lives in `data/`. Nothing crosses station boundaries — no model reads another model's augmented data.

```
Draw-C-Ai/
│
├── .agents/                  ← AI planning docs (shared by all agents)
│
├── data/                     ← Data center — shared raw source material only
│   ├── raw/                  ← Base JSONs from SVGs (binary 0/1) + original SVGs
│   └── scripts/              ← fetch_lucide_icons.py, dataset_manager.py
│
├── test_set/                 ← SACRED. 16 files. Shared by ALL models. Never train on these.
│
├── capture-tool/             ← Shared frontend drawing tool (32×32 grid)
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── KNN/                      ← KNN station
│   ├── data/
│   │   ├── raw/              ← 8 processed base JSONs (pulled from data/raw/)
│   │   └── augmented/        ← KNN's own augmented files (binary 0/1)
│   ├── augment.py            ← KNN's augmentation script (light: rotation, shift, noise)
│   ├── knn_pixel.py          ← Euclidean distance over raw 1024-pixel vectors
│   ├── knn_embedding.py      ← Distance over MobileNetV2 embeddings
│   ├── test_knn_pixel.py     ← Evaluates against test_set/
│   ├── embeddings/
│   │   ├── embeddings_v1.npy
│   │   └── labels_v1.npy
│   └── logs/
│       ├── phase2_failures.md
│       └── v1_vs_v2_comparison.md
│
├── CNN/                      ← CNN station
│   ├── data/
│   │   └── augmented/        ← CNN's own augmented data (can be 0-255 grayscale)
│   ├── augment.py            ← CNN's augmentation script (heavier than KNN's)
│   ├── model.py              ← Architecture definition
│   ├── train.py              ← Training loop
│   ├── prepare_data.py       ← Train/val/test split (70/15/15, stratified)
│   ├── weights/
│   │   └── cnn_v1.onnx       ← Exported ONNX model
│   └── logs/
│       ├── confusion_matrix.png
│       └── training_curve.png
│
├── ViT/                      ← Vision Transformer station
│   ├── data/
│   │   └── augmented/        ← ViT's own augmented data (upsampled to 224×224)
│   ├── augment.py
│   ├── model.py              ← vit_tiny_patch16_224 via timm
│   ├── train.py
│   ├── weights/
│   │   └── vit_v1.pth
│   └── logs/
│
├── ConvNeXt/                 ← ConvNeXt station
│   ├── data/
│   │   └── augmented/
│   ├── augment.py
│   ├── model.py              ← convnext_tiny via timm
│   ├── train.py
│   ├── weights/
│   │   └── convnext_v1.pth
│   └── logs/
│
├── api/
│   └── app.py                ← ONE shared FastAPI, mode-routed (pixel_knn → cnn → vit → convnext)
│
├── logs/
│   └── architecture_comparison.md  ← Cross-model results table
├── docs/
│   └── retrospective.md
├── requirements.txt
└── .gitignore
```

## Migration Map (from old `src/` structure)

| Old path | New path |
|---|---|
| `Dataset-v1/` (SVGs) | `data/raw/` |
| `dataset_v2/raw/` (8 JSONs) | `KNN/data/raw/` |
| `dataset_v2/augmented/` (72 files) | `KNN/data/augmented/` |
| `src/capture-tool/` | `capture-tool/` |
| `src/classifiers/knn_pixel.py` | `KNN/knn_pixel.py` |
| `src/classifiers/knn_embedding.py` | `KNN/knn_embedding.py` |
| `src/classifiers/test_knn_pixel.py` | `KNN/test_knn_pixel.py` |
| `src/api/app.py` | `api/app.py` |
| `src/scripts/fetch_lucide_icons.py` | `data/scripts/fetch_lucide_icons.py` |
| `src/scripts/dataset_manager.py` | `data/scripts/dataset_manager.py` |
| `src/scripts/augment_data.py` | `KNN/augment.py` (KNN owns it now) |
| `src/train/` | `CNN/` |
| `models/` | `CNN/weights/`, `ViT/weights/`, `ConvNeXt/weights/` |
| `embeddings/` | `KNN/embeddings/` |



---

## Architecture Decisions (Already Made — Don't Re-Debate)

| Decision | Choice | Why |
|---|---|---|
| Backend framework | FastAPI | Auto-docs, async, ONNX-compatible |
| Hosting | Hugging Face Spaces | Free GPU, Python-native, no infra overhead |
| Model export format | ONNX (for CNN/ViT at serving time) | Smaller image, no PyTorch at runtime |
| Training environment | Google Colab or Kaggle | Free GPU, user's constraint |
| Feature extractor (Phase 3) | MobileNetV2 (frozen, torchvision) | Smallest good backbone, 14MB |
| ViT model (Phase 5) | vit_tiny_patch16_224 via timm | Smallest ViT, Colab-runnable |
| ConvNeXt (Phase 5) | convnext_tiny via timm | Fair comparison point |
| Hand-drawn samples | Programmatically generated from SVGs | Consistent, scalable, no manual drawing |
| Dataset versioning | Auto-increment (dataset_v1, v2, ...) | Never silently overwrite |
| Portfolio site | GitHub Pages (static) | Already deployed, no changes to it |

---

## Phase Map — Quick Reference

| Phase | What You're Building | Key Output |
|---|---|---|
| 1 | Data pipeline | `dataset_v2/` with raw + augmented JSONs |
| 2 | Pixel KNN | `classifiers/knn_pixel.py` + `api/app.py` (first version) |
| 3 | Embedding KNN | `classifiers/knn_embedding.py` + `embeddings_v1.npy` |
| 4 | CNN classifier | `models/cnn_v1.onnx` + confusion matrix |
| 5 | ViT + ConvNeXt | `models/vit_v1.pth`, `models/convnext_v1.pth` + comparison table |
| 6 | API integration | Unified FastAPI + feedback endpoint + HF Spaces deploy |
| 7 | Retrospective | `docs/retrospective.md` |

---

## Known Issues / Gotchas (Do Not Step On These)

### 1. Path issue in existing scripts
`dataset_manager.py` uses `get_next_dataset_dir(base_path="dataset")` which resolves relative to cwd. If you run `fetch_lucide_icons.py` from inside `src/scripts/`, the dataset lands in the wrong place. **Fix**: pass an absolute path using `Path(__file__).parents[2]` to anchor to project root.

### 2. Dataset-v1 is a raw SVG stash, not a processed dataset
The 8 files in `Dataset-v1/` are `.svg` files. They are NOT the `{label, grid_size, vector}` JSONs. The fetch script must run first to produce those.

### 3. Thin SVG strokes may disappear at 32×32
The `< 128` threshold in `fetch_lucide_icons.py` may binarize thin strokes to empty. After running the fetch script, always do a visual ASCII sanity check before proceeding.

### 4. `augment_data.py`'s `__main__` block looks for `dataset_v*` in cwd
Same path issue as #1. It uses `glob.glob("dataset_v*")` — this will fail or pick the wrong directory depending on where you run from. Fix it the same way.

### 5. HF Spaces free tier sleeps on inactivity
The portfolio JS must ping `GET /health` on page load and show a "warming up..." state. Don't let users see a broken app — they'll think the site is broken, not the free hosting tier.

### 6. CORS must be configured for GitHub Pages
The FastAPI app must allow requests from `https://<username>.github.io`. Without this, every browser POST will be silently blocked.

---

## What the API Must Look Like

### `POST /predict`
**Request**:
```json
{
  "vector": [0, 1, 0, ...],
  "mode": "pixel_knn",
  "include_all_classes": false
}
```
**Available modes**: `pixel_knn`, `embedding_knn`, `cnn`, `vit`, `convnext`

**`include_all_classes` flag** (default: `false`):
- `false` → fast default path. Returns top-3 nearest neighbors only. This is what every draw triggers.
- `true` → on-demand breakdown path. Aggregates scores across all 8 classes and returns percentage distribution. Only called when the user explicitly requests the full breakdown (e.g. clicks "Show breakdown" in the UI). Never call this automatically — it costs more compute and defeats the purpose.

**Default response** (`include_all_classes: false`):
```json
{
  "label": "heart",
  "confidence": 0.94,
  "top_k": [
    {"label": "heart", "score": 0.94},
    {"label": "smile", "score": 0.04},
    {"label": "square", "score": 0.02}
  ],
  "mode": "pixel_knn",
  "inference_ms": 12
}
```

**Breakdown response** (`include_all_classes: true`):
```json
{
  "label": "heart",
  "confidence": 0.94,
  "top_k": [
    {"label": "heart", "score": 0.94},
    {"label": "smile", "score": 0.04},
    {"label": "square", "score": 0.02}
  ],
  "all_classes": {
    "heart": 42.1,
    "smile": 23.5,
    "x": 18.2,
    "triangle": 8.1,
    "fire": 4.3,
    "square": 2.1,
    "zap": 1.2,
    "thumbs_up": 0.5
  },
  "mode": "pixel_knn",
  "inference_ms": 18
}
```

**UI contract**: Show the top prediction immediately on draw. Render a "Show breakdown" button below it. Clicking that button fires a second POST with `include_all_classes: true` and renders the percentage bar list. Do NOT fire it automatically.

### `GET /health`
```json
{"status": "ok"}
```

### `GET /modes`
```json
{
  "modes": ["pixel_knn", "embedding_knn", "cnn", "vit", "convnext"],
  "descriptions": {
    "pixel_knn": "Euclidean distance over raw 1024-pixel vectors",
    "embedding_knn": "Distance over MobileNetV2 embeddings",
    "cnn": "Custom 2-layer CNN classifier (ONNX)",
    "vit": "Fine-tuned ViT-Tiny (timm)",
    "convnext": "Fine-tuned ConvNeXt-Tiny (timm)"
  }
}
```

### `POST /feedback`
```json
{
  "vector": [...],
  "predicted": "smile",
  "correct": "heart"
}
```
Appends to `feedback_log.jsonl`. Returns `{"status": "logged"}`.

---

## The Fixed Test Set

Before Phase 2 training, hold out 20 samples (minimum 2 per class) as the permanent test set. These are never trained on. Every model from Phase 2 through Phase 5 is evaluated on this exact same set. Store them in `test_set/` with filenames like `heart_test_001.json`.

**This is the only way the architecture comparison in Phase 5 is valid.**

---

## Ponytail Rules (Active Always)

1. Do not write code until you understand what already exists. Read it first.
2. Reuse the existing format, functions, and patterns before writing new ones.
3. No speculative abstractions. No base classes, no factory patterns, no config systems for values that never change.
4. Fewest files possible. One file per concern.
5. Add to `requirements.txt` only when you actually reach for a library. Check if numpy/PIL/OpenCV already does it first.
6. If a step isn't in the task list, it probably doesn't need to exist yet.

---

## Logging Discipline

Every phase produces logs. Don't skip this. The retrospective (Phase 7) is written from the logs, not from memory.

| File | When to Write |
|---|---|
| `logs/phase2_failures.md` | During Phase 2 — which classes get confused |
| `logs/v1_vs_v2_comparison.md` | After Phase 3 — per-class accuracy table |
| `logs/confusion_matrix_cnn.png` | After Phase 4 training |
| `logs/training_curve_cnn.png` | After Phase 4 training |
| `logs/architecture_comparison.md` | After Phase 5 — full comparison table |
| `docs/retrospective.md` | Phase 7 — written last, never skipped |
