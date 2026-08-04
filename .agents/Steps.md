# Draw-C-AI — Steps.md
## How to Execute Each Phase and Why

> **Who this is for**: A model (or developer) picking up this project for the first time.
> **How to use it**: Read the phase section *before* touching any code. Understand the goal, then execute.
> **Ponytail rule in effect**: Every step is the minimum that moves the project forward. If a step isn't here, it wasn't needed yet.

---

## Before You Touch Anything

### Read the codebase first. All of it. This is non-negotiable.
 

The project already has: a
- A working 32×32 draw grid (`src/capture-tool/`) that saves binary JSON vectors
- An SVG fetch + binarize script (`src/scripts/fetch_lucide_icons.py`)
- An augmentation script (`src/scripts/augment_data.py`)
- A dataset versioning utility (`src/scripts/dataset_manager.py`)
- 8 raw SVGs in `Dataset-v1/` (pre-script, not yet processed)

**Do not rewrite what exists. Understand what it does, then extend it.**

The format that everything speaks: `{"label": str, "grid_size": 32, "vector": [1024 ints 0 or 1]}`
Every script, every model, every API call must speak this same format. It is the contract.

---

## Phase 1 — Data Pipeline

### Goal
Produce a clean, consistent, versioned dataset that you can trust. Every bad model decision traces back to bad data. Fix data problems here or you will debug them forever in Phase 4.

---

### Step 1.1 — Fix the script paths before running anything

**Why**: `dataset_manager.py` uses a relative path (`"dataset"`). If you run `fetch_lucide_icons.py` from inside `src/scripts/`, the dataset gets created there, not at the project root. This will silently break augmentation later.

**What to do**: Before running the fetch script, check where `get_next_dataset_dir()` resolves to. If it's wrong, pass an absolute path anchored to the project root.

**How to verify**: After fixing, run the fetch script in a dry-run (print the path, don't create files yet). Confirm it points to `{project_root}/dataset_v2/`.

---

### Step 1.2 — Run the fetch script and inspect the output

**Why**: The 8 SVGs in `Dataset-v1/` are raw vector graphics. They are NOT yet the 32×32 binary JSON files the rest of the system expects. The fetch script is what converts them.

**What to do**:
1. Run `fetch_lucide_icons.py`
2. Open one output JSON (e.g., `heart_lucide_base.json`)
3. Visualize it: write 5 lines of Python that reshape the 1024-vector back to 32×32 and print it as ASCII (use `#` for 1, `.` for 0)

**Why the ASCII check matters**: At 32×32, thin SVG strokes can completely disappear after the `< 128` threshold. If "smile" looks like a blank grid, your threshold is wrong or the SVG stroke is too faint. Fix it now — you cannot fix a bad dataset after training.

**What to look for**: The icon should be recognizable as its class in ASCII. Not perfect — pixelated is expected — but identifiable.

---

### Step 1.3 — Generate stylized variants (the "hand-drawn" proxy)

**Why**: You answered that hand-drawn samples will be generated programmatically. This is the right call — it's consistent and scalable. But "stylized Lucide variants" are still cleaner than real hand drawings. That gap is what augmentation will partially close.

**What to do**: Write `generate_stylized.py`. For each of the 8 classes, produce 15–20 variants by combining:
- **Stroke width variation**: render the SVG at stroke-width 1, 2, 3 (cairosvg supports this via CSS injection)
- **Scale jitter**: render at 28×28 or 30×30 inside a 32×32 canvas (icon slightly smaller, leaves margin)
- **Perspective skew**: use OpenCV's `getPerspectiveTransform` to apply a mild 3–5% skew
- **Elastic distortion**: displace each pixel by a small random amount using a Gaussian kernel (this is what makes it look hand-drawn)

**Target**: 15–20 unique raw files per class before augmentation hits them.

**Why not just augment the 1 Lucide icon 20 times?**: Because augmentation on a single source produces correlated samples. If the base icon has a flaw, every augmentation inherits it. You need diversity at the raw level.

---

### Step 1.4 — Run augmentation and count your data

**Why**: 15–20 raw × 10–15x augmentation = 150–300 samples per class. With 8 classes that's 1200–2400 total. For a CNN doing 8-class classification on 32×32 binary images, that's enough for Phase 4. Don't generate more until you see the confusion matrix — more data in the wrong places is just noise.

**What to do**:
1. Fix `augment_data.py`'s `__main__` block (same path issue as Step 1.1)
2. Run it
3. Count the outputs: `ls dataset_v2/augmented/ | wc -l`
4. Write `manifest.json` into `dataset_v2/`: `{"created": timestamp, "raw_count": N, "aug_count": N, "classes": [...], "aug_factor": 10}`

**What NOT to do**: Do not add more augmentation types speculatively. Rotation, shift, noise, stroke jitter is enough for now. You'll revisit in Phase 4 if the confusion matrix shows specific failure modes.

---

### Step 1.5 — Dataset versioning discipline

**The rule**: Every time you run the pipeline with different parameters, it goes into a new version directory. `dataset_v2` is what Phase 2 trains on. If you retrain with more data in Phase 4, that's `dataset_v3`. Never overwrite.

**Why**: You will need to answer "did accuracy improve because of the architecture or the data?" You can only answer that if you trained both on the same dataset version.

**What to add**: A `dataset_info.json` at the root of each version:
```json
{
  "version": 2,
  "created": "2026-07-26T...",
  "raw_count": 160,
  "aug_count": 2400,
  "classes": ["smile", "square", ...],
  "notes": "Programmatic stylized variants + 10x augmentation"
}
```
Commit this file. Gitignore the actual data JSONs (they're large and binary-ish).

---

## Phase 2 — V1: Raw Pixel Nearest Neighbor

### Goal
Get a working end-to-end pipeline — draw → predict — with the dumbest possible model. This is your baseline. Everything later is measured against it.

### Why start dumb?

Because a 95% accurate KNN means you don't need a CNN. And a 40% accurate CNN means your data is broken, not your model. The baseline forces you to understand your data before you throw architecture at it.

---

### Step 2.1 — Write the KNN in the simplest way possible

**What you need**: Two functions.
- `load_dataset(aug_dir)` → loads all JSONs, returns `(X, y)` where X is shape `[N, 1024]` and y is a list of label strings
- `predict(query, X, y, k=5)` → returns top-k `[(label, distance)]`

**Distance function**: Start with Euclidean. Formula: `np.sqrt(np.sum((query - X)**2, axis=1))`. Sort ascending. Return first k.

**Why not cosine first?**: Euclidean is simpler to reason about and debug. If a class is visually distinct (square vs smile), Euclidean will separate them cleanly. If it doesn't, cosine won't save you — your data is the problem.

**What to log**: Every wrong prediction. Save the query vector, the predicted label, the correct label, and the distance to the nearest correct neighbor vs the wrong one. This is your failure analysis.

---

### Step 2.2 — Build the FastAPI endpoint (Phase 2 version)

**Two routes only**:
- `GET /health` → `{"status": "ok"}` — this is how HF Spaces health-checks your app
- `POST /predict` → accepts `{"vector": [1024 floats], "mode": "pixel_knn"}` → returns `{"label": str, "top_k": [...]}`

**Why FastAPI**: It auto-generates `/docs` (Swagger UI). You can test your API without writing a test client. Free debugging tool.

**Load the dataset on startup** using FastAPI's lifespan context. The entire augmented dataset as numpy arrays in RAM is ~10MB. That's fine.

**CORS**: Add `CORSMiddleware` allowing your GitHub Pages origin. Without this, the portfolio JS call will be blocked by the browser.

---

### Step 2.3 — Wire the capture tool to the API

**In the capture tool JS** (or a separate test page): after the user draws and hits Save, also POST the vector to `http://localhost:8000/predict`. Display the result.

**Why do this now**: You will find bugs in your vector format, your API contract, and your CORS config. Find them in Phase 2 on a dumb model — not in Phase 4 on a model that took 2 hours to train.

---

### Step 2.4 — Write the failure log

Create `logs/phase2_failures.md`. For each class, record:
- How often it was the correct answer but predicted as something else
- What it was most often confused with
- Visually: do those two classes actually look similar at 32×32?

This document drives your decision in Phase 3. If "thumbs_up" is always confused with "zap" because both have upward-pointing shapes, you know you need better features.

---

## Phase 3 — V2: Embedding-Based Nearest Neighbor

### Goal
Replace the raw 1024-pixel vector with a 1280-dimensional embedding from a pretrained neural network. Same nearest-neighbor logic, different (richer) feature space.

### Why this step before training a CNN?

Because it answers the question: "Is the problem hard because the features are bad, or because the classifier is bad?" If embedding KNN beats pixel KNN significantly, the features were the bottleneck — and a CNN might not be necessary. If embedding KNN is only marginally better, the problem is your data diversity.

---

### Step 3.1 — Pick your feature extractor

**Use MobileNetV2** (available in torchvision, ~14MB). Remove the final classification layer. The output is a 1280-dimensional vector per image.

**Why not a bigger model?**: Because your input is 32×32 binary images upsampled to 224×224. A bigger model will not extract meaningfully better features from this input. It will just take longer to run.

**Why pretrained ImageNet?**: Because even though your images look nothing like ImageNet photos, the early layers learn edges, curves, and shapes — which is exactly what you need for icon recognition.

---

### Step 3.2 — Pre-compute embeddings for the entire dataset

**What to do**:
1. Load all augmented JSONs
2. For each: reshape to 32×32, upsample to 224×224, convert to 3-channel (repeat grayscale 3 times), normalize with ImageNet mean/std
3. Pass through the frozen MobileNetV2 backbone
4. Save the resulting `[N, 1280]` array to `embeddings_v1.npy`

**Why save to disk**: Inference through even a small network takes seconds per batch. You don't want to re-run it every time you test. Pre-compute once, load on startup.

**At API serving time**: The query vector goes through the same preprocessing → embedding → KNN lookup. The precomputed embeddings are the reference database.

---

### Step 3.3 — Direct comparison: V1 vs V2

**Hold out 20 drawings** before Phase 2 training. These 20 are your fixed test set. Use them for every comparison from Phase 2 through Phase 5.

**Why a fixed test set?**: If you draw new test cases each time, you can't tell if accuracy improved because of the model or because the new test cases were easier.

**Document in `logs/v1_vs_v2_comparison.md`**:
- Per-class accuracy: pixel KNN vs embedding KNN
- Which classes improved the most?
- Which regressed (if any)?
- One sentence explanation per class: why did the embedding help (or not) here?

---

## Phase 4 — V3: Classification CNN

### Goal
Train an actual classifier on the same dataset. This is the first time you use a model that *learns* from your data rather than just comparing distances.

### Wait until you have the Phase 2 and 3 failure analysis

Do not start Phase 4 until you have written `logs/phase2_failures.md` and `logs/v1_vs_v2_comparison.md`. The confusion patterns you document there directly inform which augmentations to add for `dataset_v3` (if needed) before CNN training.

---

### Step 4.1 — Prepare the data splits

**70% train / 15% val / 15% test — stratified by class.**

Why stratified? If "fire" happens to have fewer samples (edge case in augmentation), a random split might put all of them in train and none in test. Stratified ensures every class has representation in all splits.

**Save the splits to disk as `.npy` files.** Do not re-split on every run. The test set must be identical across all experiments. If you re-split, your comparison between CNN and ViT is invalid.

---

### Step 4.2 — Keep the CNN tiny

Architecture:
```
Input: [1, 32, 32]
Conv(1→16, 3×3) → BN → ReLU → MaxPool(2×2)   → [16, 16, 16]
Conv(16→32, 3×3) → BN → ReLU → MaxPool(2×2)   → [32, 8, 8]
Flatten → Linear(2048 → 8) → Softmax
```

**Why this small?** Your input is 32×32 binary pixels. A ResNet-50 would overfit immediately. You have at most ~2400 samples. Match your model capacity to your data size. Start here and only add complexity if the val curve shows underfitting.

**Training**: 50 epochs, Adam optimizer, lr=1e-3, reduce on plateau. Log train and val loss every epoch to a CSV. You'll plot this.

---

### Step 4.3 — Evaluate properly

**Accuracy alone is not enough.** A model that gets 95% accuracy but always fails on "fire" is useless if fire is the coolest icon on your portfolio.

**What to produce**:
- Confusion matrix (saved as PNG)
- Per-class precision, recall, F1 (a table in a markdown file)
- Training curve (loss + accuracy vs epoch, saved as PNG)

**The comparison you're building toward**: CNN confidence score (softmax probability) vs KNN distance score on the same 20 test drawings. They measure different things — one is "how sure am I?" and the other is "how close is the nearest neighbor?" When they agree, you're confident. When they disagree, that's interesting — log those cases.

---

### Step 4.4 — Export to ONNX

**Why ONNX**: ONNX is a portable format. The HF Spaces API can load it with `onnxruntime` without needing the full PyTorch training stack. Smaller Docker image, faster cold start.

**How**: `torch.onnx.export(model, dummy_input, "models/cnn_v1.onnx", ...)`. Then verify: load the ONNX model with `onnxruntime`, pass the same dummy input, check the output matches PyTorch's output.

---

## Phase 5 — V4: Transformer Experiment

### Goal
Answer the question: "Does attention-based architecture outperform convolution on this specific task?" Not a general question — specific to your data, your class set, your resolution.

### The honest expectation

ViT was designed for large-scale vision. Fine-tuning `vit_tiny_patch16_224` on 2400 samples of 32×32 binary images upsampled to 224×224 is going to be a tough comparison. It might lose to your tiny CNN. That is a valid and interesting result. Document it.

---

### Step 5.1 — Use `timm` for all three models

`timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=8)` — that's your ViT.
`timm.create_model("convnext_tiny", pretrained=True, num_classes=8)` — that's your ConvNeXt.

**Freeze the backbone, train only the head for 20 epochs.** Then unfreeze and fine-tune at a lower lr for another 20 epochs (two-phase training). This prevents the pretrained weights from being destroyed by your small dataset.

**Use the exact same train/val/test split as Phase 4.** Same splits, same random seed for any stochastic elements. The only variable changing is the architecture.

---

### Step 5.2 — Architecture comparison table

| Architecture | Params | Test Accuracy | F1 (macro) | Inference (ms) | Model Size |
|---|---|---|---|---|---|
| KNN (pixel) | — | ? | ? | ? | — |
| KNN (embedding) | 14M (frozen) | ? | ? | ? | — |
| CNN (custom) | ~50K | ? | ? | ? | ~200KB |
| ViT-Tiny | 5.7M | ? | ? | ? | ~22MB |
| ConvNeXt-Tiny | 28M | ? | ? | ? | ~109MB |

Fill this table. The portfolio site can display it. It's proof you understand what you built.

---

## Phase 6 — Integration & API

### Goal
One FastAPI app that serves all modes. Portfolio JS draws, POSTs a vector, gets a prediction. User can switch modes.

---

### Step 6.1 — Single endpoint, mode-routed

```
POST /predict
{
  "vector": [1024 floats],
  "mode": "pixel_knn" | "embedding_knn" | "cnn" | "vit" | "convnext"
}
```

**Response**:
```json
{
  "label": "heart",
  "confidence": 0.94,
  "top_k": [
    {"label": "heart", "score": 0.94},
    {"label": "smile", "score": 0.04},
    {"label": "square", "score": 0.02}
  ],
  "mode": "cnn",
  "inference_ms": 12
}
```

**Why `inference_ms`?**: The portfolio can display it. It makes the "CNN vs ViT" comparison visceral — the user sees that ViT took 5x longer for 2% more accuracy.

---

### Step 6.2 — Feedback endpoint

```
POST /feedback
{
  "vector": [1024 floats],
  "predicted": "smile",
  "correct": "heart"
}
```

Append to `feedback_log.jsonl`. One JSON object per line. Never overwrite.

**Why JSONL?**: It's appendable. You can `cat` it, `grep` it, load it with one `json.loads(line)` per line. No database needed.

**The retraining script** (`retrain_from_feedback.py`) reads this file, extracts vectors with confirmed labels, and adds them to the next dataset version. This is Phase 1's `dataset_manager.py` doing its job.

---

### Step 6.3 — Hugging Face Spaces deployment

**Structure**:
```
hf_space/
  app.py              ← FastAPI app entry point
  requirements.txt    ← onnxruntime, fastapi, uvicorn, numpy, timm
  models/
    cnn_v1.onnx
  embeddings/
    embeddings_v1.npy
    labels_v1.npy
  dataset/            ← only the augmented JSONs for KNN modes
  README.md           ← HF Space metadata (title, SDK: docker or gradio)
```

**Cold start**: HF free tier pauses after inactivity. On your portfolio page, send a `GET /health` ping on page load. If it takes 10s to respond, show a loading spinner. Tell the user it's warming up — don't let them stare at a broken page.

---

## Phase 7 — Retrospective

### This is the one everyone skips. Don't.

**What to write** (`docs/retrospective.md`):

1. **What worked?** — Be specific. Not "the CNN worked." Which augmentations made the biggest difference? Which class was always easy to predict and why? What did the confusion matrix tell you that accuracy didn't?

2. **What would you do differently?** — Now that you've been through it, what was the biggest time sink? Where did you over-engineer? Where did you under-prepare?

3. **Which architecture fit the problem best and why?** — Answer this with numbers from your comparison table, not opinion. "CNN won because at 32×32 binary input, spatial locality (which convolution captures) matters more than long-range attention (which ViT captures), and our dataset is too small to fine-tune a 5.7M-param model effectively."

4. **What did distance scores vs confidence scores tell you?** — These are two different signals. A high-confidence CNN prediction with a large KNN distance means "the model is sure but the nearest training example is far away." That's a flag. Log the cases where they diverged.

5. **What would you try with more data?** — More classes? Different resolution? Color information (not just binary)? Real hand-drawn samples (not programmatically generated)? Be honest about what your current dataset can't represent.

---

## Cross-Cutting Rules (Apply Always)

### File format contract — never break this
Every sample, everywhere, is: `{"label": str, "grid_size": 32, "vector": [1024 ints]}`
API input is the same vector. Embedding and CNN receive preprocessed versions, but the contract starts here.

### Dataset versioning — never skip this
If you change anything about how data is generated or augmented, bump the dataset version. Log what changed in `dataset_info.json`.

### Logging — do it as you go, not at the end
- Phase 2: `logs/phase2_failures.md`
- Phase 3: `logs/v1_vs_v2_comparison.md`
- Phase 4: `logs/confusion_matrix_cnn.png`, `logs/training_curve_cnn.png`
- Phase 5: `logs/architecture_comparison.md`

### The test set is sacred
The 20 test drawings from Phase 2 are held out forever. Never train on them. Never swap them out. Every model is evaluated on the same 20.

### Add dependencies only when you reach for them
`requirements.txt` grows one line at a time. Before adding a library, check if numpy, PIL, or OpenCV already does it.
