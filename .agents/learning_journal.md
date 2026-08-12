# Project Learning Journal

*This document serves as a persistent record of the machine learning concepts, architectural decisions, and mathematical insights discovered during the development of Draw-C-AI.*

## Phase 1: The Data Pipeline
**Concept: Dimensionality & Flattening**
- **The Visual vs. Logical Grid:** The canvas is visually 320x320 pixels for human ease of use, but is mathematically divided by a `PIXEL_SCALE` of 10 to create a 32x32 logical grid. by a lot like I think 80 and then plus the 8 plus the removing of 8 so that's like I don't know so much 80 96 probably 
- **Flattening:** A 32x32 2D grid is flattened into a 1D vector of 1,024 items. This compresses the spatial data into a format that machine learning algorithms can rapidly process.
- **Data Standardization (fetch_lucide_icons.py):** Raw SVGs (which have infinite resolution and transparent backgrounds) must be standardized before feeding them to a model. The pipeline converts them to PNGs, applies a white background, converts to grayscale, resizes using high-quality Lanczos resampling, and enforces a strict `< 128` threshold (0 = background, 1 = drawn line).

## Phase 2: K-Nearest Neighbors (KNN) & Distance Math
**Concept: 1,024-Dimensional Euclidean Distance**
- **The Pythagorean Theorem at Scale:** To compare a newly drawn 1,024-pixel vector against a dataset sample, we don't average them. We subtract pixel 1 from pixel 1, pixel 2 from pixel 2, etc. 
- **Squaring the Errors:** We square each difference (e.g., `-1² = 1`). This ensures all mismatches count as a positive "penalty".
- **The Final Distance:** We sum all 1,024 squared penalties together and take the square root. A perfect match yields a distance of `0.0`.

**Concept: Spatial Rigidity (The Flaw of KNN)**
- **Stroke Width Vulnerability:** KNN measures exact pixel-to-pixel overlap. If a user draws a perfect 'X' but uses a 3-pixel thick stroke, while the dataset 'X' is 1-pixel thick, KNN will penalize the drawing heavily for all the non-overlapping pixels. 
- **Conclusion:** KNN lacks spatial and contextual awareness, which is why we must eventually graduate to Convolutional Neural Networks (CNNs) that recognize spatial features (edges, corners) instead of rigid coordinates.

## The "Nearest Centroid" Breakthrough
**Concept: Clustering vs. Individual Neighbors**
- When calculating confidence percentages across 8 classes, comparing the drawing to every single augmented sample (80 total) creates too much "background noise," dragging the highest confidence scores down (e.g., to ~9%).
- **The Solution:** We group the 10 augmented samples of an 'X' into a single cluster and calculate its mathematical center (`np.mean`). By measuring the distance to just the 8 class **Centroids**, we isolate the signal from the noise, resulting in a significantly cleaner and more accurate percentage breakdown.

---

## Phase 3: Embeddings, ResNet & Transfer Learning
**Date: 2026-08-12 | Time: ~04:25 IST**

### Concept: What is an Embedding?
- A raw pixel vector is 1,024 numbers that say "pixel 7 is black" — no concept of meaning.
- An embedding is a smaller, **learned** vector (512 numbers for ResNet-18) where each number encodes a meaningful visual property: diagonal line presence, symmetry, closed shapes, crossing lines.
- **Key insight:** Two drawings of an X — one thick, one thin — have high Euclidean distance in pixel space (~180) but low distance in embedding space (~8) because ResNet encodes *the concept* (crossing diagonals), not the pixels.

### Concept: ResNet & Transfer Learning
- **ResNet** = Residual Network. A CNN family trained on ImageNet (1.2M photos, 1,000 categories).
- **Residual connections:** Each layer outputs `Layer(x) + x`. The `+ x` skip gives gradients a direct highway backward, preventing vanishing gradients in deep networks.
- **ResNet variants:** 18, 34, 50, 101, 152 layers. For 80 training samples → **ResNet-18** is the right choice (smallest, fast, avoids overfitting).
- **Architecture vs. Weights:** Architecture = blueprint (layer structure). Weights = learned numbers from training. `IMAGENET1K_V1` loads the *weights*, which carry all the visual knowledge.

### Concept: Using ResNet as a Feature Extractor
- ResNet normally ends with: `[512-dim vector] → [fc layer] → [1000 class scores]`
- We replace `backbone.fc` with `torch.nn.Identity()` — this removes the ImageNet classification head and exposes the raw 512-dim embedding.
- `.eval()` turns off Dropout and BatchNorm randomness, making embeddings **deterministic** (same input = same output every time).
- `torch.no_grad()` skips building the gradient graph — we're not training, so this saves memory and speeds up inference.

### Concept: Inference vs. Training
- **Training:** model adjusts its weights to reduce error. Dropout is active. BatchNorm uses batch stats.
- **Inference:** weights are frozen. You're just asking "what is this?". `.eval()` must be called to switch mode.
- **Deterministic** = same input → same output every run. Without `.eval()`, Dropout randomly drops neurons and produces different embeddings each run.

### Concept: Image Preprocessing for ResNet
- ResNet was trained on 224×224 RGB images normalized with ImageNet mean/std.
- We must apply the **exact same transforms** — different preprocessing = different "language" = garbage embeddings.
- `T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` — these numbers come from ImageNet's pixel statistics per channel. Formula: `(pixel - mean) / std`.
- Result is NOT 0–1, it's roughly -2 to +2 — that's what ResNet expects.
- `.unsqueeze(0)` adds a batch dimension: `[3, 224, 224]` → `[1, 3, 224, 224]` because PyTorch always expects `[batch, channels, height, width]`.

### Concept: RGB vs RGBA
- R, G, B = Red, Green, Blue channels.
- A = Alpha = Transparency (0 = invisible, 255 = solid). PNG files support RGBA.
- ResNet needs exactly 3 channels. `.convert("RGB")` safely drops the A channel.

### Concept: Drop-in Replacement Architecture
- `build_embedding_dataset()` returns `X` of shape `[N, 512]` and `y` labels — **same interface** as `knn_pixel.load_dataset()`.
- `predict()` and `get_all_classes_breakdown()` in `knn_pixel.py` require **zero changes** — they just do distance math on whatever vectors they receive.
- Embeddings are just better vectors. The KNN math is identical.

### Files created this phase
- `KNN/knn_embedding.py` — ResNet-18 feature extractor, fully annotated line-by-line.
- `KNN/test_knn_embedding.py` — leave-one-out cross validation comparing pixel vs embedding KNN.

### Experimental Results (2026-08-12 | ~04:55 IST)
Leave-one-out cross validation on 72 augmented samples (8 classes × 9 each):

| Method | Accuracy |
|---|---|
| Pixel KNN (1024-dim) | 69.4% (50/72) |
| Embedding KNN (512-dim, ResNet-18) | **87.5% (63/72)** |
| **Delta** | **+18.1%** |

Per-class results:

| Class | Pixel | Embedding | Δ |
|---|---|---|---|
| fire | 67% | 100% | +33% |
| heart | 89% | 89% | = |
| smile | 44% | 78% | +34% |
| square | 78% | 78% | = |
| thumbs_up | 67% | 100% | +33% |
| triangle | 56% | 78% | +22% |
| x | 100% | 100% | = |
| zap | 56% | 78% | +22% |

**Conclusion:** ResNet embeddings, despite being trained on photographs (not drawings), boosted accuracy by 18 percentage points. The classes that suffered most from pixel-level sensitivity (smile, fire, triangle, zap) showed the largest gains. This experimentally confirms the embedding theory: learned features beat raw pixel comparisons for hand-drawn icon recognition.

---

## Live Test Finding — Distribution Mismatch
**Date: 2026-08-12 | Time: ~19:30 IST**

### What happened
The embedding KNN scored 87.5% in leave-one-out testing but performed near-randomly (~15% per class) in the live capture tool.

### Root cause: Train/Test Distribution Mismatch
- **Training data:** Thin, clean, 1-2px wide lines from programmatic Lucide SVG icons rendered to 32x32.
- **Live input:** Thick, blocky, 3-4px wide strokes drawn by hand with a mouse on the canvas.

To ResNet, these look like completely different images — even though a human would call both "an X."

### Key lesson
**Lab accuracy is meaningless if your test distribution doesn't match real-world input.** 87.5% on a dataset of programmatic icons tells you nothing about how it performs on hand-drawn inputs. This is called **covariate shift** — the input distribution changes between training and deployment.

### What this means for CNN (Phase 4)
The CNN will have the same problem if it's trained only on thin programmatic icons. The fix is training data that actually looks like user drawings — thicker strokes, imperfect lines, different entry points. This is why Steps.md Phase 3 calls for stylized variants and augmentation BEFORE training the CNN.

### Common misconception caught
"KNN can't identify shapes" — **Wrong.** KNN identified shapes at 87.5% accuracy when train and test came from the same distribution. The failure was data, not the algorithm.
