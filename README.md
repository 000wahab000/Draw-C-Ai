# Draw-C-AI

You draw a shape on a 32x32 grid. A locally-running classifier tells you what it thinks you drew. No cloud, no API key, no account.

---

## What it does

- **Draw** on a browser canvas (32x32 logical pixels, 320x320 visual)
- **Classify** using one of two modes you can switch between live:
  - `pixel_knn` - raw Euclidean distance against pixel vectors (fast, baseline)
  - `embedding_knn` - ResNet-18 embeddings + KNN (slower first run, smarter)
- **Train** by saving your own drawings as JSON vectors through the capture tool
- **Compare** multiple model backends - KNN, CNN, ViT, ConvNeXt are all wired up

Current classes: `smile`, `square`, `triangle`, `heart`, `fire`, `thumbs_up`, `zap`, `x`

---

## Architecture

```
capture-tool/     # browser UI to draw and save training data (HTML/JS/CSS)
KNN/              # pixel KNN + ResNet-18 embedding KNN implementations
CNN/              # CNN training notebook + weights
ViT/              # Vision Transformer weights/data
ConvNeXt/         # ConvNeXt weights/data
api/app.py        # FastAPI server - exposes /predict and /health
data/             # raw training JSONs
dataset_v1/       # versioned dataset snapshot
```

The API loads both datasets at startup. First run with no embedding cache takes ~30s to build ResNet-18 embeddings from your pixel vectors. After that it caches to disk and loads in under a second.

---

## Setup

**Requirements:** Python 3.10+, a browser

```bash
pip install -r requirements.txt
```

```bash
# Start the API
cd api
uvicorn app:app --reload
```

Then open `capture-tool/index.html` directly in your browser (no server needed, it hits `localhost:8000`).

---

## Quick test

```bash
# Check the API is up and datasets loaded
curl http://localhost:8000/health
```

Expected:
```json
{"status":"ok","pixel_dataset_loaded":true,"embedding_dataset_loaded":true}
```

```bash
# Send a prediction manually
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.0, ...1024 floats...], "mode": "pixel_knn"}'
```

---

## Adding training data

1. Open `capture-tool/index.html`
2. Pick a class from the dropdown
3. Draw something
4. Hit **Save Vector (JSON)** - saves a file like `smile_1234567890.json`
5. Move that file into `KNN/data/augmented/`
6. Restart the API

The KNN models have no training step - adding a file is adding a data point. Done.

---

## Prediction response

```json
{
  "label": "heart",
  "confidence": 0.87,
  "mode": "pixel_knn",
  "inference_ms": 2.4
}
```

Pass `"include_all_classes": true` to get the full centroid-distance breakdown across all 8 classes.

---

## Notes

- `embedding_knn` on first startup builds ResNet-18 embeddings for every sample in your dataset. Cached to `KNN/embeddings/`. Delete the cache files to rebuild.
- The pixel KNN uses a Nearest Centroid approach for the class breakdown, not raw vote counting. One average drawing per class, then distance-to-centroid normalized into percentages.
- CNN / ViT / ConvNeXt folders have weights and data directories but are not wired to the API yet.

---

## License

MIT
