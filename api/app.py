import os
import sys
import time
import tempfile
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from KNN.knn_pixel import load_dataset, predict, get_all_classes_breakdown
from KNN.knn_embedding import get_embedding

app = FastAPI(title="Draw-C-AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pixel KNN dataset
DATASET_X = None
DATASET_Y = None

# Embedding KNN dataset
EMBED_X = None
EMBED_Y = None

DATASET_DIR = os.path.join(PROJECT_ROOT, "KNN", "data")
EMBED_CACHE  = os.path.join(PROJECT_ROOT, "KNN", "embeddings", "embeddings_v1.npy")
LABELS_CACHE = os.path.join(PROJECT_ROOT, "KNN", "embeddings", "labels_v1.npy")


def vector_to_pil(vector) -> Image.Image:
    """Convert a 1024-element 0/1 pixel vector to a 32x32 RGB PIL image."""
    arr = np.array(vector, dtype=np.float32).reshape(32, 32)
    img_array = ((1.0 - arr) * 255).astype(np.uint8)
    return Image.fromarray(img_array, mode="L").convert("RGB")


def vector_to_embedding(vector) -> np.ndarray:
    """Convert a raw 1024-dim pixel vector to a 512-dim ResNet embedding."""
    img = vector_to_pil(vector)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        img.save(tmp_path)
        return get_embedding(tmp_path)
    finally:
        os.unlink(tmp_path)


@app.on_event("startup")
async def startup_event():
    global DATASET_X, DATASET_Y, EMBED_X, EMBED_Y

    # ── Load pixel dataset ────────────────────────────────────────────────────
    if os.path.exists(DATASET_DIR):
        try:
            DATASET_X, DATASET_Y = load_dataset(DATASET_DIR)
            print(f"Pixel dataset loaded: {len(DATASET_Y)} samples")
        except Exception as e:
            print(f"Pixel dataset load failed: {e}")
    else:
        print(f"No dataset found at {DATASET_DIR}")

    # ── Load or build embedding dataset ──────────────────────────────────────
    os.makedirs(os.path.dirname(EMBED_CACHE), exist_ok=True)

    if os.path.exists(EMBED_CACHE) and os.path.exists(LABELS_CACHE):
        EMBED_X = np.load(EMBED_CACHE)
        EMBED_Y = list(np.load(LABELS_CACHE))
        print(f"Embeddings loaded from cache: {EMBED_X.shape}")
    elif DATASET_X is not None:
        print("Building ResNet-18 embeddings (first run, this takes ~30s)...")
        embeddings = []
        for vec in DATASET_X:
            embeddings.append(vector_to_embedding(vec.tolist()))
        EMBED_X = np.array(embeddings, dtype=np.float32)
        EMBED_Y = list(DATASET_Y)
        np.save(EMBED_CACHE, EMBED_X)
        np.save(LABELS_CACHE, np.array(EMBED_Y))
        print(f"Embeddings built and cached: {EMBED_X.shape}")
    else:
        print("Cannot build embeddings: pixel dataset not loaded")


class PredictRequest(BaseModel):
    vector: List[float]
    mode: str = "pixel_knn"
    include_all_classes: bool = False


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "pixel_dataset_loaded": DATASET_X is not None,
        "embedding_dataset_loaded": EMBED_X is not None,
    }


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    start_time = time.time()

    if req.mode == "pixel_knn":
        if DATASET_X is None:
            return {"error": "Pixel dataset not loaded"}

        breakdown = get_all_classes_breakdown(req.vector, DATASET_X, DATASET_Y)
        top_label = list(breakdown.keys())[0]
        top_confidence = breakdown[top_label] / 100.0
        inference_ms = round((time.time() - start_time) * 1000, 2)

        response = {
            "label": top_label,
            "confidence": top_confidence,
            "mode": req.mode,
            "inference_ms": inference_ms,
        }
        if req.include_all_classes:
            response["all_classes"] = breakdown
        return response

    elif req.mode == "embedding_knn":
        if EMBED_X is None:
            return {"error": "Embedding dataset not loaded"}

        # Convert the incoming pixel vector → embedding → KNN
        query_emb = vector_to_embedding(req.vector)

        breakdown = get_all_classes_breakdown(query_emb, EMBED_X, EMBED_Y)
        top_label = list(breakdown.keys())[0]
        top_confidence = breakdown[top_label] / 100.0
        inference_ms = round((time.time() - start_time) * 1000, 2)

        response = {
            "label": top_label,
            "confidence": top_confidence,
            "mode": req.mode,
            "inference_ms": inference_ms,
        }
        if req.include_all_classes:
            response["all_classes"] = breakdown
        return response

    else:
        return {"error": f"Mode '{req.mode}' not implemented yet."}
