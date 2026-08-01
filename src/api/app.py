import os
import sys
import glob
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Adjust path so we can import from classifiers
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.classifiers.knn_pixel import load_dataset, predict

app = FastAPI(title="Draw-C-AI API")

# Allow portfolio site (GitHub Pages) to hit this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change to your github pages URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state to hold our dataset in memory
DATASET_X = None
DATASET_Y = None

def get_latest_dataset():
    datasets = glob.glob(os.path.join(PROJECT_ROOT, "dataset_v*"))
    if not datasets:
        return None
    return sorted(datasets)[-1]

@app.on_event("startup")
async def startup_event():
    global DATASET_X, DATASET_Y
    dataset_dir = get_latest_dataset()
    if dataset_dir:
        print(f"Loading dataset from {dataset_dir}...")
        try:
            DATASET_X, DATASET_Y = load_dataset(dataset_dir)
            print(f"Loaded {len(DATASET_Y)} samples.")
        except Exception as e:
            print(f"Failed to load dataset: {e}")
    else:
        print("No dataset_v* found in project root!")

class PredictRequest(BaseModel):
    vector: List[int]
    mode: str = "pixel_knn"

@app.get("/health")
def health_check():
    return {"status": "ok", "dataset_loaded": DATASET_X is not None}

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    start_time = time.time()
    
    if req.mode == "pixel_knn":
        if DATASET_X is None:
            return {"error": "Dataset not loaded"}
            
        top_k = predict(req.vector, DATASET_X, DATASET_Y, k=3)
        
        # Best prediction is the first one
        best = top_k[0]
        
        inference_ms = round((time.time() - start_time) * 1000, 2)
        
        return {
            "label": best["label"],
            "confidence": best["score"],
            "top_k": top_k,
            "mode": req.mode,
            "inference_ms": inference_ms
        }
    else:
        return {"error": f"Mode '{req.mode}' not implemented yet."}
