import os
import sys
import glob
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Adjust path so we can import from classifiers
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from KNN.knn_pixel import load_dataset, predict, get_all_classes_breakdown

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
    # In station-based layout, KNN dataset is in KNN/data
    dataset_dir = os.path.join(PROJECT_ROOT, "KNN", "data")
    if not os.path.exists(dataset_dir):
        return None
    return dataset_dir

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
    include_all_classes: bool = False

@app.get("/health")
def health_check():
    return {"status": "ok", "dataset_loaded": DATASET_X is not None}

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    start_time = time.time()
    
    if req.mode == "pixel_knn":
        if DATASET_X is None:
            return {"error": "Dataset not loaded"}
            
        # HF-2 Fix: Use centroid math as the single source of truth for predictions
        breakdown = get_all_classes_breakdown(req.vector, DATASET_X, DATASET_Y)
        
        # Extract the highest scoring class from the centroid breakdown
        top_label = list(breakdown.keys())[0]
        # Convert the percentage (e.g. 14.5) back to a 0.0-1.0 confidence score (0.145)
        top_confidence = breakdown[top_label] / 100.0
        
        inference_ms = round((time.time() - start_time) * 1000, 2)
        
        response_data = {
            "label": top_label,
            "confidence": top_confidence,
            "mode": req.mode,
            "inference_ms": inference_ms
        }
        
        if req.include_all_classes:
            response_data["all_classes"] = breakdown
            
        return response_data
    else:
        return {"error": f"Mode '{req.mode}' not implemented yet."}
