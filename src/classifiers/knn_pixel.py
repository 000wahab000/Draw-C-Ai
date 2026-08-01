import os
import glob
import json
import numpy as np

def load_dataset(dataset_dir):
    """
    Loads all JSON files in the dataset's 'augmented' folder.
    Returns:
        X: np.ndarray of shape [N, 1024]
        y: list of string labels of length N
    """
    aug_dir = os.path.join(dataset_dir, "augmented")
    files = glob.glob(os.path.join(aug_dir, "*.json"))
    
    if not files:
        raise ValueError(f"No JSON files found in {aug_dir}")
        
    X_list = []
    y_list = []
    
    for filepath in files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            X_list.append(data['vector'])
            y_list.append(data['label'])
            
    return np.array(X_list, dtype=np.float32), y_list

def predict(query_vector, X, y, k=5):
    """
    Euclidean nearest neighbor.
    query_vector: list or np.ndarray of shape [1024]
    X: np.ndarray of shape [N, 1024]
    y: list of length N
    Returns: list of dicts [{"label": str, "score": float}] (Top K)
    """
    query = np.array(query_vector, dtype=np.float32)
    
    # Euclidean distance (vectorized)
    distances = np.linalg.norm(X - query, axis=1)
    
    # Get indices of the k smallest distances
    # np.argsort returns indices that would sort the array
    k_indices = np.argsort(distances)[:k]
    
    # We convert distances to a pseudo-confidence score where 0 distance = 1.0 confidence.
    # score = 1 / (1 + distance)
    results = []
    for idx in k_indices:
        dist = float(distances[idx])
        score = 1.0 / (1.0 + dist) 
        results.append({
            "label": y[idx],
            "distance": round(dist, 4),
            "score": round(score, 4)
        })
        
    return results
