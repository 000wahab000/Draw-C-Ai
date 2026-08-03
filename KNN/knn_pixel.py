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

def get_all_classes_breakdown(query_vector, X, y):
    """
    Returns percentage distribution of scores using a Nearest Centroid approach.
    Instead of summing all samples, we find the average "center" of each class's cluster
    and calculate the distance to those centers for a much cleaner percentage breakdown.
    """
    query = np.array(query_vector, dtype=np.float32)
    
    # 1. Group by class and calculate the 'Average' drawing (Centroid)
    class_centroids = {}
    classes = set(y)
    y_arr = np.array(y)
    
    for c in classes:
        class_X = X[y_arr == c]
        centroid = np.mean(class_X, axis=0)
        class_centroids[c] = centroid
        
    # 2. Calculate distance from your drawing to each of the 8 Centroids
    class_scores = {}
    for c, centroid in class_centroids.items():
        dist = np.linalg.norm(centroid - query)
        # Convert distance to a confidence score
        score = 1.0 / (1.0 + float(dist))
        class_scores[c] = score
        
    # 3. Normalize into percentages
    total = sum(class_scores.values())
    if total == 0: return {}
    
    all_classes = {
        label: round((score / total) * 100, 1) 
        for label, score in class_scores.items()
    }
    
    # Sort from highest to lowest
    return dict(sorted(all_classes.items(), key=lambda item: item[1], reverse=True))
