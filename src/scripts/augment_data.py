import os
import glob
import json
import numpy as np
import cv2

def load_json_vector(filepath):
    """Loads a JSON file and returns the label and the 32x32 numpy array."""
    with open(filepath, "r") as f:
        data = json.load(f)
        
    # Reconstruct the 2D array from the 1D vector
    grid_size = data["grid_size"]
    vector = data["vector"]
    array_2d = np.array(vector).reshape((grid_size, grid_size))
    
    # We cast to float32 because OpenCV transformations work better with floats
    return data["label"], array_2d.astype(np.float32)

def save_json_vector(label, array_2d, filepath):
    """Saves a 32x32 array back into a flattened 1D JSON vector."""
    grid_size = array_2d.shape[0]
    # Threshold just to be sure it's 0 or 1
    binary_array = (array_2d > 0.5).astype(int)
    vector = binary_array.flatten().tolist()
    
    data = {
        "label": label,
        "grid_size": grid_size,
        "vector": vector
    }
    with open(filepath, "w") as f:
        json.dump(data, f)

# --- AUGMENTATION FUNCTIONS ---
# These functions manipulate the 2D image matrix to create variations.

def rotate_image(image, angle):
    """Rotates the image by a random angle."""
    center = (image.shape[1] // 2, image.shape[0] // 2)
    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Apply the rotation
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return rotated

def shift_image(image, dx, dy):
    """Shifts (translates) the image slightly up/down or left/right."""
    # Translation matrix
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def add_noise(image, noise_level=0.02):
    """Randomly flips a small percentage of pixels (salt and pepper noise)."""
    noisy = np.copy(image)
    # Generate a random matrix of same shape
    rand_matrix = np.random.rand(*image.shape)
    
    # Add noise (flip to 1)
    noisy[rand_matrix < (noise_level / 2)] = 1
    # Remove noise (flip to 0)
    noisy[rand_matrix > (1 - (noise_level / 2))] = 0
    return noisy

def alter_stroke_width(image, operation="dilate"):
    """Makes strokes thicker (dilate) or thinner (erode)."""
    # A 2x2 kernel (brush) for the morphological operation
    kernel = np.ones((2, 2), np.uint8)
    if operation == "dilate":
        return cv2.dilate(image, kernel, iterations=1)
    else:
        return cv2.erode(image, kernel, iterations=1)

# -----------------------------

def process_augmentations(dataset_version_dir, augmentations_per_file=10):
    """
    Reads all raw files in a dataset version, applies random augmentations,
    and saves them in the augmented folder.
    """
    raw_dir = os.path.join(dataset_version_dir, "raw")
    aug_dir = os.path.join(dataset_version_dir, "augmented")
    
    # Find all JSON files in the raw folder
    raw_files = glob.glob(os.path.join(raw_dir, "*.json"))
    
    if not raw_files:
        print(f"No raw files found in {raw_dir}!")
        return

    print(f"Found {len(raw_files)} raw files. Generating augmentations...")
    
    for filepath in raw_files:
        basename = os.path.basename(filepath)
        filename_no_ext = os.path.splitext(basename)[0]
        
        # Load the original
        label, image = load_json_vector(filepath)
        
        # Save the original directly into augmented as the baseline (version 0)
        save_json_vector(label, image, os.path.join(aug_dir, f"{filename_no_ext}_aug0.json"))
        
        # Generate N variations
        for i in range(1, augmentations_per_file + 1):
            aug_img = np.copy(image)
            
            # 1. Random Rotation (-15 to 15 degrees)
            angle = np.random.uniform(-15, 15)
            aug_img = rotate_image(aug_img, angle)
            
            # 2. Random Shift (-3 to 3 pixels in X and Y)
            dx = np.random.randint(-3, 4)
            dy = np.random.randint(-3, 4)
            aug_img = shift_image(aug_img, dx, dy)
            
            # 3. Random Stroke Width (Dilate, Erode, or None)
            stroke_choice = np.random.choice(["dilate", "erode", "none"], p=[0.4, 0.2, 0.4])
            if stroke_choice != "none":
                aug_img = alter_stroke_width(aug_img, stroke_choice)
                
            # 4. Add slight noise
            aug_img = add_noise(aug_img, noise_level=0.01)
            
            # Save the augmented variation
            out_name = f"{filename_no_ext}_aug{i}.json"
            save_json_vector(label, aug_img, os.path.join(aug_dir, out_name))
            
    print(f"Done! Created {len(raw_files) * (augmentations_per_file + 1)} total files in {aug_dir}")

if __name__ == "__main__":
    # To run this, you must specify WHICH dataset version you want to augment.
    # For now, let's just find the highest version folder and use it.
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_pattern = os.path.join(PROJECT_ROOT, "dataset_v*")
    datasets = glob.glob(dataset_pattern)
    if not datasets:
        print("No dataset directories found! Run fetch_lucide_icons.py first.")
    else:
        # Sort to find the latest (e.g., dataset_v2 > dataset_v1)
        latest_dataset = sorted(datasets)[-1]
        print(f"Augmenting data in: {latest_dataset}")
        
        # Create 10 augmentations per raw file
        process_augmentations(latest_dataset, augmentations_per_file=10)
