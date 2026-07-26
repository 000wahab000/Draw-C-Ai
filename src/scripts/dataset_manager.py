import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_next_dataset_dir(base_path=None):
    """
    Finds the next available dataset version directory.
    If 'dataset_v1' exists, it returns 'dataset_v2', and so on.
    This guarantees we never silently overwrite previous data.
    """
    if base_path is None:
        base_path = os.path.join(PROJECT_ROOT, "dataset")

    version = 1
    while True:
        # Construct folder name like 'dataset_v1'
        dir_name = f"{base_path}_v{version}"
        
        # Check if it already exists
        if not os.path.exists(dir_name):
            return dir_name
        
        # If it exists, increment version and check again
        version += 1

def setup_dataset_directories():
    """
    Creates a new dataset version directory with 'raw' and 'augmented' subfolders.
    Returns the paths to those subfolders.
    """
    # 1. Figure out what the next version should be
    dataset_dir = get_next_dataset_dir()
    
    # 2. Define the subdirectories
    raw_dir = os.path.join(dataset_dir, "raw")
    aug_dir = os.path.join(dataset_dir, "augmented")
    
    # 3. Create the directories
    # os.makedirs creates the parent folder (dataset_vX) AND the subfolders
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(aug_dir, exist_ok=True)
    
    print(f"Created new dataset version at: {dataset_dir}")
    print(f" - Raw data goes in: {raw_dir}")
    print(f" - Augmented data goes in: {aug_dir}")
    
    return dataset_dir, raw_dir, aug_dir

if __name__ == "__main__":
    # If this script is run directly, test the setup function
    setup_dataset_directories()
