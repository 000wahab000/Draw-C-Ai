import os
import json
import urllib.request
import fitz
from PIL import Image, ImageOps
import numpy as np

# We import the directory management function we just wrote!
from dataset_manager import get_next_dataset_dir, setup_dataset_directories

# Grid size for our images (32x32 pixels)
GRID_SIZE = 32

# The classes you requested. We map your names to the official Lucide icon names.
CLASSES = {
    "smile": "smile",
    "square": "square",
    "triangle": "triangle",
    "heart": "heart",
    "fire": "flame",        # 'flame' is the lucide equivalent for fire
    "thumbs_up": "thumbs-up",
    "zap": "zap",
    "x": "x"
}

def download_and_process_icon(class_name, lucide_name, output_dir):
    """
    Downloads a Lucide SVG, converts it to a 32x32 pixel vector, and saves it.
    """
    print(f"Processing {class_name}...")
    
    # 1. Download SVG from Lucide's GitHub repository
    url = f"https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/{lucide_name}.svg"
    
    # We save it temporarily as an SVG file
    temp_svg = f"temp_{lucide_name}.svg"
    try:
        urllib.request.urlretrieve(url, temp_svg)
    except Exception as e:
        print(f"  -> Failed to download {lucide_name}: {e}")
        return

    # 2. Convert the SVG to a PNG image
    temp_png = f"temp_{lucide_name}.png"
    # We output a 320x320 PNG first so we get a good clean render
    # Convert the SVG to a PNG image using PyMuPDF (fitz)
    doc = fitz.open(temp_svg)
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(10, 10)) # Scale 10x for a clean render
    pix.save(temp_png)
    doc.close()
    
    # 3. Open the image using Pillow (Python Imaging Library)
    img = Image.open(temp_png).convert("RGBA")
    
    # 4. Create a white background (SVGs are transparent)
    bg = Image.new("RGBA", img.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(bg, img)
    
    # 5. Convert to Grayscale (black and white)
    gray_img = alpha_composite.convert("L")
    
    # 6. Resize it down to our target grid size (32x32)
    # We use LANCZOS resizing for high quality downsampling
    resized_img = gray_img.resize((GRID_SIZE, GRID_SIZE), Image.Resampling.LANCZOS)
    
    # 7. Convert the image into a NumPy array (a matrix of numbers)
    # The pixels are 0-255 (0 = black, 255 = white)
    img_array = np.array(resized_img)
    
    # 8. We need to invert it so that 1 = drawn (black) and 0 = background (white)
    # We threshold it: anything darker than 128 becomes 1, else 0
    # Because original image has white as 255, we do: pixel < 128
    binary_array = (img_array < 128).astype(int)
    
    # 9. Flatten the 2D array (32x32) into a 1D vector (1024)
    # This is exactly what our JS script does!
    vector = binary_array.flatten().tolist()
    
    # 10. Save it to JSON format matching our capture tool
    data_obj = {
        "label": class_name,
        "grid_size": GRID_SIZE,
        "vector": vector
    }
    
    output_path = os.path.join(output_dir, f"{class_name}_lucide_base.json")
    with open(output_path, "w") as f:
        json.dump(data_obj, f)
        
    print(f"  -> Saved to {output_path}")

    # Cleanup temporary files
    os.remove(temp_svg)
    os.remove(temp_png)

if __name__ == "__main__":
    print("Setting up dataset directories...")
    # This creates dataset_v1 (or v2, v3...)
    dataset_dir, raw_dir, aug_dir = setup_dataset_directories()
    
    print("\nFetching baseline Lucide icons...")
    for class_name, lucide_name in CLASSES.items():
        download_and_process_icon(class_name, lucide_name, raw_dir)
        
    print("\nDone! You can now use the Web Capture Tool to draw more samples and put them in the 'raw' folder.")
