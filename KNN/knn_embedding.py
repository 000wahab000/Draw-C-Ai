"""
knn_embedding.py
================
Replaces raw 1,024-pixel vectors with 512-dim ResNet-18 embeddings.
Everything else (predict, get_all_classes_breakdown) in knn_pixel.py
stays exactly the same — embeddings are just better vectors.

Usage:
    from knn_embedding import get_embedding, build_embedding_dataset
    from knn_pixel import predict, get_all_classes_breakdown

    X, y = build_embedding_dataset(image_paths, labels)
    result = predict(get_embedding("drawing.png"), X, y, k=5)
"""

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torchvision.models as models
# pyrefly: ignore [missing-import]
import torchvision.transforms as T
# pyrefly: ignore [missing-import]
from PIL import Image
import numpy as np

# ── Load ResNet-18 with ImageNet weights ──────────────────────────────────────
# weights=IMAGENET1K_V1: use the weights learned from 1.2M ImageNet photos.
# Without this, weights are random and the network is completely useless.
_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# ── Remove the classification head ───────────────────────────────────────────
# The final fc layer maps 512 → 1000 (one score per ImageNet class).
# We don't want ImageNet classes — we want the 512-dim embedding before that.
# nn.Identity() just returns its input unchanged, so output is now 512-dim.
_backbone.fc = torch.nn.Identity()

# ── Evaluation mode ───────────────────────────────────────────────────────────
# Turns off Dropout and makes BatchNorm use its running stats instead of
# batch stats. Required for deterministic, consistent embeddings at inference.
_backbone.eval()

# ── Image pre-processing pipeline ────────────────────────────────────────────
# ResNet was trained on 224x224 RGB images normalised with these exact
# mean/std values (computed from ImageNet). We must match them exactly --
# different values = different "language" = garbage embeddings.
_transform = T.Compose([
    T.Resize((224, 224)),         # Upscale our 32x32 canvas to what ResNet expects
    T.ToTensor(),                 # PIL Image -> tensor, also scales 0-255 to 0.0-1.0
    T.Normalize(                  # (pixel - mean) / std, per channel
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def get_embedding(image_path: str) -> np.ndarray:
    """
    Convert a PNG file to a 512-dimensional embedding vector.

    This is a drop-in replacement for the raw pixel vector.
    Instead of flattening pixels, we pass the image through ResNet-18
    and extract its internal 512-dim representation.

    Args:
        image_path: Absolute or relative path to a PNG file.

    Returns:
        numpy array of shape (512,), dtype float32.
    """
    # Load image. Convert to RGB because:
    #   - Grayscale images have 1 channel, ResNet needs 3.
    #   - RGBA images have 4 channels, ResNet needs 3.
    #   - RGB is always safe.
    img = Image.open(image_path).convert("RGB")

    # Apply transforms: resize -> tensor -> normalise.
    # Result shape: [3, 224, 224]  (channels, height, width)
    tensor = _transform(img)

    # PyTorch models always expect a batch dimension as the first axis.
    # Even for 1 image, shape must be [batch, channels, height, width].
    # unsqueeze(0) inserts a dimension at position 0.
    # [3, 224, 224] -> [1, 3, 224, 224]
    tensor = tensor.unsqueeze(0)

    # no_grad: we are not training, so don't build the computational graph.
    # Saves memory and runs faster. Output is identical either way.
    with torch.no_grad():
        embedding = _backbone(tensor)   # shape: [1, 512]

    # Remove the batch dimension and convert to numpy.
    # [1, 512] -> [512] -> numpy array (512,)
    return embedding.squeeze().numpy()


def build_embedding_dataset(image_paths: list, labels: list):
    """
    Build X (embeddings) and y (labels) from a list of image files.

    This produces the same (X, y) format that knn_pixel.load_dataset()
    produces -- just with 512-dim vectors instead of 1024-dim pixel vectors.
    The predict() and get_all_classes_breakdown() functions work unchanged.

    Args:
        image_paths: List of paths to PNG files.
        labels:      Corresponding class labels (same length as image_paths).

    Returns:
        X: np.ndarray of shape [N, 512]
        y: list of labels, length N
    """
    X = np.array([get_embedding(p) for p in image_paths], dtype=np.float32)
    return X, list(labels)
