# Study Questions — Draw-C-AI Model Knowledge

> These are questions YOU should be able to answer after building this project.
> Cover them in order. Each section builds on the last.
> Answer in your own words first. Then check. If you hedged, ask yourself why.

---

## Section 1: Data & Representation

1. What is a "vector"? Explain it without using the word "array" or "list."
2. A 32×32 grid becomes 1,024 numbers. Where exactly does that 1,024 come from? What operation produced it?
3. We use 0 and 1 instead of 0–255. What is the name for this operation? What information do we lose by doing it, and why is that acceptable for our use case?
4. The canvas is visually 320×320 but logically 32×32. If PIXEL_SCALE were 5 instead of 10, what would the logical grid size be? Would the model get smarter or dumber? Why?
5. Why can't we feed a raw SVG into a KNN or CNN model? Give the single most important reason.
6. We have 8 icons and generated 10 augmented copies each, giving 80 samples. If we had used 100 augmented copies instead, would the model necessarily get better? What's the risk?
7. Name two things augmentation changes about a sample and two things it intentionally leaves the same.
8. What is the difference between training data and test data? Why must they never mix?

---

## Section 2: KNN (K-Nearest Neighbors)

9. Describe Euclidean distance using only a real-world analogy, no math.
10. Walk through the 4 steps: subtract → square → sum → sqrt. Do it on this tiny example: vector A = [1, 0] and vector B = [4, 4]. What is the distance?
11. Why do we square the differences instead of just taking absolute values? (There are two reasons — find both.)
12. If two drawings are identical, what is their Euclidean distance? If they are completely opposite (one is all 1s, the other all 0s), what is the distance? Calculate it for a 4-pixel vector.
13. The formula `1 / (1 + distance)` gives a score between 0 and 1. What score does a distance of 0 give? Distance of 999? Confirm with the formula, don't guess.
14. K=5 means we look at the 5 nearest samples. If 3 of them are "X" and 2 are "triangle," what does the model predict? What is the confidence?
15. Our breakdown showed all 8 classes at ~12%. In one sentence, explain exactly why this happened.
16. A centroid is the "average" of a cluster. If the X class has 3 samples: [1,0,1], [1,1,0], [0,1,1], what is the centroid? Calculate it.
17. KNN fails when you draw a thick X vs a thin X. What specifically breaks in the distance calculation? Be precise — which step fails?
18. What is the curse of dimensionality? Describe it in plain terms using our 1,024-dimension space as the example.
19. If we have 8 classes, a perfect centroid classifier should give one class ~100% and others near 0%. Ours gives ~13% to the winner. What does that tell you about our centroids?

---

## Section 3: CNN (Convolutional Neural Networks)

20. Describe what a convolution is without using the word "filter" or "kernel." Just describe what it does to the image.
21. A 3×3 filter slides over a 32×32 image. How many positions can it land in? (Don't guess — calculate it.)
22. KNN memorises samples. CNN learns features. What is the difference between memorising and learning in this context?
23. A CNN can recognise a shifted X — if you draw the X in the top-left vs bottom-right, it still recognises it. KNN cannot. Why specifically? What does the CNN have that KNN doesn't?
24. What is a feature map? If a filter detects "top-left diagonal edges," what does the feature map look like for our X icon?
25. What is pooling? Describe what max-pooling does to a 4×4 block. What is the output?
26. A CNN has convolutional layers then dense (fully connected) layers. What is the job of each? Answer in one sentence per layer type.
27. What are weights? Before training, how are they set? After training, what do they represent?
28. What is a loss function? What is it measuring in our classification problem?
29. What is overfitting? Give a concrete example using our specific 8-class dataset.
30. Our CNN was better than KNN on the same data. Explain in one sentence why. Do not say "CNN is more powerful" — be specific about what CNN does differently.

---

## Section 4: The Data Station Model (Architecture)

31. Why is there a shared `data/` folder AND separate `KNN/data/`, `CNN/data/` folders? Why not just one big shared folder for everything?
32. What is "data poisoning" in our context? Give a concrete example of how it could happen in our project if we used one shared augmented folder.
33. If the augmented JSON files were committed to Git, what specific problem would that cause for the commit history?
34. `data/raw/` contains the base SVG-derived PNGs. `data/augmented/` contains the generated copies. If you delete `data/augmented/`, can you fully recover it? If you delete `data/raw/`, can you?
35. KNN's augmentation (slight rotations, shifts, noise) is different from what CNN would need (heavy flips, colour jitter, cutout). Why does KNN need milder augmentation than CNN?

---

## Section 5: The API

36. What is FastAPI? In one sentence, what problem does it solve for us?
37. Explain the difference between GET and POST without using the words "request" or "response."
38. What is CORS? Without it, what exactly would happen when our portfolio site tries to call the API?
39. The dataset is loaded once at startup, not per-request. If the dataset changes on disk after the server is running, will the API see the new data? What would you need to do?
40. The `/predict` endpoint receives a vector of 1,024 numbers and returns a label. Trace exactly what happens in the code between receiving the vector and returning the label — every step.
41. Confidence is returned as 0.0–1.0 but displayed as a percentage. Where in the codebase does that conversion happen? Is it in Python or JavaScript?

---

## Section 6: ViT / Embeddings / Transfer Learning

42. What is an embedding? Contrast it with our raw 1,024-pixel vector. What makes an embedding "smarter"?
43. MobileNetV2 was trained on ImageNet (1.2 million photos of cats, cars, etc.). We want to use it to recognise hand-drawn icons. Why would it still work? What has it already learned that is useful to us?
44. What is transfer learning in one sentence? What are we "transferring"?
45. Why would an embedding-based nearest neighbor solve the stroke-width problem? Trace the logic from "thick X" → embedding → distance → correct prediction.
46. A Vision Transformer (ViT) splits the image into patches and treats them like words in a sentence. What does "attention" do in that analogy? Why might it outperform a CNN on our task eventually?
47. We plan to train small models first on small data, then continue training on large data. What is this called? Why is it better than training on large data from scratch?

---

## Section 7: Debugging & Engineering Thinking

48. The drawing coordinates were scattered and wrong. Before we looked at the code, what was the one question that would have diagnosed the root cause fastest?
49. The confidence was ~13% across all 8 classes. Is this a code bug or a data problem? How do you tell the difference?
50. The "Show breakdown" button disappeared permanently. Name the single line of code responsible and explain in plain English what it was doing wrong.
51. The top prediction said "triangle" but the breakdown said "X" was highest. What does it mean when two parts of the same system give conflicting answers? What is the engineering term for this?
52. If you had to explain to a non-technical person why KNN is not good enough for a real drawing app, what would you say in two sentences?

---

## Section 8: Embeddings + ResNet + KNN (Implementation Study)

> This section is specifically about HOW the code works. Read the annotated
> code block below each question before answering. The goal is that you can
> explain every single line to someone else.

---

### Part A — What IS ResNet? (Concept)

53. A "neural network" is made of layers stacked on top of each other. ResNet
    is just a CNN — it has convolutional layers. The twist is "residual
    connections." In one sentence, what problem do residual connections solve?
    (Hint: think about what happens to gradients in a very deep network.)

54. ResNet-18 means "18 layers deep." ResNet-50 means 50 layers. For our tiny
    8-class drawing problem, which should we use and why? Think about how many
    training samples we have.

55. ResNet was trained on **ImageNet**: 1.2 million photos (dogs, cars, planes,
    furniture). Our data is hand-drawn doodles on a white canvas. Name one
    thing ResNet has already learned from ImageNet that is USEFUL for our task.
    Name one thing it learned that is USELESS.

56. When we use ResNet for embeddings, we **remove the last layer** (called the
    "classification head"). After removing it, what does the network output
    instead of class probabilities? What shape is that output for ResNet-18?

---

### Part B — What IS an Embedding? (Concept)

57. ResNet-18's final layer (before we remove the head) outputs a vector of
    **512 numbers**. Our raw pixel vector is **1,024 numbers**. The embedding
    is actually SMALLER. Why is a smaller, learned vector better than a larger
    raw vector?

58. Two drawings of an X — one thick, one thin. In raw pixel space (1,024
    dims), their Euclidean distance might be 180.0. In embedding space (512
    dims), their distance might be 8.3. Why does the embedding bring them
    closer? What has ResNet "understood" that pixel math cannot?

59. An embedding is sometimes called a "feature vector." What does "feature"
    mean in this context? Give 2 concrete examples of features that ResNet's
    embedding might encode for a drawing of an X.

---

### Part C — The Code (Line by Line)

Read this code carefully. Every line is annotated. Your job is to answer the
questions below WITHOUT looking them up — use only the annotations and what
you now know.

```python
# ─── knn_embedding.py ─────────────────────────────────────────────────────────
import torch                         # PyTorch: the framework that runs neural nets
import torchvision.models as models  # Pre-trained model zoo (ResNet lives here)
import torchvision.transforms as T   # Tools to resize/normalise images before feeding them
from PIL import Image                # Opens image files (.png, .jpg, etc.)
import numpy as np

# ── STEP 1: Load a pre-trained ResNet-18 ─────────────────────────────────────
#
# weights=IMAGENET1K_V1 means: "give me the weights that were learned on
# ImageNet." Without this, the weights are random and the network is useless.
backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# ── STEP 2: Remove the classification head ───────────────────────────────────
#
# backbone.fc is the "fully connected" final layer. It normally outputs 1,000
# numbers (one per ImageNet class). We replace it with nn.Identity() which
# just passes the input through unchanged. Now the output is the 512-dim
# embedding vector instead of 1,000 class probabilities.
backbone.fc = torch.nn.Identity()

# ── STEP 3: Set to evaluation mode ───────────────────────────────────────────
#
# .eval() turns off Dropout and BatchNorm training behavior.
# If you forget this, you'll get different embeddings every run (non-deterministic).
backbone.eval()

# ── STEP 4: Define the image pre-processing pipeline ─────────────────────────
#
# ResNet was trained on images that were:
#   - Resized to 224×224 pixels
#   - Converted to a tensor (a 3D array: [channels, height, width])
#   - Normalised with specific mean/std values (these exact numbers come from ImageNet)
# We MUST apply the exact same transforms. Otherwise we're feeding the network
# a different "language" than it was trained on.
transform = T.Compose([
    T.Resize((224, 224)),       # Resize our 32×32 canvas to 224×224
    T.ToTensor(),               # PIL Image → PyTorch tensor, also scales 0-255 → 0.0-1.0
    T.Normalize(                # Subtract ImageNet mean, divide by ImageNet std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def get_embedding(image_path: str) -> np.ndarray:
    """
    Given a path to a PNG, returns a 512-dimensional embedding vector.
    This is a DROP-IN REPLACEMENT for the raw pixel vector.
    """
    # Open the image and convert to RGB.
    # Our canvas is likely saved as grayscale (L) or RGBA.
    # ResNet expects 3 channels (R, G, B), so .convert("RGB") ensures that.
    img = Image.open(image_path).convert("RGB")

    # Apply the transforms defined above. Shape becomes [3, 224, 224].
    tensor = transform(img)

    # PyTorch expects a batch dimension. .unsqueeze(0) adds it.
    # Shape goes from [3, 224, 224] → [1, 3, 224, 224]
    # (batch_size=1, channels=3, height=224, width=224)
    tensor = tensor.unsqueeze(0)

    # torch.no_grad() tells PyTorch: "don't track gradients."
    # We're not training, so this saves memory and makes it faster.
    with torch.no_grad():
        embedding = backbone(tensor)  # Shape: [1, 512]

    # .squeeze() removes the batch dimension: [1, 512] → [512]
    # .numpy() converts from PyTorch tensor to a regular numpy array
    return embedding.squeeze().numpy()  # Shape: (512,)


# ── STEP 5: Build your embedding dataset ─────────────────────────────────────
#
# Instead of storing raw 1,024-pixel vectors, we store 512-dim embeddings.
# The rest of knn_pixel.py works EXACTLY the same — you just swap the vectors.

def build_embedding_dataset(image_paths: list, labels: list):
    """
    Takes a list of image file paths and their labels.
    Returns X (embeddings) and y (labels) — same format as load_dataset().
    """
    X = np.array([get_embedding(p) for p in image_paths])  # shape: [N, 512]
    y = labels
    return X, y


# ── STEP 6: Predict — this is IDENTICAL to your current KNN ──────────────────
#
# The predict() function in knn_pixel.py takes X and a query vector.
# We just pass it embeddings instead of pixels. That's the entire change.
#
# from knn_pixel import predict
#
# query_embedding = get_embedding("path/to/user_drawing.png")
# results = predict(query_embedding, X_embeddings, y_labels, k=5)
```

---

### Questions About the Code

60. Line: `backbone.fc = torch.nn.Identity()` — What would happen if you
    FORGOT to do this and left the original `fc` layer in place? What would
    `get_embedding()` return instead of a 512-dim vector?

61. Line: `backbone.eval()` — We are not training the network. What does
    `.eval()` actually change at runtime? Name the two layers it affects.

62. The `transform` pipeline resizes our 32×32 image to 224×224. We are making
    the image BIGGER (upscaling). Is this normally a good idea? Why do we do
    it here anyway?

63. `T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` —
    These specific numbers are non-negotiable. Where do they come from? What
    breaks if you use different numbers?

64. `tensor.unsqueeze(0)` — Why does PyTorch want a batch dimension even when
    we are only processing ONE image?

65. `with torch.no_grad():` — If you removed this line, the code would still
    work and give the same output. So why include it?

66. The final function `build_embedding_dataset()` returns `X` of shape
    `[N, 512]`. Your current `predict()` in `knn_pixel.py` expects `X` of
    shape `[N, 1024]`. Do you need to change `predict()` to make it work with
    embeddings? Why or why not?

67. Right now `predict()` uses Euclidean distance. For embeddings, Cosine
    distance is often better. Describe in one sentence what Cosine distance
    measures that Euclidean distance does not.

---

### Part D — Putting It All Together

68. Describe in your own words the complete pipeline from "user finishes
    drawing on canvas" to "model returns a prediction" when using embeddings
    (not raw pixels). List every step.

69. We have 80 augmented samples (8 classes × 10 each). Each is a PNG.
    Building embeddings requires running each PNG through ResNet. How many
    "forward passes" through ResNet does `build_embedding_dataset()` perform?

70. Once the embeddings are built, do we need ResNet at inference time (when
    a user draws something)? Justify your answer.

---

*Write your answers below each question. Don't look up answers before attempting. Wrong answers are fine — write them down anyway.*
