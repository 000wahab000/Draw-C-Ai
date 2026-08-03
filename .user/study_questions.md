# Study Questions — Draw-C-AI Model Knowledge

> These are questions YOU should be able to answer after building this project.
> Cover them in order. Each section builds on the last.
> If you can answer all of these, you understand the model deeply enough to extend it.

---

## Section 1: Data & Representation

1. What is a "vector" in our context? How does a 32×32 drawing become 1,024 numbers?
2. Why do we use 0 and 1 instead of raw pixel brightness values (0–255)?
3. What does "flattening" a 2D array mean? Why do machine learning models prefer 1D input?
4. Why is the canvas visually 320×320 but logically 32×32? What is PIXEL_SCALE doing?
5. Why can't we just feed raw SVG files into a model?
6. What is "augmentation"? Why did we generate 10 copies of each icon instead of using 1?
7. What problem does augmentation solve? What problem does it NOT solve?

---

## Section 2: KNN (K-Nearest Neighbors)

8. What does "Euclidean distance" mean in plain English?
9. Walk through the 4 steps of comparing two 1,024-dimensional vectors using Euclidean distance (subtract → square → sum → sqrt).
10. Why do we square the differences? What happens if we skip that step?
11. What does a distance of 0.0 mean? What does a very large distance (e.g. 400.0) mean?
12. Why does the formula `1 / (1 + distance)` convert a distance into a confidence score between 0 and 1?
13. What is K in KNN? If K=5, what does the model actually do with the 5 nearest samples?
14. Why did the original breakdown show all 8 classes stuck at ~9–13%? What caused the noise?
15. What is a "centroid"? How does the Nearest Centroid Classifier fix the noise problem?
16. How did we calculate the centroid for the "X" class? (hint: np.mean)
17. Why does KNN fail badly when stroke width changes? What kind of spatial awareness is it missing?
18. Why does KNN get worse as the dataset grows? (curse of dimensionality)

---

## Section 3: CNN (Convolutional Neural Networks)

19. What is a "convolution"? Explain it using a 3×3 filter sliding over an image.
20. What does a CNN "learn" that KNN cannot?
21. Why can a CNN recognise an X even if it's drawn thicker or shifted slightly?
22. What is a "feature map"? What does the first layer of a CNN usually detect?
23. What is "pooling"? Why do we shrink feature maps after convolution?
24. What is the difference between the convolutional layers and the final dense (fully connected) layers in a CNN?
25. What are "weights"? How does a CNN learn them? (gradient descent, backprop — you don't need to know the full math yet, just the concept)
26. Why did CNN perform better than KNN on our data even with the same training set?
27. What is overfitting? How would you know if our CNN was overfitting?

---

## Section 4: The Data Station Model (Architecture)

28. Why do we have a shared `data/` folder but separate `KNN/`, `CNN/`, `ViT/` folders?
29. What is the "data poisoning" risk we were trying to avoid with this layout?
30. Why do we keep augmented data OUT of Git (in `.gitignore`)? What would happen if we committed it?
31. What is the difference between `data/raw/` and `data/augmented/`?
32. Why do different models (KNN vs CNN) need to augment differently?

---

## Section 5: The API

33. What does `FastAPI` do? What is the job of `api/app.py`?
34. What is a POST request? Why do we POST the vector instead of GET?
35. What is CORS? Why did we need to add `CORSMiddleware`?
36. What happens at API startup? Why do we load the dataset into memory once instead of per-request?
37. What is the job of the `/predict` endpoint? What does it return?
38. Why does the API return a confidence as a number between 0 and 1? Where do we convert it to a percentage for display?

---

## Section 6: Next Steps (ViT / Embeddings)

39. What is a "Vision Transformer" (ViT)? How is it different from a CNN at a conceptual level?
40. What is an "embedding"? How is it different from a raw 1,024-pixel vector?
41. What is MobileNetV2? Why would we use a pre-trained model instead of training from scratch?
42. What does "transfer learning" mean?
43. Why would embedding-based nearest neighbor fix the stroke-width problem that KNN failed on?

---

*Add your own answers below each question as you learn. This file is yours.*
