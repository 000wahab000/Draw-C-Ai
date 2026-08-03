# Project Learning Journal

*This document serves as a persistent record of the machine learning concepts, architectural decisions, and mathematical insights discovered during the development of Draw-C-AI.*

## Phase 1: The Data Pipeline
**Concept: Dimensionality & Flattening**
- **The Visual vs. Logical Grid:** The canvas is visually 320x320 pixels for human ease of use, but is mathematically divided by a `PIXEL_SCALE` of 10 to create a 32x32 logical grid. by a lot like I think 80 and then plus the 8 plus the removing of 8 so that's like I don't know so much 80 96 probably 
- **Flattening:** A 32x32 2D grid is flattened into a 1D vector of 1,024 items. This compresses the spatial data into a format that machine learning algorithms can rapidly process.
- **Data Standardization (fetch_lucide_icons.py):** Raw SVGs (which have infinite resolution and transparent backgrounds) must be standardized before feeding them to a model. The pipeline converts them to PNGs, applies a white background, converts to grayscale, resizes using high-quality Lanczos resampling, and enforces a strict `< 128` threshold (0 = background, 1 = drawn line).

## Phase 2: K-Nearest Neighbors (KNN) & Distance Math
**Concept: 1,024-Dimensional Euclidean Distance**
- **The Pythagorean Theorem at Scale:** To compare a newly drawn 1,024-pixel vector against a dataset sample, we don't average them. We subtract pixel 1 from pixel 1, pixel 2 from pixel 2, etc. 
- **Squaring the Errors:** We square each difference (e.g., `-1² = 1`). This ensures all mismatches count as a positive "penalty".
- **The Final Distance:** We sum all 1,024 squared penalties together and take the square root. A perfect match yields a distance of `0.0`.

**Concept: Spatial Rigidity (The Flaw of KNN)**
- **Stroke Width Vulnerability:** KNN measures exact pixel-to-pixel overlap. If a user draws a perfect 'X' but uses a 3-pixel thick stroke, while the dataset 'X' is 1-pixel thick, KNN will penalize the drawing heavily for all the non-overlapping pixels. 
- **Conclusion:** KNN lacks spatial and contextual awareness, which is why we must eventually graduate to Convolutional Neural Networks (CNNs) that recognize spatial features (edges, corners) instead of rigid coordinates.

## The "Nearest Centroid" Breakthrough
**Concept: Clustering vs. Individual Neighbors**
- When calculating confidence percentages across 8 classes, comparing the drawing to every single augmented sample (80 total) creates too much "background noise," dragging the highest confidence scores down (e.g., to ~9%).
- **The Solution:** We group the 10 augmented samples of an 'X' into a single cluster and calculate its mathematical center (`np.mean`). By measuring the distance to just the 8 class **Centroids**, we isolate the signal from the noise, resulting in a significantly cleaner and more accurate percentage breakdown.
