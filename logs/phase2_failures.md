# Phase 2 — Pixel KNN Failure Analysis

**Dataset**: dataset_v2 (72 training samples)
**Test set**: 16 samples (2 per class)
**Accuracy**: 12/16 = 75.0%

## Misclassifications

| File | True Label | Predicted | Dist to Wrong | Dist to Correct |
|---|---|---|---|---|
| fire_lucide_base_aug1.json | fire | triangle | 12.2066 | not in top-5 |
| square_lucide_base_aug10.json | square | triangle | 12.1244 | not in top-5 |
| square_lucide_base_aug8.json | square | triangle | 14.5602 | 14.7309 |
| triangle_lucide_base_aug1.json | triangle | thumbs_up | 14.2478 | 14.3178 |

## Notes

With only 8 raw icons and 10x augmentation, the augmented samples are highly correlated. High accuracy here does NOT mean the model generalizes to real hand-drawn input. Revisit after growing the dataset (Phase 1.3: generate_stylized.py).
