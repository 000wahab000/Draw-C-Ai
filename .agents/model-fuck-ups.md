# Model Fuck-Ups Log

> Track every time an AI model goes rogue and does something it wasn't asked to do.

---

## Incident #1

**Model**: Claude Opus 4.6 (Thinking)
**Date**: 2026-08-01
**Time**: ~14:23 IST

### What was asked
Audit the `.agents` files, check correctness, plan Phase 2, think through consequences. **Explicitly told: "your job is not writing code or editing it."**

### What was done instead
After the audit plan was auto-approved by the system, the model treated that as a green light to execute and did all of the following **without user permission**:

1. **Edited 4 project files** (task.md, requirements.txt, .gitignore, project_summary.md) — doc/config fixes, arguably okay but still not asked for
2. **Deleted `dataset_v1/`** — an entire directory, gone
3. **Created and ran `create_test_set.py`** — moved 16 files from `dataset_v2/augmented/` into a new `test_set/` directory
4. **Rewrote `src/classifiers/test_knn_pixel.py`** — replaced the entire file
5. **Ran the KNN evaluation** — executed code on the project

### Why it happened
The system auto-approved the implementation plan, and the model interpreted that as "proceed to execution" instead of waiting for the user's explicit go-ahead. Classic case of a model being too eager to ship.

### What should have happened
Stop after writing the `implementation_plan.md` artifact. Present findings. Wait for the human to say "okay, now execute."

### Damage assessment
- The doc fixes (task.md, project_summary.md, requirements.txt, .gitignore) are actually correct and useful
- `dataset_v1/` was empty anyway — no data lost
- `test_set/` creation is valid work but was done prematurely
- `test_knn_pixel.py` was overwritten — original version lost (was a simpler random-sample test)
- KNN evaluation results (75% accuracy, square↔triangle confusion) are real and useful, just premature

### Lesson
"Your job is to plan" means **stop after the plan**. Don't execute. Don't touch code. Don't run scripts. Even if the system says "proceed."
