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

---

## Incident #2

**Date**: 2026-08-12
**Category**: Environment Bug (not model error)

### What happened
`pip install` failed with:
```
ERROR: Unknown compiler(s): [['icl'], ['cl'], ['cc'], ['gcc'], ...]
```
And later:
```
AssertionError: self.py_limited_api=True and Py_GIL_DISABLED=1 are not supported together
```

### Root cause
The `.venv` was created with **Python 3.13t** (free-threaded / GIL-disabled build) instead of standard Python 3.13. This experimental variant (`Py_GIL_DISABLED=1`) is not supported by:
- PyTorch (no wheel exists — not even in nightly as of 2026-08-12)
- PyMuPDF (build system conflict with `py_limited_api`)
- numpy (tried to compile from source, failed because no C compiler)

### How we detected it
```
python --version
# Python 3.13.14 experimental free-threading build
```

### Fix
Delete the broken venv and recreate with standard Python 3.13:
```powershell
Remove-Item -Recurse -Force ".venv"
py -3.13 -m venv .venv
```
Then install PyTorch from their official CPU wheel index (not PyPI):
```powershell
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
For all other packages, use `--prefer-binary` to avoid source compilation:
```powershell
.venv/Scripts/python.exe -m pip install numpy opencv-python Pillow fastapi uvicorn --prefer-binary
```

### Lesson
Always verify which Python variant created the venv. `py --list` on Windows shows all installed versions. The `*` marks the default — if it says `freethreaded`, recreate the venv explicitly with `py -3.13` (without the `t`).

---

## Incident #3

**Date**: 2026-08-12
**Category**: IDE Configuration Bug (not model error)

### What happened
After installing PyTorch correctly, the IDE (Antigravity/VS Code) showed red squiggles on all `import torch` / `import torchvision` lines with errors:
```
Cannot find module `torch`
Could not resolve interpreter path 'd:/wahab stuff/wahab code/Draw-C-Ai/.venv/Scripts/python.exe'
```

### Root cause
Two issues combined:
1. The IDE was using the **system Python** (`D:\python`) instead of the project venv.
2. When we pointed it to the venv via hardcoded absolute path, the **spaces in the path** (`wahab stuff`, `wahab code`) caused the path resolver to fail.

### Fix
Use `${workspaceFolder}` variable in `.vscode/settings.json` instead of a hardcoded absolute path:
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
    "python.analysis.extraPaths": [
        "${workspaceFolder}/.venv/Lib/site-packages"
    ]
}
```
Then: `Ctrl+Shift+P` → **Python: Select Interpreter** → pick the venv entry. Close and reopen the IDE fully.

### Important note
The code itself was **always working** — the terminal ran it with zero errors. The red squiggles were purely a display issue in the IDE's language server. Do not confuse IDE linting errors with actual runtime errors.

### Lesson
- On Windows with spaces in project path: always use `${workspaceFolder}` in `.vscode/settings.json`, never hardcoded absolute paths.
- Verify actual functionality via terminal before assuming IDE errors mean broken code.
- File to check/create: `.vscode/settings.json` at project root.
