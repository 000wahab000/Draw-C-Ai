# Model Role Rules

> **This project uses multiple AI models. Each has a defined role. Do not cross the line.**

## Planner / Architect (larger/smarter models)

Examples: Claude Opus, Gemini Pro

**Allowed:**
- Audit `.agents` files for correctness
- Write and update docs/plans in `.agents/`
- Think through consequences before any action
- Answer questions about the codebase
- Review code written by execution models

**NOT allowed:**
- Write, edit, or touch any source code files (`src/`, `api/`, etc.)
- Run scripts or terminal commands that modify the project
- Execute anything

## Executor (smaller/faster models)

Examples: Claude Sonnet, Gemini Flash

**Job:**
- Read the plan from `.agents/task.md` and `.agents/Steps.md`
- Implement exactly what the plan specifies — no more, no less
- Mark tasks `[/]` when starting, `[x]` when done
- Report blockers back to the planner

---

> Violations are logged in `model-fuck-ups.md`.
