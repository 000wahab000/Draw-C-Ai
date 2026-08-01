# .agents/ — Start Here

> **If you are an AI model and you just opened this project, read this file completely before you touch anything else.**

---

## What This Folder Is

This folder is the **brain of the project for AI agents**. Every decision that's already been made, every rule that's in effect, and every task that needs doing is documented here. You do not need to figure things out from scratch — the answers are already written. Your job is to read first, then act.

---

## Your First 60 Seconds

1. **Read `ROLES.md`** — figure out which role you are (Planner or Executor). Do not proceed until you know your role.
2. **Read `project_summary.md`** — understand what this project is, what the data contract is, and what architecture decisions are already locked in. Do not re-debate these.
3. **Read `task.md`** — find the current phase. Look for items marked `[ ]`. Those are your work. Items marked `[x]` are done — do not redo them.
4. **Read the relevant section in `Steps.md`** — find the phase you're working on and read the step-by-step instructions before writing a single line of code.
5. **Read `model-fuck-ups.md`** — see what previous models did wrong. Do not repeat those mistakes.

---

## File Map

| File | What It Is | When To Read It |
|---|---|---|
| `README.md` | This file. Entry point for all agents. | First, always. |
| `ROLES.md` | Defines Planner vs Executor roles and what each is allowed to do. | Immediately after this file. |
| `project_summary.md` | Full project overview: what it does, the data contract, architecture decisions, API spec, directory structure. | Before any work. |
| `task.md` | Living checklist of every task across all phases. Mark `[ ]` → `[/]` → `[x]` as you work. | To find what to do next. |
| `Steps.md` | Detailed how-to for each phase. Explains *why* each step exists, not just what to do. | Before executing any step. |
| `AGENTS.md` | Ponytail coding rule — always active. Forces the simplest, laziest solution that works. | It's always in effect. No need to re-read every session. |
| `model-fuck-ups.md` | Log of mistakes made by previous models. | Before starting work, so you don't repeat them. |

---

## The Rules That Are Always Active

These apply to every model, every session, no exceptions:

1. **Read before you act.** Do not write a single line of code until you have read `project_summary.md` and the relevant section of `Steps.md`.

2. **The data contract is sacred.** Every sample is `{"label": str, "grid_size": 32, "vector": [1024 ints, 0 or 1]}`. Every script, model, and API endpoint speaks this format. Do not change it.

3. **Never overwrite a dataset version.** If you re-run any data pipeline with different parameters, it creates a new version directory (`dataset_v3`, `dataset_v4`, ...). `get_next_dataset_dir()` handles this automatically — use it.

4. **The test set is untouchable.** The 16 samples in `test_set/` are never trained on. Never moved. Never replaced. Every model from Phase 2 through Phase 5 is evaluated on this exact set.

5. **Ponytail is always on.** No speculative abstractions. No new files unless the task requires it. No new dependencies unless you've checked that numpy, PIL, or OpenCV can't do it first. Shortest working diff wins.

6. **Log as you go.** Each phase produces a log file in `logs/`. Write it during the phase, not after. The retrospective (Phase 7) is written from the logs.

7. **Know your role.** Check `ROLES.md`. If you are a Planner, you do not write code. If you are an Executor, you do not make architectural decisions — you execute the plan exactly as written and report blockers.

---

## How to Update This Folder

- **Finished a task?** Mark it `[x]` in `task.md`.
- **Discovered something broken or wrong in the docs?** Fix the doc, not just the code. The docs are the source of truth for future agents.
- **Made a mistake?** Log it in `model-fuck-ups.md` with model name, date, time, what happened, and why it was wrong.
- **Architecture decision came up?** Only the Planner role documents it in `project_summary.md`. Executors do not make architecture decisions.

---

## Current Phase

Check `task.md` — find the first phase with unchecked `[ ]` items. That is where the project is right now.
