# CLAUDE.md — Project Brain

**User:** Saif
**Maintained by:** Noufa (mentor) — this file is the source of truth for how Claude Code should operate on this project. Read it in full before doing anything.

---

## Project Overview

### What this project is
An **Adaptive-Compute Transformer**: a transformer language model, built from scratch, with a learned per-token router that decides — after each layer — whether a given token needs to keep going deeper or can exit early. Easy, predictable tokens (common words, repeated context) exit after a few layers. Hard, information-dense, ambiguous tokens go through the full depth of the network. Total tokens in and out never change — only how much internal computation each token receives changes.

This is inspired by real published research: DeepMind's **Mixture-of-Depths** and Google's **CALM (Confident Adaptive Language Models)**. This project is not a copy of either — it's an original from-scratch implementation, with our own routing mechanism, so every design decision is ours to defend in an interview.

### Why this project exists (the actual point)
Every company running an LLM in production pays for inference compute — every token, through every layer, every single request. That cost (GPU time, latency, energy) scales directly with how much unnecessary computation is spent on "easy" tokens. This project is a direct, hands-on attempt at that real, universal cost problem: can a model learn to spend its own compute unevenly and get the same quality output for less total work?

This is a **research-style learning project**, not a product. The deliverable is understanding, working code, real experimental results (charts, benchmarks), and a defensible story — not a deployed app.

### Core architecture
- Standard transformer block stack (e.g. 12 layers) as the baseline.
- After each layer, a lightweight learned **router** scores each token: "has this token's representation stabilized, or does it need another layer?"
- Tokens that exit early skip the remaining layers' computation for that token specifically (other tokens in the same sequence may continue).
- The router is trained end-to-end alongside the main model — it is not hand-coded or rule-based. It learns which tokens are "easy" vs "hard" from data, the same way the rest of the model learns.

### Training plan
1. Train a **baseline static-depth transformer** on a mid-size corpus (WikiText-103 or similar free dataset) as the control.
2. Train the **adaptive-depth version** on identical data and compute budget.
3. Core result: plot **accuracy vs. average layers used per token** — this curve is the central finding of the whole project.
4. **Router ablation**: compare a learned gate vs. a fixed top-k router (the actual Mixture-of-Depths approach) vs. a simple confidence threshold. Report which generalizes best and why — this is a real experimental contribution, not just a build.
5. **Real speed benchmark**: measure actual wall-clock inference speedup using `torch.compile`, not just theoretical FLOPs saved. The gap between theoretical and real speedup is itself a legitimate finding worth reporting.
6. **Visualization**: show routing decisions token-by-token on real sentences — which words got skipped vs. processed in full. This is the single most compelling visual artifact from the whole project.

### Stack (all free, open source)
| Layer | Tool |
|---|---|
| Model + training loop | PyTorch (built from scratch, no HF Trainer shortcuts for the core loop) |
| Dataset | WikiText-103 (or similar free corpus) |
| Experiment tracking | Weights & Biases (free tier) |
| Compute | Colab / Kaggle free GPU tier |
| Real-latency benchmarking | `torch.compile` |

### Definition of done
The project is complete when we have: a working baseline model, a working adaptive-depth model, the accuracy-vs-average-layers curve, the router ablation comparison, a real wall-clock speedup number, and at least one token-level routing visualization — all reproducible from logged configs and seeds.

---

## Session Control (non-negotiable)
- Do not begin any work until Saif types exactly `start`.
- Do not end a session until Saif types exactly `end`.
- Never push code to git/GitHub unless explicitly instructed to do so in that specific turn. Assume no push authority by default.

## On "start"
1. Read `memory.md` in full.
2. Summarize, in plain language, where the project left off.
3. Propose next step(s) based on current state and the Project Overview above — there's no fixed roadmap, so use judgment, not a checklist.
4. Wait for Saif's go-ahead before writing or running any code.

## On "end"
1. Append an entry to `memory.md`:
   - Session start timestamp
   - Session end timestamp
   - Summary: what was done, what was decided, what's still open
2. Update `learning.md`:
   - Concepts learned this session (not just what was coded)
   - Commands used
   - Approach/method followed
   - Blockers hit
   - How each blocker was resolved
3. Update `interview-prep.md`:
   - New Q&A generated from the actual work done this session — grounded in what was really built/decided, not generic ML trivia.

## On project completion
Generate `final-interview-prep.md` — a comprehensive, rigorous Q&A covering the entire project end to end (architecture, why it was designed this way, results, what would be done differently), built for real interview prep.

---

## File map
| File | Purpose |
|---|---|
| `memory.md` | Session log. Read at the start of every session to resume exactly where things left off. |
| `learning.md` | Session-by-session learning log — concepts, commands, approach, blockers, resolutions. |
| `interview-prep.md` | Rolling interview Q&A, updated every session. |
| `final-interview-prep.md` | Comprehensive final Q&A, created once the project is complete. |
| `DECISIONS.md` | One entry per meaningful engineering decision and the reasoning behind it (architecture choices, thresholds, why one ablation beat another). Kept separate from memory.md so it survives as a clean, standalone artifact. |

## Engineering discipline
- **Reproducibility:** every training run must log its random seed and full config (hyperparameters, data version). Never report a result without logging the run that produced it.
- **No fabricated results:** never report a metric, chart, or claim without actually running the code that produces it. If something can't be verified yet, say so explicitly instead of estimating.
- **Repo structure:** keep it consistent across sessions — `/src`, `/experiments`, `/results`, `/docs`, with `memory.md`, `learning.md`, `interview-prep.md`, `DECISIONS.md` at the project root.
- **Environment:** single `requirements.txt` (or `pyproject.toml`), kept up to date as dependencies change.
- **Colab checkpoint backup:** Colab's `/content` storage is temporary and is wiped on runtime disconnect. After any Colab training run finishes, download/back up the checkpoint and logs immediately, in the same cell block that finishes training — before doing anything else (reviewing results, starting another run, closing the tab). Do not treat a run as complete until the checkpoint has been pulled off Colab. (Lesson learned the hard way: the `baseline_v1` checkpoint was lost to a runtime disconnect before download — see `learning.md` Session 4.)

## Working style
- **No whole-codebase dumps.** For any multi-file or multi-step piece of work, propose a short module-by-module plan first, then implement and verify one module at a time — don't write the entire thing in one pass.
- Each module should be independently reviewable (and, where applicable, independently testable — e.g. a smoke test) before moving to the next.
- This applies to notebooks and scripts alike: a Colab notebook should be built cell-block by cell-block (setup → data → model → train → eval), not generated as one giant cell.

---

## claude-code-setup
Read the full codebase and current project structure, then explicitly suggest:
- MCP servers that would meaningfully help this specific project
- Claude Code skills that align with the work being done
- Hooks that would help enforce the session-control rules above automatically
- Sub-agents that would help split or parallelize the work

Do this proactively whenever the codebase changes significantly — not just once at the start.
