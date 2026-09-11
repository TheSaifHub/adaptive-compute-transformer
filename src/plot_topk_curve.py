"""Path A, Module 5: fixed_topk's own accuracy-vs-capacity curve --
the fixed_topk analog of plot_curve.py (which builds the same kind of
curve for learned_gate's target_avg_layers sweep).

Scans experiments/*/ for fixed_topk runs (identified by config.json
having router_type == "fixed_topk" -- correctly distinguishes them from
the learned_gate sweep runs, which have no router_type key at all, and
from ablation_confidence, whose router_type is "confidence_threshold"),
reads each run's final train_log.jsonl record, and plots val_lm loss
against topk_capacity (an exact, by-construction hyperparameter for this
router -- see router.py's FixedTopKRouter docstring -- so unlike
learned_gate's avg_depth there's no achieved-vs-target gap to report;
the x-axis is simply the config value each run was trained with).

Per Module 4 (2026-09-10), hard-skip reproduces dense output to float32
machine precision at every one of these capacities -- so this dense-mode
accuracy curve IS the real hard-skipped model's accuracy curve too, not
merely a proxy for it. That's the unification Path A set out to
establish, and it's what makes this curve, combined with Module
6-8's wall-clock benchmark sweep, into a genuine accuracy-vs-real-speed
proof rather than two separate, unconnected results.

Run locally; needs no GPU, just each run's config.json + train_log.jsonl
already downloaded.
"""
import glob
import json
import os

import matplotlib.pyplot as plt

EXPERIMENTS_DIR = "experiments"
RESULTS_DIR = "results"


def load_topk_runs(experiments_dir=EXPERIMENTS_DIR):
    """Returns a list of dicts, one per fixed_topk run found, each with
    run_name, topk_capacity, val_nonzero_frac, val_avg_depth, val_lm (all
    from the run's final logged step). Skips any run directory missing
    config.json, train_log.jsonl, or an empty log -- and skips any run
    whose router_type isn't "fixed_topk" (the learned_gate sweep runs
    have no router_type key at all; ablation_confidence has
    router_type == "confidence_threshold")."""
    runs = []
    for config_path in sorted(glob.glob(os.path.join(experiments_dir, "*", "config.json"))):
        run_dir = os.path.dirname(config_path)
        with open(config_path) as f:
            cfg = json.load(f)
        if cfg.get("router_type") != "fixed_topk":
            continue

        log_path = os.path.join(run_dir, "train_log.jsonl")
        if not os.path.exists(log_path):
            continue
        records = [json.loads(line) for line in open(log_path)]
        if not records:
            continue
        final = records[-1]

        runs.append({
            "run_name": cfg["run_name"],
            "topk_capacity": cfg["topk_capacity"],
            "final_step": final["step"],
            "val_nonzero_frac": final["val_nonzero_frac"],
            "val_avg_depth": final["val_avg_depth"],
            "val_lm": final["val_lm"],
        })
    return runs


def plot_curve(runs, out_path=None):
    if not runs:
        raise ValueError(f"no fixed_topk runs found under {EXPERIMENTS_DIR}/ "
                          f"(looked for config.json files with router_type == 'fixed_topk')")

    runs = sorted(runs, key=lambda r: r["topk_capacity"])
    x = [r["topk_capacity"] for r in runs]
    y = [r["val_lm"] for r in runs]

    plt.figure(figsize=(7.5, 6))
    plt.plot(x, y, "o-")
    for r in runs:
        plt.annotate(
            f"{r['run_name']}\n(step {r['final_step']})",
            (r["topk_capacity"], r["val_lm"]),
            textcoords="offset points", xytext=(6, 6), fontsize=8,
        )
    plt.xlabel("topk_capacity (fraction of tokens selected per layer)")
    plt.ylabel("val LM loss (cross-entropy)")
    plt.title("fixed_topk — accuracy vs. compute capacity", pad=14)
    plt.suptitle("hard-skip-verified lossless at every point (Module 4) — this IS the real-model curve",
                 y=0.965, fontsize=8.5, style="italic")
    # extra headroom above the highest point so its annotation clears the subtitle
    y_span = max(y) - min(y)
    plt.ylim(top=max(y) + 0.18 * y_span)
    plt.tight_layout(rect=(0, 0, 1, 0.94))

    if out_path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, "accuracy_vs_capacity_fixed_topk.png")
    plt.savefig(out_path, dpi=150)
    return out_path


if __name__ == "__main__":
    runs = load_topk_runs()
    for r in runs:
        print(f"{r['run_name']}: topk_capacity={r['topk_capacity']:.4f}  "
              f"val_nonzero_frac={r['val_nonzero_frac']:.4f}  "
              f"val_avg_depth={r['val_avg_depth']:.4f}  val_lm={r['val_lm']:.4f}  "
              f"(final step {r['final_step']})")
    path = plot_curve(runs)
    print(f"saved {path} ({len(runs)} point{'s' if len(runs) != 1 else ''})")
