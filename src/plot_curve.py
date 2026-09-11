"""Builds the project's central result: accuracy (val LM loss) vs. average
layers used per token, one point per completed adaptive-depth training run.

Scans experiments/*/ for adaptive runs (identified by config.json having a
target_avg_layers field -- the baseline run doesn't), reads each run's
final train_log.jsonl record, and plots val_lm loss against the actually
achieved val_avg_depth (not the target -- what the router really did, which
may differ slightly from what it was aimed at). Run locally; needs no GPU,
just the config.json + train_log.jsonl already downloaded per run.
"""
import glob
import json
import os

import matplotlib.pyplot as plt

EXPERIMENTS_DIR = "experiments"
RESULTS_DIR = "results"


def load_adaptive_runs(experiments_dir=EXPERIMENTS_DIR):
    """Returns a list of dicts, one per adaptive run found, each with
    run_name, target_avg_layers, val_avg_depth, val_lm (all from the run's
    final logged step). Skips any run directory missing config.json,
    train_log.jsonl, or an empty log -- and skips the baseline run, whose
    config.json has no target_avg_layers field."""
    runs = []
    for config_path in sorted(glob.glob(os.path.join(experiments_dir, "*", "config.json"))):
        run_dir = os.path.dirname(config_path)
        with open(config_path) as f:
            cfg = json.load(f)
        if "target_avg_layers" not in cfg:
            continue  # not an adaptive run (e.g. the baseline)

        log_path = os.path.join(run_dir, "train_log.jsonl")
        if not os.path.exists(log_path):
            continue
        records = [json.loads(line) for line in open(log_path)]
        if not records:
            continue
        final = records[-1]

        runs.append({
            "run_name": cfg["run_name"],
            "target_avg_layers": cfg["target_avg_layers"],
            "final_step": final["step"],
            "val_avg_depth": final["val_avg_depth"],
            "val_lm": final["val_lm"],
        })
    return runs


def plot_curve(runs, out_path=None):
    if not runs:
        raise ValueError(f"no adaptive runs found under {EXPERIMENTS_DIR}/ "
                          f"(looked for config.json files with a target_avg_layers field)")

    runs = sorted(runs, key=lambda r: r["val_avg_depth"])
    x = [r["val_avg_depth"] for r in runs]
    y = [r["val_lm"] for r in runs]

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, "o-")
    for r in runs:
        plt.annotate(
            f"{r['run_name']}\n(target {r['target_avg_layers']:g}, step {r['final_step']})",
            (r["val_avg_depth"], r["val_lm"]),
            textcoords="offset points", xytext=(6, 6), fontsize=8,
        )
    plt.xlabel("average layers used per token (val)")
    plt.ylabel("val LM loss (cross-entropy)")
    plt.title("Adaptive-Compute Transformer — accuracy vs. average depth")
    plt.tight_layout()

    if out_path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, "accuracy_vs_avg_layers.png")
    plt.savefig(out_path, dpi=150)
    return out_path


if __name__ == "__main__":
    runs = load_adaptive_runs()
    for r in runs:
        print(f"{r['run_name']}: target={r['target_avg_layers']:g}  "
              f"achieved avg_depth={r['val_avg_depth']:.3f}  val_lm={r['val_lm']:.4f}  "
              f"(final step {r['final_step']})")
    path = plot_curve(runs)
    print(f"saved {path} ({len(runs)} point{'s' if len(runs) != 1 else ''})")
