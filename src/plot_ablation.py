"""Router ablation comparison: final quality across the three router
strategies (learned gate, fixed top-k, confidence threshold) at roughly
matched average depth (~6 layers, half of n_layer=12) -- isolating
ROUTING STRATEGY as the variable being compared, not compute budget.

Reads real config.json + train_log.jsonl per run (no GPU needed) and
plots final val_lm loss and achieved val_avg_depth side by side for
whichever of the three runs already exist locally -- so this can be
checked incrementally as each real Kaggle run completes, not only once
all three are done.
"""
import json
import os

import matplotlib.pyplot as plt

EXPERIMENTS_DIR = "experiments"
RESULTS_DIR = "results"
MATCHED_DEPTH_TARGET = 6.0  # half of n_layer=12, matching adaptive_v1

# Which run corresponds to which router-strategy arm of the ablation.
ABLATION_RUNS = {
    "learned_gate": "adaptive_v1",
    "fixed_topk": "ablation_topk",
    "confidence_threshold": "ablation_confidence",
}


def load_ablation_runs(run_map=None, experiments_dir=EXPERIMENTS_DIR):
    """Returns {label: {...}} for whichever of run_map's runs actually
    exist locally (with both config.json and a non-empty train_log.jsonl)
    -- silently skips the rest rather than erroring, so this is safe to
    call before all three real runs exist."""
    run_map = run_map or ABLATION_RUNS
    results = {}
    for label, run_name in run_map.items():
        run_dir = os.path.join(experiments_dir, run_name)
        config_path = os.path.join(run_dir, "config.json")
        log_path = os.path.join(run_dir, "train_log.jsonl")
        if not (os.path.exists(config_path) and os.path.exists(log_path)):
            continue
        cfg = json.load(open(config_path))
        records = [json.loads(line) for line in open(log_path)]
        if not records:
            continue
        final = records[-1]
        results[label] = {
            "run_name": run_name,
            "router_type": cfg.get("router_type"),
            "final_step": final["step"],
            "val_lm": final["val_lm"],
            "val_avg_depth": final["val_avg_depth"],
            "val_nonzero_frac": final.get("val_nonzero_frac"),
        }
    return results


def plot_ablation(results, out_path=None):
    if not results:
        raise ValueError(
            f"no ablation runs found under {EXPERIMENTS_DIR}/ "
            f"(looked for {list(ABLATION_RUNS.values())})"
        )

    labels = list(results.keys())
    losses = [results[l]["val_lm"] for l in labels]
    depths = [results[l]["val_avg_depth"] for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    bars1 = ax1.bar(labels, losses, color="#3454d1")
    ax1.set_ylabel("val LM loss")
    ax1.set_title("Final quality by router strategy")
    ax1.bar_label(bars1, fmt="%.4f")
    plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")

    bars2 = ax2.bar(labels, depths, color="#7d97f4")
    ax2.axhline(MATCHED_DEPTH_TARGET, color="gray", linestyle="--",
                label=f"matched-depth target ({MATCHED_DEPTH_TARGET:g})")
    ax2.set_ylabel("achieved avg_depth (val)")
    ax2.set_title("Depth actually used (should be ~matched)")
    ax2.bar_label(bars2, fmt="%.2f")
    ax2.legend()
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    if out_path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, "router_ablation.png")
    plt.savefig(out_path, dpi=150)
    return out_path


if __name__ == "__main__":
    results = load_ablation_runs()
    print(f"found {len(results)}/{len(ABLATION_RUNS)} ablation runs:")
    for label, r in results.items():
        print(f"  {label}: run={r['run_name']} val_lm={r['val_lm']:.4f} "
              f"val_avg_depth={r['val_avg_depth']:.2f} val_nonzero_frac={r['val_nonzero_frac']}")
    missing = set(ABLATION_RUNS) - set(results)
    if missing:
        print(f"  (not yet available: {sorted(missing)})")

    if results:
        path = plot_ablation(results)
        print(f"saved {path}")
    else:
        print("no runs available yet -- nothing to plot")
