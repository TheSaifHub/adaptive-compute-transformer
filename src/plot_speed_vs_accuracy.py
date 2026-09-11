"""Path A, Module 8 (final): the unified accuracy-vs-real-speed chart.

Combines fixed_topk's real, verified accuracy-vs-capacity curve
(plot_topk_curve.py's data -- val_lm per topk_capacity, from real
Kaggle-trained checkpoints) with its real, measured wall-clock latency at
each of the SAME capacities (results/benchmark_speed_sweep_cuda.json,
Module 7 -- a real Kaggle GPU benchmark run) -- on one router, one
consistent benchmark session. This is the artifact the project completion
audit (DECISIONS.md, 2026-09-08, "Project completion audit") found
missing: quality-preservation evidence and real-speedup evidence had
never been demonstrated on the same router before Path A.

Per Module 4 (2026-09-10), hard-skip reproduces each checkpoint's dense
val_lm to float32 machine precision at every one of these capacities --
so the accuracy numbers plotted here (read from each run's own dense-mode
train_log.jsonl, via load_topk_runs) ARE the real hard-skipped model's
accuracy too, not merely a proxy for it.

Latency is read at the 32x256 throughput shape (matching every training/
eval run's batch_size/block_size, consistent with every prior benchmark
report in this project) in eager mode (torch.compile showed no consistent
benefit across capacities -- see DECISIONS.md 2026-09-10 -- and mixing in
a second, orthogonal variable would muddy the one comparison this chart
exists to make).

Deliberately uses ONLY this one sweep JSON's own capacity=0.5 point,
never the earlier single-capacity benchmark from Session 10
(results/benchmark_speed_cuda.json) -- those two measurements, taken in
different Kaggle sessions, show real, non-trivial absolute latency
drift (baseline: 195.21ms vs. 216.70ms, ~11% apart) that almost certainly
reflects ordinary GPU-instance/session variance, not anything about the
model. Mixing data from two sessions into one curve would silently
conflate that variance with the real, capacity-driven trend this chart
is built to show -- so every point here comes from the same single
Kaggle run, self-consistent by construction.

Run locally; needs no GPU, just plot_topk_curve.py's loader (reading
experiments/*/config.json + train_log.jsonl) plus the downloaded sweep
JSON.
"""
import json
import os

import matplotlib.pyplot as plt

from plot_topk_curve import load_topk_runs

RESULTS_DIR = "results"
SWEEP_JSON = os.path.join(RESULTS_DIR, "benchmark_speed_sweep_cuda.json")
SHAPE_KEY = "32x256"
BACKEND = "eager"


def load_sweep_latencies(sweep_json=SWEEP_JSON, shape_key=SHAPE_KEY, backend=BACKEND):
    """Returns ({capacity: hard_skip_latency_ms}, full_sweep_dict)."""
    with open(sweep_json) as f:
        d = json.load(f)
    latencies = {}
    for cap_key, variants in d["variants_by_capacity"].items():
        cap = float(cap_key)
        entry = variants["adaptive_hard_skip"][backend][shape_key]
        latencies[cap] = entry["median_s"] * 1000
    return latencies, d


def join_accuracy_and_speed(experiments_dir="experiments", sweep_json=SWEEP_JSON):
    """Joins each real fixed_topk accuracy run with its real hard-skip
    latency at the same capacity. Matches by nearest capacity within a
    small tolerance (rather than exact float equality) since the
    accuracy runs' config.json and the sweep JSON's capacities list are
    two independently-written files -- expected to carry the same
    literal values (0.1667/0.3333/0.5/0.75) but tolerance-matched
    defensively rather than assumed bit-identical."""
    accuracy_runs = load_topk_runs(experiments_dir=experiments_dir)
    latencies, sweep_meta = load_sweep_latencies(sweep_json)
    if not latencies:
        raise ValueError(f"no capacities found in {sweep_json}")

    joined = []
    for r in accuracy_runs:
        cap = r["topk_capacity"]
        nearest = min(latencies.keys(), key=lambda k: abs(k - cap))
        if abs(nearest - cap) > 1e-3:
            raise ValueError(
                f"no matching sweep capacity found for accuracy run {r['run_name']!r} "
                f"(topk_capacity={cap}) -- nearest available was {nearest}, off by {abs(nearest - cap):.6f}")
        joined.append({**r, "hard_skip_latency_ms": latencies[nearest]})

    if not joined:
        raise ValueError("no fixed_topk accuracy runs found to join against the sweep data")
    return sorted(joined, key=lambda r: r["topk_capacity"]), sweep_meta


def plot_unified(joined, sweep_meta, out_path=None):
    if not joined:
        raise ValueError("no joined accuracy+speed points to plot")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8))

    caps = [r["topk_capacity"] for r in joined]
    losses = [r["val_lm"] for r in joined]
    lat = [r["hard_skip_latency_ms"] for r in joined]

    # Panel 1: both quantities against the underlying driver, capacity --
    # shows HOW they move together, not just that a tradeoff exists.
    ax1b = ax1.twinx()
    l1, = ax1.plot(caps, losses, "o-", color="tab:red", label="val LM loss")
    l2, = ax1b.plot(caps, lat, "s--", color="tab:blue", label="real hard-skip latency")
    ax1.set_xlabel("topk_capacity")
    ax1.set_ylabel("val LM loss (cross-entropy)", color="tab:red")
    ax1b.set_ylabel(f"median latency, {SHAPE_KEY} {BACKEND} (ms)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1b.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title("Quality and real speed vs. capacity")
    ax1.legend(handles=[l1, l2], loc="center right", fontsize=8)

    # Panel 2: the direct unification this whole effort was built for --
    # accuracy plotted against REAL measured latency, one router, one
    # consistent Kaggle GPU session.
    ax2.plot(lat, losses, "o-", color="tab:purple")
    for r in joined:
        ax2.annotate(f"capacity={r['topk_capacity']:.4f}", (r["hard_skip_latency_ms"], r["val_lm"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax2.set_xlabel(f"real hard-skip latency, {SHAPE_KEY} {BACKEND} (ms) — lower is faster")
    ax2.set_ylabel("val LM loss (cross-entropy) — lower is better")
    ax2.set_title("Accuracy vs. real wall-clock speed")
    ax2.invert_xaxis()  # faster (lower latency) toward the right matches "better" being right, alongside lower loss

    fig.suptitle(
        "Path A unified proof — fixed_topk, one Kaggle GPU session, hard-skip-verified lossless at every point",
        fontsize=10.5, y=1.01)
    plt.tight_layout()

    if out_path is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, "unified_accuracy_vs_speed.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path


if __name__ == "__main__":
    joined, sweep_meta = join_accuracy_and_speed()
    print(f"device={sweep_meta['device']}  shape={SHAPE_KEY}  backend={BACKEND}\n")
    for r in joined:
        print(f"{r['run_name']}: capacity={r['topk_capacity']:.4f}  val_lm={r['val_lm']:.4f}  "
              f"hard_skip_latency={r['hard_skip_latency_ms']:.2f}ms")
    path = plot_unified(joined, sweep_meta)
    print(f"\nsaved {path}")
