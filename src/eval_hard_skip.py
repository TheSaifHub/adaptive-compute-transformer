"""Module 3: wire the real hard-skip inference path (hard_skip.py) into a
trained checkpoint and measure val_lm on real WikiText-103 validation
data, compared directly against the same checkpoint's dense (Phase 1,
soft) forward pass -- on the SAME sampled batches, same seed -- so any
difference is attributable to hard-skip's discretization, not
evaluation-sampling noise.

Originally scoped for adaptive_v1 (learned_gate) -- see DECISIONS.md
2026-09-08 for why that combination (learned_gate + hard exclusion, in
either the full_block or mlp_only mechanism) turned out to badly damage
quality: the router's trained gate scores are broadly distributed, not
sparse, so no rank-based cutoff is a safe "treat as 0" approximation.
Re-pointed at ablation_topk (fixed_topk router), whose excluded tokens
genuinely learned gate=0 during Phase 1 training itself (not an
inference-time approximation) -- see the same DECISIONS.md addendum for
the result: mlp_only mode reproduces dense output almost exactly.

Capacity is derived per router type:
  - fixed_topk: cfg.topk_capacity directly (exact, a training
    hyperparameter, not something to estimate).
  - learned_gate: the run's own real, logged final val_avg_depth (not
    the nominal target_avg_layers) divided by n_layer -- "report
    achieved, not target," same discipline as plot_curve.py.

No training happens here (inference-only). This deliberately does NOT
call data.prepare_dataset() -- that would also try to tokenize the full
train split if train.bin is missing locally, which it is (only val.bin
was cached, in Session 7). Requires data/tokenized/val.bin to already exist.
"""
import json
import os
import time

import torch

from data import get_batch
from hard_skip import HardSkipAdaptiveTransformer
from visualize_routing import load_run

EXPERIMENTS_DIR = "experiments"
RESULTS_DIR = "results"
VAL_BIN_PATH = os.path.join("data", "tokenized", "val.bin")
EVAL_ITERS = 20  # local CPU eval -- see run notes on why this is fewer than the 100 used during training
SEED = 1337


def derive_capacity(run_name, cfg, experiments_dir=EXPERIMENTS_DIR):
    """Router-type-aware capacity derivation, plus the run's final logged
    val_lm for a training-time reference point. Returns
    (capacity, capacity_source_note, trained_avg_depth_or_None, trained_val_lm).
    """
    log_path = os.path.join(experiments_dir, run_name, "train_log.jsonl")
    with open(log_path) as f:
        lines = [l for l in f if l.strip()]
    final = json.loads(lines[-1])

    if cfg.router_type == "fixed_topk":
        capacity = cfg.topk_capacity
        note = f"cfg.topk_capacity (exact, training hyperparameter)"
        trained_avg_depth = final.get("val_avg_depth")
    elif cfg.router_type == "learned_gate":
        trained_avg_depth = final["val_avg_depth"]
        capacity = trained_avg_depth / cfg.n_layer
        note = f"real logged val_avg_depth ({trained_avg_depth:.4f}) / n_layer -- achieved, not target"
    else:
        raise ValueError(f"no capacity-derivation rule for router_type={cfg.router_type!r}")

    return capacity, note, trained_avg_depth, final["val_lm"]


@torch.no_grad()
def compare(run_name, mode="mlp_only", eval_iters=EVAL_ITERS, seed=SEED):
    assert os.path.exists(VAL_BIN_PATH), (
        f"{VAL_BIN_PATH} not found -- this script deliberately does not call "
        f"data.prepare_dataset() (it would also try to tokenize the full train "
        f"split). Run visualize_routing.py once first, or tokenize val only."
    )

    dense_model, cfg = load_run(run_name, experiments_dir=EXPERIMENTS_DIR, device="cpu")
    capacity, capacity_note, trained_avg_depth, trained_val_lm = derive_capacity(run_name, cfg)
    hs_model = HardSkipAdaptiveTransformer(dense_model, capacity, mode=mode)
    hs_model.eval()

    print(f"[{run_name}] router_type={cfg.router_type}, n_layer={cfg.n_layer}, mode={mode}")
    print(f"capacity={capacity:.4f} (source: {capacity_note})")
    print(f"Evaluating {eval_iters} batches (batch_size={cfg.batch_size}, block_size={cfg.block_size}) "
          f"on real WikiText-103 validation data, CPU, seed={seed}...")

    torch.manual_seed(seed)
    dense_losses, hs_losses, hs_avg_depths = [], [], []
    t0 = time.time()
    for i in range(eval_iters):
        x, y = get_batch("val", cfg, device="cpu")

        _, dense_loss, _ = dense_model(x, y)
        dense_losses.append(dense_loss.item())

        _, hs_loss, hs_aux = hs_model(x, y)
        hs_losses.append(hs_loss.item())
        hs_avg_depths.append(hs_aux["avg_depth"].item())

        print(f"  iter {i+1}/{eval_iters}: dense_lm={dense_loss.item():.4f}  "
              f"hard_skip_lm={hs_loss.item():.4f}  (elapsed {time.time()-t0:.0f}s)")

    dense_mean = sum(dense_losses) / len(dense_losses)
    hs_mean = sum(hs_losses) / len(hs_losses)
    hs_depth_mean = sum(hs_avg_depths) / len(hs_avg_depths)

    result = {
        "run_name": run_name,
        "router_type": cfg.router_type,
        "mode": mode,
        "seed": seed,
        "eval_iters": eval_iters,
        "batch_size": cfg.batch_size,
        "block_size": cfg.block_size,
        "n_layer": cfg.n_layer,
        "capacity": capacity,
        "capacity_source": capacity_note,
        "trained_avg_depth": trained_avg_depth,
        "dense_val_lm_local": dense_mean,
        "dense_val_lm_training_logged": trained_val_lm,
        "hard_skip_val_lm": hs_mean,
        "hard_skip_val_avg_depth": hs_depth_mean,
        "delta_hard_skip_minus_dense": hs_mean - dense_mean,
    }

    print(f"\n--- Module 3 result ({run_name}, mode={mode}, {eval_iters} iters, seed {seed}) ---")
    print(f"dense (soft, Phase 1) val_lm, training-logged (100 GPU iters): {trained_val_lm:.4f}")
    print(f"dense (soft, Phase 1) val_lm, fresh local CPU estimate: {dense_mean:.4f}")
    print(f"hard-skip (Phase 2)   val_lm, same batches:             {hs_mean:.4f}")
    print(f"delta (hard_skip - dense):                              {result['delta_hard_skip_minus_dense']:+.4f}")
    print(f"hard-skip achieved avg_depth: {hs_depth_mean:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"hard_skip_eval_{run_name}_{mode}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved: {out_path}")
    return result


if __name__ == "__main__":
    import sys
    run_name = sys.argv[1] if len(sys.argv) > 1 else "ablation_topk"
    mode = sys.argv[2] if len(sys.argv) > 2 else "mlp_only"
    compare(run_name, mode)
