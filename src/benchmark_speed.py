"""Module 4: real wall-clock inference-latency benchmark.

Compares three variants at ablation_topk's real architecture (12-layer,
d_model=384, fixed_topk router, capacity=0.5):
  - baseline: BaselineTransformer, static full depth, no routing at all.
  - adaptive_dense: an AdaptiveTransformer (fixed_topk router) run
    through its normal Phase 1 forward path -- full dense compute for
    every token, every layer, exactly like baseline, plus routing/gating
    math on top. Expected to be AT LEAST as slow as baseline, not faster
    -- Phase 1 never skips compute. This is itself half of "the gap
    between theoretical and real speedup" CLAUDE.md wants documented:
    Phase 1 adds routing overhead for zero savings.
  - adaptive_hard_skip: the same model wrapped in
    HardSkipAdaptiveTransformer(mode="mlp_only") -- the (router,
    mechanism) pair verified in Module 3 to preserve quality exactly
    (val_lm matched dense to 4 decimals on every evaluated batch). This
    is where a real speedup, if any, should show up.

All three are built FRESH by default (untrained) rather than from
ablation_topk's real checkpoint -- weights don't affect latency, only
architecture/shapes/control-flow do (see below), so this is a legitimate
simplification, not a shortcut that changes what's being measured.

Benchmarks a single full-sequence forward pass (idx only, no targets --
matches real serving "prefill," not scoring against known future tokens;
see DECISIONS.md's benchmark-scope decision for why this isn't
autoregressive decode).

Local (CPU, this machine) run is a feasibility/no-crash check --
`--smoke` uses tiny shapes/iteration counts so it finishes in a couple of
minutes. torch.compile is OFF by default (`--compile` to attempt it) --
Triton/torch.compile support on Windows CPU is unreliable, and CPU timing
numbers aren't the real deliverable anyway. The real numbers come from
Module 5's Colab/Kaggle GPU notebook, running this same script with
--compile and full (non---smoke) shapes/iters.

Weights don't affect latency (only architecture/shapes/control-flow do),
so `--checkpoint <run_name>` is optional, not required: without it (the
default -- and what Module 5's notebook uses), all three variants are
built FRESH at ABLATION_TOPK_CFG (ablation_topk's real architecture and
router settings, just untrained) -- self-contained, no need to
upload/attach ablation_topk's checkpoint to a Kaggle kernel at all. Pass
`--checkpoint ablation_topk` to instead load the real trained weights
(useful locally, where the files already exist, purely as an extra
consistency check -- the numbers should not meaningfully differ).

Path A, Module 6 (2026-09-10) extended this with `run_sweep()` / `--sweep`:
times all three variants at each of several `topk_capacity` values
(default: the same four points already trained in the accuracy sweep --
0.1667, 0.3333, 0.5, 0.75) instead of one fixed capacity, so the real
wall-clock number exists at every point on the accuracy-vs-capacity
curve, not just one. `run()` (the original, single-capacity function)
and its output file (`benchmark_speed_<device>.json`, already holding
the real, reported Session 10 result) are both left completely
unchanged -- `run_sweep()` is new, additive, and writes to a
differently-named file (`benchmark_speed_sweep_<device>.json`) so it can
never silently overwrite that already-reported number.
"""
import argparse
import json
import os
import time

import torch

from config import AdaptiveConfig, BaselineConfig
from model import AdaptiveTransformer, BaselineTransformer
from hard_skip import HardSkipAdaptiveTransformer
from visualize_routing import load_run

RESULTS_DIR = "results"
REAL_SHAPES = ((1, 256), (32, 256))
SMOKE_SHAPES = ((1, 64), (4, 64))

# ablation_topk's real architecture + router config (see experiments/ablation_topk/config.json)
# -- everything except router_type/topk_capacity is just AdaptiveConfig's own defaults.
ABLATION_TOPK_CFG = AdaptiveConfig(router_type="fixed_topk", topk_capacity=0.5)

# The four capacities Path A already trained real fixed_topk checkpoints at
# (Session 12 / DECISIONS.md 2026-09-10) -- run_sweep()'s default, so the
# wall-clock benchmark covers exactly the same points as the accuracy curve
# (plot_topk_curve.py), not an arbitrarily different set.
CAPACITIES = (0.1667, 0.3333, 0.5, 0.75)


def build_variants(device, checkpoint_run_name=None, experiments_dir="experiments", topk_capacity=None):
    """Builds all three variants either fresh (default -- see module
    docstring for why this is legitimate for a latency benchmark) or from
    a real checkpoint's trained weights.

    topk_capacity: overrides the fresh-init config's capacity (ignored --
    and irrelevant -- when checkpoint_run_name is given, since a loaded
    checkpoint's capacity is baked into its own weights; see run_sweep()'s
    docstring for why sweeping a single checkpoint across OTHER capacities
    isn't done). None (the default) preserves run()'s original,
    already-reported single-capacity behavior exactly: falls back to
    ABLATION_TOPK_CFG (capacity 0.5), unchanged from before this module.
    """
    if checkpoint_run_name is not None:
        adaptive_model, cfg = load_run(checkpoint_run_name, experiments_dir=experiments_dir, device=device)
    else:
        cfg = ABLATION_TOPK_CFG if topk_capacity is None else AdaptiveConfig(
            router_type="fixed_topk", topk_capacity=topk_capacity)
        adaptive_model = AdaptiveTransformer(cfg).to(device)
    adaptive_model.eval()

    baseline_fields = {f: getattr(cfg, f) for f in
                        ("vocab_size", "block_size", "n_layer", "n_head", "d_model", "d_ff", "dropout", "bias")}
    baseline_model = BaselineTransformer(BaselineConfig(**baseline_fields)).to(device)
    baseline_model.eval()

    hard_skip_model = HardSkipAdaptiveTransformer(adaptive_model, capacity=cfg.topk_capacity, mode="mlp_only").to(device)
    hard_skip_model.eval()

    return {
        "baseline": baseline_model,
        "adaptive_dense": adaptive_model,
        "adaptive_hard_skip": hard_skip_model,
    }, cfg


@torch.no_grad()
def time_forward(model, x, warmup, iters, device):
    is_cuda = device == "cuda"
    for _ in range(warmup):
        model(x)
        if is_cuda:
            torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        if is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times.sort()
    n = len(times)
    return {"mean_s": sum(times) / n, "median_s": times[n // 2], "min_s": times[0], "max_s": times[-1], "iters": iters}


def _time_variant_shapes(model, vocab_size, shapes, warmup, iters, device, label):
    timings = {}
    for (B, T) in shapes:
        x = torch.randint(0, vocab_size, (B, T), device=device)
        t = time_forward(model, x, warmup, iters, device)
        timings[f"{B}x{T}"] = t
        print(f"[{label}] shape={B}x{T}: median={t['median_s']*1000:.2f}ms  "
              f"mean={t['mean_s']*1000:.2f}ms  (n={iters})")
    return timings


def run(device, warmup, iters, shapes, use_compile, checkpoint_run_name=None):
    variants, cfg = build_variants(device, checkpoint_run_name=checkpoint_run_name)
    results = {"device": device, "warmup": warmup, "iters": iters, "shapes": list(shapes),
               "compile_attempted": use_compile, "checkpoint_run_name": checkpoint_run_name,
               "variants": {}}

    for name, model in variants.items():
        results["variants"][name] = {
            "eager": _time_variant_shapes(model, cfg.vocab_size, shapes, warmup, iters, device, f"{name}][eager")
        }
        if use_compile:
            try:
                compiled = torch.compile(model)
                results["variants"][name]["compiled"] = _time_variant_shapes(
                    compiled, cfg.vocab_size, shapes, warmup, iters, device, f"{name}][compiled")
            except Exception as e:
                print(f"[{name}][compiled] SKIPPED -- torch.compile failed: {e}")
                results["variants"][name]["compiled"] = {"error": str(e)}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"benchmark_speed_{device}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {out_path}")
    return results


def run_sweep(device, warmup, iters, shapes, use_compile, capacities=CAPACITIES, checkpoint_run_name=None):
    """Path A, Module 6: times all three variants at each capacity in
    `capacities` (default: the four Path A accuracy-curve points), so the
    real wall-clock number exists at every point on the accuracy-vs-
    capacity curve (plot_topk_curve.py), not just the one Module 4/5
    already measured (capacity 0.5, results/benchmark_speed_<device>.json
    -- untouched by this function, see module docstring).

    baseline's and adaptive_dense's forward-pass latency are
    architecturally capacity-independent: dense mode always computes
    every token through every layer regardless of what capacity the
    router is configured to select (FixedTopKRouter's own top-k/masking
    cost doesn't meaningfully scale with k either). Only
    adaptive_hard_skip's compute path actually changes with capacity.
    Rather than assert that and skip re-timing baseline/adaptive_dense at
    every capacity (which would save time but turn an assumption into an
    unverified one), this times all three variants at every capacity
    anyway -- baseline/adaptive_dense should come back ~flat across
    capacities (up to ordinary run-to-run noise), which is then a real,
    checked confirmation of that claim rather than a bare assertion; only
    adaptive_hard_skip should show an actual capacity-dependent trend.
    """
    if checkpoint_run_name is not None:
        # A checkpoint's weights were trained at one specific capacity,
        # baked into the router's own trained scores -- sweeping a single
        # checkpoint's weights across OTHER capacities at inference time
        # is exactly the off-capacity-inference mismatch Path A's Module 1
        # diagnostic already found catastrophically breaks quality (see
        # DECISIONS.md 2026-09-09). That finding was about quality, not
        # latency, but reusing one checkpoint across capacities here would
        # imply a comparison this project has already shown isn't
        # meaningful -- so a --checkpoint run always uses ONLY that
        # checkpoint's own trained capacity, ignoring `capacities`.
        _, cfg = load_run(checkpoint_run_name, experiments_dir="experiments", device="cpu")
        capacities = [cfg.topk_capacity]
        print(f"[run_sweep] --checkpoint given: ignoring the capacities list, using "
              f"{checkpoint_run_name}'s own trained topk_capacity={cfg.topk_capacity} only")

    results = {"device": device, "warmup": warmup, "iters": iters, "shapes": list(shapes),
               "compile_attempted": use_compile, "checkpoint_run_name": checkpoint_run_name,
               "capacities": list(capacities), "variants_by_capacity": {}}

    for capacity in capacities:
        print(f"\n=== capacity={capacity:.4f} ===")
        variants, cfg = build_variants(device, checkpoint_run_name=checkpoint_run_name, topk_capacity=capacity)
        cap_key = f"{capacity:.4f}"
        results["variants_by_capacity"][cap_key] = {}

        for name, model in variants.items():
            results["variants_by_capacity"][cap_key][name] = {
                "eager": _time_variant_shapes(model, cfg.vocab_size, shapes, warmup, iters, device,
                                               f"cap={capacity:.4f}][{name}][eager")
            }
            if use_compile:
                try:
                    compiled = torch.compile(model)
                    results["variants_by_capacity"][cap_key][name]["compiled"] = _time_variant_shapes(
                        compiled, cfg.vocab_size, shapes, warmup, iters, device,
                        f"cap={capacity:.4f}][{name}][compiled")
                except Exception as e:
                    print(f"[cap={capacity:.4f}][{name}][compiled] SKIPPED -- torch.compile failed: {e}")
                    results["variants_by_capacity"][cap_key][name]["compiled"] = {"error": str(e)}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"benchmark_speed_sweep_{device}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny shapes/iters for a fast local no-crash check")
    parser.add_argument("--checkpoint", default=None,
                         help="load real trained weights from this run (e.g. ablation_topk) instead of a fresh init")
    parser.add_argument("--sweep", action="store_true",
                         help="Path A Module 6: time all variants across multiple topk_capacity values "
                              "(default: 0.1667,0.3333,0.5,0.75) instead of a single default (0.5). "
                              "Writes to a separate output file -- never touches the original single-run result.")
    parser.add_argument("--capacities", default=None,
                         help="comma-separated topk_capacity values for --sweep (default: the Path A sweep)")
    args = parser.parse_args()

    if args.sweep:
        capacities = CAPACITIES if args.capacities is None else tuple(float(c) for c in args.capacities.split(","))
        if args.smoke:
            run_sweep(device=args.device, warmup=1, iters=2, shapes=SMOKE_SHAPES, use_compile=args.compile,
                      capacities=capacities, checkpoint_run_name=args.checkpoint)
        else:
            run_sweep(device=args.device, warmup=args.warmup, iters=args.iters, shapes=REAL_SHAPES,
                      use_compile=args.compile, capacities=capacities, checkpoint_run_name=args.checkpoint)
    elif args.smoke:
        run(device=args.device, warmup=1, iters=2, shapes=SMOKE_SHAPES, use_compile=args.compile,
            checkpoint_run_name=args.checkpoint)
    else:
        run(device=args.device, warmup=args.warmup, iters=args.iters, shapes=REAL_SHAPES, use_compile=args.compile,
            checkpoint_run_name=args.checkpoint)
