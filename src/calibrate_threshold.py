"""Empirical calibration for ConfidenceThresholdRouter's threshold.

confidence_threshold has no budget loss to converge it toward a target
average depth the way learned_gate does -- depth is controlled purely by
the fixed threshold, and the threshold-to-depth relationship isn't
knowable in advance. Guessing blind wastes real Kaggle GPU-hours; this
script estimates a starting threshold locally and for free, by measuring
the actual distribution of relative delta-norms (||block(x)-x|| / ||x||,
exactly what the router thresholds) that the real architecture produces
on real WikiText data.

Round-1 only: uses a freshly-initialized (untrained) model, bypassing the
router itself to measure what every block would do if applied in full at
every layer. This is a reasonable starting point precisely because the
router hasn't been trained yet -- there's no learned feedback loop between
a layer's routing decision and what the next layer receives to account
for. Once a real Kaggle run exists at some threshold, a *better* round-2
estimate should come from that run's own actual nonzero_frac trajectory,
not from this script -- it does not attempt to simulate a trained model.
"""
import sys

import numpy as np
import torch

from config import AdaptiveConfig
from model import AdaptiveTransformer
import data as data_module


@torch.no_grad()
def measure_relative_deltas(model, idx):
    """Runs the embedding + block stack manually (bypassing every router
    entirely -- full block applied at every layer, as if gate were always
    1), collecting the relative delta norm at every (token, layer). This
    is exactly what ConfidenceThresholdRouter.forward computes internally,
    just gathered across all layers instead of thresholded and discarded.
    Returns a flat 1-D array of relative deltas, shape (B*T*n_layer,).
    """
    cfg = model.cfg
    B, T = idx.shape
    pos = torch.arange(T, device=idx.device)
    x = model.drop(model.tok_emb(idx) + model.pos_emb(pos))

    all_deltas = []
    for block in model.blocks:
        out = block.block(x)
        rel_delta = (out - x).norm(dim=-1) / (x.norm(dim=-1) + 1e-6)
        all_deltas.append(rel_delta.flatten())
        x = out  # full block applied, matching an always-gate=1 pass
    return torch.cat(all_deltas).numpy()


def suggest_threshold(deltas, target_fraction=0.5):
    """The threshold that would select exactly target_fraction of
    (token, layer) pairs, given this empirical distribution -- e.g. the
    median for target_fraction=0.5."""
    percentile = 100 * (1 - target_fraction)
    return float(np.percentile(deltas, percentile))


if __name__ == "__main__":
    torch.manual_seed(1337)

    # Real architecture, matching adaptive_v1's actual config -- the
    # point is to measure what THIS SIZE of model does, not a toy one.
    cfg = AdaptiveConfig(router_type="confidence_threshold")
    model = AdaptiveTransformer(cfg)
    model.eval()

    enc_vocab_size = cfg.vocab_size
    val_path = "data/tokenized/val.bin"
    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    print(f"loaded {len(data):,} real WikiText-103 validation tokens from {val_path}")

    batch_size = 16
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(data) - cfg.block_size - 1, size=batch_size)
    batch = np.stack([data[s:s + cfg.block_size].astype(np.int64) for s in starts])
    idx = torch.from_numpy(batch)

    deltas = measure_relative_deltas(model, idx)
    print(f"collected {len(deltas):,} relative-delta samples "
          f"({batch_size} sequences x {cfg.block_size} tokens x {cfg.n_layer} layers)")
    print(f"distribution: min={deltas.min():.4f} p10={np.percentile(deltas,10):.4f} "
          f"p50={np.percentile(deltas,50):.4f} p90={np.percentile(deltas,90):.4f} max={deltas.max():.4f}")

    for target_fraction in (0.25, 0.5, 0.75):
        # target_fraction here is nonzero_frac, which for this router
        # type equals avg_depth / n_layer exactly (binary gate) -- so
        # target_fraction=0.5 with n_layer=12 aims for avg_depth ~= 6,
        # matching the other two ablation arms.
        threshold = suggest_threshold(deltas, target_fraction)
        print(f"target nonzero_frac={target_fraction:.2f} (avg_depth ~= {target_fraction * cfg.n_layer:.1f}) "
              f"-> suggested confidence_threshold={threshold:.4f}")
