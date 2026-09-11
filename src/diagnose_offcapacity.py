"""Path A, Module 1: cheap local diagnostic -- does ablation_topk's
capacity=0.5-trained model produce plausible quality at OTHER capacities
without retraining, or does off-capacity inference collapse the way
full_block hard-skip did on learned_gate (see DECISIONS.md, 2026-09-08)?

NOT a substitute for retraining. This is a cheap (~10 min, CPU, no GPU)
early-warning gate before committing ~15 Kaggle-hours to 3 new training
runs. Even a "plausible, no collapse" result here likely still
understates what a genuinely-trained model at that capacity would
achieve, since the base model's shared weights were optimized for
capacity=0.5's selection density, not the test capacity -- that's exactly
the judgment call this diagnostic hands back to Saif, not a decision to
make unilaterally.

Technical note: FixedTopKRouter.forward() bakes in its OWN trained
capacity (0.5) when computing its selection mask -- passing a different
capacity to hard_skip.py's existing machinery would silently misbehave
above 0.5 (the router's own forward already zeroed everything beyond its
trained top-50%, discarding the information needed to select more). This
script bypasses that by pulling the raw, capacity-independent score
directly from the router's own scoring network (`router.net`) before any
masking, then applies an external top-k at the test capacity -- this
works correctly in both directions (narrower or wider than 0.5).
"""
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_batch
from hard_skip import hard_skip_mlp_only_forward, topk_indices_in_order
from visualize_routing import load_run

EXPERIMENTS_DIR = "experiments"
RESULTS_DIR = "results"
VAL_BIN_PATH = os.path.join("data", "tokenized", "val.bin")
TEST_CAPACITIES = [0.1667, 0.3333, 0.5, 0.75]  # 0.5 included as a self-consistency check
TRAINED_CAPACITY = 0.5
EVAL_ITERS = 20
SEED = 1337


def raw_router_scores(router, x):
    """FixedTopKRouter's raw, capacity-independent sigmoid score --
    bypasses FixedTopKRouter.forward's own baked-in capacity masking."""
    return torch.sigmoid(router.net(x).squeeze(-1))


class OffCapacityHardSkipBlock(nn.Module):
    """Diagnostic-only: selects top-k using the router's RAW score at an
    arbitrary test capacity, ignoring the capacity the router was
    actually trained/built with. Not part of the real Phase 2 pipeline."""

    def __init__(self, adaptive_block, test_capacity):
        super().__init__()
        self.block = adaptive_block.block
        self.router = adaptive_block.router
        self.test_capacity = test_capacity

    def forward(self, x):
        B, T, D = x.shape
        raw_scores = raw_router_scores(self.router, x)
        k = max(1, min(T, round(self.test_capacity * T)))
        active_idx = topk_indices_in_order(raw_scores, k)
        # FixedTopKRouter.forward() zeroes the score for non-selected
        # tokens before anything downstream sees it (scores * mask) --
        # the model was trained with EXACTLY zero attn+mlp contribution
        # for excluded tokens, not a small nonzero one. Replicate that
        # masking here at the test capacity, or g_full below would give
        # excluded tokens a real (if small) attention contribution the
        # model never saw during training -- this was caught by the
        # capacity=0.5 self-consistency check failing to reproduce the
        # known-exact Module 3 result (+0.083 instead of +0.000).
        mask = torch.zeros_like(raw_scores, dtype=torch.bool)
        mask.scatter_(1, active_idx, True)
        scores = raw_scores * mask
        return hard_skip_mlp_only_forward(self.block, x, active_idx, scores)


class OffCapacityHardSkipTransformer(nn.Module):
    def __init__(self, adaptive_transformer, test_capacity):
        super().__init__()
        self.cfg = adaptive_transformer.cfg
        self.tok_emb = adaptive_transformer.tok_emb
        self.pos_emb = adaptive_transformer.pos_emb
        self.drop = adaptive_transformer.drop
        self.ln_f = adaptive_transformer.ln_f
        self.head = adaptive_transformer.head
        self.hs_blocks = nn.ModuleList([
            OffCapacityHardSkipBlock(b, test_capacity) for b in adaptive_transformer.blocks
        ])

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        gates = []
        for block in self.hs_blocks:
            x, gate = block(x)
            gates.append(gate)
        gates = torch.stack(gates, dim=-1)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        avg_depth = gates.sum(dim=-1).mean()
        return logits, loss, {"gates": gates, "avg_depth": avg_depth}


@torch.no_grad()
def run(run_name="ablation_topk", capacities=TEST_CAPACITIES, eval_iters=EVAL_ITERS, seed=SEED):
    assert os.path.exists(VAL_BIN_PATH), f"{VAL_BIN_PATH} not found -- see eval_hard_skip.py's note"

    dense_model, cfg = load_run(run_name, experiments_dir=EXPERIMENTS_DIR, device="cpu")

    torch.manual_seed(seed)
    batches = [get_batch("val", cfg, device="cpu") for _ in range(eval_iters)]

    dense_losses = [dense_model(x, y)[1].item() for x, y in batches]
    dense_mean = sum(dense_losses) / len(dense_losses)
    print(f"[{run_name}] dense (trained capacity={TRAINED_CAPACITY}) val_lm, {eval_iters} iters: {dense_mean:.4f}")
    print(f"(known reference from Module 3, real training-logged: 3.9139)\n")

    results = {"run_name": run_name, "trained_capacity": TRAINED_CAPACITY, "eval_iters": eval_iters,
               "seed": seed, "dense_val_lm_local": dense_mean, "capacities": {}}

    for cap in capacities:
        model = OffCapacityHardSkipTransformer(dense_model, cap)
        model.eval()
        losses, depths = [], []
        for x, y in batches:
            _, loss, aux = model(x, y)
            losses.append(loss.item())
            depths.append(aux["avg_depth"].item())
        mean_loss = sum(losses) / len(losses)
        mean_depth = sum(depths) / len(depths)
        delta = mean_loss - dense_mean
        tag = "SELF-CONSISTENCY CHECK" if cap == TRAINED_CAPACITY else ""
        print(f"test_capacity={cap:.4f}  val_lm={mean_loss:.4f}  delta_vs_dense={delta:+.4f}  "
              f"avg_depth={mean_depth:.4f}  {tag}")
        results["capacities"][str(cap)] = {
            "val_lm": mean_loss, "avg_depth": mean_depth, "delta_vs_dense": delta,
        }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "diagnose_offcapacity.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {out_path}")
    return results


if __name__ == "__main__":
    run()
