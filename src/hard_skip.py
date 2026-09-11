"""Phase 2: real hard-skip inference for a trained learned_gate AdaptiveTransformer.

Phase 1's AdaptiveBlock computes every token through every layer, only
blending the result by a soft gate -- zero wall-clock savings. This module
wraps an already-trained AdaptiveBlock (unmodified; its weights are reused
directly, not copied or retrained -- see DECISIONS.md 2026-09-08) with a
real compute-skipping forward path.

Two mechanisms, both built the same way (gather -> compute -> blend ->
scatter on explicit indices, independent of router logic, wrapped by an
inference-time class that derives those indices from the router's own
per-token score):
  - `hard_skip_block_forward` / `HardSkipAdaptiveBlock` ("full_block"):
    skips attn+MLP together for excluded tokens -- matches the real
    Mixture-of-Depths mechanism (excluded tokens contribute no K/V at
    that layer at all). Found to catastrophically break quality on this
    project's real checkpoint (val_lm nearly doubling at the real
    capacity) -- Phase 1 training never excludes anyone from attention,
    so the model was never taught to route around missing K/V. Kept
    available, no longer the default -- see DECISIONS.md 2026-09-08 addendum.
  - `hard_skip_mlp_only_forward` / `HardSkipMLPOnlyAdaptiveBlock`
    ("mlp_only", the default): keeps attention fully dense (identical to
    Phase 1 in every way) and skips real compute only for the MLP
    sub-layer, only for excluded tokens -- carries none of the above
    mismatch, at the cost of a smaller (but still real) compute saving.

`HardSkipAdaptiveTransformer` wires either mechanism (via `mode=`) into a
full model, mirroring AdaptiveTransformer.forward()'s structure exactly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def topk_indices_in_order(scores, k):
    """Per-sequence top-k indices by score, re-sorted back into ascending
    (original sequence) order. Critical: gathering in score-rank order
    instead would corrupt the causal mask applied to the compacted
    subsequence -- a later-position token could land before an
    earlier-position one in the gathered tensor.

    scores: (B, T) -> idx: (B, k), each row strictly increasing.
    """
    idx = scores.topk(k, dim=1).indices  # (B, k), score-descending order
    idx, _ = idx.sort(dim=1)
    return idx


def hard_skip_block_forward(block, x, active_idx, scores):
    """x: (B, T, D). active_idx: (B, k) token indices, ascending per row --
    exactly the tokens that get real compute this layer. scores: (B, T),
    the full per-token gate/score (inactive positions' scores are present
    but unused) -- only the active subset's own scores are used to blend
    the active subset's own block output, the same way Phase 1's soft
    blend does.

    Excluded positions are returned bit-for-bit unchanged (gate 0, no
    compute, and no contribution to this layer's attention for anyone
    else either -- they're never part of the gathered subsequence).
    Active positions get block(x)'s real output blended by their own
    score -- identical math to Phase 1's AdaptiveBlock.forward, just
    computed only on the gathered subset.

    Returns (x_new, gate) where gate: (B, T) is 0 at excluded positions
    and the active tokens' own score elsewhere -- same shape/meaning as
    AdaptiveBlock.forward's gate output, so it aggregates into avg_depth
    the same way.
    """
    B, T, D = x.shape
    idx_expanded = active_idx.unsqueeze(-1).expand(-1, -1, D)  # (B, k, D)
    x_active = torch.gather(x, 1, idx_expanded)  # (B, k, D)

    out_active = block(x_active)  # real compute, only on the gathered subset

    scores_active = torch.gather(scores, 1, active_idx)  # (B, k)
    g = scores_active.unsqueeze(-1)
    blended_active = g * out_active + (1 - g) * x_active

    x_new = x.clone()
    x_new.scatter_(1, idx_expanded, blended_active)

    gate = torch.zeros(B, T, dtype=scores.dtype, device=scores.device)
    gate.scatter_(1, active_idx, scores_active)

    return x_new, gate


def hard_skip_mlp_only_forward(block, x, active_idx, scores):
    """Real hard-skip variant that keeps self-attention fully DENSE --
    every token computes and is visible as Q/K/V at every layer, exactly
    matching how Phase 1 training computed attention -- and skips real
    compute only for the MLP sub-layer, only for excluded tokens.

    Added after `hard_skip_block_forward` (full attn+MLP skip) was found,
    on the real adaptive_v1 checkpoint, to catastrophically break quality
    (val_lm nearly doubling even with only ~50% of tokens excluded) --
    see DECISIONS.md 2026-09-08 addendum. Root cause: Phase 1 training
    never excludes any token from attention, so a model trained that way
    was never taught to route around a token vanishing from the K/V pool.
    This variant never changes attention at all, so it carries none of
    that mismatch.

    Uses the algebraic identity that Phase 1's blend formula
    (gate*block(x) + (1-gate)*x, where block(x) = x + attn_delta +
    mlp_delta, since Block is a standard residual block) distributes
    additively: gate*(x + attn_delta + mlp_delta) + (1-gate)*x
    = x + gate*attn_delta + gate*mlp_delta. So scaling each sub-layer's
    own delta by the SAME gate and adding both to x reproduces Phase 1's
    original formula exactly -- this function just never skips computing
    attn_delta (for anyone), and only skips computing mlp_delta for
    excluded tokens (treating their mlp contribution as exactly 0, the
    same approximation `hard_skip_block_forward` already made for the
    whole block, now scoped to just the MLP half).

    Important: mlp_delta must be computed from the UNSCALED post-attention
    residual (x + attn_delta, matching what dense Block.forward actually
    feeds into ln2/mlp) -- NOT from the gate-scaled x + gate*attn_delta.
    Gate scaling is applied once, at the very end, to attn_delta and
    mlp_delta separately. (An earlier draft scaled attn_delta before
    computing mlp_delta from it -- wrong, and caught by the capacity=1.0
    exact-reproduction smoke test, which requires this to be bit-for-bit
    correct rather than merely close.)

    block: an AdaptiveBlock's own `.block` (needs .ln1, .attn, .ln2, .mlp
    -- see model.Block). x: (B, T, D). active_idx: (B, k), ascending per
    row. scores: (B, T), the full per-token score (gate).

    Returns (x_new, gate) -- same shape/meaning as hard_skip_block_forward.
    """
    B, T, D = x.shape
    g_full = scores.unsqueeze(-1)

    # Attention sub-layer: dense, every token, every layer -- identical
    # compute to Phase 1/dense AdaptiveBlock in every way.
    attn_delta = block.attn(block.ln1(x))
    x_plus_attn = x + attn_delta  # UNSCALED -- the correct mlp/ln2 input

    # MLP sub-layer: real compute only for the active subset, fed the
    # same unscaled residual dense Block.forward would use.
    idx_expanded = active_idx.unsqueeze(-1).expand(-1, -1, D)
    x_plus_attn_active = torch.gather(x_plus_attn, 1, idx_expanded)
    mlp_delta_active = block.mlp(block.ln2(x_plus_attn_active))

    scores_active = torch.gather(scores, 1, active_idx)
    g_active = scores_active.unsqueeze(-1)

    x_new = x + g_full * attn_delta  # everyone gets their gate-scaled attn contribution
    mlp_contrib = torch.zeros_like(x)
    mlp_contrib.scatter_(1, idx_expanded, g_active * mlp_delta_active)
    x_new = x_new + mlp_contrib  # excluded tokens' mlp contribution stays exactly 0

    gate = torch.zeros(B, T, dtype=scores.dtype, device=scores.device)
    gate.scatter_(1, active_idx, scores_active)

    return x_new, gate


class HardSkipMLPOnlyAdaptiveBlock(nn.Module):
    """Inference-time hard-skip wrapper using the MLP-only mechanism (see
    hard_skip_mlp_only_forward) -- reuses an already-trained
    AdaptiveBlock's weights directly, no new parameters. Same capacity=1.0
    exact-reproduction invariant as HardSkipAdaptiveBlock."""

    def __init__(self, adaptive_block, capacity):
        super().__init__()
        assert 0.0 < capacity <= 1.0, f"capacity must be in (0, 1], got {capacity}"
        self.block = adaptive_block.block
        self.router = adaptive_block.router
        self.capacity = capacity

    def forward(self, x):
        B, T, D = x.shape
        scores = self.router(x)
        k = max(1, min(T, round(self.capacity * T)))
        active_idx = topk_indices_in_order(scores, k)
        return hard_skip_mlp_only_forward(self.block, x, active_idx, scores)


class HardSkipAdaptiveBlock(nn.Module):
    """Inference-time hard-skip wrapper around an already-trained
    AdaptiveBlock. Creates no parameters of its own -- reuses the given
    block's `.block` (attn+MLP) and `.router` directly, so this is purely
    a different forward path over the same trained weights.

    capacity=1.0 (select every token, every layer) must reproduce the
    wrapped AdaptiveBlock's own dense forward() output exactly -- see the
    Module 2 smoke test. That invariant is what makes this a real
    optimization of the trained model, not a differently-behaving one.
    """

    def __init__(self, adaptive_block, capacity):
        super().__init__()
        assert 0.0 < capacity <= 1.0, f"capacity must be in (0, 1], got {capacity}"
        self.block = adaptive_block.block
        self.router = adaptive_block.router
        self.capacity = capacity

    def forward(self, x):
        B, T, D = x.shape
        scores = self.router(x)  # (B, T) -- needs only x; block_out isn't computed yet
        k = max(1, min(T, round(self.capacity * T)))
        active_idx = topk_indices_in_order(scores, k)
        return hard_skip_block_forward(self.block, x, active_idx, scores)


HARD_SKIP_MODES = {
    "full_block": HardSkipAdaptiveBlock,   # skip attn+MLP together; see DECISIONS.md -- breaks quality on a
                                            # model trained without attention exclusion (Phase 1 soft blend)
    "mlp_only": HardSkipMLPOnlyAdaptiveBlock,  # attention stays dense; only MLP is skipped -- see above
}


class HardSkipAdaptiveTransformer(nn.Module):
    """Inference-time hard-skip wrapper around an already-trained, already-
    loaded AdaptiveTransformer. Reuses every weight (token/position
    embeddings, each block's attn+MLP+router, ln_f, head) directly --
    creates no new parameters, mirrors AdaptiveTransformer.forward()'s
    structure and aux dict exactly so it's a drop-in replacement for
    evaluation/benchmarking code that already expects that shape.

    mode selects which per-layer mechanism to use -- see HARD_SKIP_MODES.
    Default is "mlp_only": the full_block mechanism was found to
    catastrophically break quality on a Phase-1-trained model (see
    DECISIONS.md 2026-09-08 addendum), so it's no longer the default,
    though it's kept available (e.g. for a future fine-tuning experiment
    that trains a model to expect real attention exclusion).
    """

    def __init__(self, adaptive_transformer, capacity, mode="mlp_only"):
        super().__init__()
        if mode not in HARD_SKIP_MODES:
            raise ValueError(f"unknown hard-skip mode: {mode!r}, expected one of {list(HARD_SKIP_MODES)}")
        block_cls = HARD_SKIP_MODES[mode]
        self.cfg = adaptive_transformer.cfg
        self.mode = mode
        self.tok_emb = adaptive_transformer.tok_emb
        self.pos_emb = adaptive_transformer.pos_emb
        self.drop = adaptive_transformer.drop
        self.ln_f = adaptive_transformer.ln_f
        self.head = adaptive_transformer.head
        self.hs_blocks = nn.ModuleList([
            block_cls(b, capacity) for b in adaptive_transformer.blocks
        ])

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"sequence length {T} exceeds block_size {self.cfg.block_size}"
        )
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        gates = []
        for block in self.hs_blocks:
            x, gate = block(x)
            gates.append(gate)
        gates = torch.stack(gates, dim=-1)  # (B, T, n_layer)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        avg_depth = gates.sum(dim=-1).mean()
        aux = {"gates": gates, "avg_depth": avg_depth}
        return logits, loss, aux
