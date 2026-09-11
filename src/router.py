"""Per-token routers for the adaptive-depth model.

Each router scores every token at every layer: has this token's
representation stabilized, or does it still need more depth? All router
variants (learned gate, fixed top-k, confidence threshold) share one
forward(x, block_out) call signature, so AdaptiveBlock never needs to
branch on which router type is plugged in -- routers that don't need
block_out (learned gate, fixed top-k) simply ignore it; only
ConfidenceThresholdRouter uses it.
"""
import torch
import torch.nn as nn


class LearnedGateRouter(nn.Module):
    """Outputs a per-token gate in (0, 1): ~1 means "this token still needs
    this layer", ~0 means "treat as converged, freeze it here". Multiplying
    a block's output by this gate (the soft-training scheme) keeps the
    router differentiable end-to-end via ordinary backprop through the
    residual stream — no REINFORCE / straight-through estimator needed.
    """

    def __init__(self, d_model, hidden_dim=None, bias=True):
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(d_model, 1, bias=bias)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim, bias=bias),
                nn.GELU(),
                nn.Linear(hidden_dim, 1, bias=bias),
            )

        self.reset_gate_init()

    def reset_gate_init(self):
        """Zero-inits the last layer so every router starts at the same
        bias-controlled gate value regardless of the (random) hidden
        state it's fed, then biases it toward "continue" (~0.88). At step
        0 the base transformer hasn't learned anything yet, so router
        decisions would otherwise be arbitrary; starting near "always
        continue" avoids randomly freezing tokens before there's any
        signal worth routing on, and lets the router earn its way toward
        skipping tokens as the base representations become meaningful.

        Callable again after the fact — a wrapping model's blanket
        weight-init pass (e.g. `model.apply(_init_weights)`) will
        re-init every nn.Linear it finds, including this one, so it
        needs to be reasserted afterward.
        """
        last_linear = self.net if isinstance(self.net, nn.Linear) else self.net[-1]
        nn.init.zeros_(last_linear.weight)
        if last_linear.bias is not None:
            nn.init.constant_(last_linear.bias, 2.0)  # sigmoid(2.0) ~= 0.88

    def forward(self, x, block_out=None):
        """x: (B, T, d_model) -> gate: (B, T), values in (0, 1). block_out
        is accepted (and ignored) so every router shares one call signature
        in AdaptiveBlock.forward() -- only ConfidenceThresholdRouter
        actually needs the block's output."""
        return torch.sigmoid(self.net(x).squeeze(-1))


class FixedTopKRouter(nn.Module):
    """Mixture-of-Depths-style router: a small learned per-token scorer
    ranks all tokens in each sequence, and exactly the top `capacity`
    fraction (rounded, per sequence) get processed -- their gate is their
    own sigmoid score; everyone else gets gate = 0 exactly. Unlike
    LearnedGateRouter, average depth doesn't need a budget loss at all --
    capacity directly and exactly determines what fraction of tokens are
    selected, by construction, every forward pass.

    Note the selected tokens' gate is their SCORE, not 1 -- selection
    (which tokens rank in the top capacity fraction) isn't differentiable,
    so scaling by the score is what gives the scorer's weights a gradient
    signal at all (only from tokens that got selected in a given batch).
    This means the "capacity fraction selected" (exact, a clean
    FLOPs-comparable number) and "avg_depth" as logged elsewhere in this
    project (gates summed across layers, which for this router is
    magnitude-weighted by the score, not a plain selection count) are two
    different quantities -- both worth logging, not interchangeable.
    """

    def __init__(self, d_model, capacity, hidden_dim=None, bias=True):
        super().__init__()
        assert 0.0 < capacity <= 1.0, f"capacity must be in (0, 1], got {capacity}"
        self.capacity = capacity
        if hidden_dim is None:
            self.net = nn.Linear(d_model, 1, bias=bias)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim, bias=bias),
                nn.GELU(),
                nn.Linear(hidden_dim, 1, bias=bias),
            )

    def forward(self, x, block_out=None):
        """x: (B, T, d_model) -> gate: (B, T). Exactly
        max(1, min(T, round(capacity * T))) positions per sequence are
        non-zero (= their own sigmoid score); the rest are exactly 0.
        block_out is accepted (and ignored) for signature parity with
        ConfidenceThresholdRouter -- see LearnedGateRouter.forward."""
        B, T, _ = x.shape
        scores = torch.sigmoid(self.net(x).squeeze(-1))  # (B, T)

        k = max(1, min(T, round(self.capacity * T)))
        topk_idx = scores.topk(k, dim=1).indices  # (B, k)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, topk_idx, True)

        return scores * mask


class ConfidenceThresholdRouter(nn.Module):
    """CALM-inspired confidence-threshold router, using a stabilization
    proxy rather than the literature's full early-exit vocab-head
    approach: gates on how much a block's own output actually changes a
    token's representation, relative to the token's own current
    magnitude -- ||block(x) - x|| / ||x||. A token whose representation
    barely moves under further processing has effectively converged.

    Relative, not absolute: the residual stream's magnitude grows across
    layers in a pre-LN transformer (nothing re-normalizes it between
    blocks), so a fixed absolute delta-norm threshold would be
    miscalibrated by depth alone -- early layers rarely tripping it, late
    layers almost always tripping it, regardless of token difficulty.
    Dividing by ||x|| makes the threshold comparable at every layer.

    No learned parameters -- nothing here is trained by gradient descent
    (there's no budget loss for this router either); average depth is
    controlled purely by the fixed threshold, calibrated empirically.
    The gate is a hard 0/1 (not continuous like the other two routers):
    there's no gradient-training reason to keep it soft, since there's no
    router weight that soft-ness would need to be differentiable for --
    gradient to the underlying block's own weights (attention/MLP) still
    flows normally through the constant gate multiplier either way.
    """

    def __init__(self, threshold, eps=1e-6):
        super().__init__()
        self.threshold = threshold
        self.eps = eps

    def forward(self, x, block_out):
        """x, block_out: (B, T, d_model) -> gate: (B, T), values in
        {0.0, 1.0} exactly."""
        delta_norm = (block_out - x).norm(dim=-1)
        relative_delta = delta_norm / (x.norm(dim=-1) + self.eps)
        return (relative_delta > self.threshold).float()


def build_router(cfg):
    """Constructs the router named by cfg.router_type. Each router class
    keeps its own primitive-argument constructor (decoupled, independently
    testable in isolation) -- this factory is the one place that knows how
    to pull the right fields off cfg for each type, so adding a new router
    type (fixed top-k, confidence threshold, ...) never requires touching
    AdaptiveBlock/AdaptiveTransformer.
    """
    if cfg.router_type == "learned_gate":
        return LearnedGateRouter(cfg.d_model, hidden_dim=cfg.router_hidden_dim)
    if cfg.router_type == "fixed_topk":
        return FixedTopKRouter(cfg.d_model, capacity=cfg.topk_capacity, hidden_dim=cfg.router_hidden_dim)
    if cfg.router_type == "confidence_threshold":
        return ConfidenceThresholdRouter(threshold=cfg.confidence_threshold)
    raise ValueError(f"unknown router_type: {cfg.router_type!r}")
