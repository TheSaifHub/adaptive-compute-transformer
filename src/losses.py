"""Compute-budget loss for the adaptive-depth model.

The LM cross-entropy alone only ever rewards MORE depth (more compute is
never wrong for next-token prediction) -- nothing pushes the model toward
using fewer layers. This adds the missing pressure: a penalty on how far
the batch's average effective depth (AdaptiveTransformer's
`aux["avg_depth"]`) is from a target, so sweeping that target across
separate training runs traces out the accuracy-vs-average-layers curve
that's the project's central result.

Deliberately penalizes the batch AVERAGE depth, not each token's
individual depth: the whole premise of this project is that depth should
vary a lot per token (easy tokens exit early, hard ones go deep). A
per-token penalty would fight that directly; an average-depth penalty
only constrains the aggregate compute budget and leaves per-token routing
free to be as heterogeneous as the data calls for.
"""
import torch


def adaptive_loss(lm_loss, avg_depth, cfg):
    """Combines the LM loss with a quadratic penalty pulling the batch's
    average effective depth toward cfg.target_avg_layers.

    Returns (total_loss, budget_loss) -- the unweighted budget_loss is
    returned separately so a training loop can log both terms.
    """
    target = torch.as_tensor(cfg.target_avg_layers, dtype=avg_depth.dtype, device=avg_depth.device)
    budget_loss = (avg_depth - target) ** 2
    total_loss = lm_loss + cfg.budget_weight * budget_loss
    return total_loss, budget_loss


def compute_loss(lm_loss, avg_depth, cfg):
    """Router-type-aware total loss (see router.build_router for the
    router types this dispatches on). Only learned_gate has a budget loss
    to add: its average depth is a free-floating quantity that needs an
    explicit target pushed toward via gradient descent. fixed_topk's
    average depth is already exact by construction (via its capacity
    hyperparameter), and confidence_threshold has no learned parameters
    to train toward a target at all -- for both, total loss is just the
    LM loss; average depth is still logged (see train_adaptive.py) but
    isn't being optimized against a target.

    Returns (total_loss, budget_loss_or_None).
    """
    if cfg.router_type == "learned_gate":
        return adaptive_loss(lm_loss, avg_depth, cfg)
    return lm_loss, None
