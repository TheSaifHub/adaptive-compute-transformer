"""Hand-written training loop for the adaptive-depth transformer.

Mirrors train.py's structure and reproducibility discipline (full config +
seed + git commit logged before training starts) but swaps in
AdaptiveTransformer and a router-type-aware loss (losses.compute_loss --
only learned_gate actually has a budget-loss term; see its docstring),
and logs the extra metrics that matter for this model regardless of
router type: avg_depth (effective layers used, magnitude-weighted for
fixed_topk -- see FixedTopKRouter's docstring), nonzero_frac (the true
fraction of (token, layer) slots the router actually applied at all --
redundant with avg_depth for learned_gate/confidence_threshold, but the
one exact, FLOPs-comparable number for fixed_topk), and the LM loss
alongside the total (LM + budget, where applicable) loss that's actually
optimized, for both train and val.

No result from this script should ever be quoted without the matching
config.json sitting next to it.
"""
import os
import json
import time
import subprocess
import math

import torch

from config import AdaptiveConfig
from model import AdaptiveTransformer
from data import prepare_dataset, get_batch
from losses import compute_loss


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
        ).decode().strip()
    except Exception:
        return "unknown"


def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step > cfg.max_steps:
        return cfg.min_lr
    decay_ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, cfg, device):
    model.eval()
    out = {}
    for split in ("train", "val"):
        total_losses = torch.zeros(cfg.eval_iters)
        lm_losses = torch.zeros(cfg.eval_iters)
        avg_depths = torch.zeros(cfg.eval_iters)
        nonzero_fracs = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = get_batch(split, cfg, device=device)
            _, lm_loss, aux = model(x, y)
            total_loss, _ = compute_loss(lm_loss, aux["avg_depth"], cfg)
            total_losses[i] = total_loss.item()
            lm_losses[i] = lm_loss.item()
            avg_depths[i] = aux["avg_depth"].item()
            nonzero_fracs[i] = (aux["gates"] > 0).float().mean().item()
        # "train"/"val" hold the TOTAL loss (LM + budget where applicable),
        # matching the key names train.py's baseline log uses for its
        # optimized loss -- so existing plotting code that reads
        # r["train"]/r["val"] keeps working unchanged on this log too.
        # LM loss, avg_depth, and nonzero_frac are separate keys.
        out[split] = total_losses.mean().item()
        out[f"{split}_lm"] = lm_losses.mean().item()
        out[f"{split}_avg_depth"] = avg_depths.mean().item()
        out[f"{split}_nonzero_frac"] = nonzero_fracs.mean().item()
    model.train()
    return out


def train(cfg: AdaptiveConfig):
    torch.manual_seed(cfg.seed)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    if device == "cpu" and cfg.device == "cuda":
        print("WARNING: CUDA requested but not available — falling back to CPU. "
              "This will be very slow; use Colab/Kaggle GPU for real runs.")

    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)

    resolved = cfg.to_dict()
    resolved["git_commit"] = get_git_commit()
    resolved["resolved_device"] = device
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(resolved, f, indent=2)

    print(f"[{cfg.run_name}] preparing dataset (cached after first run)...")
    prepare_dataset(cfg.dataset_config)

    model = AdaptiveTransformer(cfg).to(device)
    print(f"[{cfg.run_name}] model params (excl. pos emb): {model.num_params():,}")

    if cfg.compile and device == "cuda":
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    use_wandb = False
    try:
        import wandb
        os.environ["WANDB_MODE"] = cfg.wandb_mode
        wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=resolved)
        use_wandb = True
    except Exception as e:
        print(f"W&B not initialized ({e}); continuing without it.")

    log_path = os.path.join(run_dir, "train_log.jsonl")
    log_f = open(log_path, "a")

    # Checkpoint saving is decoupled from eval_interval: eval does real
    # forward passes (eval_iters batches x 2 splits) so it's expensive,
    # while a checkpoint save is just a state_dict dump -- cheap. Forcing
    # them onto the same cadence means either wasting eval compute for
    # checkpoint safety, or losing checkpoint safety when eval_interval is
    # widened for speed on a long run. ckpt_interval defaults to
    # eval_interval (unchanged behavior) but can be set independently.
    ckpt_interval = cfg.ckpt_interval or cfg.eval_interval
    ckpt_path = os.path.join(run_dir, "ckpt.pt")

    t0 = time.time()
    for step in range(cfg.max_steps + 1):
        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if step % cfg.eval_interval == 0:
            metrics = estimate_loss(model, cfg, device)
            elapsed = time.time() - t0
            record = {"step": step, "lr": lr, "elapsed_s": elapsed, **metrics}
            print(f"step {step}: train_loss {metrics['train']:.4f} "
                  f"val_loss {metrics['val']:.4f} "
                  f"train_avg_depth {metrics['train_avg_depth']:.2f} "
                  f"val_avg_depth {metrics['val_avg_depth']:.2f} lr {lr:.2e}")
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()
            if use_wandb:
                wandb.log(record, step=step)

        # Overwrites the same ckpt.pt every time, so there's always exactly
        # one recent checkpoint sitting in the run directory -- readable
        # from Colab/Kaggle's file panel even if the session disconnects
        # mid-run and train() never returns to trigger the notebook's
        # end-of-run backup/download.
        if step % ckpt_interval == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": resolved,
            }, ckpt_path)

        x, y = get_batch("train", cfg, device=device)
        for micro_step in range(cfg.grad_accum_steps):
            _, lm_loss, aux = model(x, y)
            total_loss, _ = compute_loss(lm_loss, aux["avg_depth"], cfg)
            total_loss = total_loss / cfg.grad_accum_steps
            total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    log_f.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    cfg = AdaptiveConfig()
    train(cfg)
