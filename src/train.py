"""Hand-written training loop for the baseline static-depth transformer.

Every run:
  1. resolves a BaselineConfig,
  2. dumps the full config + seed + git commit to experiments/<run_name>/config.json
     BEFORE training starts,
  3. logs train/val loss to W&B (offline mode by default) and to a local
     log file,
  4. checkpoints periodically.

No result from this script should ever be quoted without the matching
config.json sitting next to it.
"""
import os
import json
import time
import subprocess
import math

import torch

from config import BaselineConfig
from model import BaselineTransformer
from data import prepare_dataset, get_batch


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
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = get_batch(split, cfg, device=device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(cfg: BaselineConfig):
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

    model = BaselineTransformer(cfg).to(device)
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

    t0 = time.time()
    for step in range(cfg.max_steps + 1):
        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if step % cfg.eval_interval == 0:
            losses = estimate_loss(model, cfg, device)
            elapsed = time.time() - t0
            record = {"step": step, "lr": lr, "elapsed_s": elapsed, **losses}
            print(f"step {step}: train_loss {losses['train']:.4f} "
                  f"val_loss {losses['val']:.4f} lr {lr:.2e}")
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()
            if use_wandb:
                wandb.log(record, step=step)

            ckpt_path = os.path.join(run_dir, "ckpt.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": resolved,
            }, ckpt_path)

        x, y = get_batch("train", cfg, device=device)
        for micro_step in range(cfg.grad_accum_steps):
            _, loss = model(x, y)
            loss = loss / cfg.grad_accum_steps
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    log_f.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    cfg = BaselineConfig()
    train(cfg)
