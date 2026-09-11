"""Config for the baseline static-depth transformer.

Every training run resolves one of these to a dict and dumps it to
experiments/<run_name>/config.json before training starts, so no result
is ever reported without the exact config + seed that produced it.
"""
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class BaselineConfig:
    # architecture
    vocab_size: int = 50257  # gpt2 tiktoken encoding
    block_size: int = 256    # context length
    n_layer: int = 12
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 1536         # 4 * d_model
    dropout: float = 0.1
    bias: bool = True

    # data
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-103-raw-v1"

    # optimization
    batch_size: int = 32
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 500
    max_steps: int = 20000
    eval_interval: int = 500
    eval_iters: int = 100

    # run control
    seed: int = 1337
    run_name: str = "baseline_v1"
    out_dir: str = "experiments"
    device: str = "cuda"
    dtype: str = "bfloat16"  # falls back to float32 in train.py if unsupported
    compile: bool = True
    wandb_project: str = "adaptive-compute-transformer"
    wandb_mode: str = "offline"  # "online" once an API key is configured

    def to_dict(self):
        return asdict(self)


@dataclass
class AdaptiveConfig(BaselineConfig):
    """Same architecture as BaselineConfig (identical data + compute
    budget, per the project's baseline-vs-adaptive comparison) plus the
    router's own hyperparameters and the compute-budget loss (see
    losses.py). Phase 1 trains with a soft gate — the LM loss alone never
    rewards using fewer layers, so target_avg_layers/budget_weight supply
    the missing pressure. Train separate runs at different
    target_avg_layers values to trace the accuracy-vs-avg-layers curve."""

    # router
    router_type: str = "learned_gate"  # see router.build_router; other types added per ablation module
    router_hidden_dim: Optional[int] = None  # None = single Linear(d_model, 1), the lightest option
    topk_capacity: float = 0.5  # fixed_topk only: exact fraction of tokens selected per sequence per layer
    confidence_threshold: float = 0.4309  # confidence_threshold only: empirically calibrated via
                                           # calibrate_threshold.py against the real architecture on
                                           # real WikiText-103 validation data (untrained model) --
                                           # targets nonzero_frac~=0.5 i.e. avg_depth~=6, matching the
                                           # other two ablation arms. Re-calibrate if real training
                                           # behavior drifts meaningfully from this starting estimate.

    # compute budget
    target_avg_layers: float = 6.0  # midpoint of n_layer=12; sweep this across separate runs for the curve
    budget_weight: float = 0.01     # needs empirical tuning on real data; keep LM loss dominant early on

    # run control
    run_name: str = "adaptive_v1"
    ckpt_interval: Optional[int] = None  # None = checkpoint every eval_interval steps; set independently
                                          # to decouple checkpoint safety from (expensive) eval cadence
                                          # on long runs, so a mid-run Colab/Kaggle disconnect never
                                          # loses more than ckpt_interval steps of progress
