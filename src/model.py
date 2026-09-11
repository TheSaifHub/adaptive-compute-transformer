"""Baseline static-depth decoder-only transformer.

Standard GPT-style architecture (pre-LayerNorm, causal self-attention),
deliberately unremarkable: this is the control model. Later, the
adaptive-depth model will subclass / wrap this same block structure so
that any measured difference is attributable to routing, not to the
base architecture.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from router import build_router


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.dropout(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    """One transformer block. Kept as a standalone module (rather than
    inlined in the stack) because the adaptive-depth model will need to
    call blocks individually, per-token, between router decisions."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # weight tying
        self.tok_emb.weight = self.head.weight

        self.apply(self._init_weights)
        # scaled init for residual projections, per GPT-2 paper
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"sequence length {T} exceeds block_size {self.cfg.block_size}"
        )
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    def num_params(self, exclude_embeddings=True):
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.pos_emb.weight.numel()
        return n

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class AdaptiveBlock(nn.Module):
    """One transformer block wrapped with a swappable per-token router
    (see router.build_router). The router produces a gate in (0, 1) from
    the block's input and/or output (which inputs it actually needs
    depends on the router type -- see router.py); the block's contribution
    to the residual stream is blended in by that gate: gate ~= 1 applies
    the block in full, gate ~= 0 leaves the token's state unchanged (full
    attention+MLP skip, not just MLP).

    Phase 1 (soft/dense training): every token still runs through the
    full block every layer — nothing is actually skipped compute-wise
    yet, only down-weighted or masked — regardless of router type. Real
    hard-skip / token-compaction for actual compute savings is a
    separate, later phase.
    """

    def __init__(self, cfg):
        super().__init__()
        self.block = Block(cfg)
        self.router = build_router(cfg)

    def forward(self, x):
        out = self.block(x)
        gate = self.router(x, out)  # (B, T); most routers only use x, ConfidenceThresholdRouter needs out too
        g = gate.unsqueeze(-1)
        x = g * out + (1 - g) * x
        return x, gate


class AdaptiveTransformer(nn.Module):
    """Adaptive-depth counterpart to BaselineTransformer: identical
    architecture (same embeddings, ln_f, head, weight tying, init scheme)
    so any measured difference is attributable to routing — but each
    block is an AdaptiveBlock instead of a plain Block, and forward()
    additionally returns the per-layer, per-token gates so a training
    loop can log average depth used and (later) add a compute-budget
    loss term.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([AdaptiveBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # weight tying
        self.tok_emb.weight = self.head.weight

        self.apply(self._init_weights)
        # scaled init for residual projections, per GPT-2 paper
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        # self.apply() above re-initializes every nn.Linear it finds,
        # including each router's -- reassert the router's own zero-init
        # bootstrap (see LearnedGateRouter.reset_gate_init) afterward.
        # Duck-typed: not every router type has (or needs) a bootstrap
        # reset -- e.g. a non-learned confidence-threshold router has no
        # params to reassert -- so only call it where it actually exists.
        for block in self.blocks:
            if hasattr(block.router, "reset_gate_init"):
                block.router.reset_gate_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"sequence length {T} exceeds block_size {self.cfg.block_size}"
        )
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        gates = []
        for block in self.blocks:
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

        # Sum of gates across layers approximates each token's "effective
        # depth" under the soft blend (gate ~= 1 per layer contributes
        # ~1 full layer of compute); averaged over the batch, this is the
        # single number the accuracy-vs-avg-layers curve is built from.
        avg_depth = gates.sum(dim=-1).mean()
        aux = {"gates": gates, "avg_depth": avg_depth}
        return logits, loss, aux

    def num_params(self, exclude_embeddings=True):
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.pos_emb.weight.numel()
        return n
