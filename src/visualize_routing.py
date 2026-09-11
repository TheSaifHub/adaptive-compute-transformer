"""Token-level routing visualization: which tokens a trained adaptive model
actually processes deeply vs. barely touches, on real sentences.

Needs no new training -- AdaptiveTransformer.forward() already returns
per-layer, per-token gates in aux["gates"] on every call. This module just
loads already-trained checkpoints, runs inference, and renders those gates
as colored text, comparing a shallow-budget and a deep-budget checkpoint
side by side on the same sentences.
"""
import os
import json
from dataclasses import fields

import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.colors import to_hex

from config import AdaptiveConfig
from model import AdaptiveTransformer

EXPERIMENTS_DIR = "experiments"
RESULTS_DIR = "results"
COLORMAP = colormaps["YlOrRd"]

HAND_WRITTEN_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "The mitochondria is the powerhouse of the cell, according to quantum photosynthesis apparatus.",
    "She said that she would probably go to the store if it stopped raining.",
]


def load_run(run_name, experiments_dir=EXPERIMENTS_DIR, device="cpu"):
    """Reconstructs the AdaptiveConfig a run was trained with from its
    saved config.json, builds the model, and loads its checkpoint weights."""
    run_dir = os.path.join(experiments_dir, run_name)
    with open(os.path.join(run_dir, "config.json")) as f:
        raw_cfg = json.load(f)

    # config.json also carries git_commit/resolved_device, added at save
    # time (see train_adaptive.py) -- not AdaptiveConfig fields, so filter
    # to the dataclass's actual field names before reconstructing it.
    valid_fields = {f.name for f in fields(AdaptiveConfig)}
    cfg_kwargs = {k: v for k, v in raw_cfg.items() if k in valid_fields}
    cfg = AdaptiveConfig(**cfg_kwargs)

    model = AdaptiveTransformer(cfg).to(device)
    ckpt = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location=device, weights_only=False)
    state_dict = ckpt["model"]
    # torch.compile-wrapped models (the real Kaggle runs used compile=True)
    # can save state_dict keys prefixed with "_orig_mod." -- strip it if
    # present so it loads onto the plain (uncompiled) model built here.
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


def route_text(model, enc, text, device="cpu"):
    """Tokenizes text and returns [(token_text, effective_depth), ...] --
    effective_depth is a token's gates summed across layers (same quantity
    as AdaptiveTransformer's avg_depth, just per-token instead of averaged
    over the batch), continuous in [0, n_layer]."""
    ids = enc.encode_ordinary(text)
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        _, _, aux = model(idx)
    depths = aux["gates"].sum(dim=-1).squeeze(0)
    tokens = [enc.decode([tid]) for tid in ids]
    return list(zip(tokens, depths.tolist()))


def get_real_snippet(enc, cache_dir=None, start=40000, n_tokens=25):
    """Pulls a short, real WikiText-103 validation snippet, tokenizing (and
    caching) just the validation split -- not the full ~103M-token train
    split prepare_dataset() would also fetch. `start` was chosen by
    inspecting a few offsets and picking one that reads as a clean,
    coherent clause with none of WikiText-103-raw's "@-@" compound-word
    artifacts (which would be awkward to render per-token-colored)."""
    import data as data_module

    cache_dir = cache_dir or data_module.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    val_path = os.path.join(cache_dir, "val.bin")
    if not os.path.exists(val_path):
        ids = data_module._tokenize_split("validation", "wikitext-103-raw-v1", enc)
        ids.tofile(val_path)

    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    snippet_ids = data[start:start + n_tokens].astype(np.int64).tolist()
    return enc.decode(snippet_ids)


def depth_to_color(depth, max_depth):
    """Maps an effective depth in [0, max_depth] to a CSS hex color via a
    perceptual sequential colormap (light/cheap -> dark red/expensive)."""
    t = max(0.0, min(1.0, depth / max_depth))
    return to_hex(COLORMAP(t))


def render_html(run_results, sentences_by_label, max_depth, out_path):
    """run_results: {run_name: {label: [(token, depth), ...]}}
    Renders one HTML page with one column per run, sharing the SAME
    max_depth color scale across all columns -- the whole point of the
    comparison is that a shallow-budget run's tokens sit visibly lighter
    across the board, which an independently-normalized-per-run scale
    would hide."""
    parts = ["""<!doctype html><html><head><meta charset="utf-8">
<title>Adaptive routing visualization</title>
<style>
body { font-family: -apple-system, Segoe UI, sans-serif; margin: 2rem; background: #fafafa; color: #111; }
h1 { font-size: 1.3rem; }
.columns { display: flex; gap: 2rem; flex-wrap: wrap; }
.column { flex: 1; min-width: 320px; background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.25rem; }
.column h2 { font-size: 1rem; margin-top: 0; }
.sentence { line-height: 2.6; margin: 1rem 0; font-size: 1.05rem; }
.tok { padding: 0.1em 0.05em; border-radius: 3px; }
.legend { margin: 1.5rem 0; }
.legend-bar { height: 18px; width: 300px; border-radius: 3px;
  background: linear-gradient(to right, __GRADIENT_STOPS__); }
.legend-labels { display: flex; justify-content: space-between; width: 300px; font-size: 0.8rem; color: #555; }
.note { color: #555; font-size: 0.85rem; max-width: 700px; }
</style></head><body>
<h1>Adaptive-Compute Transformer &mdash; token-level routing</h1>
<p class="note">Color = effective depth used by that token (gate values summed across all 12 layers), continuous
0&ndash;__MAXDEPTH__, same scale in every column below. This is Phase 1 <em>soft</em> routing: no token is
literally skipped yet, every token still passes through every layer, but a lighter color means that layer's
contribution was mostly gated out for that token. Hover a token to see its exact depth.</p>
"""]

    gradient_stops = ", ".join(depth_to_color(d, max_depth) for d in np.linspace(0, max_depth, 6))
    parts[0] = parts[0].replace("__GRADIENT_STOPS__", gradient_stops).replace("__MAXDEPTH__", f"{max_depth:g}")

    parts.append('<div class="legend"><div class="legend-bar"></div>'
                  f'<div class="legend-labels"><span>0 layers</span><span>{max_depth:g} layers</span></div></div>')

    parts.append('<div class="columns">')
    for run_name, results in run_results.items():
        parts.append(f'<div class="column"><h2>{run_name}</h2>')
        for label, pairs in results.items():
            parts.append(f'<div class="sentence"><strong>{label}:</strong><br>')
            for token, depth in pairs:
                color = depth_to_color(depth, max_depth)
                safe_token = (token.replace("&", "&amp;").replace("<", "&lt;")
                              .replace(">", "&gt;"))
                parts.append(
                    f'<span class="tok" style="background:{color}" title="depth={depth:.2f}">{safe_token}</span>'
                )
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</div></body></html>")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    return out_path


if __name__ == "__main__":
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    run_names = ["adaptive_target2", "adaptive_target9"]

    sentences = {f"Hand-written {i+1}": s for i, s in enumerate(HAND_WRITTEN_SENTENCES)}
    sentences["Real WikiText-103 validation snippet"] = get_real_snippet(enc)

    run_results = {}
    max_depth = 0
    for run_name in run_names:
        model, cfg = load_run(run_name)
        max_depth = max(max_depth, cfg.n_layer)
        print(f"loaded {run_name}: target_avg_layers={cfg.target_avg_layers}")
        run_results[run_name] = {
            label: route_text(model, enc, text) for label, text in sentences.items()
        }

    out_path = os.path.join(RESULTS_DIR, "routing_visualization.html")
    render_html(run_results, sentences, max_depth, out_path)
    print(f"saved {out_path}")
