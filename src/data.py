"""WikiText-103 loading + tokenization + batching.

Uses HF `datasets` only for corpus I/O (streaming/download of raw text).
Tokenization (tiktoken) and batching are hand-written — no HF Trainer,
no HF data collators.
"""
import os
import numpy as np
import tiktoken

CACHE_DIR = os.path.join(os.getcwd(), "data", "tokenized")


def _tokenize_split(split, dataset_config, enc):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", dataset_config, split=split)
    eot = enc.eot_token
    ids = []
    for row in ds:
        text = row["text"]
        if not text.strip():
            continue
        ids.extend(enc.encode_ordinary(text))
        ids.append(eot)
    return np.array(ids, dtype=np.uint16)


def prepare_dataset(dataset_config="wikitext-103-raw-v1", cache_dir=CACHE_DIR):
    """Tokenizes train/validation/test splits once and caches them as
    .bin files of uint16 token ids. Safe to call every run — skips work
    if the cache already exists."""
    os.makedirs(cache_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")

    paths = {}
    for split, hf_split in [("train", "train"), ("val", "validation"), ("test", "test")]:
        path = os.path.join(cache_dir, f"{split}.bin")
        paths[split] = path
        if os.path.exists(path):
            continue
        ids = _tokenize_split(hf_split, dataset_config, enc)
        ids.tofile(path)
    return paths


def get_batch(split, cfg, cache_dir=CACHE_DIR, device="cpu"):
    """Memory-maps the cached token file and samples a random batch of
    (input, target) sequences. Re-opened per call (cheap, avoids stale
    mmap handles) — standard nanoGPT-style pattern."""
    import torch

    path = os.path.join(cache_dir, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")

    ix = np.random.randint(0, len(data) - cfg.block_size - 1, size=cfg.batch_size)
    x = torch.stack(
        [torch.from_numpy(data[i:i + cfg.block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1:i + 1 + cfg.block_size].astype(np.int64)) for i in ix]
    )
    if device != "cpu":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y
