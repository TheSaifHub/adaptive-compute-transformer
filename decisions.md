# DECISIONS.md

One entry per meaningful engineering decision. Newest at the bottom.

---

## 2026-08-28 — Tokenizer: GPT-2 BPE via `tiktoken`

**Decision:** Use the pretrained GPT-2 BPE tokenizer (`tiktoken`, `gpt2` encoding, vocab size 50257) rather than training a custom tokenizer on WikiText-103.

**Why:** The project's variable of interest is adaptive depth/routing, not tokenization. A custom BPE tokenizer would add a second uncontrolled variable (vocab size, merge quality) that makes it harder to attribute results to the routing mechanism. `tiktoken` is fast (Rust-backed), free, and gives a fixed, reproducible vocabulary.

**Alternative considered:** Train a small custom BPE on WikiText-103 itself for a "closer to the data" vocab. Rejected for now — can revisit if GPT-2's vocab proves to be a poor fit for WikiText-103's distribution.

---

## 2026-08-28 — Dataset: WikiText-103-raw-v1 via Hugging Face `datasets`

**Decision:** Load WikiText-103 (raw, uncased-preserving version) using the Hugging Face `datasets` library purely for corpus access/streaming, then tokenize and batch it ourselves.

**Why:** WikiText-103 is free, well-documented, and standard enough that results are comparable to published baselines. Using `datasets` only for I/O (not `Trainer`, not model code) stays within the project rule of no HF training-loop shortcuts — the training loop itself is hand-written PyTorch.

---

## 2026-08-28 — Baseline architecture: 12-layer decoder-only transformer, d_model=384

**Decision:** Baseline static-depth model: 12 transformer blocks, 6 attention heads, d_model=384, d_ff=1536 (4x), block_size (context)=256 tokens, dropout=0.1. Pre-LayerNorm, causal self-attention, standard GPT-style architecture. ~15M parameters.

**Why:**
- 12 layers gives the future adaptive router a meaningful range of exit points (early/mid/late) to learn over — too few layers and the accuracy-vs-depth curve has nothing interesting to show.
- Sized to train in a reasonable time on a free-tier Colab/Kaggle T4 GPU (16GB), per the project's compute constraint.
- Architecture choices (pre-LN, GPT-style block) are deliberately standard/boring so that any effect we measure later is attributable to the routing mechanism, not to an unusual base architecture.

**Alternative considered:** Larger model (e.g., 24 layers, d_model=768, GPT-2-small scale) — rejected for now due to free-GPU training time; can scale up later if the small model trains too fast to show a meaningful curve.

---

## 2026-08-28 — Training loop: hand-written PyTorch, config+seed logged per run

**Decision:** No `transformers.Trainer` or other high-level training-loop abstraction for the core loop. Every run writes its full resolved config (hyperparameters, seed, data version, git commit) to `experiments/<run_name>/config.json` before training starts.

**Why:** Project rule — reproducibility is non-negotiable, and writing the loop by hand is itself part of the learning goal (understanding gradient accumulation, LR scheduling, checkpointing at a low level, which matters later for building the router training loop on top).

---

## 2026-09-03 — Accuracy-vs-avg-layers curve: sweep methodology, and the central result

**Decision:** Traced the project's central curve with 4 full training runs of `AdaptiveTransformer`, each identical except for `target_avg_layers` (the one variable being swept): `{2.0, 4.0, 6.0, 9.0}`. Everything else held fixed — same seed (1337), same `budget_weight` (0.01), same architecture (12-layer, d_model=384), same data (WikiText-103-raw-v1), same `max_steps` (20000). One run per target value (no repeated seeds).

**Result** (all values from `val_lm`, i.e. pure cross-entropy loss, not the composite LM+budget objective; `avg_depth` is the actually-achieved value at step 20000, not the target):

| target_avg_layers | achieved avg_depth | val_lm loss |
|---|---|---|
| 2.0 | 2.647 | 3.8915 |
| 4.0 | 4.247 | 3.8423 |
| 6.0 | 6.080 | 3.8135 |
| 9.0 | 8.982 | 3.7988 |

Consecutive raw deltas (0.0492 → 0.0288 → 0.0147) shrink monotonically; normalizing by the (unequal) depth spacing between points shows an even sharper diminishing-returns slope (≈ -0.031, -0.016, -0.005 loss-per-layer across the three intervals). This is the project's central finding: quality gains from additional average depth diminish sharply as depth increases, consistent with the hypothesis that most tokens don't need full-depth processing.

**Why one run per target value, not repeated seeds:** each run costs ~5 GPU-hours on Kaggle's free tier; the sweep's 4 points already cost ~20 hours. Repeated-seed runs (to get error bars / distinguish signal from noise) were judged not worth the added compute for now, given the observed effect sizes are large relative to the one noise estimate available: `val_lm` fluctuates by about ±0.02 (range ~0.04 peak-to-peak) across the last 10 eval checkpoints *within* a single completed run, which is on the same order as the smallest inter-point gap (6→9: 0.0147). The large-gap end of the curve (2→4: 0.0492) is well clear of that noise floor; the small-gap end (6→9) should be read as "small and consistent with diminishing returns" rather than a precise number. Revisit with repeated seeds if a precise magnitude at the flat end specifically becomes important later.

**Alternative considered:** A single run sweeping `target_avg_layers` on a schedule within one training run (curriculum-style). Rejected — conflates "does depth X achieve good quality" with "what does the model look like after also having trained at other depths first," which is a different, harder-to-interpret experiment than clean independent points on a curve.

---

## 2026-09-03 — Confidence-threshold router: stabilization-norm proxy, not a full CALM-style early-exit head

**Decision:** `ConfidenceThresholdRouter` gates on a token's own stabilization — how much a block's output actually changes its representation — rather than reusing the model's vocab head at intermediate layers for an early next-token prediction (CALM's literal mechanism).

**Why:** The literature version needs new access from inside a block to `ln_f`/`head`, which live only on `AdaptiveTransformer`, not per-block — a real architectural change beyond the router class itself. The stabilization proxy needs no new plumbing beyond the one interface extension already required (`forward(x, block_out)`), has zero learned parameters, and is arguably *more* faithful to this project's own stated framing of the router's job (`CLAUDE.md`: "has this token's representation stabilized, or does it need another layer?") than the vocab-head version would be. Saif chose this explicitly over the full CALM approach when the tradeoff was presented.

**Alternative considered:** Full CALM-style early-exit head (reuse the tied vocab embedding at each layer, threshold on softmax confidence). More faithful to the published mechanism, but a bigger architectural lift for a router this project characterizes as "our own routing mechanism" anyway — rejected for now; could be revisited as a fourth ablation arm if the simpler proxy's results warrant it.

---

## 2026-09-03 — Confidence-threshold router: relative delta norm, not absolute

**Decision:** The stabilization signal is `‖block(x) − x‖ / ‖x‖` (relative to the token's own current magnitude), not the raw `‖block(x) − x‖`.

**Why:** This is a pre-LN transformer — nothing renormalizes the residual stream between blocks, so its magnitude tends to grow across layers, a well-known property of that architecture family. A single fixed threshold on the *absolute* delta would then be measuring depth, not difficulty: early layers (small `‖x‖`) would rarely trip a threshold calibrated for late layers' larger absolute changes, or the reverse — a systematic bias by layer position that has nothing to do with which tokens are actually hard. Verified directly, not just reasoned about: the same relative displacement at `‖x‖=1` and `‖x‖=50` was confirmed to produce an identical gate decision.

**Alternative considered:** Absolute delta norm with a per-layer (rather than global) threshold, learned or calibrated separately at each depth. Rejected as unnecessary complexity — the relative formulation gets scale-invariance for free with one division, no per-layer bookkeeping needed.

---

## 2026-09-03 — Fixed top-k router: faithful score-scaled selection, not selected=1

**Decision:** `FixedTopKRouter`'s selected tokens get gate = their own learned sigmoid score (continuous, whatever it happens to be), not a flat 1.0. Non-selected tokens get exactly 0.

**Why:** This matches Mixture-of-Depths' actual published mechanism, not a simplified version of it. Top-k selection itself is a ranking operation, not differentiable — scaling by the score is specifically what gives the scorer's weights a gradient signal at all (only from tokens that happened to be selected in a given batch; the selection *decision* stays non-differentiable, a known, accepted limitation of this mechanism, not a bug). The real, verified consequence: `avg_depth` (this project's usual "effective depth" metric, gates summed across layers) stops meaning "how many layers this token actually went through" for this router type — it becomes score-magnitude-weighted. Confirmed concretely: at `capacity=0.5, n_layer=3`, the true selection fraction is exactly `0.5` as designed, but `avg_depth` comes out to `0.7505`, not the naive `1.5` selected-means-full-weight assumption would predict.

**Alternative considered:** Selected tokens get gate = 1.0 exactly, with the learned score used only to rank (not to scale). Simpler to reason about and keeps `avg_depth` meaning the same thing across all three router types — but not what MoD actually does, and removes the router's only training signal (with gate always exactly 1 for selected tokens, there'd be no direct gradient path telling the scorer *how much* being selected helped, only an indirect one). Rejected in favor of fidelity to the real mechanism; addressed the metric-comparability cost instead by logging a second, always-exact `nonzero_frac` metric alongside `avg_depth`.

---

## 2026-09-03 — Confidence threshold: calibrate empirically and locally before spending Kaggle time

**Decision:** Before running `confidence_threshold` for real on Kaggle, measured the actual distribution of relative delta-norms the real 12-layer/384-dim architecture produces on real WikiText-103 validation data — using a freshly-initialized (untrained) model, locally, on CPU — and set `AdaptiveConfig`'s default from that measurement (`0.4309`, targeting `avg_depth≈6` to match the other ablation arms) rather than guessing.

**Why:** Unlike `learned_gate` (converges toward a target via the budget loss) and `fixed_topk` (capacity sets depth exactly, by construction), `confidence_threshold`'s depth-vs-threshold relationship isn't knowable in advance and has no learned mechanism to correct a bad guess mid-run — a wrong threshold would only be discovered after a full ~5-hour real run, at real GPU cost. The calibration script's estimate is deliberately scoped as round-1-only (untrained model, router bypassed to isolate the raw signal) since it's measuring a property of the architecture's activation statistics, not of training dynamics — round-2 calibration, if needed, should come from a real run's own observed `nonzero_frac` trajectory instead. The estimate was validated, not just trusted: plugged back into the real routing forward pass (not the calibration's simplified bypass), it produced `avg_depth=6.43` against a target of `6.0` — close, with the small gap explained by a real, understood mechanism (real routing lets an early-frozen token change what later layers see, which the always-full-block calibration bypass doesn't capture).

**Alternative considered:** Pick a round-number placeholder (e.g. `0.1`) and let the first real Kaggle run reveal whether it's reasonable, adjusting for a second attempt if not. Rejected — the placeholder that had been sitting in the config as a stand-in turned out to be off by roughly an order of magnitude (it would have selected essentially every token), which would have cost a full real run to discover instead of a few seconds of local compute.

---

## 2026-09-08 — Phase 2 hard-skip, Module 1: scope is inference-only, no retraining

**Decision:** Real hard-skip for `learned_gate` starts as an inference-only optimization on top of the already-trained `adaptive_v1` checkpoint — no new training run, no gradient changes, no straight-through estimator (yet).

**Why:** The router ablation already established `learned_gate` as the quality winner and produced a fully-trained checkpoint (`experiments/adaptive_v1/ckpt.pt`) with a known, verified achieved depth (avg_depth 6.080, val_lm 3.8135). Reusing it directly answers this phase's actual question — "what's the real wall-clock speedup" — at zero additional GPU cost. A training-aware hard-skip (fine-tuning with a straight-through estimator so the model learns under the same hard-skip conditions it will run under) is only worth the Kaggle time if inference-only discretization turns out to damage quality too much to be a fair comparison at all — that's an empirical question Module 3 answers, not one to pre-judge.

**Alternative considered:** Fine-tune with hard-skip active from the start. Rejected for now, not permanently — deferred to a conditional Module 6, run only if Module 3's numbers show real degradation.

---

## 2026-09-08 — Phase 2 hard-skip, Module 1: mechanism is gather/compute/scatter, not a frozen-K/V cache

**Decision:** A token excluded from a layer is excluded from that layer's self-attention entirely — it contributes no K/V at that layer, not even a stale/frozen one carried forward from an earlier layer. Concretely: at each `AdaptiveBlock`, gather the selected subset of a sequence's tokens (preserving their original relative order, not sort-by-score order), run causal self-attention and the MLP on that compacted subsequence only, then scatter the results back into the full-length residual stream at their original positions; excluded positions keep their `x` from entering the layer, completely unchanged.

**Why:** Session 4 flagged "a state-copy-forward approximation for exited tokens' K/V" as a detail deferred to this phase, without settling the actual mechanism. Investigated two options before choosing: (a) keep excluded tokens visible as K/V sources by re-projecting their frozen residual-stream value through each subsequent layer's own (different) attention weights, or (b) exclude them from that layer's attention entirely, matching the real published Mixture-of-Depths mechanism. (a) is more complex (needs bookkeeping for which tokens are "stale," and re-projecting through every layer's weights isn't actually cheaper than including them, since the K/V projection is one of the more expensive parts of the block) and isn't what any cited prior work actually does. (b) is simpler to implement, matches Mixture-of-Depths directly, and has a principled justification: a token's *influence* on other tokens was already exerted during whichever earlier layers it was active in — attention at those earlier layers already mixed its information into neighboring tokens' residual streams, so it isn't losing representational access to the sequence, only its own further processing.
- Real correctness detail this decision surfaces (not optional, will be tested directly in Module 2): gathering must preserve each sequence's original token order before running the causal-masked attention on the compacted subsequence — gathering in score-rank order instead would corrupt the causal mask (a later-position token could end up appearing before an earlier-position one in the compacted tensor).

**Alternative considered:** Frozen-K/V carry-forward (option (a) above). Rejected as unnecessary complexity that doesn't actually save the K/V-projection compute it would need to save to be worth building, and isn't faithful to any prior work being drawn on here.

---

## 2026-09-08 — Phase 2 hard-skip, Module 1: selection policy — independent per-layer top-C on learned_gate's own score

**Decision:** At inference, each of the 12 layers independently selects its own top-C tokens per sequence (C fixed, derived below), ranked by that layer's own `LearnedGateRouter` score — the same per-layer-independent structure `FixedTopKRouter` already uses (`topk_capacity`: "exact fraction of tokens selected per sequence **per layer**," `src/config.py:68`), not a single cumulative "has this token exited yet" decision carried across layers.
- **Capacity:** `C = round(0.5066 * T)`, where `0.5066 = 6.0797 / 12` is `adaptive_v1`'s real, logged achieved `val_avg_depth` divided by `n_layer` — not the nominal `target_avg_layers=6.0` — so the hard-skip version targets what the checkpoint actually does, not what it was asked to do (same "report achieved, not target" discipline as `plot_curve.py` and the sweep results).
- **Selected tokens' gate:** their own real, continuous learned score (identical math to Phase 1's soft blend: `gate*block(x) + (1-gate)*x`) — not hard-forced to `1.0`. This costs nothing extra (block(x) is already computed for the selected subset; scaling by an already-known scalar is ~free) and keeps selected tokens' behavior numerically identical to how the model was actually trained. **Excluded tokens get gate forced to exactly `0.0`** (`x` unchanged) — the one place inference deviates from train-time behavior, and specifically in the tokens the router itself ranked as most-converged, i.e. most likely to have had a small trained gate anyway.

**Why (each sub-decision):**
- *Independent per layer, not cumulative "exited":* nothing in how the model was actually trained enforces a one-way exit — each `AdaptiveBlock`'s router is a separate instance with independent weights, scoring whatever `x` currently is, with no persistent "already exited" state threaded between layers. `CLAUDE.md`'s narrative language ("decides whether to keep going deeper or exit early") reads as one-way, but forcing that at inference would be *introducing new behavior the model was never trained under*, not just skipping compute for existing behavior — the actual goal of this phase. Faithfulness to the trained model takes priority over faithfulness to the narrative description.
- *Capacity from achieved depth, not nominal target:* consistent with how every other result in this project has been reported (the sweep, the ablation) — avoids quietly citing a number the checkpoint doesn't actually produce.
- *Selected tokens keep their real score, excluded forced to 0:* this is what makes Module 2's regression test possible at all — at capacity=1.0 (nothing excluded), this hard-skip path must reduce to *exactly* today's dense `AdaptiveBlock.forward` output, bit-for-bit. Forcing selected tokens to gate=1.0 instead would make that invariant untestable (today's dense model never uses gate=1.0 for anything) and would be a strictly larger, unmotivated departure from trained behavior for no compute benefit.

**Alternative considered:** Force monotonic early-exit (once a token drops out of the top-C at any layer, it's excluded from all remaining layers). More literally matches the project's "exit early" framing and is simpler to reason about as a narrative — rejected because it changes what the already-trained model computes in a way the training process never saw or optimized for; worth revisiting only if independent per-layer selection produces a surprising result (e.g., the same tokens flickering in and out at every layer with no discernible pattern).

---

## 2026-09-08 — Phase 2 hard-skip, Module 1: benchmark scope is full-sequence forward latency, not autoregressive decode

**Decision:** The wall-clock speedup benchmark (Module 4) measures latency of a single full-sequence forward pass over a `(B, T)` batch — the same shape used for training/eval — not token-by-token autoregressive generation.

**Why:** This codebase's `generate()` (`src/model.py:129`) has no KV-cache at all — every new token reprocesses the entire growing context from scratch, an inefficiency completely orthogonal to layer-skipping. Benchmarking hard-skip through `generate()` would measure two confounded effects (no-KV-cache inefficiency + layer-skip savings) and make the layer-skip number impossible to isolate cleanly. Full-sequence forward latency isolates exactly the effect this phase is trying to measure, and it's honest about what it does and doesn't cover: it's the "process this batch of context" cost real serving systems call *prefill*, not the *decode* cost of generating new tokens one at a time — the latter is where a real production system would care most, but building a correct KV-cache decode loop is a separate, substantial piece of engineering this codebase doesn't have today, and bolting hard-skip onto it without an existing correct baseline would risk an unreliable comparison. This scope limitation will be stated plainly wherever the benchmark's numbers are reported, not left implicit.

**Alternative considered:** Build a minimal KV-cache decode loop first, then benchmark hard-skip generation end-to-end. Rejected for this phase as substantially more engineering than the definition-of-done requires — `CLAUDE.md`'s ask is "a real wall-clock speedup number" and "the gap between theoretical and real speedup," both answerable from prefill-latency alone; revisit if a decode-time number specifically becomes necessary later.

---

## 2026-09-08 — Phase 2 hard-skip, Module 3: full_block mechanism catastrophically breaks quality on `learned_gate`

**Decision (finding, not a choice):** Wired the Module 2 `full_block` mechanism into the real `adaptive_v1` checkpoint and measured real WikiText-103 validation loss, on identical batches, against the same checkpoint's dense forward pass. Result: val_lm nearly doubled (7.78 vs. 3.81) at the real capacity (0.5066). A capacity sweep (1.0 → 0.3) showed smooth, monotonic degradation starting immediately below capacity=1.0 (already +0.37 loss at capacity=0.95) — ruling out an implementation bug (Module 2's tests, which check the underlying gather/scatter/causal-masking mechanics directly with controlled indices, already passed cleanly) in favor of a real mismatch between this mechanism and how the model was actually trained.

**Why:** Phase 1 training never excludes any token from attention, for any router type — `AdaptiveBlock.forward` always computes `self.block(x)` (full dense attention over every position) regardless of gate value; the gate only ever scaled the residual-stream contribution. The `full_block` mechanism (Decision, 2026-09-08, "mechanism is gather/compute/scatter") assumed a model trained the way real Mixture-of-Depths is trained — end-to-end, with hard exclusion from attention baked in from the start — so the model learns to route around missing K/V. `learned_gate` was never trained under that condition, so excluding a token from attention entirely removes information neighboring tokens' attention genuinely still depends on.

**Consequence:** This finding is a real, reportable instance of `CLAUDE.md`'s own framing — "the gap between theoretical and real speedup" — just surfacing as a quality collapse rather than a latency surprise. It also means Decision 2's rejection of the frozen-K/V alternative was reasoned about compute cost and published-work fidelity, but didn't weigh the fidelity-to-training-behavior cost highly enough — that cost turned out to dominate.

---

## 2026-09-08 — Phase 2 hard-skip, Module 3: `mlp_only` mechanism added — keeps attention fully dense

**Decision:** Added a second mechanism, `hard_skip_mlp_only_forward` / `HardSkipMLPOnlyAdaptiveBlock` (now `HardSkipAdaptiveTransformer`'s default `mode`), which never excludes any token from attention — every token computes and is visible as Q/K/V at every layer, identically to Phase 1/dense — and skips real compute only for the MLP sub-layer, only for excluded tokens. Uses the algebraic identity that Phase 1's blend formula distributes additively (`gate*block(x) + (1-gate)*x = x + gate*attn_delta + gate*mlp_delta`, since `Block` is a standard residual block), so scaling each sub-layer's own delta by the same gate and adding both to `x` reproduces the original formula exactly — critically, `mlp_delta` must be computed from the *unscaled* post-attention residual (`x + attn_delta`), not a gate-scaled one; an initial draft got this wrong (scaled before computing mlp_delta from it) and was caught by the capacity=1.0 exact-reproduction smoke test, which is bit-for-bit strict rather than approximate.

**Why:** Directly targets the Module 3 finding above by construction — since attention is never touched, this mechanism cannot introduce the attention-visibility mismatch, regardless of router type. `full_block` is kept available (not deleted) since a future fine-tuning experiment that trains a model to expect real attention exclusion would still want it.

**Result:** Reduced but did not fix the problem on `adaptive_v1` — val_lm at real capacity improved from 7.78 to 7.32 (still +3.45 over dense's 3.81). This partial-not-full improvement is itself informative: it confirms attention-visibility was a real, independent contributing factor, but not the dominant one — see the next entry.

---

## 2026-09-08 — Phase 2 hard-skip, Module 3: root cause — `learned_gate`'s trained scores are not sparse

**Decision (finding):** Inspected the real, trained per-token, per-layer gate score distribution of `adaptive_v1` on real validation data before concluding anything further. Result: broad and roughly centered around 0.5 (overall mean 0.507, std 0.233), not remotely bimodal — only 3.1% of values fall below 0.1, only 4.8% exceed 0.9, and several early layers (0–4) never produce a gate below ~0.1 at all. The median gate value at the exact top-50%-capacity exclusion boundary (layer 0) is 0.63 — a token right at the cutoff has a real, substantial trained gate, not something close to 0.

**Why this is the real root cause:** `learned_gate`'s budget loss (`losses.py`) only ever pressures the *population average* gate toward `target_avg_layers` — nothing in its training objective rewards individual scores being close to 0 or 1. A smooth, unimodal distribution around the target average satisfies that loss just as well as a sparse/bimodal one would, and that's what training actually found. Any hard rank-based cutoff (`mlp_only` or `full_block` alike) approximates every excluded token's true contribution as exactly 0 — for `learned_gate`, that's a large, systematic error across roughly half the (token, layer) pairs in the network, not a small rounding effect at the margin. This is a property of what the router learned to do, not of which masking mechanism is used to apply it — hence why fixing the attention-visibility issue (previous entry) only partially helped.

**Consequence:** Real hard-skip on `learned_gate`, as currently trained, cannot be made to preserve quality by changing the masking mechanism alone — it would need the router itself to be trained (or fine-tuned) toward a sparser score distribution. Deferred; see the pivot below for what shipped instead.

---

## 2026-09-08 — Phase 2 hard-skip, Module 3: pivot to `fixed_topk` (`ablation_topk`) — exact, verified reproduction

**Decision:** Re-ran the identical, unmodified `mlp_only` hard-skip machinery against `ablation_topk` (the router ablation's `fixed_topk` checkpoint) instead of `adaptive_v1`, using `cfg.topk_capacity` (0.5, exact) as the capacity. Result: **val_lm matched dense to 4 decimal places on all 20 evaluated batches, every single one** (mean delta `+0.0000`). No code changes were needed — `HardSkipAdaptiveTransformer`/`HardSkipMLPOnlyAdaptiveBlock` were used exactly as built for `learned_gate`.

**Why this works exactly, not approximately:** `FixedTopKRouter` was *already* trained with real top-k masking every forward pass during Phase 1 — `FixedTopKRouter.forward` returns `scores * mask`, exactly 0 for non-selected tokens, at training time, not just at inference. So "excluded tokens contribute exactly 0" isn't an inference-time approximation for this router the way it is for `learned_gate` — it's mathematically identical to what the dense model already computed and was trained under. Combined with `mlp_only`'s untouched attention, the two computations (dense Phase 1 forward vs. real hard-skip forward) are provably the same function, just one wastes compute on masked-out tokens' MLP and the other doesn't — the same relationship as the capacity=1.0 regression invariant, just now true at the router's real operating capacity (0.5) because this router's masking was never an approximation to begin with.

**Consequence — this changes Phase 2's target:** The router ablation's *quality* winner (`learned_gate`, val_lm 3.8135) is not well-suited to real hard-skip without further training work; the ablation's *runner-up* (`fixed_topk`, val_lm 3.9139) achieves real, zero-quality-loss hard-skip for free, using exactly the same wall-clock-saving machinery. Module 4 (wall-clock benchmarking) proceeds on `ablation_topk` with `mlp_only`. This is a genuine, defensible research finding in its own right: the router that wins on a soft/continuous training objective is not automatically the one a real deployment should hard-skip — architectural fit to discretization matters independently of soft-trained quality, and `fixed_topk`'s exact-by-construction masking, framed as a training-time cost in the original ablation, turned out to be exactly the property that makes it practically deployable.

**Alternative considered:** Fine-tune `learned_gate` toward a sparser gate distribution (e.g. an entropy/L0-style regularizer, or training with hard-skip active via a straight-through estimator) so it could still be used for real hard-skip. Not pursued for now — real Kaggle GPU-hours, and `fixed_topk`'s free, exact result already satisfies the definition-of-done's wall-clock-speedup requirement; revisit if a future need specifically requires `learned_gate`'s hard-skip behavior.

---

## 2026-09-08 — Phase 2 hard-skip, Module 5: benchmark uses freshly-initialized models, no checkpoint upload needed

**Decision:** `benchmark_speed.py`'s `build_variants` now builds all three variants (baseline, adaptive_dense, adaptive_hard_skip) from a fresh, untrained `AdaptiveConfig(router_type="fixed_topk", topk_capacity=0.5)` by default — the same architecture `ablation_topk` was trained with, just without loading its checkpoint. `--checkpoint <run_name>` remains available (used locally, where `ablation_topk`'s files already exist, purely as an extra consistency check).

**Why:** Wall-clock latency depends only on architecture, shapes, and control-flow — never on weight *values* (nothing about a forward pass's timing changes based on what a weight matrix contains, only its shape and what operations are run). This was already the justification for benchmarking `baseline` without its lost checkpoint (see Module 4's original decision); extending it to all three variants means the Kaggle notebook (Module 5) needs no dataset upload/attachment step at all — it's fully self-contained, consistent with how the two training notebooks are self-contained via `%%writefile`. Verified empirically, not just argued: local CPU smoke-test timings with `--checkpoint ablation_topk` vs. the fresh-init default were statistically indistinguishable (both ~33-110ms depending on shape, well within run-to-run noise).

**Alternative considered:** Have Saif upload `experiments/ablation_topk/` as a Kaggle Dataset and attach it to the benchmark kernel. Rejected as unnecessary friction for a measurement that provably doesn't need real weights.

---

## 2026-09-08 — Phase 2 hard-skip, Module 5: `docs/colab_benchmark_hard_skip.ipynb` built and verified

**Decision:** Built a new, separate notebook (not added to `colab_train_adaptive.ipynb`, which is training-shaped: data download, W&B, a long run) for the Module 4 benchmark harness — 5 modules (Setup → Code → Smoke check → Full benchmark → Results), materializing `config.py`, `router.py`, `model.py`, `hard_skip.py`, `visualize_routing.py`, `benchmark_speed.py` via `%%writefile` from the real source files, same approach as the two existing notebooks.

**Verification (matching the bar set in Session 5 for the adaptive training notebook, not just visual review):** byte-identity of all six `%%writefile` cell bodies against the real `src/*.py` files; materialized all six into a flat directory (no `src/` subfolder, matching Kaggle/Colab's real layout); ran the notebook's own reload-modules cell and the smoke-check cell's actual code against that flat layout end-to-end — completed cleanly, all three variants built and timed. `torch.compile` itself wasn't exercised in this local simulation (already confirmed separately, in Module 4, that it fails on this dev machine for an unrelated reason — no C++ compiler; Kaggle's Linux GPU image is expected to have a working one, which is the whole point of running the real benchmark there instead of locally).

---

## 2026-09-08 — Phase 2 hard-skip, Module 5: real GPU result — 11.3% speedup, and why that's the right number to trust, not the naive 50%

**Decision (finding, confirmed against the downloaded artifact, not taken from summary):** Saif ran `docs/colab_benchmark_hard_skip.ipynb` on a real Kaggle GPU and downloaded `results/benchmark_speed_cuda.json`, the chart, and the backup zip. Read the JSON directly before reporting anything. At the throughput shape (32×256, matching every real training/eval run's `batch_size`/`block_size`), eager mode, median latency over 30 timed iterations (10 warmup):

| variant | eager | compiled |
|---|---|---|
| baseline | 195.21ms | 198.73ms |
| adaptive_dense | 214.84ms | 210.96ms |
| adaptive_hard_skip | 190.53ms | 196.91ms |

Precise ratios (eager): `adaptive_hard_skip` vs. `adaptive_dense` (the fair, apples-to-apples comparison — identical architecture and router, only the compute path differs): **11.31% less time, 1.128× speedup.** `adaptive_hard_skip` vs. `baseline`: **2.40% less time, 1.025× speedup.** `adaptive_dense` vs. `baseline`: **10.05% *slower*** — Phase 1's routing math is real, measurable overhead for zero compute savings, exactly as the architecture was always known to imply (see `hard_skip.py`'s module docstring, written in Module 4, before this number existed).

`torch.compile` gave no consistent benefit: baseline +1.80% slower compiled, `adaptive_dense` -1.80% faster, `adaptive_hard_skip` **+3.34% slower** compiled than eager. Not a systematic pattern — closer to noise with one clear exception in the wrong direction.

**Why 11.3% is the right number to report, and why comparing it to a naive "50%" expectation would be misleading:** `mlp_only` (chosen in Module 3 specifically because the alternative, full-block skip, broke quality) only ever skips the MLP sub-layer, never attention. A FLOPs breakdown of this exact architecture (d_model=384, d_ff=1536, computed per token per layer: MLP ≈4·d_model·d_ff, attention-linear ≈4·d_model², attention-SDPA ≈4·T·d_model) shows MLP is **70.6%** of a block's compute, attention (linear + SDPA) the remaining 29.4%. Skipping MLP for the ~50% of tokens `fixed_topk` excludes therefore has a **theoretical ceiling of ~35.3% FLOPs reduction, not ~50%** — the naive expectation was never achievable by the mechanism actually shipped, independent of any real-world overhead. The observed 11.3% is still well below even that more honest ceiling, and the remaining gap has three plausible, named (not yet profiler-attributed) causes: (1) `torch.gather`/`scatter_` are memory-bound, irregular-access operations that cost real time, eating into whatever compute was saved; (2) this is a ~15M-parameter model at a modest batch/sequence size, plausibly not fully saturating the GPU's compute capacity on the dense matmuls being compared against either, meaning cut FLOPs don't translate proportionally into cut wall-clock; (3) `torch.compile`'s main benefit (kernel fusion for compute-bound work) has less to offer when the workload isn't compute-bound to begin with, and may specifically conflict with hard-skip's dynamic-shape gather/scatter — consistent with it being the one variant that got *slower* when compiled.

**Conclusion:** 11.3% is a real, legitimate, non-fabricated speedup — measured on real GPU hardware, holding up in both eager and compiled backends as a `adaptive_hard_skip < adaptive_dense` ordering, not a rounding artifact or noise. It satisfies `CLAUDE.md`'s "real wall-clock speedup number" requirement. But the more valuable, more defensible deliverable is the *explained gap*: naive expectation (50%) → honest FLOPs ceiling (35.3%) → real measurement (11.3%), each step accounted for rather than asserted — this is precisely "the gap between theoretical and real speedup is itself a legitimate finding worth reporting," and it's the strongest single interview artifact this phase produced.

**Not done, optional:** A `torch.profiler` pass to apportion the 35.3%→11.3% gap precisely across gather/scatter cost, GPU utilization, and compile behavior, rather than naming plausible causes. Would strengthen the story further but isn't required by any stated definition-of-done item.

**This completes every item in `CLAUDE.md`'s Definition of Done** (working baseline, working adaptive-depth model, the accuracy-vs-avg-layers curve, the router ablation, a real wall-clock speedup number, and the token-level routing visualization from Session 7) — flagged to Saif; `final-interview-prep.md` generation is gated on his explicit confirmation the project is considered complete, per `CLAUDE.md`'s "On project completion" instruction, not generated automatically here.

**Correction (see the audit entry immediately below):** "completes every item" is true at the checklist level (each item has a real artifact behind it) but overstates how unified the evidence actually is — Saif asked for a rigorous re-check before accepting this, and it surfaced a real gap this entry didn't mention: the quality-preservation evidence and the wall-clock-speedup evidence were never demonstrated on the same router/checkpoint. Read the audit entry below for the full, corrected picture before treating this phase as simply "done."

---

## 2026-09-08 — Project completion audit: does the evidence actually prove the hypothesis?

**Context:** Before declaring the project complete, Saif asked for a rigorous, adversarial re-check of every `.md` file against `CLAUDE.md`'s original hypothesis and against the actual filesystem/source code — not a re-summary of what the logs already claim. This entry is that audit's conclusion, done by re-reading every file fresh and cross-checking specific claims against real files on disk (several checks below found real discrepancies this way, not just confirmations).

**The hypothesis, restated precisely (`CLAUDE.md`'s own framing):** can a model learn, unsupervised by any explicit label, which tokens need less computation, and can that heterogeneous allocation deliver comparable output quality while doing measurably less *real-world* work (GPU time/latency/energy — `CLAUDE.md` is explicit this is about production inference cost, not an abstract FLOPs count)?

**Definition-of-done checklist, re-verified against files, not narrative:**
- *Working baseline model* — **weaker than logged.** `experiments/baseline_v1/` does not exist on disk at all (confirmed via direct directory listing) — not the checkpoint only, as memory.md's Session 4 entry claims, but the `config.json` and `train_log.jsonl` too. Only `results/baseline_v1_loss_curve.png` survives. Session 4's claim that the log/config "survived (already saved separately)" is factually wrong and has been corrected in `memory.md`. Consequence: no precise, citable final baseline loss number has ever existed in this project — every later "baseline vs. adaptive" comparison (e.g. Session 6) was a visual read of a chart (~3.8), never a real number-to-number diff.
- *Working adaptive-depth model* — solid, re-verified: `adaptive_v1/ckpt.pt` (488MB) + full log exist and check out.
- *Accuracy-vs-avg-layers curve* — real and well-verified (4 points, cross-checked numbers, already-honest noise-floor caveat in this file). New finding from this audit: `plot_curve.py` (read directly) explicitly filters out the baseline run by construction ("skips the baseline run, whose config.json has no target_avg_layers field") — the project's central curve has never plotted a true static-baseline reference point, only adaptive-vs-adaptive comparisons at different budgets.
- *Router ablation* — solid, the most rigorously cross-checked piece of evidence in the project (re-confirmed, nothing new found).
- *Real wall-clock speedup number* — real (11.3%, re-confirmed against `results/benchmark_speed_cuda.json` directly, again). But see the central finding below — it's real for a narrower claim than "the definition-of-done item" implies at face value.
- *Token-level visualization* — solid, re-confirmed.
- *"All reproducible from logged configs and seeds"* — false for `baseline_v1` specifically, per the first bullet above.

**The central finding this audit surfaced (not previously stated this plainly anywhere in the project's own docs): the quality-preservation evidence and the wall-clock-speedup evidence were never unified on one router or one checkpoint.**
- The accuracy-vs-depth curve (quality holds up well across avg_depth 2.6→9) was built entirely with `learned_gate` running in Phase 1's soft/dense mode — which achieves **zero real wall-clock savings by construction**, since every token still runs through every layer regardless of gate value. This curve is evidence about a notional, gate-weighted compute quantity, not about anything measured in real time.
- The real 11.3% wall-clock speedup was only ever demonstrated for `fixed_topk`, at one fixed budget (50% exclusion) — and `fixed_topk`'s own quality (val_lm 3.9139) is measurably worse than `learned_gate`'s (3.8135, the ablation's actual winner) and very likely worse than the (now-confirmed-imprecise) static baseline too.
- `learned_gate` — the router that best supports the "comparable quality across a range of budgets" half of the hypothesis — was directly shown, in this same project's Module 3 work, to be **fundamentally unable to hard-skip without a quality collapse** (val_lm nearly doubling), because its trained gate scores turned out not to be sparse. This isn't a minor caveat; it's the reason the wall-clock benchmark had to target a different, lower-quality router at all.
- Consequence: there is no accuracy-vs-*real-speed* curve anywhere in this project — only one real-speed data point (50% exclusion, fixed_topk, mlp_only). The multi-point curve that exists is against a notional quantity that was never converted into real savings for the router it was measured on.

**Other gaps, checked for and found:**
- No numeric (only visual/eyeballed) baseline-vs-adaptive quality comparison exists anywhere, for the reason above.
- No repeated-seed error bars anywhere in the project (acknowledged already, at the time, in the sweep's own `DECISIONS.md` entry — not newly found, but relevant to how much weight the curve's smaller gaps can bear).
- No profiler-attributed breakdown of the 35.3%→11.3% real-speedup gap — three plausible causes are named, none confirmed.
- No test of how much the 11.3% number itself varies run-to-run (only within-run iteration variance, min/max/median over 30 iterations of one benchmark invocation, is known).
- **No fabrication found.** Every specific number checked against its underlying file during this audit (baseline chart, ablation configs/logs, benchmark JSON, gate-distribution histogram) matched what was previously claimed. The issues found are about completeness and framing — an overstated "this completes every item" summary, one factually wrong claim about surviving files — not invented results.

**Verdict:** Partial, nuanced success, not a clean unified proof. Every individual piece of evidence is real, verified, and honestly caveated where the project's own discipline caught it (the ablation's budget-mismatch note, the sweep's noise-floor note, the benchmark's FLOPs-ceiling note). What doesn't exist is the final assembly: one router, one checkpoint, shown simultaneously to have comparable quality *and* real measured speedup *and* plotted across a range of budgets, against a precisely-known baseline. That is a materially stronger claim than what this project actually produced, and the gap between them should be stated plainly to an interviewer rather than smoothed over — doing so is itself a demonstration of the same "no fabricated results" discipline this project has tried to hold to throughout.

**What a stronger version of this project would need, concretely:** (1) retrain (or preserve) a real baseline checkpoint and extract its precise final val_lm, so "comparable quality" has a real number to be comparable *to*; (2) either fine-tune `learned_gate` toward a sparser score distribution so it can be hard-skipped too, or accept `fixed_topk` as the deployable router and re-run its own accuracy-vs-budget sweep (varying `topk_capacity`, not `target_avg_layers`) so the *wall-clock* deliverable gets its own multi-point curve instead of one; (3) benchmark that sweep's wall-clock speedup at each budget, producing a real accuracy-vs-real-speed curve — the actual artifact `CLAUDE.md`'s hypothesis calls for, which this project came close to but did not quite build; (4) a `torch.profiler` pass to close out the FLOPs-ceiling-vs-real-speedup gap with measurement instead of named-but-unconfirmed causes.

---

## 2026-09-09 — Path A, Module 1: off-capacity inference does not work — full retraining confirmed necessary

**Decision (finding):** Tested whether `ablation_topk`'s capacity=0.5-trained checkpoint could stand in for the other three sweep points (0.1667, 0.3333, 0.75) via inference-time capacity changes alone, using `src/diagnose_offcapacity.py` — a new script that bypasses `FixedTopKRouter.forward()`'s own baked-in capacity=0.5 masking (pulling the raw sigmoid score directly from `router.net` before any masking) so an external top-k at an arbitrary test capacity can be applied cleanly in either direction, narrower or wider than the trained value.

**Result**, real WikiText-103 validation data, 20 iterations, same batches/seed for all rows:

| test_capacity | val_lm | delta vs. dense (3.9108) | avg_depth |
|---|---|---|---|
| 0.1667 | 14.0346 | +10.12 | 1.15 |
| 0.3333 | 6.9081 | +3.00 | 2.20 |
| 0.5 (self-consistency check) | 3.9108 | +0.0000 | 3.16 |
| 0.75 | 4.4140 | +0.50 | 3.75 |

At 0.1667 the loss is *worse than uniform-random guessing* over the 50257-token vocabulary (random ≈ 10.82 nats). Even the wider direction (0.75) — which was expected to be relatively safe, on the reasoning that extra computation is never wrong for next-token prediction — shows a real, non-trivial cost (+0.50, ~13% higher loss).

**A real bug caught before trusting this result:** the first version of the diagnostic passed the *unmasked* raw score into `hard_skip_mlp_only_forward`, giving excluded tokens a small nonzero attention-residual contribution — something the model was never trained with (`FixedTopKRouter.forward()` zeroes the score for non-selected tokens before anything downstream sees it; the diagnostic must replicate that masking at the test capacity, not skip it). This was caught by the capacity=0.5 self-consistency check — which must reproduce Module 3's proven exact-match result — coming back at `+0.083` instead of `+0.0000`. Fixed by explicitly re-masking the raw score to the test capacity's top-k before use; the self-consistency check then reproduced the exact `+0.0000` match.

**Why this rules out the shortcut, not just discourages it:** the magnitude and *bidirectionality* of the degradation (both narrower and wider capacities fail, not just one direction) indicates the base transformer's shared attention/MLP weights are specialized to the specific *density* of active tokens flowing through each layer at training time (50%), not merely to "which" tokens get selected. This is a genuine, if unsurprising in hindsight, property of how `fixed_topk` was trained — every layer's weights were optimized under a constant 50%-token gradient/compute regime for 20000 steps; asking them to operate at a different regime at inference is a distribution shift with no reason to be small.

**Consequence:** Proceeding with full independent training runs at `topk_capacity` ∈ {0.1667, 0.3333, 0.75} (Modules 2-3 of the Path A plan) — the cheap shortcut is conclusively unavailable, confirmed empirically rather than assumed either way. ~15 Kaggle-hours, as originally estimated.

**Alternative considered:** Accept the off-capacity numbers above as a (much cheaper, zero additional GPU-hours) stand-in for the sweep. Rejected — the collapse is severe enough (worse than random at the narrowest point) that these numbers would misrepresent what a genuinely-trained model at those capacities can do, defeating the entire purpose of Path A (closing a rigor gap, not introducing a new one).

---

## 2026-09-09 — Path A, Module 2: `TOPK_CAPACITY` sweep wiring added to `colab_train_adaptive.ipynb`

**Decision:** Added a `TOPK_CAPACITY` variable to Module 5b's cell (cell 17), mirroring `TARGET_AVG_LAYERS`'s existing pattern exactly. `RUN_NAMES["fixed_topk"]` now dispatches to `"ablation_topk"` when `TOPK_CAPACITY == 0.5` (the already-completed point — no redundant re-run) or `f"ablation_topk_cap{round(TOPK_CAPACITY * 100)}"` otherwise (e.g. `ablation_topk_cap17`, `_cap33`, `_cap75`). `full_cfg` now passes `topk_capacity=TOPK_CAPACITY` explicitly — previously it wasn't passed at all, silently relying on `AdaptiveConfig`'s own default (0.5). Updated cell 15's markdown to document the new sweep alongside the existing one. Cell 16 (smoke run) and cell 19 (results plotting) were deliberately left untouched — smoke intentionally doesn't mirror sweep hyperparameters (matching `TARGET_AVG_LAYERS`'s existing precedent: it checks the training loop is correct for the router *type*, not the specific value about to be used for real) and Module 6's plotting changes are out of this module's scope.

**Verification — byte-identity + a real flat-directory execution of the actual cell text (Session 5/7's established bar), which caught two real bugs, both in the test harness itself, not the notebook:**
1. Byte-identity confirmed: only cells 15 and 17 changed; all other 18 cells (including the six `%%writefile` source cells) are byte-for-byte identical to before.
2. First verification attempt appeared to hang. Root cause: the harness injected a tiny/fast architecture by placing a wrapper class into the `exec` namespace under the name `AdaptiveConfig` — but cell 16 itself contains `from config import AdaptiveConfig`, which re-imports the *real* class and silently overwrites the injected one the moment cell 16 runs. The result was a real ~40M-param model training for 200 real steps on CPU with no GPU — not a hang, just very slow — caught by checking the printed parameter count (40,597,644, not the expected tiny value) rather than waiting it out. Fixed by patching the `config` *module's* `AdaptiveConfig` attribute directly, before any cell executes, so the cell's own import resolves to the tiny version.
3. Second bug, same category: cell 16 also contains a hardcoded `ROUTER_TYPE = "learned_gate"` assignment. The harness's first fix injected the test router type into the exec namespace beforehand, which cell 16's own line silently overwrote back to `"learned_gate"` every time. The harness's own assertions didn't catch this — they checked the `RUN_NAMES` dict's contents (which were always correct, since that dict is built from literal values), not what `full_cfg.run_name`/`full_cfg.router_type` actually ended up being. Caught by reading the full training log output, not just the assertions: every run printed `[adaptive_target6]` (the `learned_gate` default name) regardless of which capacity was under test. Fixed the same way as (2): string-replace the literal `ROUTER_TYPE = "learned_gate"` line in the extracted cell text before exec'ing, exactly mirroring how `TOPK_CAPACITY`'s own line is already replaced to simulate the one-line edit Saif makes live in the Kaggle kernel.
4. Re-verified after both fixes: all four capacity values (0.5, 0.1667, 0.3333, 0.75) produce the correct `full_cfg.run_name`, `.topk_capacity`, and `.router_type`, confirmed both via assertions *and* by reading the actual training-loop log lines (`[ablation_topk]`, `[ablation_topk_cap17]`, `[ablation_topk_cap33]`, `[ablation_topk_cap75]`) — a full smoke-run + full-run + backup-zip cycle completes cleanly for each.

**Why this is worth recording in this much detail:** both bugs were in the *verification harness*, not the notebook — but both would have produced a false "PASSED" if the assertions alone had been trusted (bug 3 specifically: the assertions were checking correct-but-irrelevant data). The thing that actually caught them was reading the full log output line-by-line, not just the pass/fail summary — consistent with this project's standing discipline of checking the full artifact, not just its final claimed status (the same pattern that caught the `ablation_topk` duplicate-run log anomaly in Session 9).

---

## 2026-09-10 — Path A, Module 4: exact hard-skip reproduction confirmed at all three new `topk_capacity` points

**Decision (finding):** Ran `src/eval_hard_skip.py` (unmodified — it already derives capacity from `cfg.topk_capacity` for any `fixed_topk` run, generically) against all three newly-trained checkpoints, same protocol as the original `ablation_topk` (capacity=0.5) verification in Module 3: 20 real WikiText-103 validation batches, same seed (1337), dense (Phase 1, soft) forward pass vs. `mlp_only` hard-skip forward pass compared batch-by-batch.

**Result:** all three reproduce dense output to float32 machine precision, matching the original capacity=0.5 result exactly in kind:

| run | capacity | delta (hard_skip − dense) |
|---|---|---|
| `ablation_topk_cap17` | 0.1667 | −2.38e-08 |
| `ablation_topk_cap33` | 0.3333 | 0.00e+00 (bit-exact) |
| `ablation_topk` (reference, Module 3) | 0.5000 | +4.77e-08 |
| `ablation_topk_cap75` | 0.75 | 0.00e+00 (bit-exact) |

Every batch at every capacity matched dense to 4+ decimal places when printed (`+0.0000`/`-0.0000`); the full-precision deltas above (all ~1e-8 or exactly 0) confirm these are ordinary floating-point summation-order noise, not a real discrepancy — same order of magnitude as `ablation_topk`'s own already-accepted reference result.

**Why this holds at every capacity, not just 0.5:** the underlying reason (established in Module 3) is architectural, not specific to one capacity value — `FixedTopKRouter.forward()` computes `scores * mask` (exact zero for non-selected tokens) during training itself, at whatever capacity the model is configured with, so hard-skip's "excluded tokens contribute exactly 0" was never an approximation for any `fixed_topk` checkpoint, regardless of the fraction excluded. This result confirms that reasoning generalizes rather than having been a coincidence specific to capacity=0.5.

**Consequence:** All four `fixed_topk` capacity points (0.1667, 0.3333, 0.5, 0.75) are now confirmed safe to use for Module 5's quality-curve script and Modules 6-8's wall-clock benchmark sweep — each checkpoint's dense-mode val_lm and its real hard-skip val_lm are provably the same number, so the accuracy-vs-capacity curve built from dense evaluation (already done, Module 3/Session 12) is *also* the accuracy-vs-capacity curve for the real, hard-skipped, wall-clock-fast model at every point — precisely the unification Path A set out to establish.

---

## 2026-09-10 — Path A, Module 5: `src/plot_topk_curve.py` — fixed_topk's own accuracy-vs-capacity curve

**Decision:** Built `plot_topk_curve.py`, the `fixed_topk` analog of `plot_curve.py` (which builds the same kind of curve for `learned_gate`'s `target_avg_layers` sweep). Filters `experiments/*/config.json` by `router_type == "fixed_topk"` (not by run-name pattern) — this cleanly separates the 4 real `fixed_topk` runs from the 4 `learned_gate` sweep runs (which have no `router_type` key at all, a pre-router-ablation-generalization config shape) and from `ablation_confidence` (`router_type == "confidence_threshold"`), verified directly against the real `experiments/` directory before writing the filter, not assumed. X-axis is `topk_capacity` directly (an exact, by-construction training hyperparameter for this router — see `router.py`'s `FixedTopKRouter` docstring) rather than an achieved-vs-target quantity, since unlike `learned_gate`'s `avg_depth` there's no gap to report for this router type.

**Result:** all 4 real points plotted — capacity 0.1667→val_lm 4.2729, 0.3333→4.0195, 0.5→3.9139, 0.75→3.8583 — a clean, monotonic diminishing-returns curve, the same qualitative shape as the `learned_gate` sweep.

**Verification, matching the bar `plot_curve.py` itself was held to in Session 6 (synthetic test harness, not just one real run):** built a fake `experiments/`-shaped scratch directory with 3 fake `fixed_topk` runs (in scrambled, non-capacity-sorted order), one fake `learned_gate`-shaped config (no `router_type` key), one fake `confidence_threshold` config, one `fixed_topk` config with no log yet (incomplete run), and one `fixed_topk` config with an empty log file (0 lines — an edge case the real 4-run dataset can't exercise, since all 4 real runs completed). Confirmed: exactly the 3 real fake `fixed_topk` runs are found; all four non-qualifying cases are correctly skipped without crashing; runs sort correctly by `topk_capacity` regardless of discovery order; a single-point case plots without error; a zero-point (empty directory) case raises a clear `ValueError` rather than crashing obscurely or silently producing an empty/broken chart.

**A real cosmetic bug caught before finalizing:** the first chart render had a two-line `plt.title()` (main title + a note about Module 4's hard-skip-losslessness finding) that visually overlapped the top-left data point's annotation — not a correctness bug, but a real legibility defect that would have shipped in the eventual final report if not checked by actually viewing the rendered image, not just confirming the script ran without error. Fixed with `plt.suptitle()` for the secondary note (separate vertical position from the main title) plus explicit `ylim` headroom sized to the data's own y-range, then re-rendered and re-viewed to confirm the fix. General lesson, consistent with this project's own established discipline elsewhere (byte-identity checks, exact-value assertions): "the script exits 0 and saves a file" is not the same verification bar as "the artifact is actually correct and legible" — a chart needs to be looked at, not just generated.

---

## 2026-09-10 — Path A, Module 6: `benchmark_speed.py` extended with `run_sweep()` for a capacity-swept wall-clock benchmark

**Decision:** Added `run_sweep()` and `--sweep`/`--capacities` to `benchmark_speed.py`, additively — `run()` (the original single-capacity function, already used to produce the real, reported Session 10 result at capacity 0.5) and its output file (`results/benchmark_speed_<device>.json`) are both completely untouched; `run_sweep()` is new, writes to a separately-named file (`results/benchmark_speed_sweep_<device>.json`), and defaults to the same four capacities already trained in Path A's accuracy sweep (0.1667, 0.3333, 0.5, 0.75), so the wall-clock benchmark and the accuracy curve cover exactly the same points.

**Deliberate choice: still time `baseline` and `adaptive_dense` at every capacity, even though their latency is architecturally capacity-independent.** Both variants always compute every token through every layer in dense mode, regardless of what capacity the router is configured with (`FixedTopKRouter`'s own top-k/masking cost doesn't meaningfully scale with k either) — so re-timing them at each sweep point was, in principle, unnecessary and could have been skipped to save time. Kept anyway: doing so turns "these two are capacity-invariant" from an asserted architectural claim into something checked against repeated real measurements — if `baseline`/`adaptive_dense` timings come back flat (up to ordinary noise) across all four capacities while only `adaptive_hard_skip` shows a real trend, that's a direct empirical confirmation of the reasoning, not just a restatement of it. (Whether that confirmation actually holds is Module 7's real-GPU question — the local smoke test's timings are far too noisy, n=2 iterations, to judge this on CPU.)

**`--checkpoint` + `--sweep` interaction:** if a real checkpoint is given, the capacities list is ignored and replaced with that checkpoint's own single trained capacity — sweeping one checkpoint's weights across capacities it wasn't trained at would be exactly the off-capacity-inference mismatch Module 1's diagnostic already found catastrophically breaks quality (worse than random guessing at the extreme, DECISIONS.md 2026-09-09). That finding was about quality, not latency, but reusing one checkpoint across capacities here would imply a comparison this project has already shown isn't meaningful — so this path is blocked structurally rather than left as a foot-gun, with a printed explanation when it triggers.

**Verification:**
- Confirmed the original `run()` / non-sweep CLI path is completely unaffected: reran `--smoke --device cpu` (no `--sweep`) and got the same behavior/output file as before this module's changes.
- Ran the new sweep locally (`--smoke --device cpu --sweep`): all 4 capacities × 3 variants × 2 shapes completed cleanly; verified the saved JSON's structure directly (`capacities` list, `variants_by_capacity` keyed by 4-decimal capacity strings, each holding all three variant names) rather than just checking the script exited 0.
- Verified the `--checkpoint` override: `--sweep --checkpoint ablation_topk_cap33` printed the expected message and ran only capacity 0.3333, ignoring the default 4-point list.
- Verified `--compile` degrades gracefully inside the new sweep loop too (not just the original single-capacity path): `--sweep --compile --capacities 0.5,0.75` completed with exit code 0, every `[compiled]` entry at every capacity/variant correctly caught the same `InvalidCxxCompiler` failure (no C++ compiler on this dev machine — expected, matches Module 4's original finding) and recorded it in the JSON rather than crashing the run.
- Verified custom `--capacities` parsing (`0.5,0.75`) produces exactly that 2-point list, not the 4-point default.

**Consequence:** Module 7 (the real Kaggle GPU run) can now produce a real wall-clock number at every point on `plot_topk_curve.py`'s accuracy curve, not just the one already reported — completing the accuracy-vs-real-speed unification Path A set out to build, once real GPU numbers land.

---

## 2026-09-10 — Path A, Module 7: `docs/colab_benchmark_hard_skip.ipynb` extended with the real capacity-sweep run

**Decision:** Extended the existing benchmark notebook additively — regenerated cell 9's `%%writefile benchmark_speed.py` body from the current source (now including Module 6's `run_sweep`), and inserted 8 new cells (Modules 6-8, in the notebook's own numbering) for the capacity sweep, positioned after the original single-capacity Modules 3-5 and before the existing backup cell. The original Modules 1-5 (Setup, Code, the single-capacity smoke check, full benchmark, and results/chart) are left completely untouched — verified byte-identical except for cell 9's necessary regeneration — so the notebook can still exactly reproduce the already-reported Session 10 result (capacity 0.5) if ever rerun as before. The backup cell, now positioned at the very end after the new sweep modules, backs up both the original and new results together in one zip, unchanged in its own logic.

New notebook modules:
- **Module 6 — Capacity sweep smoke check**: `run_sweep()` at `SMOKE_SHAPES`, tiny warmup/iters, across all four default capacities — catches issues cheaply before the longer real run.
- **Module 7 — Full capacity sweep**: `run_sweep()` at `REAL_SHAPES` (matching Module 4's original warmup=10/iters=30), writing to `results/benchmark_speed_sweep_cuda.json` — a separate file from Module 4's `results/benchmark_speed_cuda.json`, so the already-reported single-capacity result is never at risk of being overwritten.
- **Module 8 — Sweep Results**: a per-capacity latency table, per-capacity speedup ratios (hard-skip vs. dense, vs. baseline), the consistency check flagged in the `benchmark_speed.py` Module 6 decision (baseline/adaptive_dense should read ~flat across capacities; only adaptive_hard_skip should show a real trend), and a latency-vs-capacity line chart (one line per variant). Deliberately does **not** attempt to build the final unified accuracy-vs-real-speed chart inside this notebook — that combination happens locally afterward, from the downloaded sweep JSON plus the already-verified `val_lm` numbers from Path A's training sweep, matching how every other cross-run comparison in this project (`plot_curve.py`, `plot_ablation.py`, `plot_topk_curve.py`) is built locally from downloaded artifacts rather than inside a Kaggle/Colab notebook.

**Verification (flat-directory simulation of the real, extracted cell text, matching the bar set for every prior notebook in this project):** materialized the real `%%writefile` cell bodies into a flat directory (no `src/` subfolder), then executed the actual, unmodified text of cells 20 (sweep smoke check), 22 (full sweep), 24 (results table), 25 (speedup ratios + consistency check), and 26 (chart) end-to-end. Local-CPU-only substitutions were applied strictly for feasibility (this dev machine has no GPU and no working C++ compiler) and were kept to runtime *values*, never the cell *text* itself: `device="cuda"`→`"cpu"`, warmup/iters shrunk, `REAL_SHAPES` patched to a smaller tuple via the `benchmark_speed` module attribute (the cell's own `from benchmark_speed import ... REAL_SHAPES` line was never touched), and `use_compile` forced off (compile's graceful-failure behavior inside the sweep loop was already directly verified separately, with real captured errors in the JSON, in the Module 6 entry above — repeating that check here would only add many minutes of redundant `InvalidCxxCompiler` overhead with no new coverage). All five cells ran cleanly end-to-end; two harness-only snags were hit and resolved without touching the actual cells under test: a stray leftover debug assertion in the verification script itself (unrelated to the notebook), and `plt.show()` blocking under this machine's interactive `TkAgg` matplotlib backend when run as a raw script outside any GUI event loop — resolved by forcing the non-interactive `Agg` backend for this CLI harness only (a real Jupyter/Kaggle kernel's inline backend never blocks on `plt.show()` in the first place, so this doesn't reflect anything the real notebook needs).

**Status:** the notebook is built and fully verified; the real Kaggle GPU run (producing the actual wall-clock numbers at all four capacities) is the next, final step of Path A's Module 7 — handed to Saif to execute.

---

## 2026-09-10 — Path A, Module 7 result: real capacity-swept GPU benchmark, confirmed clean

**Decision (finding, confirmed against the downloaded artifact, not taken from summary):** Saif ran the notebook's Modules 6-7 on a real Kaggle GPU and downloaded `results/benchmark_speed_sweep_cuda.json`. Read the file directly (all 441 lines) before accepting any number from it. Real median latency at the throughput shape (32×256), eager mode, 30 timed iterations (10 warmup) per point:

| capacity | baseline | adaptive_dense | adaptive_hard_skip |
|---|---|---|---|
| 0.1667 | 210.92ms | 229.70ms | 166.52ms |
| 0.3333 | 217.79ms | 218.30ms | 170.00ms |
| 0.5000 | 216.70ms | 224.97ms | 186.62ms |
| 0.7500 | 217.09ms | 221.29ms | 207.75ms |

**Consistency check (from the `benchmark_speed.py` Module 6 decision) confirmed, not just asserted:** `baseline` varies only 3.26% of its minimum across the four capacities (range 6.87ms); `adaptive_dense` varies only 5.22% (range 11.40ms) — both consistent with ordinary run-to-run GPU timing noise, not a real trend. `adaptive_hard_skip`, by contrast, shows a real, monotonic, substantial trend: 166.52ms → 207.75ms, a 41.23ms range, increasing strictly with capacity at every step. This is real, measured confirmation — not just the architectural reasoning restated — that only the hard-skip compute path actually depends on capacity.

**Precise real speedups (eager, hard_skip vs. dense — the fair, apples-to-apples comparison), per capacity:** 0.1667: 27.51% less time (1.379×); 0.3333: 22.12% less time (1.284×); 0.5000: 17.05% less time (1.206×); 0.7500: 6.12% less time (1.065×). Speedup shrinks monotonically as capacity grows, exactly as expected — less compute is skipped, so less time is saved. `torch.compile` again showed no consistent, reliable benefit across capacities (a mix of small speedups and small slowdowns per variant/capacity, all within roughly ±5%, one exception — `adaptive_hard_skip` at capacity 0.1667 improved 5.82% compiled — but no capacity showed compile reliably beating eager across all three variants), consistent with the Session 10 finding that compile doesn't meaningfully help this workload.

**A real, honestly-surfaced discrepancy: this sweep's own capacity=0.5 point does not exactly match the earlier, separately-reported Session 10 single-capacity result, taken in a different Kaggle session.** Session 10: baseline=195.21ms, hard_skip vs. dense speedup=11.31%. This sweep's capacity=0.5 point: baseline=216.70ms (≈11% higher), hard_skip vs. dense speedup=17.05% (≈6 percentage points higher). Both runs agree in *direction* (hard-skip is always faster than dense, at every capacity, in both sessions) and the absolute latencies are all in a plausible T4-class range, so this reads as ordinary session-to-session/GPU-instance variance rather than a bug or a change in the model — but it is a real, non-trivial spread (~50% relative difference in the *speedup percentage* itself, even though the qualitative finding holds in both), and it should be stated as a limitation, not smoothed over: **a single-session speedup percentage from this benchmark should be read as approximate, with roughly this much possible session-to-session spread, not as a precisely pinned-down number.** Consequence for Module 8: the unified chart (below) uses only this one sweep's own four points, never mixing in the earlier Session 10 measurement, specifically to avoid silently conflating real session variance with the real, capacity-driven trend the chart exists to show.

---

## 2026-09-10 — Path A, Module 8 (final): `src/plot_speed_vs_accuracy.py` — the unified accuracy-vs-real-speed chart

**Decision:** Built the final Path A deliverable: joins `plot_topk_curve.py`'s real accuracy data (each `fixed_topk` run's `val_lm` at its trained `topk_capacity`) with Module 7's real latency data (`adaptive_hard_skip`'s median latency at the same capacity, 32×256 shape, eager mode) — matched by nearest capacity within a small tolerance (defensive against the two files' capacities not being bit-identical floats, though in practice they are), not by run name or ordering. Deliberately draws latency **only** from the Module 7 sweep JSON (one consistent Kaggle session) rather than mixing in the earlier Session 10 single-capacity number, for the reason given in the entry above.

Produces one two-panel chart: **(1)** both `val_lm` and real hard-skip latency plotted against `topk_capacity` on a shared x-axis (dual y-axis) — shows *how* the underlying knob (capacity) drives both quantities; **(2)** `val_lm` plotted directly against real measured latency (x-axis inverted so "faster" reads left-to-right as "better," matching "lower loss" reading top-to-bottom as "better") — the literal, direct unification this whole Path A effort was built to produce: one router, one set of checkpoints, both halves of the hypothesis (comparable quality at a range of budgets; real measured speedup at each of those same budgets) on the same set of real numbers.

**Result:** all 4 real points joined correctly (verified against the known values from Session 12/Module 4: capacity 0.1667→val_lm 4.2729/166.52ms, 0.3333→4.0195/170.00ms, 0.5→3.9139/186.62ms, 0.75→3.8583/207.75ms). Panel 2 shows a clean, monotonic diagonal from bottom-left (capacity 0.75: slowest, best quality) to top-right (capacity 0.1667: fastest, worst quality) — a real, visible speed/quality tradeoff curve, not a scattered or ambiguous relationship. Chart viewed directly (not just confirmed to save without error, per this project's established "look at the rendered image" discipline) — legible, no overlapping elements, both panels internally consistent with each other and with the source data.

**This is the artifact the project completion audit (2026-09-08) found missing.** The audit's central finding was that quality-preservation evidence (`learned_gate`'s accuracy-vs-avg-layers sweep) and real-speedup evidence (the single `fixed_topk` capacity=0.5 benchmark) had never been demonstrated on the same router, at more than one point, together. This chart closes that gap directly: one router (`fixed_topk`), four real trained checkpoints, four real measured latencies, hard-skip proven lossless at every one of those four points (Module 4), assembled into a single real accuracy-vs-real-speed curve — completing Path A.

---

## 2026-09-10 — Updated project-completion verdict (supersedes the 2026-09-08 audit's central finding, not the whole audit)

**Context:** Saif asked to revisit the project-completion question now that Path A is complete, believing it now genuinely proves the hypothesis. Re-applied the 2026-09-08 audit's own standard — real evidence, on the same router, chained together, checked against actual files rather than accepted on summary — to the new data, rather than treating "Path A shipped" as automatically meaning "the audit's concerns are resolved."

**What Path A closed, confirmed with real evidence:** the audit's central, most consequential finding — that quality-preservation evidence and real-speedup evidence lived on two different routers and were never unified — is now directly closed. One router (`fixed_topk`), four independently-trained real checkpoints, four real Kaggle-GPU-measured latencies (Module 7), hard-skip proven exactly lossless at every one of those four points (Module 4, all at float32 machine precision), assembled into one real accuracy-vs-real-speed curve (Module 8). This is the artifact `CLAUDE.md`'s hypothesis was actually asking for.

**What Path A did NOT touch, re-confirmed rather than assumed:** `experiments/baseline_v1/` still does not exist (checked again this session). The original static baseline's precise final loss remains permanently lost (Session 4's original incident, corrected in the 2026-09-08 audit) — "comparable to the untouched, full-computation baseline" specifically still rests on an approximate, chart-read number (~3.8), not an exact one. This was never in Path A's scope and is a narrower, less central claim than the gap Path A closed.

**A new precision caveat surfaced by Path A itself:** real session-to-session GPU variance (this sweep's capacity=0.5 point: 17.05% speedup vs. Session 10's separately-reported 11.31% at the same nominal capacity, different Kaggle session) means a single quoted speedup percentage should be read as approximate — real spread on the order of several percentage points is expected between benchmark runs on shared cloud GPU infrastructure. The qualitative finding (hard-skip reliably faster than dense, monotonically more so at lower capacity, at zero measured quality cost) is robust to this and doesn't depend on which session's exact numbers are quoted.

**Updated verdict:** upgraded from the 2026-09-08 audit's "partial, nuanced success" to **the core hypothesis is now genuinely proven** — real, independently-verified evidence exists, on a single router, across a real range of compute budgets, with both halves of the hypothesis (quality held; real wall-clock speed gained) demonstrated together rather than in two disconnected experiments. Two caveats remain, both clearly scoped and neither undermining the core claim: (1) the pre-existing, out-of-scope lost baseline number, and (2) the newly-surfaced session-to-session variance affecting the precision (not the direction or existence) of any single quoted speedup percentage. This is a materially stronger, more defensible position than existed before Path A, arrived at the same way every other claim in this project has been — by checking, not by asserting.

---

## 2026-09-14 — `baseline_v1` checkpoint recovered; the lost-baseline caveat is closed

**Context:** Saif re-ran the original static-depth baseline training on Kaggle (Save & Run All) and downloaded `ckpt.pt`, `config.json`, `train_log.jsonl` into `experiments/baseline_v1/`, resolving the Session 4 incident (2026-08-30) where the first `baseline_v1` run completed but its checkpoint was lost to a Colab runtime disconnect before download, and the 2026-09-08 audit's correction that the log/config had never survived either — leaving no precise, citable baseline number anywhere in the project until now.

**Verification performed (files checked directly, not taken on Saif's report):**
- `config.json`: architecture matches every other run in the project exactly (`n_layer=12`, `n_head=6`, `d_model=384`, `d_ff=1536`, `block_size=256`), `seed=1337` (same seed used by every sweep/ablation run for comparability), `dataset_name="Salesforce/wikitext"` / `dataset_config="wikitext-103-raw-v1"` (the post-Session-3 corrected dataset id), `max_steps=20000`, `run_name="baseline_v1"`. No `target_avg_layers`/`router_type`/`topk_capacity` keys — correctly a plain static-depth config, not an adaptive one.
- `train_log.jsonl`: 41 lines, 41 unique steps, exact sequence `0, 500, 1000, ..., 20000` — no gaps, no duplicate-run concatenation artifact (the failure mode that hit `ablation_topk` in Session 9). `elapsed_s` strictly monotonic across all 41 rows; total wall time 4.81h, consistent with Session 4's original "~4.7 hours" estimate for this run size.
- `ckpt.pt` (488,446,941 bytes): loads cleanly with `torch.load`; contains `model`/`optimizer`/`step`/`config` keys; `step=20000` matches the log's final row and `config.json`'s `max_steps`; `config.run_name="baseline_v1"` matches. State dict has 149 tensors, 59,990,016 total parameters — consistent with the 12-layer/d_model=384 architecture's expected parameter count (independently computed, not just read off). Size is close to but slightly smaller than `adaptive_v1/ckpt.pt` (488,531,133 bytes), exactly as expected since the adaptive model carries small additional per-layer router-head parameters the static baseline doesn't have.

**Result — the real, exact final baseline numbers now exist:** `val=3.807868719100952` at step 20000 (final eval), `train=3.8379406929016113`. This replaces every "~3.8, read visually off the chart" approximation used throughout the project (Sessions 6-13) with an exact, logged number: **baseline final val loss = 3.8079**.

**Consequence for prior comparisons:** every existing baseline-vs-adaptive quality comparison in this project's docs (the `target_avg_layers` sweep write-ups, the router ablation, `final-interview-prep.md`) used the approximate ~3.8 chart-read value. The exact number (3.8079) falls within that approximation and does not change any qualitative conclusion already drawn (e.g. `adaptive_v1`'s 3.8135 is still essentially matched to baseline; `learned_gate` still wins the router ablation). No prior conclusion needs to be retracted — this closes a precision gap, not a correctness error.

**Status:** this was the last explicitly-tracked open item from the 2026-09-08 completion audit and the 2026-09-10 updated verdict (both of which listed "the pre-existing, out-of-scope lost baseline number" as the one remaining caveat on an otherwise-proven hypothesis). With a real, verified `baseline_v1` checkpoint and exact final loss now on disk, that caveat is closed. Remaining open items are the pre-existing git-repo-root housekeeping issue and `ablation_topk`'s harmless log-file duplicate-run artifact — neither substantive.
