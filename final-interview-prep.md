# final-interview-prep.md

Comprehensive, end-to-end interview Q&A for the Adaptive-Compute Transformer project, generated on project completion (2026-09-10) per `CLAUDE.md`'s "On project completion" instruction. Synthesized from `memory.md`, `learning.md`, `interview-prep.md`, and `DECISIONS.md` — every number here is grounded in a real, verified artifact somewhere in this repo, not restated from memory.

**One-paragraph summary, if asked to describe the project in 60 seconds:** This project builds an adaptive-compute transformer from scratch — a 12-layer decoder-only language model with a learned per-token router that decides, after each layer, whether a token needs more computation or has converged. The central finding is that most tokens don't need full depth: a real, trained router can cut the average compute per token by more than half for a small, well-characterized quality cost. Three router designs were built and compared in a real ablation (a jointly-trained gate, a fixed top-k selector, and a fixed confidence threshold), and a real hard-skip inference path was built and benchmarked on GPU hardware — with an honest, non-obvious finding along the way: the router that wins on soft-trained quality (the jointly-trained gate) turns out to be *incompatible* with real hard-skip inference, while the runner-up (fixed top-k) hard-skips perfectly, for a specific, well-understood structural reason. The project's final deliverable — closing a real gap found by its own internal audit — is a single, unified accuracy-vs-real-wall-clock-speed curve, built on one router, with real Kaggle GPU numbers at four compute budgets, showing real speedups of 6–28% depending on how aggressively compute is skipped.

---

## 1. The hypothesis, motivation, and why this project exists

**Q: What was this project actually trying to prove?**
A: That a transformer can learn, without any explicit per-token labels, which tokens need heavy computation and which don't — and that exploiting this can reduce real inference cost (GPU time, latency, energy) while keeping output quality comparable. The real-world motivation is direct: every production LLM call costs GPU-seconds proportional to tokens × layers, paid regardless of whether a given token actually needed that much computation. This is the same problem DeepMind's Mixture-of-Depths and Google's CALM are both attacking — conditional computation is one of the only concrete levers against inference cost that doesn't mean shrinking the model itself.

**Q: Why build this from scratch instead of using an existing library or paper's code?**
A: The stated goal was understanding, not output — the training loop, the model, and the routing mechanism are all hand-written PyTorch, no `transformers.Trainer`. This mattered concretely later: the adaptive-depth training loop needs custom per-token masking/routing logic in both the forward and backward pass that a high-level trainer abstraction doesn't cleanly support. Building it by hand meant full control from the start instead of fighting an abstraction midway through the harder engineering (Phase 2's hard-skip work in particular).

**Q: Why WikiText-103 and the GPT-2 tokenizer specifically?**
A: Both choices are deliberately "boring" so any effect measured later is attributable to the routing mechanism, not to an unusual setup. WikiText-103 is free, well-documented, and standard enough to be comparable to published baselines. The GPT-2 BPE tokenizer (`tiktoken`, vocab 50257) was used as-is rather than training a custom one — introducing a second uncontrolled variable (vocab size, merge quality) would make it harder to attribute any change in the accuracy-vs-depth curve specifically to the router.

---

## 2. Architecture

**Q: Describe the baseline architecture and why it was sized this way.**
A: A standard 12-layer decoder-only transformer: 6 attention heads, `d_model=384`, `d_ff=1536` (4×), 256-token context, dropout 0.1, pre-LayerNorm, causal self-attention — roughly 15M parameters. The sizing is a balance between two constraints: 12 layers gives the router a meaningful range of early/mid/late exit points to learn over (2-3 layers wouldn't show an interesting curve), and the whole thing has to train in a reasonable time on a free-tier Colab/Kaggle T4 GPU. Architecture choices (pre-LN, GPT-style block) are deliberately standard so nothing unusual about the base model confounds the routing results.

**Q: Walk me through exactly how a token "skips" a layer in the adaptive model.**
A: Each `AdaptiveBlock` wraps the same dense `Block` (attention + MLP) used in the baseline, plus a per-token router. The router looks at the block's input (and, for one router type, its output too) and produces a gate value. The block's output is blended with its input using that gate: `x_new = gate * block(x) + (1 - gate) * x` — a highway-network-style blend. Gate near 1 applies the block in full; gate near 0 leaves the token essentially untouched. Critically, in this "Phase 1" soft-training scheme, *every* token still computes through the full block, every layer — nothing is actually skipped compute-wise, only down-weighted. This keeps the whole thing differentiable by ordinary backprop, no REINFORCE or straight-through estimator needed, and lets routing *behavior* be validated before the much harder problem of real compute savings is tackled separately (that's Phase 2).

**Q: Why does the router start biased toward "keep going" instead of neutral, and what did testing that reveal?**
A: The learned-gate router's last layer is zero-initialized with a bias so it starts at gate ≈ 0.88 for every token, regardless of the (still-random) hidden state feeding it — before the base model has learned anything, router decisions would otherwise be arbitrary, so starting near "always continue" avoids randomly freezing tokens with no real signal to route on yet. Testing this surfaced something genuinely non-obvious: because the last layer's weight is exactly zero, the gradient flowing *back into the hidden state* is also exactly zero on the very first backward pass — the zero weight matrix blocks the chain rule going backward, even though the router's own weight still gets a valid gradient (since that depends on the activations feeding it, not on the weight's current value). Expected, self-resolving after one optimizer step, but the first draft of the test asserted the opposite and had to be corrected.

**Q: Why did `AdaptiveTransformer` duplicate `BaselineTransformer`'s boilerplate instead of sharing a base class?**
A: A deliberate risk-scoping decision — a few similar lines of embedding/init/weight-tying code is a smaller risk than restructuring the already-validated baseline's class hierarchy for the sake of DRY-ing it up mid-project. After every edit to the shared `model.py` file, the original baseline smoke test was rerun to explicitly confirm zero regression, rather than assuming new code was purely additive just because it lived in a new class.

**Q: What's the compute-budget loss, and why is cross-entropy alone not enough?**
A: Cross-entropy never penalizes using more layers — more compute is never wrong for next-token prediction, so nothing in the LM loss alone pushes the model toward using fewer. The budget loss adds a quadratic penalty pulling the batch's *average* effective depth toward a target value. The key design choice: it penalizes the batch *average*, not each token's individual depth — a per-token penalty would push every token toward the same depth, directly contradicting the whole premise that depth should vary by difficulty. An average-depth penalty constrains only the aggregate budget and leaves per-token routing free to be as heterogeneous as the data calls for.

---

## 3. The three routers, and why each was built the way it was

**Q: Why three router designs instead of just the one that worked first?**
A: This was always meant to be a real ablation, not just a working system — `CLAUDE.md`'s own plan called for comparing a learned gate against a fixed top-k router (Mixture-of-Depths' actual mechanism) and a confidence-threshold router (CALM's idea) to see which routing *strategy* generalizes best and why, not just build one and stop.

**Q: `FixedTopKRouter` scales a selected token's output by its own score rather than setting it to exactly 1 — why, and what's the consequence?**
A: This matches Mixture-of-Depths' actual published mechanism. Top-k *selection* itself isn't differentiable (it's a ranking operation), so scaling the selected tokens' contribution by their own continuous score is what gives the scorer any gradient signal at all — only from tokens that happened to be selected in a given batch. The real, verified consequence: this project's usual "average depth" metric (gates summed across layers) stops meaning "how many layers this token actually went through" for this router type — it becomes score-magnitude-weighted. Demonstrated concretely, not just argued: at `capacity=0.5, n_layer=3`, the true selection fraction was exactly 0.5 as designed, but reported `avg_depth` came out to 0.7505, not the naive 1.5 a selected-means-full-weight assumption would predict. Resolved by logging a second, always-exact metric (`nonzero_frac`) alongside `avg_depth` rather than forcing one number to mean two different things.

**Q: Why does `ConfidenceThresholdRouter` use a *relative* delta norm instead of absolute, and how was that verified?**
A: This is a pre-LN transformer — nothing renormalizes the residual stream between blocks, so its magnitude tends to grow across layers, a known property of that architecture family. An absolute threshold on "how much did this block change the token" would then measure depth, not difficulty: early layers (small `‖x‖`) would rarely trip a threshold calibrated for late layers' larger absolute changes. Dividing by the token's own current magnitude makes the comparison meaningful at every layer. Verified directly, not just reasoned about: the same relative displacement at `‖x‖=1` and `‖x‖=50` was confirmed to produce an identical gate decision.

**Q: Why calibrate `confidence_threshold` locally before spending Kaggle GPU-hours on it?**
A: Unlike `learned_gate` (which converges toward a target via its own budget loss) and `fixed_topk` (whose capacity sets depth exactly, by construction), this router's depth-vs-threshold relationship isn't knowable in advance and has no learned mechanism to self-correct a bad guess mid-run — a wrong threshold would only be discovered after a full ~5-hour real run. A local, free calibration script measured the real distribution of relative delta-norms this exact architecture produces on real WikiText-103 validation data (untrained model, since this measures architecture activation statistics, not training dynamics), and found the placeholder default (0.1) was off by roughly an order of magnitude — it would have selected essentially every token. The empirically-derived value, 0.4309, was then validated against the real routing forward pass (not just the calibration bypass) and came out close to the target (`avg_depth=6.43` vs. a target of 6), with the small gap explained by a real, understood mechanism.

---

## 4. Central Result 1 — the accuracy-vs-average-depth curve

**Q: Walk me through the sweep methodology and result.**
A: Four full training runs of the `learned_gate` router, identical except for one swept variable, `target_avg_layers` ∈ {2, 4, 6, 9} — same seed (1337), same architecture, same data, same 20,000 training steps. Result (`val_lm`, achieved `avg_depth`, not the nominal target):

| target_avg_layers | achieved avg_depth | val_lm |
|---|---|---|
| 2.0 | 2.647 | 3.8915 |
| 4.0 | 4.247 | 3.8423 |
| 6.0 | 6.080 | 3.8135 |
| 9.0 | 8.982 | 3.7988 |

A clean, monotonic diminishing-returns curve: going from full-ish depth (9 layers) down to less than a quarter of the network's total capacity (2.6 layers) costs only about 0.09 in cross-entropy loss. This is the project's central finding — quality gains from additional average depth diminish sharply, consistent with most tokens not needing full-depth processing.

**Q: How confident are you in the precision of that curve, given only one run per point?**
A: High confidence in the qualitative shape — the largest gap (2→4 layers: 0.0492 loss) is well clear of noise. Lower confidence in the precise magnitude at the flat end: the smallest gap (6→9: 0.0147) is close to the ~0.04 peak-to-peak fluctuation measured in a single run's own late-training checkpoints (a real noise-floor estimate computed from the data already on hand, not assumed). I'd state the small-gap end as "small, consistent with diminishing returns" rather than a precisely pinned-down number, and would want repeated seeds before treating it as more than that. This is a real, acknowledged limitation, not glossed over.

**Q: Normalizing by the unequal spacing between sweep points — did that change your conclusion?**
A: The four points aren't evenly spaced in achieved depth (2.6→4.2 is a 1.6-layer gap; 6.1→9.0 is a 2.9-layer gap), so raw loss deltas alone slightly conflate "returns are diminishing" with "the last interval just covers more ground." I checked which way normalizing would cut before citing either number — it made the diminishing-returns pattern look *even sharper*, not weaker, so the correction reinforced the finding rather than undermining it.

---

## 5. Central Result 2 — the router ablation

**Q: Which router won, and why?**
A: `learned_gate` won clearly on quality: val_lm 3.8135 at avg_depth 6.08, vs. `fixed_topk`'s 3.9139 at nonzero_frac 0.5, vs. `confidence_threshold`'s 4.0682 at nonzero_frac 0.255. The mechanistic "why" matters more than the ranking itself: `learned_gate` is the only router whose selection mechanism co-adapts with the base model during training, via the budget-loss gradient. The two static routers were each calibrated or fixed once and held constant — and one of them visibly drifted as a result (see below).

**Q: None of the three routers landed at the same actual compute budget, even though all were targeting ~6 layers — how did that happen, and does it undermine the ablation?**
A: It happened because two of the three routers can't respond to how training changes the model. `fixed_topk` is exactly 50% selection by construction — a rank-based rule, immune to this regardless of how scores are distributed. `confidence_threshold`'s magnitude-based rule was calibrated against an *untrained* model to hit avg_depth≈6.4; once real training happened, layer-to-layer representations apparently stabilize much more than an untrained model's do, so the same fixed threshold let far more tokens exit early than intended — achieved budget fell to nonzero_frac 0.255 (avg_depth 3.06), roughly half the target. This doesn't undermine the ablation — it strengthens it, since it's direct evidence for *why* a jointly-trained router generalizes better: it tracks a moving target that a router calibrated once, offline, cannot. I also checked the "unfair budget" objection directly: `confidence_threshold` uses *less* compute than `fixed_topk` and still loses, so it isn't trading quality for efficiency — it's dominated on both axes.

**Q: What's the practical, real-system takeaway from this ablation?**
A: If you can afford to train the router jointly with the model, it's worth it — a real, non-trivial quality gap in its favor, not just parity. If you need a hard, guaranteed compute ceiling for deployment (an SLA-style constraint), fixed top-k is the safer static choice, specifically because its guarantee is rank-based and doesn't degrade as the model or inputs drift. A fixed confidence/magnitude threshold is the one I'd be most cautious shipping without periodic re-calibration — this ablation is direct evidence its effective behavior can silently drift from its intended operating point, with no self-correction mechanism.

---

## 6. Phase 2 — real hard-skip, and the pivot that became the project's most interesting finding

**Q: The ablation's winner was `learned_gate` — why doesn't the project just hard-skip that one?**
A: I tried that first — it's not an oversight, it's a real, load-bearing result. Real hard-skip on `learned_gate` catastrophically broke quality (val_lm nearly doubled: 7.78 vs. dense 3.81, at the real capacity ≈0.5066). I traced this to two separate, independently-confirmed causes rather than stopping at "it's bad":
1. **Attention-exclusion mismatch:** the literal Mixture-of-Depths mechanism (excluding a token from a layer's attention entirely) assumes a model trained to expect that exclusion. Phase 1 training never excludes anyone from attention for any router type — every token is always visible as K/V. Fixing this (a new `mlp_only` mode: attention stays fully dense, only the MLP sub-layer is skipped for excluded tokens) helped, but not enough — val_lm improved to 7.32, still nearly double dense.
2. **The real root cause:** inspecting `learned_gate`'s actual trained gate-score distribution (not just its average) showed it's broad and centered near 0.5 (mean 0.507, std 0.233) — not remotely sparse; only ~3–5% of values sit near the extremes. The router's budget loss only ever pressures the *population average* toward the target, never individual scores toward 0 or 1, so training found a smooth, non-sparse solution that satisfies that loss just as well as a sparse one would. Any hard rank-based cutoff then approximates roughly half of every layer's tokens as contributing exactly 0 when their true trained contribution is large — a systematic error, not a rounding effect.

**Q: So what did you do instead?**
A: Tested the identical, unmodified hard-skip machinery against `fixed_topk`'s checkpoint (the ablation's runner-up) instead. It worked essentially perfectly: val_lm matched dense to four decimal places on every evaluated real-data batch (mean delta +0.0000). The reason it works exactly, not approximately: `FixedTopKRouter.forward()` already computes `scores * mask` — exact zero for non-selected tokens — *during training itself*, every forward pass. "Excluded tokens contribute exactly 0" was never an inference-time approximation for this router; it's mathematically identical to what the dense model was trained under.

**Q: Doesn't pivoting away from the ablation's own quality winner undercut the ablation's conclusion?**
A: No — it's a second, independent axis, not a contradiction. The ablation answered "which router generalizes best under its own soft training objective" (`learned_gate`, clearly). Hard-skip compatibility answers a different question: "which router's *training-time* behavior is structurally compatible with a real, hard, post-hoc compute-skipping mechanism" (`fixed_topk`, clearly, for a specific, well-understood reason). Both are true and both are useful — the real lesson is that soft-training quality and hard-deployment compatibility are separate axes that don't automatically move together, and a real system should check both before committing to a router design based on offline quality metrics alone.

---

## 7. Real wall-clock speedup, the FLOPs-ceiling honesty check, and Path A's capacity sweep

**Q: What was the first real GPU speedup number, and was it good?**
A: At the real training-matching shape (32×256), eager mode: baseline 195.21ms, dense adaptive 214.84ms, real hard-skip 190.53ms — hard-skip is 11.31% faster than dense (the fair, apples-to-apples comparison) and 2.40% faster than the plain static baseline. Whether "good" depends entirely on the ceiling you compare it to. The naive expectation ("50% of tokens skipped → ~50% faster") is the wrong ceiling: the mechanism that actually preserves quality (`mlp_only`) only ever skips the MLP sub-layer, never attention, and a FLOPs breakdown of this exact architecture shows MLP is ~70.6% of a block's compute. So the honest theoretical ceiling at 50% token exclusion is ~35.3% FLOPs reduction, not 50%. Against that more precise number, 11.3% is still a real gap — but a far more explainable one.

**Q: Where does the remaining gap between ~35% theoretical and ~11% real likely come from?**
A: Three plausible, named causes, none profiler-confirmed (a real, acknowledged limitation, not claimed as measured): (1) `torch.gather`/`scatter_` are memory-bound, irregular-access GPU operations that cost real time, eating into whatever compute was saved; (2) this is a ~15M-parameter model at a modest batch/sequence size, plausibly not fully saturating the GPU's compute capacity on the dense matmuls being compared against either — when you're not compute-bound, cutting FLOPs doesn't cut wall-clock proportionally; (3) `torch.compile`'s main benefit (kernel fusion for compute-bound work) has less to offer here and may specifically conflict with hard-skip's dynamic-shape gather/scatter — consistent with hard-skip's compiled variant coming out 3.34% *slower* than eager in that first benchmark, the one clear anomaly in an otherwise noise-level compile effect across variants.

**Q: A project-completion audit found a real gap in this story — what was it?**
A: Before declaring the project done, I re-read every log file cold and cross-checked specific claims against the actual filesystem — not a re-summary of what the logs already said. The audit's central finding: the accuracy-vs-depth curve (quality holds up across a wide budget range) was built entirely on `learned_gate`, which by construction gets *zero* real wall-clock savings in its soft-trained form. The real 11.3% speedup was only ever demonstrated on a *different* router (`fixed_topk`), at *one* fixed budget. So quality-preservation evidence and real-speedup evidence had never been unified on the same router, at more than one point, together — two separate experiments, not one continuous proof. The audit also caught a real factual error in an earlier log entry (a claim that the original static baseline's config/log "survived" a lost checkpoint — verified directly that neither file exists, only a chart image does), and confirmed the router ablation and the sweep methodology itself held up cleanly under scrutiny.

**Q: How was that gap actually closed — walk me through "Path A."**
A: By sweeping `fixed_topk`'s own `topk_capacity` (0.1667, 0.3333, 0.5, 0.75) the way `learned_gate`'s `target_avg_layers` was swept, and getting real GPU numbers at every point on that curve, on one router:
1. **Checked for a shortcut first, and it failed decisively:** tested whether the existing capacity=0.5 checkpoint could stand in for other capacities via inference-time capacity changes alone. It couldn't — at capacity 0.1667 the loss came out *worse than uniform-random guessing* over the vocabulary, and even the wider direction (0.75, expected to be safer since "more compute is never wrong") showed a real cost. The bidirectional collapse showed the base model's shared weights specialize to the specific *density* of active tokens seen during training, not just to which tokens get picked — full retraining at each capacity was genuinely necessary.
2. **Trained three new checkpoints** at those capacities (~15 Kaggle-hours), each verified against its own downloaded files with the same anomaly checks used throughout the project (log-line-count vs. unique-step-count, `elapsed_s` monotonicity, checkpoint file size).
3. **Re-verified exact hard-skip reproduction at every new capacity**, not just the original one — all three came back at float32 machine precision (deltas of ~1e-8 or exactly 0), confirming the "exact zero for excluded tokens" property is architectural (a fact about how `fixed_topk` is trained), not a coincidence at one setting.
4. **Built `fixed_topk`'s own accuracy-vs-capacity curve** — same diminishing-returns shape as the original `learned_gate` sweep, an independent cross-check that the finding isn't specific to one router's training dynamics.
5. **Extended the wall-clock benchmark to sweep the same four capacities on a real Kaggle GPU.** The result showed a real, monotonic trend: `adaptive_hard_skip` latency ranged from 166.5ms (capacity 0.1667) to 207.7ms (capacity 0.75), while `baseline` and `adaptive_dense` — architecturally incapable of depending on capacity — stayed flat (3.26% and 5.22% range respectively, consistent with ordinary GPU noise). This is a *measured* confirmation that only the hard-skip path actually depends on capacity, not just the architectural reasoning restated. Precise speedups (hard-skip vs. dense): 27.51% at capacity 0.1667, 22.12% at 0.3333, 17.05% at 0.5, 6.12% at 0.75 — shrinking monotonically, exactly as expected.
6. **Joined the accuracy and speed data into one chart** — one router, four real checkpoints, four real measured latencies, hard-skip proven lossless at every one of them.

**Q: You found a discrepancy between this new benchmark run and the earlier one — how did you handle it?**
A: This sweep's own capacity=0.5 point showed a 17.05% speedup; the earlier, separately-reported single-capacity benchmark (a different Kaggle session) showed 11.31% at the same nominal capacity — baseline latency itself shifted about 11% between sessions (195ms vs. 217ms). Both runs agree on *direction* (hard-skip always faster, more so at lower capacity) — they disagree on the exact percentage by about 6 points, almost certainly ordinary GPU-instance/session variance rather than a bug. I reported both numbers rather than picking the more favorable one, named the likely cause, and built the final unified chart entirely from one session's self-consistent data rather than silently blending two sessions with different absolute latencies into one curve.

---

## 8. What is proven, what isn't, and the honest final verdict

**Q: Does the final evidence actually prove the project's core hypothesis?**
A: Yes, with two clearly-scoped caveats that don't undermine the core claim. What's now proven, on real, independently-verified evidence, on a single router (`fixed_topk`), across a real ~4.5× range of compute budgets: (1) a router can allocate a controllable, non-uniform amount of per-token computation and quality degrades smoothly, not catastrophically, as the average budget shrinks; (2) that same router's hard-skip inference path is provably lossless relative to its own dense forward pass at every tested budget (float32 machine precision); (3) that lossless hard-skip produces real, measured, monotonically budget-dependent wall-clock speedups on real GPU hardware (6–28% depending on capacity), with the "why the trend exists" independently confirmed via a consistency check on the non-varying baseline/dense timings. Both halves of the hypothesis — comparable quality at a range of budgets, and real reduced real-world compute at those same budgets — are now demonstrated together, on the same object, not in two disconnected experiments.

**Q: What's still not closed?**
A: One thing, narrower than the gap Path A closed. Real session-to-session GPU variance (the 11.3%-vs-17.1% discrepancy) means any single quoted speedup percentage should be read as approximate, with real spread on that order between benchmark runs — the qualitative finding is robust to this, the precise digit isn't a fixed constant. (The other caveat that stood here at project completion — the original static baseline's exact final loss being permanently lost, Session 4's Colab checkpoint incident — was independently closed after completion: Saif re-ran the baseline on Kaggle, Session 14 (2026-09-14) verified the recovered `ckpt.pt`/`config.json`/`train_log.jsonl` directly against the filesystem, and the project now has an exact number — val 3.8079, train 3.8379 at step 20000 — instead of the ~3.8 chart-read approximation used throughout Sessions 6-13. The exact value falls within that approximation and changes no prior conclusion.)

**Q: If someone pushed back and said "you got two different speedup numbers for the same thing, why should I trust either" — what would you say?**
A: That's the right thing to be suspicious of, and it's exactly why I reported the discrepancy instead of hiding it. A benchmark that always produces the identical number, run after run, on shared cloud GPU infrastructure, would actually be more suspicious than one with real variance. What matters is whether the qualitative claim holds across that variance — and it does: in both sessions, at every capacity tested in both, hard-skip was reliably faster than dense, and the direction of the capacity-speedup relationship was consistent. "Proven" in this project has always meant "verified with real evidence, discrepancies included," not "produces one unchanging number regardless of when you run it."

**Q: Given everything, would you call this project a success?**
A: Yes, unambiguously — both as an engineering exercise and as a piece of research storytelling. The individual findings are real, mostly non-obvious, and each was arrived at by actually testing something rather than assuming it: the attention-exclusion failure, the gate-sparsity root cause, the FLOPs-ceiling calculation, the off-capacity bidirectional collapse, the baseline-files discrepancy the completion audit itself caught. That process — build a claim, test it hard, find it doesn't fully hold, trace the actual mechanism, adjust scope honestly, and go build the missing piece rather than declare victory early — is a stronger interview story than a clean "it worked exactly as hypothesized" narrative would have been, and it's genuinely how the project actually went.

---

## 9. What would be done differently, in hindsight

**Q: What's the single biggest avoidable mistake in this project?**
A: Losing the baseline checkpoint. Colab's `/content` storage is ephemeral and gets wiped on any runtime disconnect, and the checkpoint wasn't downloaded immediately after training finished — the resulting "Colab checkpoint backup" rule (download immediately, same cell block, before doing anything else) was added to `CLAUDE.md` *after* this happened, and followed without incident for every run afterward. In hindsight, that discipline should have been in place from the very first training run, not learned from losing one. It's a good illustration of how a single early process failure can leave a mark on a project's evidence that persists well past the point the rest of the project is considered done — Path A's entire completion effort never touched this gap, and it was only closed afterward, in Session 14, when the baseline was re-run on Kaggle and its recovered checkpoint/log verified directly against the filesystem (exact final loss: val 3.8079, train 3.8379 — previously only known approximately, ~3.8, from a chart).

**Q: If you were redesigning the router-selection process from scratch, what would you do differently?**
A: I'd treat "is this router's trained behavior compatible with real hard-skip inference" as a design criterion to check *before* committing to full ablation training runs, not something discovered after the fact. `learned_gate`'s incompatibility (its scores are trained to be non-sparse, since its loss never rewards sparsity) is a predictable structural consequence of its own loss function, not a surprise that needed a failed hard-skip attempt to find — a quick check of the score distribution's entropy, or an explicit sparsity-encouraging regularizer added from the start, could have either caught this earlier or made `learned_gate` itself hard-skip-compatible, potentially avoiding the pivot to a lower-quality router entirely.

**Q: What would you add if you had more Kaggle-hours to spend?**
A: In priority order: (1) a real `torch.profiler` pass to apportion the 35.3%→11.3%(-ish) FLOPs-ceiling-to-real-speedup gap precisely across gather/scatter cost, GPU utilization, and compile behavior, instead of naming plausible causes without confirming them; (2) repeated-seed runs at the sweep's flattest points, to put real error bars on the smallest observed gaps rather than relying on a single run's own late-training noise floor as a proxy; (3) a real KV-cache decode loop, so the wall-clock number could extend from prefill latency (what's measured now) to actual autoregressive generation cost, which is what a real serving system would care about most; (4) either a real fine-tune of `learned_gate` toward a sparser score distribution (an entropy or L0-style regularizer) or a full retraining sweep of `learned_gate` itself under hard-skip conditions via a straight-through estimator, to see whether the ablation's quality winner can be made hard-skip-compatible rather than accepting the pivot to `fixed_topk` as final.

**Q: What process lessons from this project would you bring to a new one?**
A: Three, all demonstrated repeatedly rather than just stated: (1) verify claims against actual files, not against your own or someone else's summary of them — this caught a real factual error in an early session log, a real duplicate-run artifact in a training log, and two real bugs in test harnesses that would otherwise have produced false "PASSED" results; (2) when a notebook or script is meant to run somewhere else (Colab, Kaggle), actually simulate that target environment (a flat directory with no subfolder structure, a tiny injected architecture run end-to-end) rather than trusting that code which looks right will behave right — this caught real import-order and namespace-shadowing bugs that byte-identity checks alone would have missed; (3) report negative and null results as plainly as positive ones — `torch.compile` not helping, a router's trained scores not being sparse, a benchmark showing session-to-session variance — since burying an inconvenient finding is a worse outcome for a research project's credibility than a clean story with one fewer positive result in it.

---

## 10. Quick reference — key numbers

| Quantity | Value |
|---|---|
| Architecture | 12 layers, 6 heads, d_model=384, d_ff=1536, block_size=256, ~15M params |
| Static baseline final loss (`baseline_v1`, step 20000) | val 3.8079, train 3.8379 (exact — recovered and verified Session 14, 2026-09-14) |
| `learned_gate` accuracy sweep | avg_depth 2.65→val_lm 3.8915; 4.25→3.8423; 6.08→3.8135; 8.98→3.7988 |
| Router ablation (≈50% budget) | learned_gate 3.8135 (avg_depth 6.08) < fixed_topk 3.9139 (nonzero 0.5) < confidence_threshold 4.0682 (nonzero 0.255) |
| Hard-skip on learned_gate | val_lm 7.78 (full_block) → 7.32 (mlp_only) vs. dense 3.81 — quality collapse, root cause: non-sparse trained scores |
| Hard-skip on fixed_topk | exact match to dense at all 4 tested capacities (deltas ~1e-8 or 0, float32 precision) |
| `fixed_topk` accuracy-vs-capacity sweep | 0.1667→4.2729; 0.3333→4.0195; 0.5→3.9139; 0.75→3.8583 |
| Real GPU speedup (hard-skip vs. dense, eager, 32×256) | 27.51% @ cap 0.1667; 22.12% @ 0.3333; 17.05% @ 0.5; 6.12% @ 0.75 |
| FLOPs ceiling at 50% exclusion (mlp_only) | ~35.3%, not the naive 50% (MLP is ~70.6% of block compute) |
| Known limitations | session-to-session GPU variance affects precision of any single speedup %; no profiler-confirmed FLOPs-ceiling breakdown; no repeated-seed error bars |
