# learning.md

Session-by-session learning log — concepts, commands, approach, blockers, resolutions.

---

## Session 1 — 2026-08-28

**Concepts:**
- Project framing: adaptive compute / conditional computation in transformers, inspired by DeepMind's Mixture-of-Depths (fixed top-k token routing per layer) and Google's CALM (confidence-threshold early exit). Distinction matters for the later router ablation.
- Why reproducibility discipline (seed + full config logged per run) matters more once a second axis (routing depth) is added — a result is meaningless if you can't say exactly what config produced it.

**Commands used:**
- `python -m py_compile ...` to catch syntax errors before runtime.
- `pip install torch --index-url https://download.pytorch.org/whl/cpu` — CPU-only wheel, used only for local shape/correctness smoke tests since this environment has no GPU.
- Wrote and ran a standalone smoke test (tiny random config: 2 layers, d_model=32) exercising forward pass, backward pass, and `generate()` — checked output shapes and that loss/gradients are finite (no NaNs). This validates the code is wired correctly; it says nothing about model quality, which requires real training data + GPU.

**Approach:**
- Scaffold repo structure and lock in founding architecture/dataset/tokenizer decisions before writing model code, so the baseline and future adaptive model share one settled foundation.
- Adopted a module-by-module working style for anything multi-step (now codified in `CLAUDE.md`): plan the modules first, build and validate one at a time rather than generating everything in one pass.
- Built the Colab notebook (`docs/colab_train_baseline.ipynb`) programmatically — a small Python script read the real `src/*.py` files and inserted them into `%%writefile` cells via `json.dump`, rather than hand-typing notebook JSON. Avoids escaping bugs and guarantees the notebook's code is byte-identical to the reviewed source.

**Blockers:**
- No GPU available in this working environment — can write and review code here, but actual training runs must happen on Colab/Kaggle free tier.

**Resolution:**
- Treated this session as code-writing + review only; no training run attempted or claimed.

---

## Session 2 — 2026-08-29

**Concepts:**
- Framed the project's real-world relevance clearly for interview purposes: inference cost in production LLMs is paid per token, per layer, regardless of whether that computation was actually needed — conditional computation (this project's routing mechanism, plus Mixture-of-Depths, CALM, MoE) is one of the few concrete levers against that cost without shrinking the model.
- Portability gotcha: code written assuming a fixed repo layout (`src/data.py` relative to a sibling `data/` directory via `os.path.dirname(__file__)`) breaks silently when the same files are flattened into a different directory structure (Colab's `/content`, no `src/` subfolder) — the relative `..` now walks up one directory too far. General lesson: prefer `os.getcwd()`-based paths (or an explicit passed-in path) over `__file__`-relative paths when code is meant to run in more than one directory layout.

**Commands used:**
- None this session (review/discussion only, no code run).

**Approach:**
- Reviewed the smoke-run config and the notebook cell that will actually execute on Colab tomorrow, before Saif runs anything — caught the `CACHE_DIR` path bug by manually simulating the path resolution logic rather than waiting to discover it after a live Colab run.

**Blockers:**
- Same as Session 1: no GPU here, so review was code-level only — the smoke run itself still needs to happen on Colab.

**Resolution:**
- N/A — Saif deferred the actual Colab run (and the fix decision) to tomorrow. Nothing left half-applied; the bug is fully documented in `memory.md` for a clean pickup.

---

## Session 3 — 2026-08-30

**Concepts:**
- HF Hub deprecated bare, namespace-less dataset ids for script-based loaders (`load_dataset("wikitext", ...)` used to work; now requires the canonical `"namespace/name"` form, `"Salesforce/wikitext"`). A dataset that worked in an older tutorial or cached example can silently stop resolving on the Hub side with no code change on our end — worth checking the actual Hub page rather than trusting a remembered id, especially for anything not touched in a while.
- Distinguished two outcomes of the same root-cause bug class (`..`-relative / `__file__`-relative paths breaking across directory layouts): a *silent* failure (last session's `CACHE_DIR`, which wrote files to the wrong place with no error) vs. a *fail-safe* one (`train.py`'s git-commit lookup, which throws inside a `try/except` and degrades to `"unknown"` with no side effect). Same underlying pattern, but only one actually needed fixing — worth checking whether a given instance of a known bug class is dangerous or already contained before "fixing" it reflexively.
- Distinguished "regenerate a notebook cell from its `src/*.py` source" (config.py, data.py — must stay byte-identical to source) vs. "edit a notebook-native cell directly" (the Module 6 results-plotting cell, which has no corresponding source file) — same verification bar (JSON re-validated, content checked) but a different mechanism for applying the fix.

**Commands used:**
- `python -c "..."` one-off scripts (not saved) to: regenerate specific notebook cells from `src/*.py` via `json.load`/`json.dump` with `ensure_ascii=False`, and to assert byte-for-byte equality between a notebook cell's body and its source file after regeneration — programmatic verification instead of eyeballing a diff.
- `git rev-parse --show-toplevel` and `git ls-files -- <paths>` to check whether files are actually tracked in the repo — surfaced that this repo's root is the whole home directory and the project files aren't tracked yet.
- `python -m py_compile src/data.py src/config.py` as a fast post-edit sanity check.
- WebFetch against the live HF Hub dataset page to confirm `wikitext-103-raw-v1` is still a valid config under `Salesforce/wikitext`, rather than assuming the rename was a pure namespace swap.

**Approach:**
- For each fix: locate every occurrence repo-wide via grep first (not just the one reported instance), confirm the fix against a live external source of truth when available (HF Hub page) rather than assuming, apply the source-of-truth (`src/*.py`) change, then regenerate and byte-verify any notebook cells derived from it, then re-validate the whole notebook JSON structure. Showed diffs before/alongside regenerating so Saif could review the actual change, not just be told "fixed."
- When a second instance of a previously-fixed bug class showed up (`results_dir = "../results"`), did a full repo sweep for the same pattern (`..`-relative and `__file__`-relative paths) rather than just patching the one reported line — surfaced one more instance (`train.py`'s git-commit lookup) and let Saif decide whether it needed fixing, since it was already fail-safe.

**Blockers:**
- Still no GPU here — every fix this session was a static code/notebook change, verified by grep, compile, and JSON structure checks, not by an actual Colab run. The real test of both fixes (dataset loads without `HfUriError`, chart appears in `/content/results`) still has to happen on Saif's next Colab run.

**Resolution:**
- Both bugs fixed and verified statically (source changed, notebook cells regenerated/byte-checked, JSON re-validated). Saif explicitly decided not to fix the third relative-path spot found (git-commit lookup in `train.py`) since it fails safe and Colab never clones the repo — documented as a deliberate non-fix, not an oversight.

---

## Session 4 — 2026-08-30

**Concepts:**
- Colab's `/content` filesystem is ephemeral and tied to the runtime session, not the notebook — it's wiped on disconnect (idle timeout, manual disconnect, crash), regardless of whether the training run itself completed successfully. A completed run and a *safely retrieved* run are two different things; only the second one actually counts as done.
- What's actually lost vs. safe after a runtime disconnect, for this project's baseline run: `train_log.jsonl` (loss curve data) and `config.json` were already saved locally/downloaded during Module 6, so the accuracy/loss numbers and the chart are intact and citable. What's gone is the model weights (`ckpt.pt`) itself — the trained parameters, not the record of how well they performed.
- **Soft vs. hard routing.** True per-token hard skipping (variable-shaped compute per layer) needs real token compaction (gather/scatter) to get actual FLOPs savings, and doesn't save anything during training anyway since the compute budget is fixed regardless. A *soft* gate — scale the block's contribution to the residual stream by a continuous learned score, still compute everything — is fully differentiable by ordinary backprop (no REINFORCE / straight-through estimator needed) and lets routing *behavior* be validated before ever touching the harder, real-compute-savings implementation. Separating "does the router learn something sensible" from "is the sparse-compute path correct" is a real engineering simplification, not just theory.
- **Zero-init-last-layer bootstrap tricks have a real, predictable transient cost.** Zero-initializing a network's final layer (so it starts at a controlled, input-independent output) blocks gradient from flowing to everything *upstream* of that layer on the very first backward pass — the zero weight matrix kills the chain rule going backward, even though the layer's own weight still gets a valid gradient (since `dL/dW` depends on the activations feeding it, not on `W`'s current value). This is expected and self-resolving after one optimizer step, but it's a general pattern worth recognizing whenever a "gradient is exactly zero" surprise shows up right after a zero-init.
- **A blanket weight-init pass (`model.apply(fn)`) is dangerous for a model with hand-tuned submodule inits.** `AdaptiveTransformer.__init__` runs the same `self.apply(self._init_weights)` pattern `BaselineTransformer` uses (reinitializes every `nn.Linear`/`nn.Embedding` it finds) — but this recursively hits every router's `nn.Linear` too, silently overwriting the router's deliberate zero-init bootstrap. The general lesson: when composing a hand-initialized submodule into a larger model that does a blanket re-init pass, the submodule's init must be explicitly reasserted afterward (or the blanket pass must be scoped to exclude it) — and this kind of bug is easy to miss by code review alone; it only surfaced because the smoke test checked the *actual numeric value* of `avg_depth` at init against a hand-computed expectation, not just "is it finite."
- **Compute-budget regularization: penalize the aggregate, not the individual.** A loss term meant to control an aggregate resource (average depth, average compute) needs to be computed as a penalty on the aggregate statistic, not decomposed into a per-unit penalty — the latter silently imposes uniformity (every token pushed toward the same depth) even when the actual goal is to allow maximum per-unit heterogeneity while only constraining the total budget. Easy to get backwards if you don't stop to check which quantity the project's hypothesis actually depends on.

**Commands used:**
- `python -m py_compile src/*.py` after every module edit, including ones that touched already-shipped files (`model.py`, `config.py`), as a fast pre-test sanity gate.
- Wrote standalone smoke-test scripts (not saved to the repo) for each module: `router.py` (shape/range/gradient checks, including the zero-init transient), the `AdaptiveTransformer` integration (shape/gradient checks across all layers' routers, plus the exact-value `avg_depth` init check that caught the blanket-reinit bug, plus a full rerun of the original baseline smoke test as a regression check), `losses.py` (pure-math gradient-sign checks plus a real 40-step optimizer integration test on random data), and `train_adaptive.py` (a full `train()` run with `prepare_dataset`/`get_batch` monkeypatched to synthetic data, checking the actual files written to disk).
- `pip install tiktoken` — was needed for `data.py`'s import chain during the `train_adaptive.py` smoke test but hadn't actually been installed in this local dev environment before; already a listed dependency, just never previously exercised locally.

**Approach:**
- Planned the adaptive model's architecture *before* writing any code: worked through three real design forks with Saif (block-skip granularity, soft-vs-hard training mechanism, which router variant to build first) with explicit recommendations and tradeoffs, got sign-off on the plan and module order, then built one module at a time — router, then the model extension, then the loss term, then the training script — with Saif reviewing and approving each one before moving to the next, per the project's own module-by-module working-style rule.
- For every module, verification went beyond "does it run without erroring": checked exact numeric values against hand-computed expectations (init gate mean, `avg_depth = 0.88 * n_layer`), verified gradient signs and magnitudes in isolation before wiring modules together, and for `losses.py` and `train_adaptive.py` ran real (if tiny/synthetic) end-to-end integration tests proving the mechanism actually moves model behavior, not just that gradients exist on paper.
- Whenever a module edit touched a file underlying an already-shipped/validated result (editing `model.py`, which the already-completed baseline run depends on), reran the original baseline-specific smoke test afterward as an explicit regression check, rather than assuming the new code was additive-only.

**Blockers:**
- The Colab runtime disconnected after the baseline training run finished but before Saif downloaded `experiments/baseline_v1/ckpt.pt`. Colab's temporary storage was then wiped, so the trained checkpoint is unrecoverable — only the loss curve and config survived (they'd already been saved/downloaded separately).
- No GPU locally, same as every prior session — every module built and tested this session was verified on tiny/synthetic CPU-only data. The real test (does the adaptive model's `avg_depth` actually converge sensibly on real WikiText data, at real scale) still requires the Colab run Saif has deferred to tomorrow.

**Resolution:**
- Checkpoint loss: no retraining right now. The already-logged loss curve and final train/val numbers are sufficient to document the baseline result and move forward; the actual checkpoint weights aren't needed yet since nothing planned next (building the adaptive-depth model) depends on loading the baseline's trained weights. Retraining is deferred to if/when a specific need for the baseline checkpoint itself comes up (e.g., a direct side-by-side generation/inference comparison against the adaptive model, as opposed to a same-architecture full-depth-vs-routed comparison run on the adaptive model alone).
- Process fix to prevent recurrence: added an explicit "Colab checkpoint backup" rule to `CLAUDE.md`'s Engineering discipline section — download/back up checkpoint + logs immediately after training finishes, in the same cell block, before anything else.
- Blanket-reinit-clobbers-router-init bug: fixed by extracting the router's init logic into a re-callable `reset_gate_init()` method, called again per-block after `AdaptiveTransformer`'s blanket `self.apply(_init_weights)` pass — verified via the smoke test's exact `avg_depth` value check at init.
- Still-no-GPU blocker: accepted as a standing constraint, same as every prior session — all four modules were built and verified as thoroughly as CPU-only, tiny/synthetic testing allows; real-scale validation is explicitly deferred to tomorrow's Colab run, not claimed as already proven.

---

## Session 5 — 2026-09-01

**Concepts:**
- **Kaggle's session-persistence model is meaningfully different from Colab's, not just a smaller version of the same thing.** Colab lets you download any file from a live interactive session's Files panel at any point, even after a partial disconnect/reconnect in some cases. Kaggle does not: an interactive session that disconnects (including from ~40 minutes of *browser tab* inactivity — independent of whether the kernel itself is actively computing) leaves `/kaggle/working` files completely inaccessible afterward, with no post-hoc recovery. This is a platform-specific fact, not something guessable from Colab experience — worth verifying per-platform rather than assuming backup/recovery mechanics transfer.
- **"Non-interactive" and "not subject to inactivity timeout" are not the same guarantee as "output is always captured."** Kaggle's "Save Version → Save & Run All" (a background/batch commit) is Kaggle's own recommended path for long runs specifically because it removes the *inactivity*-driven disconnect risk — but multiple independent Kaggle user reports describe partial/no output capture when the notebook actually *crashes* (OOM, exception) mid-commit, as opposed to completing or being stopped cleanly. Two different failure modes (idle timeout vs. actual crash) can have different recovery guarantees on the same platform, so a fix aimed at one doesn't automatically cover the other.
- **A safety mechanism designed against one platform's failure model doesn't automatically transfer to a different platform.** `ckpt_interval` (this session's own change) was designed around Colab-style recoverability (a periodically-updated file you can grab from a live, always-downloadable file browser). On Kaggle, that same file sitting in `/kaggle/working` is only actually recoverable under specific conditions (a successfully *completed or cleanly stopped* commit) — it does not, by itself, close the gap for an actual mid-run crash. Building a mitigation is necessary but not sufficient; it has to be re-validated against the specific environment it'll actually run in.
- **First real (non-synthetic) evidence the router mechanism works as designed**: on real WikiText data, `avg_depth` was observed genuinely converging toward `target_avg_layers=6.0` while loss decreased healthily, up to step 3000. Every previous verification of this exact mechanism (Sessions 4-5's smoke tests) used random tokens, where there's no real language signal for the LM loss to compete against the budget loss with — this is qualitatively different evidence, even though the run didn't finish.

**Commands used:**
- `WebSearch` (Kaggle community/product-feedback threads, since `kaggle.com/docs/notebooks` itself wasn't crawlable via `WebFetch` — no body content, JS-rendered) to verify actual platform behavior before relying on it, rather than assuming Colab-style semantics apply. Followed up with a second, more targeted search specifically on whether a genuine committed background job is affected by local wifi/tab state.
- Same notebook-build-and-verify pattern as prior sessions: a Python script writes the `.ipynb` JSON directly (`json.dump`, `ensure_ascii=False`), followed by a separate verification pass — this session added a new rung to that ladder: not just byte-identity checks, but writing all `%%writefile` cell bodies to a real flat directory and running an actual `train()` call against that simulated layout, catching anything a byte-identity check alone couldn't (e.g., import-order issues that only show up when the files are actually flat, not nested under `src/`).
- `torch.load(ckpt_path, map_location="cpu", weights_only=False)` in smoke tests to inspect a saved checkpoint's actual `step` field — used to prove decoupled cadence empirically (checkpoint step is a multiple of `ckpt_interval`, not `eval_interval`) rather than trusting the code by inspection alone.

**Approach:**
- Before writing any code for the `ckpt_interval` request, checked what already existed rather than assuming a gap — found periodic checkpointing was already present (just coupled to eval cadence), which meant the actual fix was smaller and more targeted (decouple two things) than what was asked for at face value (add periodic saving from scratch). Verifying-before-building applies to features, not just bugs.
- Before answering a platform-behavior question (Kaggle backup semantics, then later the wifi/tab question), searched for real, sourced evidence rather than reasoning from Colab-derived intuition or general notebook-platform assumptions — flagged explicitly when the evidence was strong (multiple consistent community reports) versus when it was suggestive but not fully confirmed by a quotable primary source (the wifi/tab-closure question specifically).
- When simulating a "mid-run disconnect" in a smoke test, simulated the actual mechanism of failure (an eval interval wide enough to never fire again) rather than just artificially truncating a loop — this proves the checkpoint survives independent of the logging/eval path, which is the property that actually matters for the real incident this was built to prevent.

**Blockers:**
- The real Kaggle training run stopped around step 3000 with the exact cause still unconfirmed (local wifi disconnect vs. a manual stop vs. an actual kernel crash are all still possible) — this session's research narrows the likely causes (a properly-committed background job shouldn't be killed by local wifi) but doesn't pin down which one actually happened. Genuinely blocked without checking Kaggle's own version/commit status log for that specific run, which wasn't done this session.
- Couldn't get a fully authoritative, directly-quotable primary source confirming that closing the browser tab / losing wifi has literally zero effect on a genuine "Save and Run All" commit — the evidence (search-engine-summarized snippets citing "non-interactive" and "not subject to the 1 hour inactivity limit") is reasonably strong but secondhand, not a first-party Kaggle doc excerpt.

**Resolution:**
- Not fully resolved this session — deliberately left open rather than guessed at. Logged as the first thing to check before the retry: look at Kaggle's Versions/Output tab for the stopped run's actual recorded status, which should disambiguate crash vs. stop vs. something else far more reliably than continuing to reason about it secondhand.

---

## Session 6 — 2026-09-02

**Concepts:**
- **The accuracy-vs-compute curve needs an aggregation step that's genuinely separate from the training loop itself.** `train_adaptive.py` and the notebook's Module 6 both work at the level of one run's own trajectory (loss/depth over steps). The actual central-result curve is a different axis entirely — final quality vs. final compute, one point *per run*, not per step — and needed its own script (`plot_curve.py`) reading across many runs' already-finished logs, not a modification to how any single run trains or logs.
- **A per-token-heterogeneous compute budget can buy a lot of quality for a little depth, at least at the extremes tested so far.** Going from avg_depth 6.08 down to 2.65 (a >2x reduction) only cost 0.08 in val LM loss (3.8135 → 3.8915). That's the first genuinely quantitative evidence for the project's whole premise, not just a mechanism check — worth being precise that this is only 2 of 4 planned points, so the *shape* of the tradeoff (is it this flat everywhere, or does it fall off a cliff below some threshold?) isn't visible yet.
- **Filtering runs by structural shape (a field's presence/absence in config.json), not by naming convention.** `plot_curve.py` identifies "is this an adaptive run" by checking for `target_avg_layers` in the loaded config, not by pattern-matching the run name — more robust, since it doesn't depend on Saif always naming runs a particular way, and correctly generalizes to the baseline's config (which structurally has no such field) without needing a baseline-specific exclusion rule.

**Commands used:**
- Read/inspected real downloaded artifacts directly (`json.load` on `config.json`, iterating `train_log.jsonl`) before trusting a verbal summary of what a training run produced — same discipline as checking `git ls-files` or `git rev-parse --show-toplevel` in earlier sessions rather than assuming.
- Built and ran a synthetic multi-run test harness for `plot_curve.py` (fake `experiments/*/config.json` + `train_log.jsonl` files in a scratch directory) to validate the aggregation logic before it had to work on ~15 hours of real, not-yet-generated data — the same "prove it on cheap synthetic data before trusting it on expensive real data" pattern used for `losses.py` and `train_adaptive.py` in Session 4.

**Approach:**
- Verified the prior session's claimed retry success against real local files before treating it as fact and building on top of it — `experiments/adaptive_v1/` having a real `ckpt.pt` this time, not just an entry in `memory.md`, is what actually confirms the checkpoint-backup fix worked in practice, not just in the smoke tests that preceded it.
- When a genuine resource/time tradeoff came up (how many sweep points, at ~5 Kaggle-hours each), asked rather than picked a number — this is a budget decision only Saif can make, unlike the earlier architecture-design forks which had a clear best-recommendation to lead with.
- Caught and self-corrected an unnecessary complexity in my own first draft of the notebook's sweep-parameterization edit (a conditional special-casing `target=6.0`'s run name) before it went in — worth noting as a small but real instance of applying "don't add complexity beyond what's needed" to my own code, not just flagging it in someone else's.

**Blockers:**
- None new this session — the Session 5 root-cause question (why the step-3000 run stopped) remains unresolved, but didn't block progress since the retry succeeded anyway. Revisit if a future sweep run stops unexpectedly, since the diagnostic step (checking Kaggle's Versions/Output status) still hasn't actually been done.

**Resolution:**
- N/A — no new blocker to resolve this session.

---

## Session 7 — 2026-09-03

**Concepts:**
- **Convergence and noise are two separate questions, and a single final-step number can't distinguish them.** "Did `avg_depth` actually stabilize" (checked by looking at the trailing window of a run's own trajectory) and "how much does `val_lm` fluctuate at a fixed, converged state" (checked the same way) are different questions that both matter for trusting a single reported number — the first says the reported value is representative of the converged model, the second says how much uncertainty that single value carries even so.
- **A noise floor estimated from one run's own late-training fluctuation is a legitimate, cheap substitute for repeated-seed error bars when repeated seeds aren't affordable.** It's not as rigorous (it doesn't capture run-to-run variance from a different seed/init), but it's a real, non-zero, already-available signal for whether an observed gap between two points is likely to be signal or noise — free to compute from data that already exists, versus another ~5 GPU-hours per additional seed.
- **Normalizing a delta by the (unequal) x-axis spacing between points changes the story, and can either weaken or strengthen a qualitative claim depending on which way the spacing is skewed.** Here, the naive raw deltas (0.0492 → 0.0288 → 0.0147) already look like diminishing returns, but the actual per-layer slope is an even sharper decline, because the last interval (6→9) spans *more* x-distance (2.9 layers) than the first (2→4: 1.6 layers) — so the raw-delta comparison was, if anything, understating the effect, not overstating it. Worth checking which direction an unequal-spacing artifact cuts before assuming it's a threat to a finding.

**Commands used:**
- Sliced each run's `train_log.jsonl` to its last 10 records (`records[-10:]`) and computed `max - min` for both `val_avg_depth` and `val_lm` to get a cheap, real convergence-stability and noise-floor estimate — no new training or GPU time needed, just re-reading data that was already downloaded.

**Approach:**
- Before accepting "the sweep is done and the curve is real" at face value, went one level deeper than re-verifying the reported final numbers (already standard practice by this point) — checked whether those final numbers were actually *stable*, and quantified how much noise exists at that stability plateau, before making any claim about the shape or magnitude of the differences between points. This is the same "verify against real artifacts, not the summary" discipline as always, applied one layer deeper: not just "is the number real" but "is the number meaningful at the precision being read off it."
- When asked for a next-step recommendation under real time/compute constraints, ranked the three remaining deliverables (visualization, router ablation, Phase 2 hard-skip) by a concrete cost/value comparison (GPU-hours needed, engineering risk, whether it's blocked on nothing vs. needs existing artifacts vs. is genuinely open-ended engineering) rather than just listing them neutrally — gave a clear, defended recommendation since that's what was actually asked for.

**Blockers:**
- None this session — the sweep completed successfully and the noise-floor question was answered from existing data, not left open.

**Resolution:**
- N/A.

**Concepts (continued — visualization + router ablation, same session):**
- **A per-chip data-encoded color needs its own contrast-aware text color, not a page-level theme token.** The routing-visualization artifact's token backgrounds range continuously from light yellow to dark red (encoding depth, not decoration) — a single fixed text color would go illegible at one end regardless of which page theme (light/dark) is active, since the chip color is absolute, not theme-relative. Solved by computing each chip's own text color from that chip's actual luminance, at generation time, in Python where the RGB values already exist.
- **A "swappable interface" built early doesn't automatically cover every future variant cleanly, and it's worth saying so rather than assuming.** `router.py`'s `forward(x) -> gate` contract was explicitly designed back in Session 4 to make later router types pluggable — and it genuinely helped — but `FixedTopKRouter` needed a fundamentally different *shape* of decision (ranking across a whole sequence, not a per-token independent function) and `ConfidenceThresholdRouter` needed information (the block's own output) that wasn't in the original interface at all. Both were still buildable with modest, well-scoped changes (a factory function; one signature extension), but pretending the original design had already solved these would have been dishonest and would have hidden real complexity from the planning conversation.
- **MoD's actual training mechanism (score-scaled selection, not selection-as-1) has a real, non-obvious consequence**: the metric everywhere else in the project calls "avg_depth" stops meaning "how many layers this token was processed at" for a top-k router — it becomes magnitude-weighted by the (still-training) score. Concretely demonstrated with real numbers (0.7505 vs. a naive 1.5), not just reasoned about, and resolved by logging a second, genuinely exact metric (`nonzero_frac`) rather than trying to force one number to mean two different things well.
- **The residual stream in a pre-LN transformer isn't renormalized between blocks, so its magnitude drifts across depth** — a fact that would have silently broken an absolute-threshold confidence router (early layers rarely tripping it, late layers almost always, for reasons having nothing to do with token difficulty) if not caught during design rather than discovered empirically later. Normalizing by the token's own current magnitude (a relative delta) fixes this and was verified directly, not just asserted, via a scale-invariance test.
- **Empirical local calibration can stand in for expensive-platform calibration when the thing you're calibrating is a property of a fixed, known-shape function (like an untrained model's activation statistics) rather than of training dynamics.** `calibrate_threshold.py` measures real relative-delta statistics from the actual architecture on actual data, entirely locally and for free, and the resulting starting threshold transferred well when plugged back into the real routing forward pass (target 6.0, got 6.43) — a good example of de-risking an expensive resource (Kaggle GPU-hours) with a cheap one (local CPU inference) before spending the expensive one.
- **Testing "the real committed cell text," not a hand-retyped reproduction of it, is the difference between testing your understanding of the code and testing the code.** For the notebook verification with real Kaggle time on the line tomorrow, extracted the actual cell source from the `.ipynb` JSON and `exec()`'d it (with a transparent tiny-architecture injection for speed) rather than writing a script that "does the same thing" — catches drift between what's described and what's actually saved that a description-based test cannot.

**Commands used (continued):**
- `pip install datasets` (a listed but not-yet-locally-installed dependency, same situation as `tiktoken`/`datasets` in earlier sessions) to tokenize the real WikiText-103 validation split locally for the first time, for both the visualization's real snippet and later the threshold calibration.
- Loaded the `artifact-design` skill before publishing the routing-visualization artifact, per its own trigger requirement — produced a genuine design plan (named palette, type pairing, layout concept) rather than treating a technical/data artifact as not warranting one.
- `exec(compile(cell_source, ...), namespace)` on cell text extracted directly from the notebook's JSON, with a monkeypatched `AdaptiveConfig` wrapper that transparently injects a tiny architecture under whatever kwargs the real cell passes — used to test the *actual* committed notebook cell logic end-to-end without waiting for a real 20000-step run.

**Approach (continued):**
- For the router ablation, ran a real planning conversation before writing any code — explicitly surfaced where the existing swappable-router interface would and wouldn't hold up, rather than starting to build and discovering the gaps mid-implementation. One clarifying question (which confidence-threshold design to use) was asked directly rather than picked unilaterally, since it was a genuine literature-fidelity-vs-engineering-effort tradeoff with no clearly-better default.
- Continued the by-now-established rhythm for every code change touching the shared `AdaptiveBlock`/`AdaptiveTransformer` machinery: rerun the *exact*, *unmodified* original Session 4 test file and require byte-for-byte identical numeric output, not just "no crash." This caught nothing new this session (everything held), which is itself the point — the discipline's value is in catching the one time it wouldn't.
- Before spending real Kaggle GPU-hours on two ~5-hour ablation runs, built a local, free calibration step for the one hyperparameter that couldn't otherwise be reasoned about in advance (the confidence threshold), rather than accepting "we'll find out on Kaggle" as the plan.

**Blockers (continued):**
- None — both new router types were built, tested, calibrated where needed, and wired into the notebook without any unresolved issue at session end.

**Resolution (continued):**
- N/A.

---

## Session 8 — 2026-09-06

**Concepts:**
- **A live edit made directly in a Kaggle kernel doesn't automatically propagate back into the repo's own copy of a notebook.** `docs/colab_train_adaptive.ipynb`'s committed `ROUTER_TYPE` was still `"learned_gate"` even after Saif had (per his account) edited it to `"fixed_topk"` live on Kaggle to attempt the ablation run — the two copies diverged silently, with nothing in the repo to indicate a live run had used a different config than what's on disk. This is a real gap in the reproducibility discipline: the notebook file is the source of truth for what *should* run, but only the run's own downloaded `config.json` (which never arrived here, since the run didn't complete) is the source of truth for what *actually* ran.
- **When a user states a fact as settled ("this did NOT run successfully"), re-verify from files anyway rather than defer entirely to their account or to prior in-conversation analysis.** The first verification pass (same session, earlier) had already concluded the ablation state was ambiguous and recommended checking Kaggle directly; Saif's follow-up asserted the failure as fact and asked for a full fresh re-check rather than accepting that ambiguous conclusion — the correct response was to re-run every check from scratch (not reuse earlier tool outputs from memory) rather than simply agreeing based on the user's statement, since the two are complementary: his account explains *why* no artifact exists, but the file check is what actually confirms it.

**Commands used:**
- Re-ran the same folder-listing + per-config `router_type`/`target_avg_layers`/`topk_capacity`/`confidence_threshold` read loop against `experiments/` a second time from a clean state, deliberately not reusing the first pass's tool output, per Saif's explicit instruction to read files rather than rely on conversation memory.
- Re-extracted notebook cells 15-17 directly from `docs/colab_train_adaptive.ipynb`'s saved JSON a second time (UTF-8 file write + Read, to route around the Windows console's cp1252 crash on `≈`) rather than trusting the first extraction's summary.
- Searched `Downloads` root for any un-extracted Kaggle backup zip (`*backup*`, `*ablation*`, `*target6*`, `*.zip`) that might contain a completed-but-unfiled `fixed_topk` run — none found, ruling out "it's done but just not organized yet" as an explanation.

**Approach:**
- Treated the user's stated fact ("fixed_topk didn't run successfully") as the reason a fresh check was warranted, not as a substitute for it — re-verified independently from files both times requested, rather than short-circuiting to "yes, confirmed" based on the user's own report.
- Synthesized the two independent verification results (local `experiments/` state + notebook's committed default) into one single, unambiguous next action, per Saif's explicit request for "one clear next action" rather than a menu of considerations.

---

## Session 9 — 2026-09-07

**Concepts:**
- A completed final training step is not the same thing as a clean log file. `train_log.jsonl`'s step count and its line count can diverge if a run is interrupted and re-executed against the same file path — verifying "does the final row look right" is necessary but not sufficient; verifying "is every step number unique" catches a different class of problem entirely.
- `elapsed_s` (or any wall-clock-since-start field) is a good tell for "this log actually contains more than one process's output" — a value that resets back near zero partway through a monotonically increasing step sequence means a new process started writing to the same file, not that the same process paused and resumed.
- Two router types can be compared on genuinely equivalent footing even when their headline metrics (`avg_depth` vs. `nonzero_frac`) aren't the same quantity — `fixed_topk`'s `nonzero_frac` (exact selection fraction) and `learned_gate`'s `avg_depth / n_layer` (average gate value) both approximate "fraction of full compute used," so dividing one by `n_layer` makes them directly comparable even though neither router logs the other's native metric.

**Commands used:**
- `wc -l`, `head -n 1`, `tail -n 5` on `train_log.jsonl` for a first-pass shape check.
- A short inline `python3 -c "..."` script parsing every JSON line to compare `len(steps)` vs. `len(set(steps))`, then printing `(line_index, step, elapsed_s)` for every row to visually spot exactly where the sequence resets — this is what surfaced the two-concatenated-runs anomaly; `wc -l` alone would have looked like a plausible log length and hidden it.
- `du -h` to sanity-check checkpoint file size against a known-good reference (`adaptive_v1`'s `ckpt.pt`) as a cheap, meaningful integrity check before trusting a 466MB binary blob at all.

**Approach:**
- Treated "Saif says the files landed" as a claim to verify, not a fact to log — read the actual bytes on disk (file sizes, config contents, full log contents) before writing anything to `memory.md`, consistent with the project's standing "no fabricated results" discipline.
- When a config/log matched expectations on the surface (final row looked right, config fields were correct), still checked the *full* log rather than stopping at the last line — that's specifically what caught the concatenated-run anomaly, which a final-row-only check would have missed entirely.
- Made a preliminary cross-run comparison honestly bounded: flagged it explicitly as one-of-two-ablation-arms and not a final conclusion, rather than stating a router-quality verdict off a single data point.

**Blockers:**
- None — this was a local, file-verification-only session; no GPU/Kaggle work was run from here.

**Resolution:**
- N/A.

**Blockers:**
- None — this was a pure verification session, no code changes made, no ambiguity left unresolved by session end.

**Resolution:**
- N/A.

---

## Session 10 — 2026-09-08

**Concepts:**
- **A router's compute budget calibrated on an untrained model doesn't necessarily hold once real training happens** — `confidence_threshold`'s threshold (0.4309) was empirically tuned in Session 7 against an untrained 12-layer model to hit avg_depth≈6.4, but after 20000 real training steps its actual selection rate fell to `nonzero_frac=0.255` (avg_depth≈3.06), about half the intended budget. The likely mechanism: a trained model's per-layer representations change less from block to block (smaller relative delta-norm) than an untrained model's do, so a fixed magnitude threshold measured pre-training becomes systematically more permissive toward early-exit as training progresses.
- **A router's *type* of budget guarantee matters, not just its target value.** `fixed_topk`'s budget is a rank-based selection rule (always exactly the top 50% of tokens by score, whatever those scores are) — invariant to how the underlying score distribution shifts during training. `confidence_threshold`'s budget is a magnitude-based rule (score above/below a fixed number) — directly sensitive to the same distribution shift. Same nominal target, structurally different robustness to the thing that actually changes during training.
- **A router with zero learned parameters cannot correct for its own miscalibration.** `learned_gate` and `fixed_topk` both have a mechanism to adapt (an explicit budget-loss gradient for the former, an always-exact selection fraction by construction for the latter); `confidence_threshold` has neither — once its one hyperparameter is set, nothing in training can move it back toward the intended budget if the underlying representation statistics drift.
- Completing a 3-arm ablation doesn't automatically produce an apples-to-apples comparison — the *actually achieved* budgets have to be checked per arm before trusting the loss comparison as a clean "which router is best," which is exactly why `plot_ablation.py` reports both val_lm and achieved depth/nonzero_frac side by side rather than loss alone.

**Commands used:**
- Re-verified all three ablation run directories from scratch (`ls -la` on each of `experiments/adaptive_v1`, `experiments/ablation_topk`, `experiments/ablation_confidence`) before trusting any of them, per the project's standing file-first verification discipline.
- Read `ablation_confidence/config.json` directly to confirm `router_type`, `confidence_threshold`, and `target_avg_layers` matched what was expected from the Session 7 calibration, rather than assuming the downloaded run matched its intended config.
- A short inline `python3 -c "..."` script comparing line count vs. unique `step` count on `ablation_confidence/train_log.jsonl` — same check that caught `ablation_topk`'s duplicate-run anomaly in Session 9 — this time confirming the log is clean (41/41).
- `python src/plot_ablation.py` — the actual real 3-way comparison run, not a synthetic/dev-test invocation.

**Approach:**
- Treated "the run is downloaded" as a claim to verify per-file (directory listing, then config contents, then full log integrity) before running the comparison script, consistent with Sessions 8-9's discipline, rather than jumping straight to `plot_ablation.py` on trust.
- When the plotted numbers showed all three routers at different achieved budgets rather than the intended matched ~6, treated that as a finding worth explaining rather than a discrepancy to gloss over — the "why" (calibration staleness under distribution shift) is a genuine ablation insight, not a caveat to bury.
- Read the ranking two ways before concluding anything: by raw val_lm, and separately by val_lm-per-unit-compute — confirming `confidence_threshold` loses on both axes (not just nominal loss) before calling it the weakest arm.

**Blockers:**
- None — Kaggle-side training work for this ablation was already complete; this session was local verification, comparison, and analysis only.

**Resolution:**
- N/A.

**Phase 2 continuation (same session) — concepts, commands, approach, blockers:**

**Concepts:**
- **A soft-blend formula can be algebraically decomposed per sub-layer.** `gate*block(x) + (1-gate)*x`, where `block(x) = x + attn_delta + mlp_delta` (a standard residual block), distributes additively into `x + gate*attn_delta + gate*mlp_delta`. This is what let attention stay fully dense (untouched from training) while only the MLP sub-layer's compute gets skipped for excluded tokens — but the decomposition only holds if `mlp_delta` is computed from the *unscaled* post-attention residual (`x + attn_delta`), not a gate-scaled one. Got this wrong on the first attempt (fed the scaled version into `mlp`'s LayerNorm) and it was caught immediately by a bit-for-bit exact-reproduction test at capacity=1.0 — a "close but not exact" match (max diff 0.02) was the tell that something was subtly wrong, not just floating-point noise.
- **A masking mechanism can only be as good as what the model was actually trained to expect.** Excluding a token from attention entirely (the literal Mixture-of-Depths mechanism) devastated `adaptive_v1`'s quality (val_lm nearly doubled) because Phase 1 training never excludes anyone from attention — every token is always visible as K/V, regardless of gate value. The published mechanism assumes a model trained end-to-end with that exclusion built in from the start; retrofitting it onto a differently-trained model is a much bigger distribution shift than it looks on paper.
- **"Zero out low-ranked tokens" is only a safe approximation if the router was actually trained to be sparse.** Inspecting `adaptive_v1`'s real per-token, per-layer gate distribution (not just its average) showed it's broad and centered near 0.5 — only ~3-5% of values sit near the extremes. The router's budget loss only ever pressures the *population average*, never individual scores toward 0/1, so training found a smooth, non-sparse solution that satisfies that loss just as well as a sparse one would. A hard rank-based cutoff on a distribution like that discards real, substantial signal for roughly half of every layer's tokens — no masking mechanism fixes this; it's a property of what the router learned, not of how the mask is applied.
- **A router's own training-time behavior can make it a better or worse fit for a totally different, later objective (real hard-skip) than its original training objective (soft quality) would predict.** `fixed_topk` lost the router ablation on quality, but its training-time masking (`scores * mask`, exact zero for non-selected tokens, every forward pass during training) turned out to make it the ONLY one of the three routers whose real hard-skip reproduces dense output exactly — because "excluded tokens contribute exactly 0" was never an approximation for this router to begin with.
- **Wall-clock speedup has a lower, more honest theoretical ceiling than "fraction of tokens skipped."** Skipping only the MLP sub-layer (necessary for quality, since full-block skip broke it) means the real ceiling is `capacity_excluded × mlp_fraction_of_block_flops`, not `capacity_excluded` alone. For this architecture (d_model=384, d_ff=1536), MLP is ~70.6% of a block's FLOPs — so 50% token exclusion caps out at ~35.3% FLOPs reduction, not ~50%. The real observed speedup (11.3%) should be compared against that more precise ceiling, not the naive one.
- **Real wall-clock speedup can fall well short of even an honest FLOPs-based ceiling**, and the reasons are themselves diagnosable without a full profiler: gather/scatter are memory-bound, irregular-access operations that eat into savings from skipped compute; a small model (15M params) may not fully saturate the GPU on its dense matmuls either, so cutting FLOPs doesn't cut wall-clock proportionally; and `torch.compile`'s main benefit (kernel fusion for compute-bound work) has less to offer when the workload isn't compute-bound to begin with, and may specifically struggle with hard-skip's dynamic-shape gather/scatter operations.
- **`torch.compile` can make things slower, not just fail to help** — observed directly (`adaptive_hard_skip` compiled was 3.34% slower than eager on the real GPU run), not hypothesized. A "no meaningful benefit" finding is itself worth reporting precisely (with real numbers per variant) rather than summarized as "compile didn't help."

**Commands used:**
- Local CPU smoke tests with controlled, explicit indices (bypassing the router entirely) to isolate and verify the causal-masking-after-gather property independent of any router logic — same "test the mechanism, not the whole system at once" approach as Session 4's router tests.
- A capacity sweep (1.0 → 0.3) on fixed batches, before concluding a result was a real effect rather than a bug — smooth, monotonic degradation ruled out an implementation error in favor of a genuine mismatch.
- Inspected the real trained gate distribution directly (`torch.histogram`, per-layer mean/std/min/max, fraction below 0.1 / above 0.9) rather than reasoning about it in the abstract — this is what actually found the root cause, not the capacity sweep alone.
- `python -m py_compile` after every edit, before running anything, consistent with the project's standing discipline.
- A flat-directory notebook simulation (materialize all `%%writefile` cell bodies into one directory with no `src/` subfolder, then actually run the reload + smoke-check logic against it) — same rigor Session 5 used for the adaptive training notebook, now applied to a benchmark notebook instead of a training one.
- Computed a real FLOPs breakdown by hand (per-token, per-layer, split into MLP / attention-linear / attention-SDPA) to get an honest theoretical ceiling to compare the real GPU number against, rather than comparing against the naive "50% of tokens skipped = 50% faster" assumption.

**Approach:**
- When a result looked catastrophically wrong (val_lm nearly doubling), didn't assume a bug OR silently accept the number — ran a cheap, targeted diagnostic (the capacity sweep) first to distinguish "implementation error" from "real effect," before spending more engineering effort in either direction.
- When the cause turned out to be more fundamental than the current mechanism (a property of the router's trained score distribution, not the masking approach), surfaced it explicitly and let Saif choose the direction (fine-tune vs. absolute-threshold vs. pivot routers) rather than unilaterally picking a fix — this was a genuine fork with real cost tradeoffs, not a call to make alone.
- After the pivot to `fixed_topk` worked essentially perfectly, still ran the full 20-iteration real-data comparison (not just the quick diagnostic) before calling Module 3 done — matching the project's standing "don't report a result you haven't actually run at full rigor" discipline.
- When the real GPU speedup number (11.3%) was well below a naive expectation, worked out the more precise theoretical ceiling by hand (FLOPs breakdown) before either dismissing 11.3% as disappointing or accepting it uncritically as "the number" — this is what turned a single number into a defensible, explainable finding.

**Blockers:**
- No local GPU, and `torch.compile` doesn't work at all on this dev machine (no C++ compiler) — both expected and already documented in `DECISIONS.md`'s benchmark-scope decision; not a blocker to building/verifying the harness and notebook, only to getting real numbers locally.

**Resolution:**
- Local work (Modules 1-4, notebook verification) proceeded entirely via CPU smoke tests and flat-directory simulation, exactly as scoped from the start — real numbers came from Saif running the verified notebook on a real Kaggle GPU, exactly per the original plan.

---

## Session 11 — 2026-09-09

**Concepts:**
- **A test harness that injects a fake/tiny object into an `exec` namespace can be silently defeated if the executed code itself re-imports or reassigns that same name.** `from config import AdaptiveConfig` inside the real cell text re-binds the namespace entry to the genuine class, overwriting whatever was placed there beforehand — the fix is to patch the *module's* attribute (`config.AdaptiveConfig = Tiny...`) before exec, so the code's own import resolves to the patched version instead of restoring the original.
- **Assertions can all pass while testing the wrong thing.** Checking that a dictionary's literal values are correct (`RUN_NAMES["fixed_topk"] == expected`) is not the same as checking that the value which actually got *used* downstream is correct (`full_cfg.run_name == expected`) — when the executed code can silently overwrite an injected variable via its own hardcoded assignment (cell 16's `ROUTER_TYPE = "learned_gate"` line), the dict can be perfectly correct while the thing built from a *different*, clobbered variable is wrong. Only reading the full log output, not the assertion summary, caught it.
- **A router's off-distribution failure can be bidirectional, not just one-sided.** Off-capacity inference on a `fixed_topk` model trained at capacity=0.5 collapsed both narrower (worse than random guessing at 0.1667) and wider (a real, non-negligible cost even at 0.75, contrary to the hypothesis that "extra compute is never wrong") — indicating the base model's shared attention/MLP weights specialize to the specific *density* of active tokens per layer during training, not merely to which tokens get selected.
- **A diagnostic that bypasses a router's normal forward path has to replicate its internal masking exactly, not just its scoring.** `FixedTopKRouter.forward()` zeroes the score for non-selected tokens before anything downstream sees it; a bypass that recovers the raw score but forgets to re-apply that masking gives excluded tokens a small, real (if unintended) contribution the model never trained with — a subtle correctness bug, not a rounding difference, and one with a real, checkable symptom (a "should be exact" self-consistency check coming back non-zero).

**Commands used:**
- Built a raw-score bypass (`torch.sigmoid(router.net(x))`, computed before `FixedTopKRouter.forward()`'s own capacity-based masking) to test off-capacity token selection cleanly in both directions (narrower and wider than the trained capacity) — something the router's own `forward()` method can't do above its baked-in capacity, since it discards the information needed to select more.
- A capacity=0.5 self-consistency check (must reproduce Module 3's proven exact `+0.0000` match) as the go/no-go gate on the diagnostic's own correctness before trusting any off-capacity number from it.
- String-replacement of a cell's literal hardcoded line, simulating the exact one-line edit a user would make in a live Kaggle kernel, in place of exec-namespace pre-injection once that approach was shown to be silently defeated by the cell's own import/assignment statements.
- `TaskStop` to end a background process once direct, unbuffered, isolated reproduction established it was running a real (not tiny) architecture rather than genuinely hung — diagnosis before termination, not termination as the diagnosis.

**Approach:**
- Treated the diagnostic's own capacity=0.5 self-consistency failure (`+0.083` instead of the required `+0.0000`) as a bug report worth investigating immediately, not noise to average away — "should be mathematically exact" is a strict, checkable bar, and a real gap against it is a signal, not a rounding artifact.
- When a background verification appeared to hang, didn't assume infrastructure failure and didn't just wait longer — ran a direct, unbuffered, isolated repro of the suspected slow path first, which correctly localized the problem to the test harness rather than the real training code.
- After fixing the first harness bug, re-ran and read the *full* log output rather than trusting the pass/fail assertions alone — this is specifically what caught the second, independent bug, which the assertions had already (wrongly) blessed as passing.

**Blockers:**
- None from GPU/Kaggle this session — entirely local diagnostic and notebook-prep work, as scoped. The real ~15 Kaggle-hours of training is deliberately deferred to the next session.

**Resolution:**
- N/A.

---

## Session 12 — 2026-09-10

**Concepts:**
- **Extending an established sweep methodology to a new router type mostly just works when the prep step generalized the wiring correctly.** Module 2's notebook verification effort (catching two harness bugs, confirming all four capacity values dispatch correctly) paid off directly here — three real Kaggle runs, zero notebook-side surprises, zero config drift, straight to clean results.
- **A second, independently-trained router showing the same qualitative curve shape as the first is a real (if informal) cross-check.** `fixed_topk`'s own capacity sweep shows the same diminishing-returns pattern (biggest gains at low capacity, shrinking as capacity grows) as `learned_gate`'s `target_avg_layers` sweep — two structurally different routers, trained completely independently, agreeing on the shape strengthens confidence that it reflects something real about the data/task, not an artifact of one router's specific training dynamics.
- **A small, expected discrepancy should still be explained, not just waved off.** `nonzero_frac` (0.1680, 0.3320) doesn't exactly equal the configured `topk_capacity` (0.1667, 0.3333) — traced to `round(capacity × block_size)` integer rounding at `block_size=256` (0.1667×256=42.68→round to 43→43/256=0.1680, exactly matching what was logged) before accepting it as clean, rather than either ignoring the mismatch or treating it as an unexplained anomaly requiring investigation.

**Commands used:**
- Directory listing + `config.json` reads for all three new run directories, matching the standing per-run verification protocol used for every previous run in the project.
- A single inline Python script checking, for all three logs at once: line count vs. unique step count, first/last step against the expected `0..20000` step-500 sequence, and `elapsed_s` strict monotonicity — the same anomaly-detection checks that caught `ablation_topk`'s duplicate-run issue in Session 9, run proactively this time rather than only after something looked suspicious.

**Approach:**
- Verified all three new runs fully before treating any of them as trustworthy for the curve, even though nothing about Saif's report suggested a problem — the check is unconditional, applied the same way regardless of whether the report sounds clean or not.
- Confirmed the small `nonzero_frac`-vs-`topk_capacity` deviations were explainable by a specific, checkable mechanism (rounding) before calling the runs clean, rather than either dismissing them without checking or over-investigating a fully expected effect as if it were a real anomaly.

**Blockers:**
- None this session — the real ~15 Kaggle-hours of training happened entirely on Saif's side; this session's work was setup confirmation before the runs and file-level verification after.

**Resolution:**
- N/A.

---

## Session 13 — 2026-09-10

**Concepts:**
- **A property proven at one point can and should be re-verified at every point a claim will rest on, even when the underlying mechanism is architectural (should generalize) rather than incidental.** The exact hard-skip reproduction was only ever directly measured at capacity=0.5 before this session; re-running the same check at the three new capacities (Module 4) wasn't redundant busywork — it's the difference between "this router type's masking is lossless" as an argued generalization and as a directly confirmed one.
- **A consistency check embedded in a benchmark design only has value if it's actually looked at, not just assumed to pass.** `benchmark_speed.py`'s `run_sweep()` was built (Module 6) specifically so that baseline/adaptive_dense's expected capacity-invariance could be checked empirically rather than asserted — and when the real Kaggle data came back, checking it directly (3.26%/5.22% range vs. a real 41ms monotonic trend in hard_skip) is what turns "this should hold" into "this does hold."
- **Two measurements of nominally the same quantity, taken in different sessions on shared infrastructure (a Kaggle GPU), can legitimately disagree by a non-trivial amount without either one being wrong.** This sweep's capacity=0.5 point (17.05% speedup) and the earlier Session 10 single-capacity result (11.31% speedup) are both real, both correctly measured, and still ~6 percentage points apart — a reminder that a single wall-clock benchmark run is a sample, not a ground truth constant, and that reporting a speedup number honestly means acknowledging its session-to-session spread rather than treating whichever number was measured most recently as definitive.
- **Closing a real, previously-identified gap is a different kind of update than completing a new feature — it warrants going back and re-running the same critical evaluation that found the gap in the first place, not just checking off that new work was done.** Revisiting the project-completion question this session meant re-applying the 2026-09-08 audit's own standard (real evidence, on the same router, chained together) to the new data, and separately confirming what the new work did NOT touch (the still-missing precise baseline number), rather than treating "Path A shipped" as automatically meaning "the gap is closed."

**Commands used:**
- Reran `src/eval_hard_skip.py` against all three new checkpoints (unmodified — it already generalizes capacity derivation per router type) to get the Module 4 exact-reproduction numbers.
- Built and ran a synthetic edge-case test harness for `plot_topk_curve.py` (scrambled-order fake runs, wrong-router-type configs, incomplete/empty-log cases) before trusting it on the real 4-run dataset.
- Ran `benchmark_speed.py --sweep`/`--smoke` locally in several configurations (default sweep, `--checkpoint` override, custom `--capacities`, `--compile`) to verify each new code path before any of it touched real Kaggle time.
- A flat-directory notebook simulation (materializing the real `%%writefile` cell bodies, executing the actual extracted text of the new sweep cells) — same rigor as every prior notebook verification in this project, this time applied to an additive extension of an already-shipped notebook rather than a from-scratch build.
- Direct, full-file reads of the downloaded `benchmark_speed_sweep_cuda.json` (all 441 lines) before accepting any number from it, plus a from-scratch recomputation of every percentage quoted, rather than trusting a summary.
- Re-ran `ls experiments/baseline_v1` (expecting failure) as a final sanity check before writing the updated completion verdict — confirming the one gap Path A didn't touch is still exactly where the 2026-09-08 audit left it, not assuming it from memory.

**Approach:**
- Treated the local flat-directory simulation hitting two real snags (a stray debug assertion, `plt.show()` blocking under an interactive matplotlib backend) as bugs in the *test harness* to diagnose and fix, not as reasons to weaken the verification — resolved both without touching the actual cells under test, then completed the full run to confirm.
- When the real Kaggle numbers came back showing a real discrepancy against an earlier session's result, didn't pick the more favorable number or average them away silently — reported both, explained the likely cause (session/instance variance), and adjusted how the final chart was built (single-session data only) rather than the number reported.
- For the final "is the project complete" reassessment, deliberately separated "what Path A closed" from "what Path A left untouched" as two different questions, rather than letting a successful build session blur into "and therefore the audit's concerns are all resolved" — the lost baseline number is a real, separate, still-open item that doesn't get closed just because a different gap did.

**Blockers:**
- None — Modules 4-6 and the local Module 7 verification were entirely local/no-GPU; the real Kaggle run for Module 7 was Saif's side, and Module 8 was local-only once the real data was downloaded.

**Resolution:**
- N/A.
