# product.md — Turning this into a YC-backed product

Started: 2026-09-06. This is a living strategy doc, separate from the research log (`memory.md`/`learning.md`/`interview-prep.md`/`DECISIONS.md`), which stays focused on the research project itself. This file is where the "what if this becomes a company" conversation lives.

---

## 1. The honest starting point

What exists today: a from-scratch 12-layer transformer with a learned per-token router that decides how many layers each token needs, trained on WikiText-103. Real result: at `target_avg_layers=6` (half depth), loss is within ~0.03 of the 9-layer point and not dramatically worse than full depth — i.e., a real diminishing-returns curve showing most tokens don't need full depth.

**This is a research prototype that proves a mechanism, not a product.** The gap between the two is the entire content of this document. Two things have to change for this to become a company:

1. **From-scratch → retrofit.** No company is going to pretrain their own LLM to save inference cost. They already have models (Llama 3, Mistral, Qwen, or their own fine-tunes) running in production. The sellable version teaches an *existing* pretrained model to route depth via a small amount of additional fine-tuning (router-only, or router + light adapter), without touching the base weights' knowledge. This is a materially different ML problem from what's built so far — closer to "distillation with a routing objective" than "train from zero."
2. **Theoretical FLOPs saved → real wall-clock $ saved.** The project's own plan already flags this: `torch.compile` benchmarking exists specifically because theoretical savings and measured savings diverge. For a product, this divergence isn't a caveat in a results section — it's the central engineering problem, because it lives inside GPU batching.

---

## 2. Why this problem is real (the "why now")

- Inference cost, not training cost, is now the dominant recurring cost for any company running LLMs at scale — it scales with usage, forever, unlike a one-time training run.
- Every production LLM call burns full compute through every layer for every token, regardless of whether that token is "the" (trivial) or a rare technical term in a dense sentence (hard). That uniformity is pure waste, and it's structural — it doesn't go away with a bigger GPU or a better model, only with a smarter execution strategy.
- The academic groundwork is public and credible (DeepMind's Mixture-of-Depths, Google's CALM) — the ideas are de-risked scientifically. What's missing is productization: nobody has shipped an easy-to-adopt way for a team running open-weight models to get this benefit without becoming a systems research lab themselves.

---

## 3. The real technical risk (say this in the YC interview, don't hide it)

**Heterogeneous-depth batching.** In production serving, many requests' tokens are processed together in one batch for GPU efficiency. If token A exits after layer 3 and token B needs all 12, you can't just "skip work" for A inside a batched matmul without either (a) padding/masking waste that erodes the savings, or (b) re-architecting the serving loop to dynamically regroup tokens by remaining depth mid-sequence — which is exactly the kind of systems work vLLM/TensorRT-LLM/SGLang do at the kernel level, and exactly why Mixture-of-Depths and CALM stayed papers instead of becoming products.

This is the fork in the road for the company:
- **Solve it well → real defensibility.** A kernel/scheduler that makes depth-heterogeneous batching actually fast is hard, is genuine IP, and is not something a customer can trivially replicate.
- **Don't solve it → feature, not company.** If the honest wall-clock number ends up close to zero after batching overhead, this becomes a paper-worthy footnote, not a business — and a serving-engine team could absorb the routing idea as one more optimization flag in an existing open-source engine.

This is the single most important thing to de-risk before spending real conviction (or a YC application) on this. Don't skip it to get to a nicer pitch deck faster.

---

## 4. Product surface options (pick one wedge, don't try to be all of these)

| Option | What it is | Who buys it | Pros | Cons |
|---|---|---|---|---|
| **A. Open-source retrofit library + hosted "auto-tune" service** | OSS tool that takes a HF checkpoint + your data, fine-tunes a router onto it, outputs a depth-adaptive checkpoint. Hosted version does the fine-tune-and-benchmark for you. | AI infra/ML platform teams at startups with meaningful inference spend | Cheapest path to distribution (OSS credibility), matches how vLLM/Neural Magic grew | Monetization is the hard part of pure OSS-first plays; must have a real hosted upsell |
| **B. Serving-engine plugin** | A plugin/patch for vLLM or TensorRT-LLM that adds depth-adaptive execution as a serving-time feature | Same buyer, but integrates into infra they already run | Rides existing adoption, lowest integration friction for customer | You're one PR away from the host project shipping this natively and erasing your wedge |
| **C. Hosted inference API** | You run inference yourself, cheaper than competitors because of the technique, compete directly on $/token | Teams who want an API, not infra to run themselves | Direct monetization, clearest story | You're now competing with Together/Fireworks/Groq on their home turf — capital-intensive, not a research-driven moat by itself |
| **D. Enterprise optimization engagement** | Bespoke: come in, retrofit + tune their specific model + workload, guarantee a cost reduction % | Large enterprises running open-weight models on-prem/VPC for compliance reasons | High price per deal, real ACV | Services-heavy, doesn't scale like software, harder VC story |

**Recommendation for a YC application specifically:** start as **A**, because it's the cheapest way to get a credible, public, reproducible benchmark (which doubles as your traction story) before you've raised anything, and it naturally produces design partners for a future paid tier. B is a plausible pivot once you have a benchmark people trust. C and D are later-stage, not YC-application-stage.

---

## 5. Business model (once past A's OSS phase)

- **Usage-based**: % of measured inference cost savings, or $/million tokens processed through the optimized path — ties price directly to delivered value, easiest to sell against a CFO's inference bill.
- **Platform/licensing fee**: flat fee for the optimization SDK + support, for enterprises that won't do usage-based metering for compliance/procurement reasons.
- Avoid pure seat-based pricing — this isn't a per-user tool, it's an infra cost lever, and pricing should read that way to buyers.

---

## 6. Competitive landscape (be precise, don't hand-wave)

- **Inference-optimization incumbents**: vLLM, TensorRT-LLM, SGLang (continuous batching, KV-cache tricks, speculative decoding) — none of these currently ship depth-adaptive routing as a first-class feature. That's the opening, and also the acquirer/absorber risk.
- **Quantization-first companies** (e.g., Neural Magic-style plays): different lever (numerical precision, not compute path), largely complementary — a real adaptive-depth product should stack with quantization, not compete with it.
- **Hosted inference platforms** (Together, Fireworks, Groq, Baseten): potential customers/partners for A/B, potential competitors if you go C.
- **The research itself** (Mixture-of-Depths, CALM): public, unpatented, not productized by their authors as far as is known. The differentiation has to be engineering execution (the batching problem in §3) and a credible, reproducible benchmark — not the routing idea itself, which isn't novel or defensible on its own.

---

## 7. Path from current prototype to an MVP worth showing design partners

Concrete, in order — this is real ML/systems engineering work, not a rebrand of what exists:

1. **Retrofit pipeline**: take one real open-weight model (start small — a 1B-3B class open model, not Llama-70B, to keep iteration cheap) and build the router-only (or router+LoRA) fine-tuning procedure that adds depth-adaptive routing without materially degrading base-model quality on standard evals (MMLU or similar, not just WikiText perplexity — WikiText loss alone won't be credible to a buyer).
2. **Cross-domain robustness check**: verify the router generalizes beyond the fine-tuning corpus — a router that only knows how to skip layers on WikiText-shaped text is a demo, not a product. Needs eval on code, chat-style data, at minimum.
3. **Real serving-time execution**: this is where §3's batching problem has to be confronted directly, not deferred again. Even a rough first version (e.g., group same-depth-bucket tokens within a batch rather than solving the fully general case) that produces a *measured* wall-clock number on a real GPU (A100/H100) is the single most important artifact for a pitch — theoretical FLOPs savings will not be believed by an infra buyer or a YC partner.
4. **Benchmark + writeup**: the accuracy-vs-avg-layers curve (already have the method from this project) run on the retrofitted real model, plus $/million-tokens and real latency numbers, published openly. This is simultaneously the credibility artifact and the YC-application traction artifact.
5. **Design partners**: 2-3 teams with real inference spend, willing to try the retrofit on their own model/workload and report back a real number. This is what turns "interesting research" into "traction" for an application.

---

## 8. What to say in a YC application (draft framing, not final copy)

- **One-liner**: "We cut LLM inference cost by teaching existing models to spend less compute on easy tokens and more on hard ones — a drop-in retrofit for models teams already run, not a new model to train."
- **Why now**: inference spend scales with usage and never stops, unlike training spend; teams are actively hunting for cost levers; the research technique is proven in papers but nobody has shipped the systems work to make it real.
- **Why us**: [Saif's specific technical credibility from actually having built the routing mechanism, the ablation, and the honest benchmark methodology from scratch — this is real, defensible ML engineering depth, not a "we read the paper" pitch.]
- **Traction ask**: a real, reproducible benchmark on an open-weight model plus at least one design partner's measured number, not the WikiText prototype alone.

---

## 9. Open questions / decisions still needed from Saif

- [ ] Which product surface (A/B/C/D in §4) to actually commit to — recommend A, but this is Saif's call.
- [ ] Which open-weight base model to use for the first retrofit attempt (needs to be small enough to iterate on with Colab/Kaggle-class compute).
- [ ] How much of the current from-scratch research (baseline, sweep, ablation) is worth finishing as-is for the interview-prep/portfolio story, versus redirecting remaining project time toward the retrofit pipeline once the ablation runs land.
- [ ] Whether to treat the batching/wall-clock problem (§3) as a go/no-go gate before any further product conversation, or to keep it as parallel-track research risk.

---

## 10. Where this stands right now

This is a strategy discussion, not a decision. Nothing here has been built. The research project's own next steps (finishing the `fixed_topk`/`confidence_threshold` ablation runs — see `memory.md` Session 7) are unaffected by this document and remain the active lab work unless Saif decides to redirect.
