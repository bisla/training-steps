# Plan — Rebrand + Expand the Fine-Tuning Cookbook

## Context

`cookbook/` is a ~107k-word, 27-chapter book that currently teaches **only one-shot SFT**
(Unsloth + TRL `SFTTrainer`) for a memory-extraction task, branded **"Engram."** We are
turning it into a broader, progressive, practical handbook:

1. **De-brand** "Engram" → **"The Domain Fine-Tuning Handbook"**, and reframe the
   memory-extraction task as *one example* of domain fine-tuning (alongside org/company
   "brain", a "dream"/recall feature, etc.) rather than a product.
2. **Add a cheap (<$30) front-loaded Quickstart** so a reader gets a *working* fine-tuned
   model in one afternoon, then **progressively** layers on sophistication (better data →
   eval → preference/RL → continual system).
3. **Add Part 7 — Preference & RL** (reward modeling, DPO, GRPO; PPO treated *conceptually*
   as "why not PPO, why GRPO"; decision guide).
4. **Add Part 8 — Continuous Learning as a System** (loop architecture, data curation,
   how-much/how-often, catastrophic forgetting, production ops).
5. **Re-thread** existing Ch1, Ch3, Ch13, Ch22, Ch23, Appendix A.
6. Add **`AUTHORING_CHARTER.md`** (mission + acceptance bar + WRITER/EDITOR templates) and
   build new chapters via a **writer → editor → fix** agent loop.

**Outcome:** a single de-branded book that takes a zero-ML Python dev from a $20 working
model to preference/RL and a production continual-learning system, intuition-first and
fully runnable, anchored on the memory-extraction example.

**Fixed decisions (do not re-litigate):** the six goals above, the running example, the
de-brand target name, intuition-before-code voice, real TRL/Unsloth APIs.

---

## Decisions locked from clarifying questions

- **PPO:** No runnable hand-rolled PPO loop. The PPO chapter explains the full RL loop
  (value head / KL / advantage) *conceptually*, argues **why PPO is impractical** for this
  reader (two extra models, reward-model training, instability, cost), and positions
  **GRPO as the practical answer**. Weave in **Karpathy's framing**: SFT = imitation; RL =
  the model *discovering* its own strategies ("the research/exploration value"), plus his
  caution that RLHF "is barely RL." (Writer must verify the exact Karpathy source/quote.)
- **Running example:** Keep memory extraction as the spine, but Ch1 reframes it as one of
  several domain-FT use cases. Remove the product name "Engram"; refer to it as
  "the memory extractor" / "our memory-extraction model."
- **Synthetic data:** Expand existing **Ch13** (don't add a redundant chapter); intuition
  on *how much data* lives partly here (initial dataset sizing) and partly in new **Ch32**
  (ongoing loop sizing/replay). Keep the boundary explicit to avoid duplication.
- **Quickstart:** A **separate front "Chapter 0" speedrun** in its own `quickstart/` dir,
  before Part 0 — no renumbering of existing chapters.

---

## API reality check (verified via web research, late-2025/2026 TRL + Unsloth)

Treat exact version numbers/dates from research as **unconfirmed**; the durable facts:

- **`RewardTrainer` (+ `RewardConfig`)** — current/stable. Wraps a
  `AutoModelForSequenceClassification` reward model. ✅ runnable.
- **`DPOTrainer` (+ `DPOConfig`)** — current/stable. Dataset columns `prompt`/`chosen`/
  `rejected`; key knobs `beta`, `loss_type`. ✅ runnable.
- **`GRPOTrainer` (+ `GRPOConfig`)** — current/stable. `reward_funcs` is a callable (or
  list) with signature `fn(completions, **kwargs) -> list[float]` (dataset columns arrive
  as kwargs); `num_generations` controls group size. Unsloth integrates via
  `PatchFastRL("GRPO", FastLanguageModel)` + vLLM fast generation. ✅ runnable — **this is
  the RL centerpiece.**
- **PPO — CHANGED, do NOT trust old tutorials:** the legacy manual loop
  (`AutoModelForCausalLMWithValueHead` + `trainer.step(query_tensors, response_tensors,
  rewards)`) is **deprecated/relocated**; the modern `PPOTrainer` is `.train()`-based
  (policy/ref/value/reward models). Because we treat PPO conceptually, **code in the PPO
  chapter is illustrative/annotated pseudocode, explicitly not claimed runnable** — point
  readers to the GRPO chapter for the runnable path.
- **`KTOTrainer` / `ORPOTrainer`** — exist, more experimental; cover briefly in the DPO
  chapter and the decision guide, not as full runnable chapters.
- **Unsloth + TRL:** `FastLanguageModel.from_pretrained(... load_in_4bit=True,
  fast_inference=True)` → `get_peft_model(...)` → `PatchFastRL` → standard TRL trainer.
  `PatchDPOTrainer()` for DPO.

**Mandatory mitigation:** the implementation **pins exact versions** in a new
`code/requirements.txt` (trl, unsloth, transformers, peft, datasets, vllm, bitsandbytes).
Every RL code block is written against those pins, and the **EDITOR must verify import
paths/signatures against the pinned version, not from memory.**

---

## Pinned "how much data" numbers (defensible ranges — present as rules of thumb, not laws)

For a **~4B model, narrow structured-extraction task, LoRA/QLoRA**:

| Stage | Rows | Notes |
|---|---|---|
| Proof-of-life (signal) | **200–500** | Speedrun lower bound; enough to see format learned |
| Solid baseline | **1,000–3,000** | Speedrun target ~500–1,000; most projects live here |
| Strong | **3,000–10,000** | Diversity/quality matters more than count |
| Diminishing returns | **>~10,000** | For one narrow task; spend effort on quality/eval instead |

- **Epochs:** 2–3 for ≤2k rows, 1–2 for larger; stop on eval-loss plateau (cross-ref Ch17).
- **Tokens rule of thumb:** ~1–5M training tokens is ample for narrow LoRA (rows × ~500–1.5k tok).
- **DPO preference pairs:** 500–5,000; start ~1,000. **Reward model:** similar, a few thousand pairs.
- **GRPO:** 500–2,000 *prompts*, `num_generations` 4–8 each (no gold completions needed).
- **Replay ratio (continual rounds):** mix **~10–30% prior/general data** with new data
  (default ~20%); even 1–10% replay sharply reduces forgetting. Keep a **frozen general
  "canary" eval set** to detect drift.
- **Cadence:** retrain when you accumulate enough *new* high-quality rows (~few hundred–1k)
  or on a fixed schedule (weekly/monthly), **gated by eval**.
- **Speedrun cost:** rented L4 (~$0.5–0.8/hr) or A100 (~$1–2/hr) × 1–3 GPU-hrs + teacher-API
  synth gen for ~1k rows (~$1–5) → **~$5–30 all-in**; free Colab T4 path ≈ $0 compute.

---

## Final structure (ordered)

```
Quickstart (NEW, front)        quickstart/00-speedrun.md
Part 0  (edit Ch1, Ch3)        existing
Parts 1–5                      existing (Ch13 expanded in Part 3)
Part 6  (edit Ch22, Ch23)      existing
Part 7  NEW Preference & RL    part-7-preference-rl/24..29
Part 8  NEW Continual System   part-8-continual-learning/30..34
Appendices (expand A)          existing
```

---

## Execution checklist

### Phase 0 — Scaffolding (do first; no prose yet)
- [ ] Create `cookbook/AUTHORING_CHARTER.md` (mission, acceptance bar, WRITER + EDITOR
      templates — see spec below).
- [ ] Create dirs: `cookbook/quickstart/`, `cookbook/part-7-preference-rl/`,
      `cookbook/part-8-continual-learning/`.
- [ ] Create `cookbook/code/requirements.txt` with pinned versions (writer verifies latest
      compatible pins at build time).

### Phase 1 — De-brand (mechanical, repo-wide)
- [ ] `build/book.toml`: title/authors → "The Domain Fine-Tuning Handbook".
- [ ] `build/build.sh`: title strings (lines ~3, 117–118, 166–167, 198–199) and output
      filenames `engram-cookbook.pdf/.epub` → `domain-finetuning-handbook.*`.
- [ ] `README.md` + `SUMMARY.md`: title, "running example: Engram" section, Ch23 title.
- [ ] Body refs to "Engram" the product → de-branded phrasing in: `01-why-finetune.md`,
      `22-serving-and-integration.md`, `23-toward-continual-learning.md`,
      `16-hyperparameters.md` (Ch23 cross-ref), `D-cost-time-and-checklist.md`.
- [ ] Verify zero remaining brand hits: grep `-i engram` across `cookbook/` returns only
      intentional/historical mentions (or none).

### Phase 2 — Author NEW chapters (writer → editor → fix loop, charter injected)

**Quickstart — `quickstart/00-speedrun.md` — "Chapter 0: Fine-Tune a Model This Afternoon for Under $30"**
Must cover:
- The whole loop end-to-end, copy-paste runnable: pick base (Qwen3-4B / Gemma 3 4B, 4-bit),
  generate ~500–1,000 synthetic memory-extraction rows with a teacher LLM, QLoRA SFT with
  Unsloth, quick eval (eyeball + a few metrics), and **serve it** (Ollama or a 10-line vLLM/
  FastAPI call).
- A concrete **cost/time budget** (GPU-hrs, teacher-API $, total ≈ $5–30; free Colab path).
- Explicit "this is the fast path; here's the deep dive" cross-refs into Parts 3–6.
- Sets up the **progressive arc**: working model now → better data/eval next → preference/RL →
  continual system. Intuition-first, minimal theory, every command runnable.

**Part 7 — Preference & RL** (`part-7-preference-rl/`)
- [ ] **Ch24 `24-why-preference.md` — "Beyond Imitation: Why Preference and RL"**
  - SFT teaches *imitation of one right answer*; many tasks have *better/worse* answers.
  - Intuition via the running example: SFT nails JSON format but you want to *prefer* more
    complete, faithful, non-hallucinated extractions.
  - Karpathy framing: RL lets a model *discover* strategies beyond the demos (the
    "research"/exploration value); his "RLHF is barely RL" caution. (Verify source.)
  - Map of the toolbox to come: reward model → DPO → PPO(concept) → GRPO.
- [ ] **Ch25 `25-reward-modeling.md` — "Rewards: Functions and Reward Models (`RewardTrainer`)"**
  - Two kinds of reward: cheap **programmatic reward functions** (JSON-valid? schema match?
    entity overlap / F1 vs gold?) vs a **learned reward model**.
  - Runnable: build a programmatic reward for memory extraction; train a small reward model
    with `RewardTrainer`/`RewardConfig` on preference pairs. Real, pinned API.
- [ ] **Ch26 `26-dpo.md` — "DPO: Learning From Preference Pairs (`DPOTrainer`)"**
  - Intuition: skip the separate reward model + RL loop; learn directly from
    chosen/rejected. `beta` intuition.
  - Runnable: build `prompt/chosen/rejected` pairs for the running example; `DPOTrainer` +
    `DPOConfig` + Unsloth `PatchDPOTrainer()`. Brief, honest note on **KTO/ORPO** cousins.
- [ ] **Ch27 `27-ppo-why-not.md` — "PPO and the Full RL Loop: Why We Don't Use It Here"**
  - Conceptual walkthrough of the classic loop: policy, **reference model + KL penalty**,
    **value head**, **advantage/GAE**, reward model.
  - Honest case **against** PPO for this reader: 3–4 models in memory, reward-model training,
    instability, cost, brittleness; modern TRL deprecated the hand-rolled loop.
  - Karpathy's RL value vs RLHF caveats reprised. **Annotated illustrative code only —
    explicitly NOT runnable;** points to Ch28.
- [ ] **Ch28 `28-grpo.md` — "GRPO: Practical RL With Reward Functions"** ← runnable star
  - Intuition: GRPO drops the value model; scores a *group* of sampled answers and pushes
    toward the better ones — simpler, cheaper, stable.
  - Fully runnable: Unsloth `PatchFastRL("GRPO", ...)` + `FastLanguageModel(fast_inference=True)`
    + `GRPOTrainer`/`GRPOConfig`; `reward_funcs(completions, **kwargs)->list[float]` reusing
    Ch25's programmatic rewards; `num_generations`; VRAM/cost notes.
- [ ] **Ch29 `29-choosing-a-method.md` — "Choosing Your Method"**
  - Decision guide/table: **SFT vs DPO vs KTO/ORPO vs GRPO vs PPO** across data you have,
    cost, stability, when-to-reach-for-it; concrete recommendation path for the running
    example (SFT first → DPO if you have pairs → GRPO if you have a good reward signal).

**Part 8 — Continuous Learning as a System** (`part-8-continual-learning/`)
- [ ] **Ch30 `30-the-loop-architecture.md` — "The Continual Learning Loop as a System"**
  - Architecture: collect → select/curate → (re)label → train → eval-gate → canary → deploy
    → monitor → repeat. Expands Ch23's teaser into real component design + interfaces.
- [ ] **Ch31 `31-data-selection-curation.md` — "Selecting and Curating Data"**
  - **Dedup** (exact + near-dup/embedding), **quality scoring**, **hard-example mining**,
    **importance sampling**; runnable utilities; what to keep vs drop.
- [ ] **Ch32 `32-how-much-how-often.md` — "How Much Data, How Often"**
  - **Rows vs tokens**, **~4B scaling** intuition, **cadence**, **replay mix ratios** (use
    the pinned numbers above); when to retrain; budget framing. (Boundary: Ch13 sizes the
    *initial* dataset; Ch32 sizes the *ongoing loop*.)
- [ ] **Ch33 `33-catastrophic-forgetting.md` — "Catastrophic Forgetting Over Many Rounds"**
  - What it is, how to *measure* it (frozen canary set), how to *mitigate* (replay,
    lower LR, LoRA isolation, regularization); worked multi-round example.
- [ ] **Ch34 `34-production-ops.md` — "Production Ops: Monitoring, Versioning, Gating, Rollback"**
  - Monitoring/observability, **dataset + adapter versioning**, **eval gating / canary**
    deploys, **rollback**; ties back to serving (Ch22).

### Phase 3 — Re-thread existing chapters
- [ ] **Ch1 (`01-why-finetune.md`):** de-brand; reframe memory extraction as *one* domain-FT
      use case (give 3–4 examples: company/org brain, personal recall/"dream" feature,
      domain extraction, structured-output tasks); set up the progressive arc (Quickstart →
      SFT → preference/RL → continual system); point to Chapter 0.
- [ ] **Ch3 (`03-landscape-when-to-use-what.md`):** add a **post-training axis**
      (SFT vs preference/RL) and a **continual-learning axis** to the landscape/decision
      matrix and tradeoff table.
- [ ] **Ch13 (`13-synthetic-data-generation.md`):** expand — deeper generation pipeline,
      teacher prompts, diversity/coverage, plus **how-much-data intuition for the initial
      dataset** (the SFT rows in the table above) and cost.
- [ ] **Ch22 (`22-serving-and-integration.md`):** ensure hosting/serving is concrete and
      current (Ollama, vLLM, TGI, cloud endpoints) with cost notes; align with the
      Quickstart's serve step and Ch34's rollback/canary.
- [ ] **Ch23 (`23-toward-continual-learning.md`):** de-brand title; rewrite ending as a
      **bridge into BOTH Part 7 (preference/RL) and Part 8 (system)**, not a standalone teaser.
- [ ] **Appendix A (`A-glossary.md`):** add terms — preference optimization, reward function,
      reward model, RLHF, DPO, KTO, ORPO, PPO, GRPO, RLOO, KL divergence, advantage/GAE,
      value head, policy model, reference model, preference pair, replay buffer/replay ratio,
      importance sampling, hard-example mining, canary/gating, rollout. (Catastrophic
      forgetting likely already present — extend it.)

### Phase 4 — Wire the build & index
- [ ] `build/build.sh` **CHAPTERS**: insert `quickstart/00-speedrun.md` first; append the six
      Part-7 and five Part-8 files in order (before appendices). Keep order == TOC.
- [ ] `build/build.sh` **CONTENT_DIRS**: add `quickstart part-7-preference-rl part-8-continual-learning`.
- [ ] `SUMMARY.md`: add a `# Quickstart` section + `# Part 7 — Preference & RL` +
      `# Part 8 — Continuous Learning as a System` with chapter links (Ch24–34).
- [ ] `README.md`: add Quickstart + Part 7 + Part 8 to the TOC tables; update the running-
      example section to the reframed multi-use-case description.

### Phase 5 — Build & verify
- [ ] Run `cookbook/build/build.sh html` (and `epub`); confirm no "chapter not found"
      warnings and that new chapters render in order.
- [ ] Re-grep `-i engram`; confirm clean.
- [ ] Spot-check: every new RL code block imports from the pinned versions and matches the
      verified signatures (DPO/GRPO/Reward runnable; PPO clearly marked illustrative).

---

## `AUTHORING_CHARTER.md` spec (acceptance bar + reusable templates)

**Mission statement (top of file):** "The Domain Fine-Tuning Handbook teaches a Python dev
with zero ML background to fine-tune, align (preference/RL), and continually improve a small
domain model — practically and runnably — using the memory-extraction example as the spine."

**Acceptance bar (the EDITOR enforces; a chapter PASSES only if all hold):**
1. **Intuition before code** — every new concept opens with a plain-English analogy/story
   before any code or math; no naked equations.
2. **Practical & runnable** — code uses **real, current TRL/Unsloth APIs** against the
   pinned `requirements.txt`; imports shown; no pseudo-code *unless explicitly labeled*
   (the PPO chapter is the only sanctioned conceptual-code exception).
3. **On-example** — uses the memory-extraction running example (or an explicitly-flagged
   parallel use case), consistent schema `[{text, type, entities}]`.
4. **De-branded** — no "Engram" product references; "domain fine-tuning" framing.
5. **Progressive** — assumes only prior chapters; cross-references rather than re-explains;
   advances the arc (working → better → preference/RL → continual).
6. **Honest** — numbers given as ranges/rules-of-thumb with rationale; tradeoffs stated;
   API caveats (esp. PPO deprecation) called out.
7. **Voice/format** — matches existing chapters: opens with "What you'll learn" +
   "Concepts you need first"; conversational, for-dummies-but-respectful; heavy inline
   comments; sample outputs shown.

**WRITER template (injected into each chapter-writing agent):** role, the mission, the
chapter's title + "must cover" bullets (from this plan), the pinned-version list, 2–3
style exemplar passages from existing chapters, the running-example schema, and the rule
"verify every API/signature/quote — do not write from memory."

**EDITOR template (injected into each review agent):** the 7-point acceptance bar as a
checklist; must return PASS/FAIL per item with specific line-level fixes; explicitly must
verify TRL/Unsloth import paths and signatures against pinned versions and flag any
hallucinated API; for the PPO chapter, confirm conceptual code is labeled non-runnable.

---

## Verification

- **Build:** `cd cookbook/build && ./build.sh html && ./build.sh epub` — zero
  "chapter not found" warnings; HTML TOC shows Quickstart + Parts 7–8 in order.
- **De-brand:** `grep -ri engram cookbook/` returns nothing unintended.
- **Runnable code (sampled, in a GPU env if available):** the Quickstart script, the DPO
  (Ch26), GRPO (Ch28), and RewardTrainer (Ch25) snippets import and instantiate against the
  pinned versions without `ImportError`/signature errors. PPO (Ch27) snippets are clearly
  marked illustrative.
- **Editor gate:** each new/edited chapter has a recorded EDITOR PASS on all 7 acceptance
  items before it's considered done.
```
