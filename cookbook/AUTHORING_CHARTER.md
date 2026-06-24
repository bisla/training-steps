# Authoring Charter — The Domain Fine-Tuning Handbook

This charter governs every chapter of the book. It is **injected verbatim into every
writer agent and every editor agent**. New or rewritten chapters are not "done" until an
editor records a PASS on all seven acceptance items below.

---

## Mission

**The Domain Fine-Tuning Handbook teaches a Python developer with zero machine-learning
background to fine-tune, align (preference & RL), and continually improve a small domain
model — practically and runnably — using a memory-extraction model as the running example.**

The book is intuition-first and relentlessly practical. The reader should always be able to
copy a code block, run it, and see something work. Theory exists only to make the next
command make sense.

### The progressive arc (every chapter advances it; never breaks it)

1. **Quickstart (Ch0):** get a *working* fine-tuned model this afternoon for under $30.
2. **Parts 0–2:** the mental models and setup that explain *why* the speedrun worked.
3. **Parts 3–6:** do it properly — data, SFT training, evaluation, deployment.
4. **Part 7 — Preference & RL:** go beyond imitation; make the model *prefer* better answers.
5. **Part 8 — Continuous Learning:** run it as a living system that improves over many rounds.

Reader profile: writes Python, has never trained a neural net, has felt "LoRA / loss curve /
gradient" not click. No math prerequisites. A single consumer GPU or a rented cloud instance
is enough.

---

## Framing: memory extraction is ONE example of domain fine-tuning

Domain fine-tuning means teaching a small model a *skill specific to your world*. The book
uses **memory extraction** (read a conversation → emit structured facts) as its running
example, but it is one of many. When motivating *why* a reader would fine-tune, draw on this
broader set so the reader sees the technique, not just the one task:

- **Company / org "brain"** — extract decisions, owners, and commitments from meeting and
  Slack transcripts into structured records.
- **Personal recall / "dream" feature** — turn a person's notes and chats into durable,
  queryable memories for a personal assistant.
- **Domain extraction** — pull structured fields from clinical notes, contracts, support
  tickets, or research papers.
- **Any reliable structured-output task** — where prompting a big general model is too
  inconsistent, too slow, or too expensive to run at scale.

The *running code* always stays on memory extraction (one consistent schema, below). Use the
other examples only to build motivation and intuition, and flag clearly when you do.

---

## The running example — pinned schema and system prompt (DO NOT DRIFT)

Every code example uses this exact memory schema and system prompt. Copy them verbatim; do
not invent variant field names or a different prompt.

```python
# A single memory object:
{
    "text": "Sarah prefers dark roast coffee in the morning",   # the fact, as a complete sentence
    "type": "preference",                                        # one of: preference | fact | decision | relationship
    "entities": ["Sarah"]                                        # named people, places, or things involved
}
# The model's output is a JSON array of zero or more such objects ([] is valid).
```

```python
SYSTEM_PROMPT = """You are a memory extraction assistant. Your job is to read a conversation and extract every memorable piece of information as a list of atomic memory objects.

Each memory object must follow this exact JSON schema:
{
  "text": "<the fact, written as a complete, standalone sentence>",
  "type": "<one of: preference | fact | decision | relationship>",
  "entities": ["<list of named people, places, or things involved>"]
}

Rules:
- One fact per memory object. Do not bundle multiple facts into one.
- Write "text" as a sentence someone could read without any surrounding context.
- If there are no memorable facts in the conversation, return an empty list: []
- Return ONLY a valid JSON array. No explanation, no markdown fences, no extra text.
"""
```

Training rows use TRL's conversational format: `{"messages": [{"role": "system", ...},
{"role": "user", ...}, {"role": "assistant", ...}]}`. The system prompt used in training
MUST be reused verbatim at inference.

---

## Pinned library versions (write all code against these)

The book's code is authored and tested against this frozen snapshot. Pin these in
`code/requirements.txt`. Do **not** write code from memory of older tutorials — verify
import paths and signatures against these versions.

```
unsloth==2026.6.9
trl==1.6.0
transformers   # version compatible with trl 1.6.0 (verify with `pip show trl`)
peft
datasets
accelerate
bitsandbytes
vllm           # for fast GRPO generation and serving
```

### API facts that are easy to get wrong (verified live against TRL 1.6.0)

**The #1 gotcha — `processing_class`, NOT `tokenizer`.** In modern TRL, every trainer
constructor takes `processing_class=tokenizer`, not the old `tokenizer=tokenizer` kwarg.
This applies to `RewardTrainer`, `DPOTrainer`, and `GRPOTrainer`. (Unsloth's `SFTTrainer`
path in earlier chapters still shows `tokenizer=`; the new RL chapters must use
`processing_class=`.)

- **SFT:** `from trl import SFTTrainer, SFTConfig`; `from unsloth import FastLanguageModel`;
  `train_on_responses_only` masks prompt tokens. (Already used throughout Parts 3–4.)
- **Reward model:** `from trl import RewardTrainer, RewardConfig` — train an
  `AutoModelForSequenceClassification` (num_labels=1). Constructor:
  `RewardTrainer(model=..., args=RewardConfig(...), train_dataset=..., processing_class=tokenizer, peft_config=...)`.
  Expects preference data (`chosen`/`rejected`). `RewardConfig.max_length` default 1024. ✅ runnable.
- **DPO:** `from trl import DPOTrainer, DPOConfig`; dataset columns `prompt`/`chosen`/
  `rejected`. Constructor: `DPOTrainer(model=..., ref_model=None, args=DPOConfig(...),
  train_dataset=..., processing_class=tokenizer, peft_config=...)`. `DPOConfig.beta` default
  0.1; `loss_type` default `"sigmoid"`. Unsloth: `from unsloth import PatchDPOTrainer;
  PatchDPOTrainer()` before constructing. ✅ runnable.
- **GRPO:** `from trl import GRPOTrainer, GRPOConfig`. Constructor:
  `GRPOTrainer(model=..., reward_funcs=<callable or list>, args=GRPOConfig(...),
  train_dataset=..., processing_class=tokenizer)`. `reward_funcs` signature is
  `fn(completions, **kwargs) -> list[float]` (extra dataset columns arrive as kwargs; a list
  of funcs is weighted by `GRPOConfig.reward_weights`). Config defaults to verify/teach:
  `num_generations=8`, `max_completion_length=256`, `beta=0.0` (KL coefficient — **0 means no
  KL penalty by default**, raise it to stay near the reference model), `use_vllm=False` (set
  True for fast generation). Unsloth: `from unsloth import FastLanguageModel, PatchFastRL;
  PatchFastRL("GRPO", FastLanguageModel)` + `FastLanguageModel.from_pretrained(...,
  fast_inference=True)`. ✅ runnable — the RL centerpiece. (Verify the exact Unsloth patch
  name against the installed `unsloth==2026.6.9`; if `PatchFastRL` is absent, GRPO works with
  a plain Unsloth-loaded PEFT model + the standard TRL `GRPOTrainer`.)
- **PPO — relocated to experimental; NOT a runnable hand-rolled loop.** In TRL 1.6.0,
  `PPOTrainer`, `PPOConfig`, and `AutoModelForCausalLMWithValueHead` are **not** top-level —
  they live in `trl.experimental.ppo` and importing them prints a `TRLExperimentalWarning`.
  The legacy `trainer.step(query_tensors, response_tensors, rewards)` manual loop is gone; the
  current `PPOTrainer` is `.train()`-based (policy/ref/value/reward models). The PPO chapter is
  **conceptual**: explain the full loop (value head, KL penalty, advantage), argue why PPO is
  impractical for this reader, and hand off to GRPO. Any code there is **explicitly labeled
  illustrative / not runnable**.
- **KTO / ORPO:** `KTOTrainer` is top-level (`from trl import KTOTrainer, KTOConfig`).
  **ORPO is experimental:** `from trl.experimental.orpo import ORPOTrainer, ORPOConfig`.
  Mention both briefly in the DPO chapter and the decision guide; do not build full runnable
  chapters on them.
- **RLOO:** `from trl import RLOOTrainer` also exists (a lighter online-RL alternative);
  worth a one-line mention in the decision guide alongside GRPO.

---

## "How much data" — defensible rules of thumb (present as ranges, not laws)

For a ~4B model, narrow structured-extraction task, LoRA/QLoRA:

| Stage | Rows | Notes |
|---|---|---|
| Proof-of-life | 200–500 | Speedrun lower bound; format gets learned |
| Solid baseline | 1,000–3,000 | Most projects live here; speedrun targets ~500–1,000 |
| Strong | 3,000–10,000 | Diversity/quality matters more than raw count |
| Diminishing returns | >~10,000 | Spend effort on quality/eval instead |

- Epochs: 2–3 for ≤2k rows, 1–2 for larger; stop on eval-loss plateau.
- Tokens rule of thumb: ~1–5M training tokens is ample for narrow LoRA.
- DPO pairs: 500–5,000 (start ~1,000). Reward model: a few thousand pairs.
- GRPO: 500–2,000 *prompts*, `num_generations` 4–8.
- Replay ratio (continual rounds): mix ~10–30% prior/general data (default ~20%); keep a
  frozen "canary" eval set to detect forgetting.
- Speedrun cost: ~$5–30 all-in (rented L4/A100 1–3 GPU-hrs + ~$1–5 teacher-API synth gen);
  free Colab T4 ≈ $0 compute.

Always present numbers as ranges with the *reason* behind them. Never state a single magic
number as if it were a law.

---

## Voice and format (match the existing chapters exactly)

Every chapter:
1. Opens with a 1–2 paragraph hook, then a **"What you'll learn"** bullet list, then a
   **"Concepts you need first"** section that explains prerequisites in plain English.
2. Introduces each new idea with an **analogy or story before any code or math**.
3. Uses **real, runnable code** with heavy inline comments and shown sample output.
4. Cross-references other chapters by name (e.g. *Ch17 - Watching Training*) instead of
   re-explaining.
5. Conversational, for-dummies-but-respects-your-intelligence tone. No naked equations.

### Voice exemplars (the target register)

> "Imagine you buy a medical textbook and it's perfect — except it knows nothing about your
> hospital's specific procedures. You have two options: (1) Reprint the entire textbook with
> your procedures woven in. Expensive. Slow. (2) Write your procedures on sticky notes and
> attach them to the relevant pages. Fast, cheap, and you can peel them off if you change
> your mind. LoRA is option two." — *Ch6*

> "Think of it like training a new employee. Reading every company document gives them
> background knowledge. But they only learn *how to do the job* when you show them real
> examples of tasks being done correctly." — *Ch12*

---

## Acceptance bar (the EDITOR scores PASS/FAIL on each; all 7 must PASS)

1. **Intuition before code** — every new concept opens with a plain-English analogy/story
   before any code or math; no naked equations.
2. **Practical & runnable** — code uses the real, pinned TRL/Unsloth APIs; imports shown;
   no pseudo-code *unless explicitly labeled* (the PPO chapter is the only sanctioned
   conceptual-code exception, and its code must say so).
3. **On-example** — uses the memory-extraction running example with the pinned schema
   `[{text, type, entities}]` and the pinned system prompt.
4. **De-branded** — no "Engram" product references anywhere; "domain fine-tuning" framing.
5. **Progressive** — assumes only earlier chapters; cross-references rather than re-explains;
   advances the arc (working → better → preference/RL → continual).
6. **Honest** — numbers given as ranges/rules-of-thumb with rationale; tradeoffs stated;
   API caveats (especially PPO) called out; any external quote (e.g. Karpathy) is verified
   or clearly attributed as paraphrase.
7. **Voice/format** — opens with "What you'll learn" + "Concepts you need first"; matches the
   exemplar register; heavy inline comments; sample outputs shown.

---

## WRITER prompt template (injected into each chapter-writing agent)

```
You are writing one chapter of "The Domain Fine-Tuning Handbook." The full Authoring Charter
is below — follow it exactly (mission, running example, pinned schema + system prompt, pinned
library versions, data rules of thumb, voice, and the 7-point acceptance bar).

CHAPTER: <number + title>
MUST COVER (from the approved plan):
  - <bullet>
  - <bullet>

Before writing, read these existing chapters for voice and continuity: <paths>.
Write the full chapter as a single Markdown file at <path>. Target ~3,500–5,000 words.
Rules:
- Verify every API call, signature, import path, and external quote — do NOT write from
  memory of old tutorials. Match the pinned versions (trl==1.6.0, unsloth==2026.6.9).
- Stay on the memory-extraction example with the pinned schema and system prompt.
- Intuition/analogy before code. Open with "What you'll learn" + "Concepts you need first."
- Cross-reference other chapters by name; don't re-explain them.
- For the PPO chapter ONLY: code is conceptual and must be labeled "illustrative, not runnable."

<full charter pasted here>
```

## EDITOR prompt template (injected into each review agent)

```
You are the editor for "The Domain Fine-Tuning Handbook." Review the chapter at <path>
against the 7-point acceptance bar in the Authoring Charter (below). For EACH of the 7 items,
return PASS or FAIL with specific, line-referenced reasons. Then give a prioritized fix list.

Hard checks you must perform:
- API correctness: every TRL/Unsloth import path and constructor signature matches trl==1.6.0
  / unsloth==2026.6.9. Flag any hallucinated or deprecated API (especially the old PPO
  .step() loop, which must NOT appear as runnable).
- De-brand: zero "Engram" product references.
- Example fidelity: pinned memory schema + system prompt used unchanged.
- Numbers: stated as ranges with rationale, consistent with the charter's table.
- Quotes (e.g. Karpathy): verified or clearly marked as paraphrase.

Return: per-item PASS/FAIL table, then the fix list. A chapter passes only if all 7 PASS.

<full charter pasted here>
```
