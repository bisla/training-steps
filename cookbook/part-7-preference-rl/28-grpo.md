# Chapter 28 - GRPO: Practical RL With Reward Functions

In *Ch27 - PPO and the Full RL Loop* you learned how the classic reinforcement-learning-from-human-feedback loop works — and why, for a developer with one GPU and a structured-output task, it is a beast you do not want to wrestle. Four models in memory at once. A value head you have to train. A reward model you have to collect preference data for. A KL penalty you have to babysit. The code in that chapter was explicitly *illustrative, not runnable*, because the honest answer is: you almost certainly should not hand-roll PPO.

This chapter is the runnable alternative. **GRPO — Group Relative Policy Optimization — is the RL method you will actually use.** It throws away the most painful piece of PPO (the value model), replaces it with a trick so simple you will wonder why it took the field years to land on it, and — thanks to Unsloth — runs on a single consumer GPU. And the reward "judge" is not some separately-trained neural network. It is the *same Python scoring function you already wrote in Ch25*: JSON validity, schema match, entity F1 against the gold answer. You graded the model's homework with code. GRPO turns that grade into a training signal.

By the end of this chapter you will have a complete, top-to-bottom script that takes your memory-extraction model and pushes it to extract memories *better* — not by showing it more correct answers (that was SFT, *Ch15*), and not by showing it pairs of better/worse answers (that was DPO, *Ch26*), but by letting it try, scoring its tries, and nudging it toward the good ones.

---

## What you'll learn

- The one-sentence intuition behind GRPO and why "the group average is the critic"
- Why GRPO is the practical RL choice for this book's reader, point by point against PPO
- How to load a 4-bit model + LoRA for GRPO with Unsloth (and how to check the right entry point for your installed version)
- How to turn the programmatic reward functions from *Ch25* into TRL `reward_funcs` — with the exact `fn(completions, **kwargs) -> list[float]` signature
- How to combine several reward functions with `GRPOConfig.reward_weights`
- The `GRPOConfig` fields that actually matter, with their **real** defaults: `num_generations=8`, `max_completion_length=256`, `beta=0.0`, `use_vllm`
- Why the GRPO training dataset is **just prompts** — no gold completions — and why that is a big deal
- VRAM and cost expectations, plus how to read the reward curve and run a before/after eval

---

## Concepts you need first

### The critic problem (a one-paragraph recap of Ch27)

PPO needs to know, for a given answer, "was that better or worse than what I'd normally expect here?" The thing that estimates "what I'd normally expect" is called the **value model** or **critic** — a second neural network, the same size as your policy, trained alongside it. It is the source of half of PPO's pain: extra VRAM, extra instability, extra things to get wrong. Hold onto that word — *critic* — because GRPO's entire idea is a clever way to not have one.

### Reward functions are just graders (recap of Ch25)

In *Ch25 - Rewards: Functions and Reward Models* you wrote plain Python functions that take a model output and return a number: higher = better. Does it parse as JSON? +1. Does every object match the `{text, type, entities}` schema? +1. How well do the extracted entities match the gold entities (F1)? Add that. Did it wrap the output in a markdown fence we told it not to? Small penalty. No neural network, no training — just code you can unit-test. GRPO consumes exactly these functions. If you skipped Ch25, the short version is: a reward function is a deterministic grader that returns a float, and you already know how to grade memory extraction because you built the whole eval harness in *Ch18*.

### "Group relative" in plain English

That is the whole idea, and the next section makes it concrete.

---

## The intuition: a classroom with no answer key

Imagine a teacher who wants to improve how a student writes a particular kind of essay, but has no answer key — only a rubric (a grader). Here is the PPO way and the GRPO way.

**The PPO way (with a critic):** The teacher hires a second person — a "predictor" — whose entire job is to look at an essay prompt and guess, in advance, what score the student is *likely* to get. The student writes one essay. The grader scores it. The teacher compares the actual score to the predictor's guess: "You scored a 7, but the predictor expected an 8 here — that's below par, write it differently next time." This works, but now you are paying two salaries (two models), and the predictor itself needs constant training to stay accurate. If the predictor is wrong, the whole signal is wrong.

**The GRPO way (group relative):** Fire the predictor. Instead, for each prompt, make the student write **eight essays** in one sitting. Grade all eight with the rubric. Now you do not need anyone to *guess* the expected score — you can just *measure* it: the average of those eight scores **is** the expected score for this prompt. Then the feedback is dead simple:

> *"Of your eight attempts, these three scored above the group average — do more of that. These three scored below — do less of that. The middle two were about par — leave them alone."*

The group's own average replaces the predictor. That is the entire trick. "Group Relative" = each answer is judged relative to the average of its sibling answers for the same prompt. **The critic is gone, replaced by arithmetic you get for free from sampling a group.**

A subtle but important consequence: GRPO never needs to know the *absolute* "right score." It only needs *within-group ranking*. If all eight attempts are mediocre, it still pushes toward the least-mediocre ones. If all eight are great, it barely nudges at all (they are all near the average, so the signal is near zero). The model is always being pulled toward "better than your own current average for this prompt," which is exactly what continuous improvement looks like.

> **Where the name comes from.** GRPO was introduced in the DeepSeekMath paper (Shao et al., 2024) and popularized by the DeepSeek-R1 work. The formal version computes a per-answer *advantage* by subtracting the group mean and dividing by the group standard deviation — a "z-score within the group." You do not need the equation. The intuition above *is* the equation, minus the notation.

---

## Why GRPO is the practical choice for you (vs PPO)

Lay the two methods side by side for our reader — one GPU, a 4–8B model, a structured-extraction task:

| | PPO (*Ch27*) | GRPO (this chapter) |
|---|---|---|
| Models in memory | Policy + **value head** + reference + reward model | Policy + reference only (reference is cheap/shareable) |
| Separate value network to train? | **Yes** — same size as the policy | **No** — the group average replaces it |
| Reward source | Usually a *trained* reward model (Ch25) | **Programmatic rewards from Ch25** — just Python |
| Preference data needed? | Yes, to train the reward model | **No** — your grader is code |
| Stability for a beginner | Fiddly; many ways to diverge | Much more forgiving |
| VRAM | Highest of all the methods | Dramatically lower, especially with Unsloth |
| Unsloth support | Limited | First-class, optimized |

Read the second and third rows together, because they compound. PPO's pain is not just the value head — it is that the value head sits *on top of* a reward model you also had to build. GRPO lets you skip **both**: no value network, and no reward model, because your reward is the Ch25 grader. You go from "train two extra neural networks and keep four models resident" to "load one model, write a Python function." That is the difference between an afternoon and a research project.

There is one honest tradeoff to name. GRPO samples a *group* of completions per prompt (the default is eight), so generation is the expensive part — you are generating 8× the tokens per training step. This is exactly why Unsloth's fast generation path (and optionally vLLM) matters so much here, and why we cover the VRAM math carefully below. It is a real cost, but it is *compute* you can rent cheaply, not *complexity* you have to debug.

> **Two lighter cousins, one line each (see the decision guide in *Ch29*).** `RLOO` (REINFORCE Leave-One-Out, `from trl import RLOOTrainer`) is a similar group-based online method worth knowing about. `KTO` (`from trl import KTOTrainer`) learns from single thumbs-up/thumbs-down labels instead of pairs. For this book's running example, GRPO is the workhorse; the others are alternatives, not prerequisites.

---

## The key mental shift: GRPO trains on PROMPTS, not answers

This trips up everyone coming from SFT and DPO, so we will say it plainly before any code.

- **SFT (*Ch15*)** needs `{prompt, gold answer}`. You show the model the right answer and it imitates.
- **DPO (*Ch26*)** needs `{prompt, chosen, rejected}`. You show the model a better and a worse answer and it learns the preference.
- **GRPO needs only `{prompt}`.** That's it. No gold answer in the training row at all.

Why can it get away with this? Because the model **generates its own answers during training** (the group of eight), and your **reward function judges them on the fly**. The "correct answer" knowledge lives inside the reward function, not inside the dataset rows.

For memory extraction, this is liberating. Your training data is just conversations — the *inputs* — formatted as chat prompts with the pinned system prompt. You do not need to have hand-labeled or teacher-generated the gold memory JSON for every one of them. The reward function does the grading.

> **A wrinkle worth flagging.** Some of our reward signals (like entity F1) *do* compare against a gold answer. When you have gold answers available, GRPO lets you carry them along as **extra dataset columns** — they arrive in your reward function as keyword arguments (we will see exactly how). So in practice: prompt columns are *required*; gold columns are *optional helpers* that ride along for any reward function that wants them. Pure format rewards (valid JSON? schema-conforming?) need no gold at all.

---

## Building the reward functions (reusing Ch25)

TRL's contract for a custom reward function — verified against `trl==1.6.0` — is:

```python
def reward_fn(completions, **kwargs) -> list[float]:
    ...
```

The trainer calls it with `completions` (the batch of generated answers) and passes **every extra column in your dataset as a keyword argument**. So if your dataset rows have a `gold_memories` column, your function receives `gold_memories=[...]` automatically. It must return one float per completion. A list of these functions is allowed, and their outputs are combined using `GRPOConfig.reward_weights`.

One format detail: when your prompts are *conversational* (a list of `{"role", "content"}` messages — which ours are), TRL hands each completion to your function as a list of message dicts too, e.g. `[{"role": "assistant", "content": "[...JSON...]"}]`. We pull the text out of the last message. (For plain-string prompts, completions are plain strings. The helper below handles both.)

We will lift the grading logic straight from *Ch25 / Ch18* and wrap it in three reward functions, then weight them.

```python
# rewards.py
# Reward functions for GRPO memory-extraction training.
#
# Each function follows TRL's contract EXACTLY:
#     def reward_fn(completions, **kwargs) -> list[float]
#
# - `completions` is a list, one entry per generated answer in the batch.
#     * conversational prompts -> each entry is a list of message dicts
#       e.g. [{"role": "assistant", "content": "<the JSON>"}]
#     * string prompts         -> each entry is a plain string
# - Any extra dataset columns (e.g. "gold_memories") arrive as kwargs,
#   each a list aligned with `completions`.
# - Return a list[float] of the same length as `completions`.
#
# These reuse the parsing/validation/F1 logic from Ch18's evaluate.py and
# the reward design from Ch25. Nothing here is new machinery.

import json
import re
from difflib import SequenceMatcher

VALID_TYPES = {"preference", "fact", "decision", "relationship"}


def _extract_text(completion) -> str:
    """Normalize a TRL completion (conversational OR string) into raw text."""
    if isinstance(completion, list):
        # Conversational format: take the content of the final message.
        return completion[-1]["content"]
    return completion  # already a plain string


def _parse(raw: str):
    """Best-effort parse to a list of memory dicts; None on failure.
    Same three-strategy approach as Ch18's parse_model_output."""
    raw = raw.strip()
    # Strategy 1: clean JSON array
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        pass
    # Strategy 2: strip a markdown code fence the model was told not to use
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if fence:
        try:
            data = json.loads(fence.group(1))
            return data if isinstance(data, list) else None
        except json.JSONDecodeError:
            pass
    # Strategy 3: grab the outermost [...] and try that
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw[start:end + 1])
            return data if isinstance(data, list) else None
        except json.JSONDecodeError:
            pass
    return None


# ── Reward 1: valid JSON array? ──────────────────────────────────────────────
# The most basic signal. We reward a clean parse and gently penalize output
# that needed fence-stripping or bracket-hunting to rescue. This is what stops
# the model from drifting back into prose or markdown.
def reward_json_valid(completions, **kwargs) -> list[float]:
    scores = []
    for c in completions:
        raw = _extract_text(c).strip()
        try:
            data = json.loads(raw)            # strict, clean parse
            scores.append(1.0 if isinstance(data, list) else -0.5)
        except json.JSONDecodeError:
            # Could we rescue it? Salvageable but not clean -> small penalty.
            scores.append(-0.5 if _parse(raw) is not None else -1.0)
    return scores


# ── Reward 2: schema conformance ─────────────────────────────────────────────
# Of the objects we CAN parse, what fraction obey the pinned schema:
#   keys text/type/entities present, type in VALID_TYPES, entities is a list.
# Returns a value in [0, 1]; an empty list [] is a legitimate, schema-valid
# answer and scores 1.0 (a conversation with nothing to remember).
def reward_schema(completions, **kwargs) -> list[float]:
    scores = []
    for c in completions:
        mem = _parse(_extract_text(c))
        if mem is None:
            scores.append(0.0)
            continue
        if len(mem) == 0:
            scores.append(1.0)            # empty list is valid
            continue
        good = 0
        for m in mem:
            if (isinstance(m, dict)
                    and {"text", "type", "entities"} <= set(m)
                    and m.get("type") in VALID_TYPES
                    and isinstance(m.get("entities"), list)):
                good += 1
        scores.append(good / len(mem))
    return scores


# ── Reward 3: entity F1 vs gold (uses the optional gold column) ──────────────
# This reward needs the gold answer, which rides along as a dataset column
# named "gold_memories". TRL passes it in as the kwarg `gold_memories`,
# a list aligned with `completions`. If it's absent (pure-prompt dataset),
# we return 0.0 contributions so this reward simply sits out.
def _entities_of(mem_list) -> set:
    ents = set()
    for m in mem_list or []:
        if isinstance(m, dict):
            for e in m.get("entities", []) or []:
                ents.add(str(e).strip().lower())
    return ents


def reward_entity_f1(completions, gold_memories=None, **kwargs) -> list[float]:
    if gold_memories is None:
        # No gold available in this dataset — contribute nothing.
        return [0.0] * len(completions)
    scores = []
    for c, gold in zip(completions, gold_memories):
        pred = _parse(_extract_text(c)) or []
        pred_ents = _entities_of(pred)
        gold_ents = _entities_of(gold)
        if not pred_ents and not gold_ents:
            scores.append(1.0)            # both empty: agree perfectly
            continue
        tp = len(pred_ents & gold_ents)
        precision = tp / len(pred_ents) if pred_ents else 0.0
        recall    = tp / len(gold_ents) if gold_ents else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        scores.append(f1)
    return scores
```

A quick sanity check before training is always worth it — reward bugs are silent and devastating (a function that returns the *same* score for every completion gives GRPO nothing to learn from, because every answer looks "average"):

```python
# Smoke-test the rewards on hand-written completions (conversational form).
good = [{"role": "assistant",
         "content": '[{"text": "Sarah prefers dark roast coffee in the morning", '
                    '"type": "preference", "entities": ["Sarah"]}]'}]
junk = [{"role": "assistant", "content": "Sure! Here are the memories: ..."}]

print(reward_json_valid([good, junk]))   # -> [1.0, -1.0]
print(reward_schema([good, junk]))       # -> [1.0, 0.0]
print(reward_entity_f1([good, junk],
      gold_memories=[[{"text": "x", "type": "fact", "entities": ["Sarah"]}],
                     [{"text": "y", "type": "fact", "entities": ["Sarah"]}]]))
# -> [1.0, 0.0]   (good matches the gold entity "sarah"; junk parses to nothing)
```

If the numbers vary across your test inputs, your grader has a usable gradient. If they are all identical, fix the reward before you waste a GPU-hour.

---

## The `GRPOConfig` fields that matter

Before the full script, here are the knobs you will actually touch. **Every default below was verified against the installed `trl==1.6.0`** (`GRPOConfig` dataclass defaults), not remembered from a blog post:

- **`num_generations=8`** — the size of the *group*. This is the "write eight essays" number. Larger groups give a more reliable average (a steadier learning signal) but multiply generation cost linearly. The charter's rule of thumb for this task is **4–8**; 8 is the default and a fine starting point, drop to 4 if you are VRAM- or time-constrained. (Constraint: your effective batch size must be divisible by `num_generations`.)
- **`max_completion_length=256`** — the token budget for each generated answer. Our memory JSON is typically 100–300 tokens, so 256 is snug; bump to 512 if you see answers getting truncated mid-array (truncated answers parse as invalid and get punished unfairly). Bigger means more generation cost.
- **`beta=0.0`** — the **KL-penalty coefficient**, and this default surprises people. KL is the leash that keeps the model from wandering too far from the *reference* model (its starting point). `beta=0.0` means **no leash by default** — TRL's current GRPO trusts the reward signal and lets the model move freely, which trains faster and is what recent recipes (à la DeepSeek-R1) found works well. **When should you raise it (e.g. `0.01`–`0.05`)?** If you watch the reward climb but your eval quality *degrades* — a classic sign of reward hacking, where the model games the grader (e.g. emits empty `[]` arrays because that reliably scores 1.0 on schema validity) and forgets how to actually extract. A small `beta` pulls it back toward the sensible model you started with. Start at `0.0`; reach for KL only if you see drift.
- **`use_vllm`** (default `False`) — set `True` to generate the group with vLLM, which is dramatically faster for the 8×-per-step sampling GRPO demands. With Unsloth's fast-inference path this is the recommended setting on a single GPU (vLLM runs *colocated* with training by default — `vllm_mode="colocate"`). Leave it `False` if vLLM is not installed or you hit a setup snag; training still works, just slower.

A few more you will see in the config but rarely change: `temperature=1.0` (sampling diversity within the group — you *want* variety here, unlike the `0.0` you use at inference), `scale_rewards="group"` (the z-score-within-group normalization described earlier — this is the "relative" in Group Relative), and `epsilon=0.2` (the PPO-style clip that prevents any single update from being too large). The defaults are sensible; do not touch them on your first run.

> **A note on naming.** This trl 1.6.0 `GRPOConfig` does **not** expose a `max_prompt_length` field, so do not set one — keep your prompts within `max_seq_length` instead. (Older tutorials reference it; it is not here. This is exactly the kind of thing the charter warns about — verify against the pinned version, never write from memory.)

---

## Loading the model for GRPO with Unsloth

You load the model almost exactly as in *Ch15* — 4-bit base + LoRA adapters — with two GRPO-specific touches. First, you want fast generation, because GRPO generates constantly. Second, recent Unsloth versions provide a GRPO patch that you should *check for at runtime* rather than assume.

**Verify the entry point against your installed `unsloth==2026.6.9`.** The charter notes the canonical pattern is:

```python
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
```

…but it also notes the patch name has moved around across releases. **So do not hard-code an assumption — detect it.** The block below tries `PatchFastRL`; if it is not importable in your installed build, it falls back to a plain Unsloth-loaded PEFT model fed to the standard TRL `GRPOTrainer`, and it **prints which path it took** so you (and your logs) know exactly what ran. Both paths are valid — `GRPOTrainer` works fine on an ordinary Unsloth PEFT model; `PatchFastRL` just wires in extra speed optimizations when present.

```python
# ── GRPO model loading with a version-safe Unsloth entry point ──────────────
# Unsloth MUST be imported before transformers/trl for its patches to take hold.
from unsloth import FastLanguageModel

# Try the dedicated GRPO patch. It exists in recent Unsloth builds; if your
# installed unsloth==2026.6.9 doesn't expose it, we fall back cleanly.
USING_PATCH_FAST_RL = False
try:
    from unsloth import PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)   # wires in Unsloth's GRPO fast path
    USING_PATCH_FAST_RL = True
    print("[load] Using Unsloth PatchFastRL('GRPO', ...) — fast GRPO path enabled.")
except ImportError:
    print("[load] PatchFastRL not found in this Unsloth build — falling back to "
          "a plain Unsloth-loaded PEFT model + standard TRL GRPOTrainer.")

MAX_SEQ_LENGTH = 2048
LORA_RANK      = 16

# fast_inference=True turns on Unsloth's vLLM-backed generation, which is what
# makes sampling a group of 8 completions per step affordable. If vLLM isn't
# installed, set this to False (and use_vllm=False in the config below).
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Qwen3-8B-bnb-4bit",  # same family as Ch15
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit   = True,            # QLoRA: frozen base in 4-bit (see Ch6)
    dtype          = None,            # auto: bfloat16 on Ampere+, float16 on T4
    fast_inference = True,            # Unsloth's vLLM generation path
    gpu_memory_utilization = 0.6,     # cap vLLM's share so training has room
)

# Attach LoRA adapters — only ~1% of params train; the 4-bit base is frozen.
model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_RANK,
    lora_alpha     = LORA_RANK * 2,   # alpha = 2 x rank, the Ch15 rule of thumb
    target_modules = "all-linear",
    bias           = "none",
    use_gradient_checkpointing = "unsloth",   # big VRAM saver for long context
    random_state   = 42,
)
print("Model + LoRA ready for GRPO.")
```

> **If `from_pretrained` rejects `fast_inference` or `gpu_memory_utilization`** in your build, drop those two kwargs and set `use_vllm=False` in the config. The training loop is identical; only generation speed changes. This is the same "verify against your installed version" discipline from *Ch15* — the load signature is the one place Unsloth releases differ most.

---

## The full GRPO training script

This is the centerpiece. It assumes `rewards.py` (above) is in the same directory and your prompt dataset lives at `data/splits/train.jsonl`.

```python
# train_grpo.py
# Improve a memory-extraction model with GRPO + programmatic rewards.
#
# Expected VRAM:  ~12-18 GB for an 8B model in 4-bit with num_generations=8,
#                 max_completion_length=256, Unsloth gradient checkpointing.
#                 Fits a single 24 GB card (RTX 3090/4090, A10) comfortably;
#                 tight-but-doable on 16 GB if you drop num_generations to 4.
# Expected time:  ~1-3 GPU-hours for ~500-1,000 prompts (generation-bound).
# Output:         data/grpo_adapter/  — a LoRA adapter, same format as Ch15.

import json
import torch

# ── 1. Load model (version-safe Unsloth entry point) ────────────────────────
from unsloth import FastLanguageModel

USING_PATCH_FAST_RL = False
try:
    from unsloth import PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)
    USING_PATCH_FAST_RL = True
    print("[load] Unsloth PatchFastRL('GRPO', ...) enabled.")
except ImportError:
    print("[load] PatchFastRL absent — plain Unsloth PEFT model + TRL GRPOTrainer.")

MAX_SEQ_LENGTH = 2048
LORA_RANK      = 16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Qwen3-8B-bnb-4bit",
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit   = True,
    dtype          = None,
    fast_inference = True,            # drop this (and use_vllm below) if no vLLM
    gpu_memory_utilization = 0.6,
)
model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_RANK,
    lora_alpha     = LORA_RANK * 2,
    target_modules = "all-linear",
    bias           = "none",
    use_gradient_checkpointing = "unsloth",
    random_state   = 42,
)

# ── 2. The pinned system prompt — identical to Ch15 / Ch18 training ─────────
# Reused VERBATIM. A drifted prompt here is the #1 cause of "GRPO made it worse."
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

# ── 3. Build the PROMPT-ONLY dataset ────────────────────────────────────────
# This is the big conceptual difference from SFT/DPO (see "key mental shift"):
# GRPO rows need a "prompt" column and NOTHING ELSE is required. The model will
# generate its own answers; the reward functions grade them.
#
# We carry an OPTIONAL "gold_memories" column too — it rides along as a kwarg
# into reward_entity_f1. If you have no gold answers, just omit it; the F1
# reward returns 0.0 and sits out, while the JSON/schema rewards still drive
# learning. (TRL applies the chat template to the "prompt" column itself when
# the prompt is conversational, so we hand it the message list directly.)
from datasets import load_dataset

def to_grpo_row(example):
    msgs = example["messages"]                 # [system, user, assistant]
    conversation = msgs[1]["content"]          # the user turn = the input
    # The "prompt" is the conversation in conversational form, WITHOUT the
    # assistant turn — that's what the model must produce.
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": conversation},
    ]
    # Optional gold answer (for the entity-F1 reward). Parse it from the
    # assistant turn we already have in our SFT-format files. If a row has no
    # usable gold, store an empty list.
    try:
        gold = json.loads(msgs[2]["content"])
        if not isinstance(gold, list):
            gold = []
    except (json.JSONDecodeError, IndexError, KeyError):
        gold = []
    return {"prompt": prompt, "gold_memories": gold}

raw = load_dataset("json", data_files={"train": "data/splits/train.jsonl"})["train"]
# Charter rule of thumb for GRPO: 500-2,000 prompts is plenty. More prompts
# helps coverage; it does NOT need to be huge because each prompt is "mined"
# num_generations times per pass.
dataset = raw.map(to_grpo_row, remove_columns=raw.column_names)
print(f"GRPO prompts: {len(dataset)}")
print("Sample prompt column keys:", dataset[0].keys())   # dict_keys(['prompt', 'gold_memories'])

# ── 4. Reward functions (from rewards.py) ────────────────────────────────────
from rewards import reward_json_valid, reward_schema, reward_entity_f1

# ── 5. GRPOConfig — defaults verified against trl==1.6.0 ────────────────────
from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    output_dir          = "data/grpo_adapter",
    # --- the GRPO-specific knobs (defaults shown; all real) ---
    num_generations     = 8,      # group size: "write 8 essays" (default 8; 4-8)
    max_completion_length = 256,  # token budget per generated answer (default 256)
    beta                = 0.0,    # KL coefficient. 0.0 = NO KL leash (default).
                                  # raise to ~0.01-0.05 if eval drops while reward
                                  # climbs (reward hacking — pull back to reference).
    use_vllm            = True,   # fast group generation; set False if no vLLM.
    temperature         = 1.0,    # want VARIETY within the group (default 1.0)
    # --- ordinary training knobs ---
    learning_rate       = 1e-6,   # RL likes a small LR; this is the trl default
    per_device_train_batch_size = 8,   # must be divisible by num_generations
    gradient_accumulation_steps = 1,
    num_train_epochs    = 1,      # one pass is often enough; each prompt is
                                  # sampled num_generations times anyway
    max_steps           = 300,    # or cap by steps for a first run
    logging_steps       = 1,      # log reward EVERY step — you want the curve
    save_steps          = 100,
    max_grad_norm       = 0.1,    # tight clip keeps RL stable
    optim               = "adamw_8bit",
    bf16                = torch.cuda.is_bf16_supported(),
    fp16                = not torch.cuda.is_bf16_supported(),
    # --- COMBINE multiple rewards: one weight per reward function, in order ---
    # Final reward per completion = 4*json + 2*schema + 4*entity_f1.
    # Heavier weight on the things you care about most. Format first (the model
    # must produce valid JSON before content scoring is even meaningful),
    # content close behind.
    reward_weights      = [4.0, 2.0, 4.0],
    seed                = 42,
)

# ── 6. Construct the trainer ─────────────────────────────────────────────────
# CRITICAL: processing_class=tokenizer, NOT tokenizer=tokenizer.
# (Modern TRL trainers use processing_class. The old `tokenizer=` kwarg is gone
#  for the RL trainers — using it raises a TypeError. See the charter's #1 gotcha.)
# reward_funcs is a LIST here; their outputs are weighted by reward_weights above.
trainer = GRPOTrainer(
    model            = model,
    reward_funcs     = [reward_json_valid, reward_schema, reward_entity_f1],
    args             = config,
    train_dataset    = dataset,
    processing_class = tokenizer,    # <-- the one everyone gets wrong
)

print(f"Trainer ready. PatchFastRL active: {USING_PATCH_FAST_RL}")
print(f"Group size (num_generations): {config.num_generations}")
print(f"Reward weights: {config.reward_weights}")

# ── 7. Train ─────────────────────────────────────────────────────────────────
# You'll see a per-step log. The number to WATCH is `reward` — it should trend
# up. `kl` will be ~0 because beta=0.0 (no KL term). `completion_length` tells
# you whether answers are getting truncated against max_completion_length.
print("Starting GRPO training…")
trainer.train()

# ── 8. Save the improved adapter (same format as Ch15) ──────────────────────
model.save_pretrained("data/grpo_adapter")
tokenizer.save_pretrained("data/grpo_adapter")
print("GRPO adapter saved to data/grpo_adapter/")
```

### What the training log looks like

GRPO's per-step log is different from SFT's. There is no `eval_loss` to chase — you watch the **reward** climb. With `logging_steps=1` you will see something like:

```
{'loss': 0.0, 'reward': 3.91, 'reward_std': 2.04, 'kl': 0.0, 'completion_length': 142.0, 'epoch': 0.01}
{'loss': 0.0012, 'reward': 4.55, 'reward_std': 1.88, 'kl': 0.0, 'completion_length': 138.5, 'epoch': 0.03}
{'loss': 0.0019, 'reward': 5.62, 'reward_std': 1.61, 'kl': 0.0, 'completion_length': 131.0, 'epoch': 0.05}
{'loss': 0.0027, 'reward': 7.10, 'reward_std': 1.24, 'kl': 0.0, 'completion_length': 129.5, 'epoch': 0.08}
...
{'loss': 0.0041, 'reward': 8.83, 'reward_std': 0.71, 'kl': 0.0, 'completion_length': 127.0, 'epoch': 0.30}
```

How to read it (this is your *Ch17 - Watching Training* skill, applied to RL):

- **`reward` trends up.** This is the whole game. Ours can range roughly 0 → 10 given the weights `4*json + 2*schema + 4*f1`. A steady climb means the policy is learning to satisfy the graders. A flat line from step 1 means something is wrong — usually a reward function returning a constant (no gradient) or a prompt-format mismatch making *every* answer fail.
- **`reward_std` shrinks over time.** Early on, the eight group members disagree wildly (some great, some garbage) → high std → strong learning signal. As the model converges, the group clusters near the top → low std → the signal naturally fades. That is healthy.
- **`loss` is small and not very meaningful.** Unlike SFT, GRPO's "loss" is just the policy-gradient surrogate; do not read tea leaves in it. Watch reward.
- **`kl` is ~0** because `beta=0.0`. If you raise `beta`, this becomes the leash tension — non-trivial KL means the model is being held near the reference.
- **`completion_length`** drifting toward `max_completion_length` (256) is a warning: answers may be truncating. Ours settling around 127–142 tokens is exactly the 100–300 range we expect for memory JSON. Good.

> **Reward-hacking smell test.** If `reward` rockets to its ceiling in a handful of steps and `completion_length` collapses toward a tiny number, suspect the model discovered that emitting `[]` scores well on JSON+schema for free. That is when you (a) check whether your dataset is too heavy on empty-answer conversations, and (b) raise `beta` a touch to keep it honest. Climbing-but-believable beats instant-max every time.

---

## VRAM and cost: why Unsloth changes the math

The headline tension with GRPO is generation: `num_generations=8` means you generate eight answers per prompt per step. Naively that is brutal. Unsloth makes it tractable in two ways — its `use_gradient_checkpointing="unsloth"` slashes the activation memory of long-context training, and its vLLM-backed `fast_inference` path generates the group far faster and *colocated* on the same GPU as training (no second card, no separate server).

Rough, honest ranges for the running example (an 8B model in 4-bit, single GPU) — treat these as starting expectations, not guarantees, since they swing with sequence length and group size:

| Setup | VRAM (approx.) | Notes |
|---|---|---|
| 8B, 4-bit, `num_generations=4`, `max_completion_length=256` | ~12–14 GB | Fits a 16 GB card if you keep batch small |
| 8B, 4-bit, `num_generations=8`, `max_completion_length=256` | ~14–18 GB | Comfortable on a 24 GB card (3090/4090/A10) |
| 8B, 4-bit, `num_generations=8`, `max_completion_length=512` | ~18–22 GB | Longer answers; watch `gpu_memory_utilization` for vLLM |

Two levers if you are tight on VRAM: drop `num_generations` to 4 (halves generation memory and time, slightly noisier average), and lower the vLLM share via `gpu_memory_utilization` so the trainer keeps enough room. Going to 4-bit and Unsloth checkpointing is what brings long-context GRPO down from "needs an A100" to "runs on a gaming GPU."

**Cost.** GRPO is generation-bound, so wall-clock — and therefore rental cost — is dominated by how many `prompts × num_generations × max_completion_length` tokens you generate. For a first run of ~500–1,000 prompts, one epoch, expect **~1–3 GPU-hours**, which on a rented A10/A100 at well under $1–2/hr lands in the **single-digit-dollars** range — comfortably inside the book's "~$5–30 per round" envelope from *Ch0*. There is no teacher-API cost here at all, because your reward is code, not an LLM judge. (You *could* use an LLM-as-judge reward — the Ch18 judge wrapped as a `reward_fn` — but then you pay per-call API costs and lose determinism; for memory extraction the programmatic rewards are cheaper and sharper.)

---

## Before & after: did GRPO actually help?

GRPO produces a LoRA adapter in the **exact same format** as *Ch15*, which means your *Ch18* evaluation harness already knows how to score it. No new eval code — just point `evaluate.py` at the new adapter and compare against the pre-GRPO model.

```bash
# Score the SFT model and the GRPO-improved model on the SAME held-out test set,
# using the SAME metrics from Ch18 (parse-rate, schema, P/R/F1, optional judge).
python code/evaluate.py \
    --test_file   data/splits/test.jsonl \
    --finetuned   data/sft_adapter \
    --base_model  unsloth/Qwen3-8B \
    --max_examples 100 --skip_judge

python code/evaluate.py \
    --test_file   data/splits/test.jsonl \
    --finetuned   data/grpo_adapter \
    --base_model  unsloth/Qwen3-8B \
    --max_examples 100 --skip_judge
```

A typical, *honest* before/after on this task (your numbers will vary with data quality and how much room the SFT model left to improve):

```
Metric            SFT only (Ch15)     after GRPO (Ch28)
------------------------------------------------------------
Parse rate        96.0%               99.0%
Schema valid      94.0%               98.0%
Avg Precision     0.812               0.851
Avg Recall        0.788               0.829
Avg F1            0.800               0.840
```

What to expect, stated plainly: GRPO's biggest, most reliable wins on a structured task are on the things your reward functions *directly* measure — **parse-rate and schema validity climbing toward 100%**, because you literally rewarded valid JSON every step. Content metrics (F1) improve more modestly, driven by the entity-F1 reward. A **few points of F1 and a near-elimination of format failures is a realistic, worth-it result** — not a miracle. If your SFT model was already at 96% parse / 0.80 F1, do not expect 0.95 F1; expect the format failures to nearly vanish and content to tick up. If GRPO makes a metric *worse*, that is your cue to inspect for reward hacking and consider a small `beta` (see above).

> **The honest tradeoff, one more time.** GRPO is not magic content knowledge — it cannot teach the model facts it never saw in SFT. It is a *polishing* and *preference-shaping* step: it makes the model reliably do more of what your graders reward and less of what they punish. If your model is fundamentally missing a capability, the fix is better SFT data (*Ch13*), not more RL.

---

## Recap

- **GRPO drops the value model.** For each prompt it samples a *group* of answers, grades them with your reward function(s), and pushes toward the above-average ones and away from the below-average ones. The group average *is* the critic — that is the whole idea.
- **It is the practical RL choice** for this book's reader: one policy + a cheap reference, no value head, no separately-trained reward model (reuse the *Ch25* programmatic rewards), more stable than PPO, and Unsloth-optimized for a single GPU.
- **Reward functions follow `fn(completions, **kwargs) -> list[float]`.** Extra dataset columns (like `gold_memories`) arrive as kwargs. Pass a list of them via `reward_funcs=[...]` and weight them with `GRPOConfig.reward_weights`.
- **The dataset is just prompts** — a `prompt` column, no gold completions required. The model generates its own answers; the reward grades them. Optional gold columns ride along for rewards that want them.
- **Config defaults that matter (verified against trl 1.6.0):** `num_generations=8` (group size, 4–8), `max_completion_length=256`, `beta=0.0` (no KL leash by default — raise it only if reward climbs while eval drops), `use_vllm` (set `True` with Unsloth fast-inference for affordable group sampling).
- **Construct with `processing_class=tokenizer`**, not the legacy `tokenizer=` kwarg.
- **Watch the reward curve, not the loss.** Reward should trend up; `reward_std` should shrink; `completion_length` should stay clear of the truncation ceiling.
- **Costs ~$5–30 / 1–3 GPU-hours** for a 500–1,000-prompt run on a single 24 GB card; Unsloth's checkpointing + vLLM generation are what make long-context GRPO that cheap.
- **Evaluate with the *Ch18* harness unchanged.** Expect format metrics to climb toward 100% and content F1 to improve modestly — a real, honest win, not a miracle.

## Next

*Ch29 - Choosing Your Method* — a decision guide that puts everything in Part 7 side by side, so you can pick the right tool (and the right *order* — SFT then preference/RL) for your own project, with RLOO, KTO, and ORPO placed on the map.
