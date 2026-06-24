# Chapter 26 - DPO: Learning Directly From Preference Pairs

Up to now, every model you have trained has learned by imitation. In *Ch15 - Your First Fine-Tune with Unsloth* you showed the model thousands of examples of conversations paired with correct memory JSON, and it learned to copy that behavior. Supervised fine-tuning (SFT) is teaching by demonstration: "here is the right answer, become more likely to produce it."

But demonstration has a ceiling. SFT can tell the model what a *good* answer looks like. It cannot easily tell the model what a *bad* answer looks like, or — more importantly — what makes one good-ish answer **better than another good-ish answer**. Both of these outputs might parse as valid JSON and both might look reasonable to a glance, yet one quietly drops a fact and the other captures everything. SFT treats every training answer as equally worth imitating. It has no notion of "this one is better than that one."

This is exactly the gap that preference learning fills, and **Direct Preference Optimization (DPO)** is the simplest, most reliable way to close it. This chapter teaches you how.

---

## What you'll learn

- The core intuition behind preference learning, and why DPO is the "lazy genius" version of it — it skips both the separate reward model *and* the reinforcement-learning loop
- What a preference pair is — `(prompt, chosen, rejected)` — and how to build a good one for memory extraction
- How to generate realistic "rejected" answers programmatically and with a teacher model, so chosen-vs-rejected teaches the *specific* failures you care about
- The one knob that matters most, `beta`, and how to reason about it in plain English
- A complete, runnable DPO training script using Unsloth + TRL 1.6.0, with the exact import paths and constructor signature
- How to run a quick before/after comparison so you can see DPO actually move the model
- An honest, brief tour of DPO's cousins (KTO and ORPO) and when you might reach for them instead

---

## Concepts you need first

### Why "make the right answer more likely" is not enough

When you trained with `SFTTrainer`, the model saw one target output per example and learned to raise the probability of that output. Picture the model's behavior as a landscape of possible answers, with hills where it likes to land. SFT builds a hill on top of the demonstrated answers. It does nothing about the *neighboring* hills — the plausible-but-wrong answers sitting right next to the good ones.

For memory extraction, the dangerous answers are not the obviously broken ones (those got filtered out in *Ch14 - Cleaning, Splitting, and Sanity-Checking Data*). The dangerous ones are the *near misses*: an extraction that gets four facts right and silently drops the fifth, or one that invents an entity name that was never said, or one that crams two facts into a single `text` field. These look fine. SFT, asked only "imitate the good answer," never learns to actively avoid them — because it was never shown the contrast.

Preference learning shows the contrast directly. For each prompt you provide **two** answers: a better one (`chosen`) and a worse one (`rejected`). The model learns the *direction* between them: push toward the chosen, away from the rejected.

### The two-and-a-half ways to do preference learning

There is a family of techniques here, and it helps to know where DPO sits before you write any code.

**The classic recipe (RLHF) has three stages.** First you do SFT (you already did this). Then you train a *separate* model — a **reward model** — whose only job is to look at an answer and output a score: how good is this? You train it on your preference pairs so it learns to score `chosen` higher than `rejected`. Then you run **reinforcement learning** (PPO or GRPO): the model generates answers, the reward model scores them, and an RL loop nudges the model toward higher-scoring answers. This is powerful but it is three moving parts — a reward model that can be miscalibrated, plus an RL loop that is fiddly and easy to destabilize. We covered the reward model in *Ch25 - Rewards: Functions and Reward Models*, cover PPO conceptually in *Ch27 - PPO and the Full RL Loop: Why We Don't Use It Here*, and GRPO — the RL centerpiece — in *Ch28 - GRPO: Practical RL With Reward Functions*.

**DPO collapses all of that into one training run.** This is the punchline of the whole chapter, so it is worth saying slowly:

> DPO skips the separate reward model **and** the reinforcement-learning loop. You hand it the preference pairs directly, and a single, ordinary-looking training loop — the same shape as the SFT loop you already ran — adjusts the model to make `chosen` more likely and `rejected` less likely.

There is no second model to train. There is no sampling-and-scoring loop. It is a clever piece of math (you do not need it) that proves: *training directly on the preference pairs with the right loss function gives you almost the same result as the full reward-model-plus-RL pipeline, for a fraction of the complexity.* That is why DPO became the default first thing people try after SFT.

### An analogy: the wine tasting

Imagine teaching someone to taste wine.

The **RLHF way** is to first hire a sommelier (train a reward model) who can score any glass from 1 to 100. Then you sit your student down and have them taste glass after glass, with the sommelier scoring each one and the student slowly adjusting their palate toward whatever scores high (the RL loop). Two experts, a lot of glasses, a lot of time, and if the sommelier has bad taste, the student inherits it.

The **DPO way** skips the sommelier entirely. You just put two glasses in front of the student — "this one is better than that one" — over and over, hundreds of pairs. The student learns the *difference* directly from the comparisons. No scoring scale, no expert in the loop. Just: this beats that, this beats that, this beats that.

DPO is the second way. The pairs *are* the lesson.

### Where DPO sits in your workflow

DPO is a **second course**, not a starting point. You SFT first to get a model that reliably produces valid memory JSON (*Ch15*). DPO is what you reach for *after* that, once your evaluation (*Ch18 - Evaluating Memory Extraction*) shows the model is structurally fine but makes consistent *quality* mistakes — dropped facts, occasional hallucinated entities, bundled facts. DPO is the tool for sanding down those specific rough edges.

You need a model that already works before DPO can make it work *better*. Do not skip SFT and start here.

---

## The `beta` knob: how hard to pull

DPO has exactly one hyperparameter you will actually think about, and it is `beta`. Get the intuition right and everything else is plumbing.

Here is the tension DPO is balancing. On one hand, you want the model to move toward your preferences — strongly prefer chosen, strongly avoid rejected. On the other hand, you do *not* want it to lurch so far that it forgets everything it learned during SFT and starts producing garbage. Remember, the model already produces valid JSON; you are refining it, not retraining it from scratch. DPO keeps a frozen copy of the original (called the **reference model**) and gently penalizes the training model for drifting too far from it.

`beta` controls how much that leash matters.

- **Low `beta` (e.g. 0.05):** a long leash. The model is allowed to move far from the original to satisfy the preferences. It learns the preferences aggressively, but it risks "over-optimizing" — chasing the preference signal so hard that it degrades on things the pairs do not cover.
- **High `beta` (e.g. 0.3–0.5):** a short leash. The model stays close to the original and only nudges toward preferences. Safer, more conservative, but slower to actually change behavior. If the leash is too short, DPO barely does anything.
- **`beta = 0.1`** is the default in TRL, and it is the right place to start for almost everyone. It is a moderate leash — enough freedom to learn the preferences, enough restraint to not wreck the base behavior.

A clean mental model: **`beta` is how confident you are that your preference pairs are right and complete.** If your pairs are gold and cover the full range of mistakes, you can afford a longer leash (lower `beta`) and let the model lean in. If your pairs are noisy or narrow, keep the leash short (higher `beta`) so a few bad pairs cannot drag the whole model off a cliff.

Start at `0.1`. If after training the model barely changed, try `0.05`. If it changed too much and started producing weird or degraded output, try `0.2` or `0.3`. That is the entire tuning story for `beta`.

---

## Building a preference dataset for memory extraction

This is where most of the real work lives. The DPO training code is short; the dataset is what determines whether DPO helps or hurts. Garbage pairs teach garbage preferences.

A DPO dataset has exactly three columns:

- **`prompt`** — the input. For us, this is the conversation to extract memories from, formatted with the chat template.
- **`chosen`** — the better answer. For us, the *complete, faithful* extraction: every fact captured, atomic, correctly typed, no invented entities.
- **`rejected`** — the worse answer. A *plausible-but-worse* extraction: still valid JSON, still looks reasonable, but exhibits one of the specific failures we want to train away.

The whole art is in the `rejected` column. A rejected answer that is obviously broken (not JSON, empty, nonsense) teaches the model nothing useful — it already avoids those after SFT. The rejected answer must be a **realistic near-miss**, the kind of mistake your model actually makes. That way the contrast teaches a real lesson.

### The failure modes worth targeting

From evaluation (*Ch18*), these are the four near-misses that matter most for memory extraction. Each becomes a recipe for generating a `rejected` answer from a known-good `chosen`:

1. **Missing fact** — drop one of the memory objects entirely. Teaches: capture *everything*.
2. **Hallucinated entity** — add a name to an `entities` list that never appeared in the conversation. Teaches: only name what was actually said.
3. **Bundled facts** — merge two atomic memories into one `text` field. Teaches: one fact per object.
4. **Wrong type** — relabel a `preference` as a `fact`, or a `decision` as an `event`. Teaches: type discipline.

### Generating rejected variants programmatically

The most reliable way to manufacture realistic near-misses is to start from your *known-good* extractions (the assistant outputs from your SFT data, which you already trust) and corrupt them in controlled ways. You know exactly what failure each rejected example represents, because you injected it.

```python
# build_dpo_pairs.py
# Turn known-good memory extractions into (prompt, chosen, rejected) preference pairs
# by programmatically corrupting the "chosen" answer into a realistic near-miss.

import json
import random
import copy

random.seed(42)  # reproducible corruptions

# The pinned system prompt — identical to Ch12/Ch13/Ch15 training and to inference.
# Reusing it verbatim is non-negotiable: the model must see the same dialect everywhere.
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

VALID_TYPES = ["preference", "fact", "decision", "relationship"]

# A pool of plausible-but-absent names to use for the "hallucinated entity" corruption.
# These are generic names that are very unlikely to appear verbatim in a conversation,
# which is exactly what makes adding them a hallucination.
FAKE_ENTITIES = ["David", "Priya", "Acme Corp", "Berlin", "the Q3 report"]


def corrupt_missing_fact(memories):
    """Drop one memory object at random. Needs at least 2 to leave something behind."""
    if len(memories) < 2:
        return None  # can't drop a fact and still have a meaningful extraction
    out = copy.deepcopy(memories)
    out.pop(random.randrange(len(out)))
    return out


def corrupt_hallucinated_entity(memories):
    """Add a name that never appeared in the conversation to one memory's entities."""
    if not memories:
        return None
    out = copy.deepcopy(memories)
    target = random.choice(out)
    fake = random.choice(FAKE_ENTITIES)
    if fake in target["entities"]:
        return None  # don't accidentally add a name that's already there
    target["entities"] = target["entities"] + [fake]
    return out


def corrupt_bundled_facts(memories):
    """Merge two atomic memories into one bundled object — breaks 'one fact per object'."""
    if len(memories) < 2:
        return None
    out = copy.deepcopy(memories)
    i, j = random.sample(range(len(out)), 2)
    a, b = out[i], out[j]
    # Join the two facts into a single run-on text and union their entities.
    bundled = {
        "text": f"{a['text'].rstrip('.')}, and {b['text'][0].lower() + b['text'][1:]}",
        "type": a["type"],
        "entities": list(dict.fromkeys(a["entities"] + b["entities"])),
    }
    # Remove the two originals (drop the higher index first to keep indices valid)
    for idx in sorted([i, j], reverse=True):
        out.pop(idx)
    out.insert(min(i, j), bundled)
    return out


def corrupt_wrong_type(memories):
    """Relabel one memory with a different (wrong) type."""
    if not memories:
        return None
    out = copy.deepcopy(memories)
    target = random.choice(out)
    wrong = random.choice([t for t in VALID_TYPES if t != target["type"]])
    target["type"] = wrong
    return out


CORRUPTIONS = [
    corrupt_missing_fact,
    corrupt_hallucinated_entity,
    corrupt_bundled_facts,
    corrupt_wrong_type,
]


def make_rejected(memories):
    """Try corruptions in random order until one succeeds (returns a changed list)."""
    for fn in random.sample(CORRUPTIONS, len(CORRUPTIONS)):
        candidate = fn(memories)
        if candidate is not None and candidate != memories:
            return candidate, fn.__name__
    return None, None  # this example can't be corrupted (e.g. a single-memory output)


def to_dpo_row(conversation_text, chosen_memories):
    """Build one {prompt, chosen, rejected} row in TRL's conversational DPO format.

    TRL accepts DPO data either as plain strings or as message lists. We use message
    lists so the trainer applies the chat template itself — this keeps the prompt
    formatting identical to how the model was trained and how it's served.
    """
    rejected_memories, kind = make_rejected(chosen_memories)
    if rejected_memories is None:
        return None  # skip examples we couldn't corrupt

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": conversation_text},
    ]
    # chosen / rejected are the assistant turn only — TRL pairs them with the prompt.
    chosen = [{"role": "assistant",
               "content": json.dumps(chosen_memories, ensure_ascii=False)}]
    rejected = [{"role": "assistant",
                 "content": json.dumps(rejected_memories, ensure_ascii=False)}]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected, "_corruption": kind}


# ── Build the file ────────────────────────────────────────────────────────────
# Input: the same gold extractions you used for SFT. Each line is a {messages: [...]}
# row from Ch12. We pull the user conversation and the trusted assistant JSON out of it.
def gold_rows(path):
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            msgs = row["messages"]
            user_msg = next(m["content"] for m in msgs if m["role"] == "user")
            asst_msg = next(m["content"] for m in msgs if m["role"] == "assistant")
            yield user_msg, json.loads(asst_msg)


pairs = []
for conversation_text, chosen_memories in gold_rows("data/splits/train.jsonl"):
    row = to_dpo_row(conversation_text, chosen_memories)
    if row is not None:
        pairs.append(row)

print(f"Built {len(pairs)} preference pairs.")
# Sample output:
#   Built 947 preference pairs.

# Quick sanity check: how balanced are the corruption types?
from collections import Counter
print(Counter(p["_corruption"] for p in pairs))
# Sample output:
#   Counter({'corrupt_missing_fact': 281, 'corrupt_bundled_facts': 246,
#            'corrupt_hallucinated_entity': 232, 'corrupt_wrong_type': 188})

with open("data/dpo/pairs.jsonl", "w") as f:
    for p in pairs:
        p.pop("_corruption")  # not needed by the trainer
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print("Wrote data/dpo/pairs.jsonl")
```

This gives you a few hundred to a few thousand pairs cheaply, with a known mix of failure types. Per the rules of thumb in *Ch13 - Generating a Synthetic Dataset*, **500–5,000 pairs is the useful range for DPO, and ~1,000 is a sensible starting point** — diversity of failure modes matters far more than raw count.

### Generating rejected variants with a teacher

Programmatic corruption is reliable but slightly artificial — the rejected answers are mechanically perfect except for the one injected flaw. Real model mistakes are messier. A complementary approach is to let a **teacher model** generate the rejected answer for you, by asking a strong general model to produce a *deliberately mediocre* extraction. (This is the same teacher-model idea you used to build synthetic SFT data in *Ch13*.)

The point is not to use the teacher's *best* answer — your gold `chosen` is already that. The point is to harvest realistic, organically-flawed near-misses.

```python
# teacher_rejected.py  (illustrative pattern — fill in your own client)
# Use a strong teacher model to produce a realistic, deliberately-mediocre extraction
# that becomes the "rejected" half of a preference pair.

import json
from anthropic import Anthropic  # the book's teacher is Claude; see Ch13 for setup

client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# This instruction asks the teacher to make ONE realistic mistake — not garbage.
# A rejected answer that's obviously broken teaches the model nothing it doesn't
# already know after SFT. We want a believable near-miss.
REJECTED_INSTRUCTION = """You will be given a conversation and a CORRECT memory extraction.
Produce a SLIGHTLY WORSE extraction that a careless model might output: still valid JSON,
still mostly right, but containing exactly ONE realistic mistake — for example, dropping a
single fact, bundling two facts into one object, mislabeling one type, or adding one entity
name that was not actually mentioned. Do NOT explain. Return ONLY the JSON array."""

def teacher_rejected(conversation_text, chosen_memories):
    """Ask the teacher for a believable worse extraction. Returns a list or None."""
    resp = client.messages.create(
        model="claude-opus-4-5",          # a strong teacher; see Ch13 for model choice
        max_tokens=1024,
        system=REJECTED_INSTRUCTION,
        messages=[{
            "role": "user",
            "content": (
                f"CONVERSATION:\n{conversation_text}\n\n"
                f"CORRECT EXTRACTION:\n{json.dumps(chosen_memories, ensure_ascii=False)}"
            ),
        }],
    )
    raw = resp.content[0].text.strip()
    try:
        rejected = json.loads(raw)
    except json.JSONDecodeError:
        return None  # teacher slipped — skip rather than feed bad data to DPO
    # Guard: a "rejected" identical to "chosen" is useless. Skip those.
    if rejected == chosen_memories:
        return None
    return rejected
```

> **Practical advice:** mix both sources. Programmatic corruptions guarantee coverage of each failure mode you care about; teacher-generated rejections add realistic, organic flaws. A 50/50 blend gives the model both targeted contrasts and natural ones. And always keep a small held-out set of *real* model outputs you have hand-labeled as worse — those are the truest rejected examples, and they make excellent eval material later.

---

## Training with DPO

Now the part that surprises people: after all that dataset work, the training script is barely longer than your SFT script. Same shape — load model, attach LoRA, configure, train — with TRL's `DPOTrainer` swapped in for `SFTTrainer`.

Two Unsloth-specific things to know up front:

- You call `PatchDPOTrainer()` **before** constructing the trainer. Unsloth patches TRL's DPO internals to be faster and more memory-efficient; the patch only takes effect if it runs first.
- You pass `ref_model=None`. The reference model (the frozen original — your leash anchor from the `beta` discussion) is normally a second copy of the model in memory. With a LoRA setup, TRL is smart enough to reconstruct the reference behavior by simply *disabling the adapter*, so it does not need a separate model. Passing `None` tells it to do exactly that, which roughly halves your memory use.

```python
# train_dpo.py
# Refine a memory-extraction model with Direct Preference Optimization.
# Run this AFTER you have an SFT adapter from Ch15. DPO sharpens it; it does not
# replace SFT.
#
# Expected VRAM:  ~9-11 GB for a 7B model in 4-bit (a bit more than SFT — two
#                 forward passes per step, one with the adapter, one without).
# Expected time:  ~20-40 minutes for ~1,000 pairs x 1 epoch on an A100.
# Output:         data/dpo_adapter/  — a new LoRA adapter, preference-tuned.

import torch

# ── 1. Imports ──────────────────────────────────────────────────────────────
# unsloth must be imported before transformers/trl so its patches apply.
# PatchDPOTrainer() rewrites TRL's DPOTrainer with Unsloth's faster kernels.
# Call it BEFORE you import or construct DPOTrainer.
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()

# DPOTrainer / DPOConfig are top-level in TRL 1.6.0 (unlike ORPO, which is
# experimental — see the "cousins" section at the end).
from trl import DPOTrainer, DPOConfig

from datasets import load_dataset

# ── 2. Configuration ─────────────────────────────────────────────────────────
SFT_ADAPTER_PATH = "data/adapter"      # the adapter you trained in Ch15
OUTPUT_DIR       = "data/dpo_adapter"  # where the DPO-tuned adapter is saved
DPO_DATA_PATH    = "data/dpo/pairs.jsonl"
MAX_SEQ_LENGTH   = 2048

BETA             = 0.1    # the leash. 0.1 is the TRL default — start here.
LORA_RANK        = 16     # match what you used in SFT for a clean continuation
LORA_ALPHA       = 32
LEARNING_RATE    = 5e-6   # DPO uses a MUCH smaller LR than SFT (SFT used 2e-4).
                          # You are nudging an already-good model, not training one
                          # from scratch — small steps avoid wrecking it.
NUM_EPOCHS       = 1      # 1 epoch is usually enough for DPO; 2 at most.
BATCH_SIZE       = 2
GRAD_ACCUM       = 4      # effective batch size = 8

# ── 3. Load the SFT model and re-attach LoRA ─────────────────────────────────
# We load the SFT adapter as our starting point: DPO refines the SFT model.
# Loading the adapter path pulls the base model + your SFT weights together.
print("Loading SFT model…")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = SFT_ADAPTER_PATH,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit   = True,
    dtype          = None,   # auto-detect (bfloat16 on Ampere+)
)

# Attach fresh trainable LoRA adapters for the DPO phase. (As in Ch15, the base
# stays frozen in 4-bit; only these adapter weights move during DPO.)
model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_RANK,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = 0.0,            # 0 is standard for DPO
    target_modules = "all-linear",
    bias           = "none",
    use_gradient_checkpointing = "unsloth",
    random_state   = 42,
)

# ── 4. Load the preference dataset ───────────────────────────────────────────
# Each row has prompt / chosen / rejected (the columns DPOTrainer expects).
print("Loading preference pairs…")
dataset = load_dataset("json", data_files=DPO_DATA_PATH, split="train")
print(f"Preference pairs: {len(dataset)}")
# Sample output:
#   Preference pairs: 947

# ── 5. Configure DPO ─────────────────────────────────────────────────────────
# DPOConfig wraps the same TrainingArguments you know, plus DPO-specific knobs.
dpo_config = DPOConfig(
    output_dir                  = OUTPUT_DIR,
    beta                        = BETA,        # default 0.1 — the leash strength
    # loss_type defaults to "sigmoid" (the original DPO loss). Leave it unless
    # you have a specific reason to change it. "sigmoid" is the right default.
    num_train_epochs            = NUM_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    learning_rate               = LEARNING_RATE,
    lr_scheduler_type           = "cosine",
    warmup_ratio                = 0.1,
    bf16                        = torch.cuda.is_bf16_supported(),
    fp16                        = not torch.cuda.is_bf16_supported(),
    logging_steps               = 10,
    optim                       = "adamw_8bit",
    seed                        = 42,
    max_length                  = MAX_SEQ_LENGTH,  # max length of prompt + answer
    max_prompt_length           = 1024,            # cap on the prompt portion
)

# ── 6. Build the trainer ─────────────────────────────────────────────────────
# THE GOTCHA: it is `processing_class=tokenizer`, NOT `tokenizer=tokenizer`.
# Modern TRL renamed this argument. The old kwarg will raise a TypeError.
#
# ref_model=None: with a LoRA setup, TRL reconstructs the reference model by
# disabling the adapter, so no second model copy is loaded. This halves memory.
trainer = DPOTrainer(
    model           = model,
    ref_model       = None,                 # use the adapter-disabled model as reference
    args            = dpo_config,
    train_dataset   = dataset,
    processing_class = tokenizer,           # NOT tokenizer=  ← the #1 mistake
)

# ── 7. Train ─────────────────────────────────────────────────────────────────
# DPO logs are different from SFT. The numbers to watch:
#   rewards/chosen     — the model's "implicit reward" for chosen answers (want ↑)
#   rewards/rejected   — same for rejected answers (want ↓)
#   rewards/margins    — (chosen − rejected). This is the headline number.
#                        It should climb steadily above 0. That IS learning.
#   rewards/accuracies — fraction of pairs where chosen > rejected (want → 1.0)
print("Starting DPO training…")
trainer.train()
# Sample output (early → late):
#   {'loss': 0.6931, 'rewards/chosen': 0.01,  'rewards/rejected': -0.00,
#    'rewards/margins': 0.01, 'rewards/accuracies': 0.55, 'epoch': 0.08}
#   {'loss': 0.4123, 'rewards/chosen': 0.38,  'rewards/rejected': -0.71,
#    'rewards/margins': 1.09, 'rewards/accuracies': 0.91, 'epoch': 0.85}
# Loss starts near 0.693 (= ln 2 — the model is initially 50/50 on the pairs)
# and falls as margins grow. Rising margins + rising accuracy = DPO working.

# ── 8. Save the preference-tuned adapter ─────────────────────────────────────
print(f"Saving DPO adapter to {OUTPUT_DIR}…")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done.")
```

### Reading the DPO logs

SFT had one number to watch: loss going down. DPO gives you a richer dashboard, and you should learn to read it because it tells you *whether the preferences are being learned* — not just whether some loss is shrinking.

- **`rewards/margins`** is the headline. It is `rewards/chosen − rewards/rejected` — how much more the model now prefers the good answer over the bad one. It starts near zero and should climb steadily. A rising margin is the single clearest sign DPO is doing its job.
- **`rewards/accuracies`** is the fraction of pairs where the model correctly scores chosen above rejected. It starts around 0.5 (a coin flip) and should rise toward 0.9+.
- **`loss`** starts near **0.693** (that is `ln 2`, exactly what you get when the model is 50/50 on every pair) and falls as the margins grow.

If margins stay flat near zero, DPO is not learning — usually because your `beta` is too high (leash too short) or your pairs do not actually differ in a learnable way (chosen ≈ rejected). If margins explode and accuracy hits 1.0 in the first few steps, your pairs may be *too* easy (rejected is obviously broken) and you are not teaching the subtle lesson you wanted.

---

## A quick before/after eval

Numbers in the training log are necessary but not sufficient. The real test is whether the model behaves better on the running example. Run the *same* conversation through the SFT model and the DPO model and look at the outputs side by side. (For the full evaluation methodology — schema validity, fact recall, entity precision — see *Ch18 - Evaluating Memory Extraction*; here we just want a quick eyeball.)

```python
# before_after.py
# Compare the SFT model and the DPO model on one fresh conversation.

import json
from unsloth import FastLanguageModel

MAX_NEW_TOKENS = 512

# Same pinned system prompt as everywhere else.
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

# A conversation neither model trained on. It deliberately contains several
# distinct facts (easy to drop one) and named people (easy to hallucinate or miss).
TEST_CONVERSATION = """User: My sister Lena just started a new job at a fintech startup in Lisbon.
Assistant: Congratulations to her! How's she finding it?
User: She loves it, though she misses our weekend hiking trips. We used to go every Saturday.
User: I'm thinking of flying out to visit her next month. I hate flying, but for her I'll do it.
"""

def extract(adapter_path):
    """Load an adapter, run the test conversation, return the raw model output."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = adapter_path, max_seq_length = 2048,
        load_in_4bit = True, dtype = None,
    )
    FastLanguageModel.for_inference(model)   # enable Unsloth fast inference (see Ch15)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": TEST_CONVERSATION},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to("cuda")
    out = model.generate(
        inputs, max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

print("── SFT model (before DPO) ─────────────────────────────")
print(extract("data/adapter"))

print("\n── DPO model (after DPO) ──────────────────────────────")
print(extract("data/dpo_adapter"))
```

A representative result. The SFT model produces valid JSON but commits exactly the near-misses DPO was trained to fix — here it bundles the hiking habit with the relationship, and drops the "hates flying" preference entirely:

```json
// SFT model (before DPO) — bundled fact, one fact missing
[
  {"text": "The user's sister Lena started a new job at a fintech startup in Lisbon, and they used to go hiking every Saturday.",
   "type": "fact", "entities": ["Lena", "Lisbon"]},
  {"text": "The user is thinking of visiting Lena next month.",
   "type": "decision", "entities": ["Lena"]}
]
```

```json
// DPO model (after DPO) — atomic, complete, correctly typed
[
  {"text": "The user's sister Lena started a new job at a fintech startup in Lisbon.",
   "type": "fact", "entities": ["Lena", "Lisbon"]},
  {"text": "The user and Lena used to go hiking every Saturday.",
   "type": "relationship", "entities": ["Lena"]},
  {"text": "The user dislikes flying.",
   "type": "preference", "entities": []},
  {"text": "The user is planning to visit Lena next month.",
   "type": "decision", "entities": ["Lena"]}
]
```

That is the DPO payoff in one picture: same structurally-valid model, but the preference pairs taught it to stop bundling and stop dropping facts. Run this on a held-out set of 50–100 conversations and score it properly with the *Ch18* harness before declaring victory — a single example is an anecdote, not an evaluation. Watch in particular for the failure mode where DPO improves on the targeted mistakes but quietly *regresses* somewhere else (this is the "over-optimization" you tuned `beta` against). If recall went up but, say, the model now over-extracts trivia, raise `beta` and retrain.

---

## DPO's cousins: KTO and ORPO (and when to care)

DPO is the default, but it is not the only preference method, and you will see two cousins mentioned constantly. Here is the honest, one-paragraph-each version. The full decision guide — which method for which situation — is *Ch29 - Choosing Your Method*; this is just enough to recognize the names and know they exist.

**KTO (Kahneman-Tversky Optimization).** DPO needs *pairs*: for each prompt, a chosen and a rejected answer side by side. That is often annoying to collect — real feedback usually arrives as isolated thumbs-up or thumbs-down on a single answer, not as neat comparisons. KTO learns from exactly that: **binary, unpaired feedback.** Each example is one answer plus a label, "good" or "bad," and you do not need a matching counterpart. If your production system logs thumbs-up/thumbs-down on individual extractions, KTO can consume that log directly with no pairing step. The API is top-level in TRL — `from trl import KTOTrainer, KTOConfig` — and the trainer shape mirrors DPO's. Reach for KTO when your feedback is naturally unpaired; see *Ch29* for the tradeoffs.

**ORPO (Odds Ratio Preference Optimization).** DPO assumes you already did SFT — it refines an existing model and needs a reference model to stay anchored to. ORPO's pitch is to **fold the preference signal directly into the SFT stage**, so you do SFT and preference learning in a *single* pass, with **no separate reference model** at all (it is "reference-free"). That is appealing when you want one training run instead of two, and it saves the memory a reference model would cost. The catch: in TRL 1.6.0, **ORPO lives in the experimental namespace** — `from trl.experimental.orpo import ORPOTrainer, ORPOConfig` — and importing it prints a `TRLExperimentalWarning`. Treat it as promising-but-less-settled than DPO. Reach for it when you want to merge SFT and alignment into one step; again, *Ch29* covers when that is worth it.

Both are real, both are supported, and both solve a specific inconvenience of DPO rather than beating it outright. For most readers, on most projects, **DPO first** is the right call — it is the most battle-tested, the data format is the most intuitive, and it slots cleanly after the SFT you already did.

---

## Recap

- **SFT teaches by imitation; DPO teaches by comparison.** SFT can only say "be more like this answer." DPO shows the model two answers and teaches the *direction* from worse to better — exactly what you need to fix near-misses.
- **DPO skips the reward model and the RL loop.** It learns directly from `(prompt, chosen, rejected)` pairs in one ordinary training run — the "two glasses of wine" instead of "hire a sommelier and run an RL loop."
- **The dataset is the hard part.** Rejected answers must be realistic near-misses (missing fact, hallucinated entity, bundled facts, wrong type), not obvious garbage. Build them by corrupting known-good extractions programmatically and/or with a teacher. Use 500–5,000 pairs; start around 1,000.
- **`beta` is the only knob you'll think about.** It is the leash on how far the model drifts from the original. Default 0.1; lower for a longer leash, higher for a shorter one.
- **The API, exactly:** `from unsloth import FastLanguageModel, PatchDPOTrainer; PatchDPOTrainer()` first; then `from trl import DPOTrainer, DPOConfig`; construct with `DPOTrainer(model=..., ref_model=None, args=DPOConfig(beta=0.1, ...), train_dataset=..., processing_class=tokenizer)`. It is `processing_class`, **not** `tokenizer=`. `ref_model=None` reuses the adapter-disabled model to save memory.
- **Watch `rewards/margins` and `rewards/accuracies`** during training, then confirm with a real before/after eval. Margins climbing and accuracy heading toward 1.0 means it is working.
- **Cousins:** KTO for unpaired thumbs-up/down feedback (`from trl import KTOTrainer, KTOConfig`); ORPO for folding preference into SFT, reference-free but experimental (`from trl.experimental.orpo import ORPOTrainer, ORPOConfig`). DPO first for almost everyone.

## Next

*Ch27 - PPO and the Full RL Loop: Why We Don't Use It Here* — DPO collapsed the classic RLHF pipeline into one training run. Next we open up that pipeline to see the full PPO loop it replaces — the policy, reward model, value head, and KL penalty — and explain, honestly, why it is the wrong machine for this reader before handing off to GRPO.
