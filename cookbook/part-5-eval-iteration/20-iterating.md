# Ch20 - Iterating: From a Mediocre Model to a Good One

Your first fine-tuned model is rarely your best one. That is not a failure — it is the plan.

The real workflow is a loop, not a single pass: train → evaluate → find the specific things that went wrong → fix exactly those things → retrain. Each cycle is faster and cheaper than the one before, because you're targeting known weaknesses instead of guessing.

This chapter teaches you that loop. By the end, you will have run a full mini-iteration on the memory-extraction task: found real failure patterns, generated targeted data to patch them, retrained, and confirmed the improvement.

---

## What you'll learn

- How to do structured error analysis — not just "it scored 0.72" but "it consistently misses relationship-type memories in short conversations"
- How to generate targeted synthetic data for specific failure patterns (hard negatives, edge cases, underrepresented types)
- How to version your datasets and LoRA adapters so you can always compare and roll back
- When to add more data versus switching to a bigger base model
- How to run a full mini-iteration cycle from scratch, end to end

---

## Concepts you need first

### Data-centric AI — the 20% explanation

Imagine you're training a dog to identify squirrels. You could spend months researching better training techniques — new reward schedules, new clicker timings. Or you could just show the dog more squirrels: squirrels behind trees, squirrels at night, squirrel silhouettes. The second approach usually works faster.

Data-centric AI is that second approach applied to machine learning. Instead of changing *how* you train (the algorithm, the hyperparameters, the architecture), you change *what* you train on. Most practitioners who iterate successfully on fine-tuned models spend 80% of their time on data and 20% on training settings.

Why it matters for our task: if your model misses `relationship`-type memories, you probably don't have enough examples of relationship memories in your training data. Adding 150 targeted examples of that type will almost certainly outperform any hyperparameter tweak you could make.

### Hard negatives — the 20% explanation

Think about how you learned to proofread. If every sentence you practiced on was either obviously correct or obviously wrong, you'd struggle with the subtle ones — sentences that *sound* right but have a quiet error.

Hard negatives are the subtle ones. In memory extraction, a hard negative is an example where the right answer is *close* to a wrong answer, and the model needs to learn the precise distinction. Examples: a conversation that mentions someone's name in passing (not a relationship memory) versus one where the relationship is explicitly stated. Or a preference stated with uncertainty ("I might try sushi sometime") versus a real preference ("I always order sushi").

Training on hard negatives teaches the model those precise boundaries. Without them, the model learns the easy cases and fails on anything at the edge.

### Adapter versioning — the 20% explanation

A LoRA adapter (from Ch6) is just a folder of files — a small set of weight adjustments on top of your base model. You can save multiple versions of it, just like you save multiple versions of code with git. Each version corresponds to a specific dataset and set of training settings.

Versioning matters because iteration without versioning is chaos. You won't remember whether `v3` was better than `v2` once you're looking at `v5`. Keep every adapter. They're small (typically 100–400 MB).

### Catastrophic forgetting — the 20% explanation

When you train a model on new examples, the training process nudges its weights toward fitting those new examples. The problem is that those same weight changes can overwrite patterns the model had already learned from its original training — like cramming for a new exam so hard that you forget everything from the last one.

This is called catastrophic forgetting. In practice it means: if you take your `v1` adapter and keep training it on relationship-focused data, the model may get better at relationships but quietly get worse at facts or preferences it previously handled well. The fix is simple — always start each training run from the frozen base model weights, not from a previous adapter. The "Retraining from a previous adapter" mistake in Common Mistakes below is catastrophic forgetting in action.

---

## The iteration loop

Here is the full cycle, drawn simply:

```
[evaluate current model]
         ↓
[error analysis: what specifically fails?]
         ↓
[generate targeted data for those failures]
         ↓
[merge new data with existing dataset, version it]
         ↓
[retrain with Unsloth, version the adapter]
         ↓
[evaluate again, compare to previous version]
         ↓
[repeat until good enough]
```

You ran a version of evaluation in Ch18. This chapter picks up where that left off and closes the loop.

---

## Step 1 — Structured error analysis

"The model scored 0.68 F1" tells you nothing actionable. (F1 is the score we defined in Ch18 — it combines precision and recall into a single number between 0 and 1, where 1.0 is perfect.) You need to know *where* it scored 0.68 — which types of memories it misses, which it hallucinates, which it correctly extracts.

Here is an error analysis script. It loads your test set, runs the model on each conversation, and breaks down failures by memory type, conversation length, and failure mode.

```python
# error_analysis.py
"""
Run this after Ch18 evaluation to understand WHERE the model fails.
Input: the test JSONL from Ch14, plus your trained adapter.
Output: a breakdown of errors by type, length, and failure mode.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from unsloth import FastLanguageModel
import torch


# ── Load the model (same as Ch18) ────────────────────────────────────────────

ADAPTER_PATH = "outputs/memory-extraction-v1"   # path to your saved LoRA adapter
BASE_MODEL   = "unsloth/Qwen3-4B-bnb-4bit"       # same base you fine-tuned from
TEST_DATA    = "data/memories_test.jsonl"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)


# ── Inference helper ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a memory extraction system. Given a conversation, extract a JSON array of memory objects.
Each memory must have: text (string), type (one of: fact, preference, decision, relationship, goal), entities (list of strings), confidence (high/medium/low).
Output ONLY the JSON array, no other text."""
# NOTE — schema extension: this chapter adds a fourth field, 'confidence', to the
# {text, type, entities} schema established in Ch14 and evaluated in Ch18.
# 'confidence' is a plain-English certainty label: "high" means the conversation
# states the fact clearly; "medium" means it is implied; "low" means it is a guess.
# IMPORTANT: your Ch14 dataset rows and Ch18 test set must also carry this field —
# if they do not, the gold outputs in retrain_v2.py will not match the test set rows
# and evaluation will silently break. If you generated data before this chapter,
# re-run Ch14's generation script with the updated schema, or add a "confidence"
# field (defaulting to "high") to each existing row with a one-liner:
#   import json; rows = [json.loads(l) for l in open("data/memories_test.jsonl")]
#   for r in rows:
#       for m in r["memories"]: m.setdefault("confidence", "high")
# Then overwrite the file. Do the same for memories_train_v1.jsonl and memories_val.jsonl.


def extract_memories(conversation: str) -> list:
    """Run the fine-tuned model on one conversation and return parsed memories."""
    user_msg = f"Extract memories from this conversation:\n\n{conversation}"

    # Format as a chat message — must match the chat template used during training.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=512,
            temperature=0.1,   # low temperature for deterministic extraction
            do_sample=True,
        )

    # Decode only the new tokens (not the prompt).
    new_tokens = outputs[0][inputs.shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Strip markdown fences if present (the model sometimes adds them).
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$",     "", raw)

    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []   # model produced unparseable output — counts as a failure


# ── Matching logic ────────────────────────────────────────────────────────────

def memories_match(pred: dict, gold: dict) -> bool:
    """
    Loose match: the predicted memory is considered correct if its text
    overlaps significantly with the gold memory text.
    This mirrors the Ch18 evaluation logic — keep it consistent.
    """
    pred_words = set(pred.get("text", "").lower().split())
    gold_words = set(gold.get("text", "").lower().split())
    if not gold_words:
        return False
    # At least 50% word overlap counts as a match.
    overlap = len(pred_words & gold_words) / len(gold_words)
    return overlap >= 0.5


# ── Error analysis ────────────────────────────────────────────────────────────

# These counters will tell us exactly where the model struggles.
by_type     = defaultdict(lambda: {"total": 0, "found": 0, "missed": 0})
by_length   = defaultdict(lambda: {"total": 0, "found": 0, "missed": 0})  # short/medium/long
hallucinated = 0   # predicted memories with no matching gold memory
total_gold   = 0
total_pred   = 0

failure_examples = defaultdict(list)  # store a few examples per failure type


def conv_length_bucket(conversation: str) -> str:
    """Bucket a conversation by word count for length-based analysis."""
    words = len(conversation.split())
    if words < 120:
        return "short (<120 words)"
    elif words < 300:
        return "medium (120-300 words)"
    else:
        return "long (>300 words)"


rows = [json.loads(l) for l in open(TEST_DATA)]
print(f"Analyzing {len(rows)} test examples...\n")

for row in rows:
    conv  = row["conversation"]
    golds = row["memories"]
    preds = extract_memories(conv)

    bucket = conv_length_bucket(conv)
    total_gold += len(golds)
    total_pred += len(preds)

    # For each gold memory: was it found in predictions?
    for gold in golds:
        mem_type = gold.get("type", "unknown")
        by_type[mem_type]["total"]   += 1
        by_length[bucket]["total"]   += 1

        matched = any(memories_match(p, gold) for p in preds)
        if matched:
            by_type[mem_type]["found"]   += 1
            by_length[bucket]["found"]   += 1
        else:
            by_type[mem_type]["missed"]  += 1
            by_length[bucket]["missed"]  += 1
            # Save an example of this failure for later inspection.
            if len(failure_examples[mem_type]) < 5:
                failure_examples[mem_type].append({
                    "conversation": conv[:300],   # truncate for readability
                    "missed_memory": gold,
                })

    # For each predicted memory: does it correspond to anything real?
    for pred in preds:
        if not any(memories_match(pred, g) for g in golds):
            hallucinated += 1


# ── Print the report ──────────────────────────────────────────────────────────

print("=" * 60)
print("ERROR ANALYSIS REPORT")
print("=" * 60)

print(f"\nOverall: {total_gold} gold memories, {total_pred} predicted")
print(f"Hallucinations (predicted with no gold match): {hallucinated}")

print("\n── By memory type ──")
for mem_type, counts in sorted(by_type.items()):
    recall = counts["found"] / counts["total"] if counts["total"] > 0 else 0
    print(f"  {mem_type:15s}  total={counts['total']:4d}  "
          f"found={counts['found']:4d}  missed={counts['missed']:4d}  "
          f"recall={recall:.2f}")

print("\n── By conversation length ──")
for bucket, counts in sorted(by_length.items()):
    recall = counts["found"] / counts["total"] if counts["total"] > 0 else 0
    print(f"  {bucket:30s}  recall={recall:.2f}")

print("\n── Example failures by type ──")
for mem_type, examples in failure_examples.items():
    print(f"\n  [{mem_type}] — {len(examples)} sample failures:")
    for ex in examples[:2]:   # show max 2 per type
        print(f"    Missed: {ex['missed_memory']['text']}")
```

Run it:

```bash
python error_analysis.py
```

A typical first-run output looks something like this:

```
── By memory type ──
  decision        total= 180  found= 152  missed=  28  recall=0.84
  fact            total= 420  found= 378  missed=  42  recall=0.90
  goal            total=  95  found=  70  missed=  25  recall=0.74
  preference      total= 310  found= 261  missed=  49  recall=0.84
  relationship    total= 140  found=  81  missed=  59  recall=0.58

── By conversation length ──
  short (<120 words)          recall=0.71
  medium (120-300 words)      recall=0.87
  long (>300 words)           recall=0.88
```

This tells you everything. `relationship` recall is 0.58 — the model misses nearly half of all relationship memories. And short conversations are hurting recall badly. Those are your two targets.

Don't skip this step. Every iteration cycle starts here. Without it, you are guessing.

---

## Step 2 — Understanding why failures happen

Before you generate more data, read the actual failure examples that the script collected. The `failure_examples` dict has up to five real cases per memory type.

For `relationship` failures, you'll typically see one of three patterns:

1. **Implicit relationships** — "She mentioned her sister was visiting" → the model doesn't extract `[Alice's sister is visiting her]` because the relationship is not stated outright.
2. **Relationships buried late in a conversation** — the model's attention on short sequences sometimes misses context from the first few lines.
3. **Weak entity recognition** — the model extracted the text but listed no entities, so the memory wasn't counted as a match.

For short-conversation failures, the pattern is usually that the model generates fewer memories than exist — it seems to "not bother" on short inputs. This is a training distribution problem: your original synthetic data probably skewed toward longer conversations.

Write down your top two or three failure patterns before moving on. You need this list to design the targeted data you'll generate in Step 3.

Example notes from a real iteration:

```
Failure pattern 1: relationship-type memories recalled at 0.58
  - Root cause: only ~9% of training rows had relationship-type memories
  - Model has seen too few examples of this type to recognize it reliably

Failure pattern 2: short conversation recall 0.71
  - Root cause: training data had mostly medium/long conversations
  - Short conversations were only ~12% of training rows

Failure pattern 3: hallucination rate higher than expected (~18% of predictions)
  - Root cause: some training examples had overly aggressive memory lists
  - Model learned to over-extract
```

---

## Step 3 — Generating targeted data

Now you generate a batch of new training rows that directly address your failure patterns. This is different from the broad generation in Ch13 — here you are surgical.

### 3a — Targeted: relationship-type memories

```python
# targeted_data/generate_relationship_focus.py
"""
Generates conversations where relationship information is the primary content.
These directly address the recall=0.58 weakness on relationship-type memories.
"""

import json
import random
import time
from pathlib import Path

from anthropic import Anthropic

client = Anthropic()

# Seeds designed specifically to surface relationship information.
# These are narrower than the general seeds in Ch13 — that is intentional.
RELATIONSHIP_TOPICS = [
    "catching up and sharing family news",
    "introducing a mutual friend and explaining how they know each other",
    "discussing a falling out with a colleague",
    "talking about a new romantic relationship",
    "explaining a complicated family dynamic",
    "reuniting with an old mentor",
    "discussing a friendship that has grown distant",
    "talking about a new neighbor they've gotten to know",
]

RELATIONSHIP_STYLES = [
    "warm and personal, full of names and backstory",
    "matter-of-fact, names mentioned naturally in context",
    "one person catching the other up on people they don't know",
]

PROMPT_TEMPLATE = """Generate a realistic chat conversation and extract its memories.

The conversation should feature RICH relationship information — people mentioning family members,
friends, colleagues, or romantic partners by name, and describing how they know each other
or how those relationships work.

Topic: {topic}
Style: {style}
Length: {turns} messages, alternating A: and B:

Output format:
<conversation>
[the conversation here]
</conversation>

<memories>
[JSON array — include AT LEAST 2 relationship-type memories, plus any others that are genuinely present]
Each memory: {{"text": "...", "type": "fact|preference|decision|relationship|goal", "entities": [...], "confidence": "high|medium|low"}}
</memories>

Relationship memories should look like:
  {{"text": "Alex's sister Emma lives in Portland and they talk every week", "type": "relationship", "entities": ["Alex", "Emma", "Portland"], "confidence": "high"}}
  {{"text": "Jordan and Sam met in college and have been close friends since", "type": "relationship", "entities": ["Jordan", "Sam"], "confidence": "high"}}
"""


def generate_relationship_row():
    """Generate one conversation focused on relationship content."""
    topic = random.choice(RELATIONSHIP_TOPICS)
    style = random.choice(RELATIONSHIP_STYLES)
    turns = random.choice([6, 8, 10])

    prompt = PROMPT_TEMPLATE.format(topic=topic, style=style, turns=turns)

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text
    except Exception as e:
        print(f"  API error: {e}")
        return None

    import re
    conv_match = re.search(r"<conversation>(.*?)</conversation>", raw, re.DOTALL)
    mem_match  = re.search(r"<memories>(.*?)</memories>",      raw, re.DOTALL)
    if not conv_match or not mem_match:
        return None

    try:
        memories = json.loads(mem_match.group(1).strip())
    except json.JSONDecodeError:
        return None

    # Quality gate: must contain at least one relationship memory.
    has_relationship = any(m.get("type") == "relationship" for m in memories)
    if not has_relationship:
        return None

    return {
        "conversation": conv_match.group(1).strip(),
        "memories": memories,
        "meta": {"topic": topic, "style": style, "targeted": "relationship"}
    }


output = Path("data/targeted_relationship.jsonl")
target = 200   # 200 rows focused on this failure pattern
accepted = 0

print(f"Generating {target} relationship-focused rows...")
with open(output, "a") as f:
    while accepted < target:
        row = generate_relationship_row()
        if row:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            accepted += 1
            print(f"  {accepted}/{target}")
        time.sleep(0.3)

print(f"Done. Wrote {accepted} rows to {output}")
```

### 3b — Targeted: short conversations

```python
# targeted_data/generate_short_convs.py
"""
Generates SHORT conversations (4-6 messages) with valid memory extractions.
Addresses the recall=0.71 on short conversations.
"""

import json
import random
import re
import time
from pathlib import Path
from anthropic import Anthropic

client = Anthropic()

# These topics work well in short format — quick, punchy exchanges.
SHORT_TOPICS = [
    "quick check-in about weekend plans",
    "brief update on a job application",
    "short exchange about a restaurant recommendation",
    "quick question about a shared project",
    "brief mention of a trip coming up",
    "short conversation about a health update",
    "quick exchange about a life decision",
    "brief mention of a new hobby starting",
]

PROMPT_TEMPLATE = """Generate a SHORT realistic chat conversation (exactly {turns} messages) and extract its memories.

Keep it concise and natural — this is a quick exchange, not a long catching-up session.
Despite the brevity, there should still be {n_memories} meaningful memories to extract.

Topic: {topic}
Format: alternate A: and B: for exactly {turns} messages.

Output:
<conversation>
[conversation here]
</conversation>

<memories>
[JSON array of {n_memories} memories]
Each: {{"text": "...", "type": "fact|preference|decision|relationship|goal", "entities": [...], "confidence": "high|medium|low"}}
</memories>
"""


def generate_short_row():
    topic   = random.choice(SHORT_TOPICS)
    turns   = random.choice([4, 5, 6])
    # Short conversations realistically yield 1-3 memories.
    n_memories = random.choice([1, 2, 2, 3])

    prompt = PROMPT_TEMPLATE.format(
        topic=topic, turns=turns, n_memories=n_memories
    )

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text
    except Exception as e:
        print(f"  API error: {e}")
        return None

    conv_match = re.search(r"<conversation>(.*?)</conversation>", raw, re.DOTALL)
    mem_match  = re.search(r"<memories>(.*?)</memories>",      raw, re.DOTALL)
    if not conv_match or not mem_match:
        return None

    try:
        memories = json.loads(mem_match.group(1).strip())
    except json.JSONDecodeError:
        return None

    conv = conv_match.group(1).strip()
    words = len(conv.split())

    # Quality gate: must actually be short.
    if words > 140 or len(memories) == 0:
        return None

    return {
        "conversation": conv,
        "memories": memories,
        "meta": {"topic": topic, "turns": turns, "targeted": "short_conversation"}
    }


output  = Path("data/targeted_short.jsonl")
target  = 150
accepted = 0

print(f"Generating {target} short-conversation rows...")
with open(output, "a") as f:
    while accepted < target:
        row = generate_short_row()
        if row:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            accepted += 1
            print(f"  {accepted}/{target}")
        time.sleep(0.3)

print(f"Done. Wrote {accepted} rows to {output}")
```

### 3c — Hard negatives: teaching the model what NOT to extract

Hard negatives are examples that demonstrate restraint. The model needs to learn that not everything in a conversation is worth extracting — passing comments, filler, and uncertain statements are not memories.

```python
# targeted_data/generate_hard_negatives.py
"""
Generates conversations where MOST of the content should NOT be extracted,
but a small number of genuine memories are present.

This trains the model to be selective — the hallucination fix.
"""

import json
import random
import re
import time
from pathlib import Path
from anthropic import Anthropic

client = Anthropic()

HARD_NEG_TOPICS = [
    "small talk about the weather and weekend",
    "brief chat mostly about nothing in particular",
    "catching up where most topics are vague or hypothetical",
    "talking about a movie plot without revealing preferences",
    "chatting about general plans without committing to anything",
]

PROMPT_TEMPLATE = """Generate a realistic chat conversation and extract ONLY the memories that are genuinely durable and meaningful.

The conversation should contain a lot of small talk, vague comments, and passing mentions — things that are NOT worth extracting.
But it should contain exactly {n_real} genuine memory (or memories) worth keeping.

Topic: {topic}
Length: {turns} messages (A: and B: alternating)

A good memory: "Jordan is training for a marathon in April" — specific, standalone, durable.
NOT a memory: "they talked about exercise" — too vague.
NOT a memory: "A said it might be nice to travel sometime" — uncertain, hypothetical.
NOT a memory: "they discussed the weather" — ephemeral, not meaningful.

Output:
<conversation>
[conversation]
</conversation>

<memories>
[JSON array — only {n_real} memory/memories that actually meet the bar above]
Each: {{"text": "...", "type": "fact|preference|decision|relationship|goal", "entities": [...], "confidence": "high|medium|low"}}
</memories>
"""


def generate_hard_negative():
    topic    = random.choice(HARD_NEG_TOPICS)
    turns    = random.choice([8, 10, 12])
    n_real   = random.choice([1, 1, 2])   # deliberately few genuine memories

    prompt = PROMPT_TEMPLATE.format(
        topic=topic, turns=turns, n_real=n_real
    )

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text
    except Exception as e:
        print(f"  API error: {e}")
        return None

    conv_match = re.search(r"<conversation>(.*?)</conversation>", raw, re.DOTALL)
    mem_match  = re.search(r"<memories>(.*?)</memories>",      raw, re.DOTALL)
    if not conv_match or not mem_match:
        return None

    try:
        memories = json.loads(mem_match.group(1).strip())
    except json.JSONDecodeError:
        return None

    # Quality gate: the "hard negative" value comes from sparse extraction.
    # Reject rows where the model generated too many memories anyway.
    if len(memories) > 3:
        return None

    return {
        "conversation": conv_match.group(1).strip(),
        "memories": memories,
        "meta": {"topic": topic, "turns": turns, "targeted": "hard_negative"}
    }


output  = Path("data/targeted_hard_negatives.jsonl")
target  = 150
accepted = 0

print(f"Generating {target} hard-negative rows...")
with open(output, "a") as f:
    while accepted < target:
        row = generate_hard_negative()
        if row:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            accepted += 1
            print(f"  {accepted}/{target}")
        time.sleep(0.3)

print(f"Done. Wrote {accepted} rows to {output}")
```

---

## Step 4 — Versioning datasets and adapters

Before you touch anything, version what you already have.

### Dataset versioning

Keep every version of your training data. The naming convention below is simple and unambiguous:

```
data/
  memories_train_v1.jsonl        ← your original training set
  memories_train_v2.jsonl        ← v1 + targeted data (what you're building now)
  memories_val.jsonl             ← validation set — NEVER CHANGES between versions
  memories_test.jsonl            ← test set — NEVER CHANGES between versions
```

The validation and test sets must stay frozen. If you change them between iterations, you cannot compare scores across versions.

```python
# build_v2_dataset.py
"""
Merges the original training data with the targeted additions to form v2.
Run this once before retraining.
"""

import json
import random
from pathlib import Path

# Source files
ORIGINAL   = Path("data/memories_train_v1.jsonl")
TARGETED_1 = Path("data/targeted_relationship.jsonl")
TARGETED_2 = Path("data/targeted_short.jsonl")
TARGETED_3 = Path("data/targeted_hard_negatives.jsonl")
OUTPUT     = Path("data/memories_train_v2.jsonl")


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


rows = (
    load_jsonl(ORIGINAL)   +
    load_jsonl(TARGETED_1) +
    load_jsonl(TARGETED_2) +
    load_jsonl(TARGETED_3)
)

# Shuffle so targeted data is distributed throughout, not all at the end.
# A model trained on data sorted by type will overfit to the ordering.
random.shuffle(rows)

with open(OUTPUT, "w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"v2 dataset: {len(rows)} rows → {OUTPUT}")

# Print a breakdown so you know what you're training on.
from collections import Counter
sources = Counter(r.get("meta", {}).get("targeted", "original") for r in rows)
print("\nSource breakdown:")
for src, count in sources.most_common():
    pct = count / len(rows) * 100
    print(f"  {src:30s}  {count:5d} rows  ({pct:.1f}%)")
```

### Adapter versioning

Save each trained adapter with a version tag. In your Unsloth training script (from Ch15), change the output directory:

```python
# In your training script (Ch15), update this line:

# Version 1 (original):
# trainer_config = {"output_dir": "outputs/memory-extraction-v1", ...}

# Version 2 (after adding targeted data):
trainer_config = {"output_dir": "outputs/memory-extraction-v2", ...}
```

Keep both `v1` and `v2` folders. They're typically 100–300 MB each for a 4B model with LoRA. Storage is cheap; being unable to compare is expensive.

A simple log file keeps you sane across iterations:

```bash
# adapters/VERSIONS.txt  — keep this updated manually

v1  |  data: memories_train_v1.jsonl (500 rows)   |  f1=0.79  |  2025-07-15
v2  |  data: memories_train_v2.jsonl (1000 rows)  |  f1=?     |  (training now)
```

---

## Step 5 — Retraining with the new dataset

The retraining script is nearly identical to Ch15. The only things that change are the input dataset path and the output adapter path.

```python
# retrain_v2.py
"""
Retrains the memory-extraction model on the v2 dataset.
Identical to Ch15 except for dataset path and output directory.
Takes roughly the same time as the first training run.
"""

from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

# ── Model loading ─────────────────────────────────────────────────────────────

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-bnb-4bit",   # always start from the BASE model
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Apply LoRA — same rank as Ch15 for a fair comparison.
# rank controls how many extra weights LoRA adds — higher rank = more capacity
# but more memory. Changing it between v1 and v2 would mean you can't tell whether
# any improvement came from the new data or the different rank. Keep it the same.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ── Dataset ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a memory extraction system. Given a conversation, extract a JSON array of memory objects.
Each memory must have: text (string), type (one of: fact, preference, decision, relationship, goal), entities (list of strings), confidence (high/medium/low).
Output ONLY the JSON array, no other text."""


def format_row(row):
    """
    Converts a JSONL row into the chat-template format the model expects.
    Same function as Ch15 — keeping it identical avoids a confound.
    """
    import json
    user_content = f"Extract memories from this conversation:\n\n{row['conversation']}"
    gold_output  = json.dumps(row["memories"], ensure_ascii=False)

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": gold_output},
    ]
    return {"text": tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )}


# Load v2 — the only line that differs from the Ch15 script.
dataset = load_dataset("json", data_files={"train": "data/memories_train_v2.jsonl"})
dataset = dataset["train"].map(format_row, remove_columns=dataset["train"].column_names)

# ── Training ──────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="outputs/memory-extraction-v2",   # ← version bump
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        dataset_text_field="text",
        max_seq_length=2048,
    ),
)

trainer.train()

# Save the final adapter.
model.save_pretrained("outputs/memory-extraction-v2")
tokenizer.save_pretrained("outputs/memory-extraction-v2")
print("v2 adapter saved.")
```

Run it:

```bash
python retrain_v2.py
```

Training time on the v2 dataset (~1000 rows, 3 epochs, Qwen3-4B) is roughly 25–40 minutes on an A100 or equivalent. See Ch8 for hardware reference.

---

## Step 6 — Comparing v1 and v2

Run the Ch18 evaluation script on both adapters with the same frozen test set. Then compare.

```python
# compare_versions.py
"""
Evaluates two adapter versions on the same test set and prints a side-by-side comparison.
Replace ADAPTER_V1 and ADAPTER_V2 with your actual paths.
"""

import json
import re
from collections import defaultdict
from unsloth import FastLanguageModel
import torch

TEST_DATA   = "data/memories_test.jsonl"
ADAPTER_V1  = "outputs/memory-extraction-v1"
ADAPTER_V2  = "outputs/memory-extraction-v2"
BASE_MODEL  = "unsloth/Qwen3-4B-bnb-4bit"

SYSTEM_PROMPT = """You are a memory extraction system. Given a conversation, extract a JSON array of memory objects.
Each memory must have: text (string), type (one of: fact, preference, decision, relationship, goal), entities (list of strings), confidence (high/medium/low).
Output ONLY the JSON array, no other text."""


def load_adapter(adapter_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def run_eval(model, tokenizer, rows):
    """Returns per-type recall scores for a model on the given rows."""
    by_type = defaultdict(lambda: {"found": 0, "total": 0})

    for row in rows:
        conv  = row["conversation"]
        golds = row["memories"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Extract memories:\n\n{conv}"},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
        raw = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        raw = re.sub(r"^```json\s*", "", raw); raw = re.sub(r"\s*```$", "", raw)

        try:
            parsed = json.loads(raw)          # parse once into a variable
            preds = parsed if isinstance(parsed, list) else []
        except Exception:
            preds = []

        for gold in golds:
            mtype = gold.get("type", "unknown")
            by_type[mtype]["total"] += 1
            pred_words = lambda p: set(p.get("text","").lower().split())
            gold_words = set(gold.get("text","").lower().split())
            matched = any(
                len(pred_words(p) & gold_words) / max(len(gold_words), 1) >= 0.5
                for p in preds
            )
            if matched:
                by_type[mtype]["found"] += 1

    return {t: c["found"] / c["total"] for t, c in by_type.items() if c["total"] > 0}


rows = [json.loads(l) for l in open(TEST_DATA)]
print(f"Evaluating on {len(rows)} test rows...\n")

print("Loading v1...")
m1, t1 = load_adapter(ADAPTER_V1)
scores_v1 = run_eval(m1, t1, rows)
del m1, t1  # free VRAM before loading v2
torch.cuda.empty_cache()

print("Loading v2...")
m2, t2 = load_adapter(ADAPTER_V2)
scores_v2 = run_eval(m2, t2, rows)

print("\n" + "=" * 55)
print(f"{'Memory type':18s}  {'v1 recall':>10}  {'v2 recall':>10}  {'delta':>8}")
print("=" * 55)

all_types = sorted(set(scores_v1) | set(scores_v2))
for t in all_types:
    v1 = scores_v1.get(t, 0.0)
    v2 = scores_v2.get(t, 0.0)
    delta = v2 - v1
    arrow = "▲" if delta > 0.02 else ("▼" if delta < -0.02 else " ")
    print(f"  {t:16s}  {v1:10.3f}  {v2:10.3f}  {arrow}{abs(delta):+.3f}")
```

A successful iteration might look like this:

```
Memory type           v1 recall   v2 recall     delta
=======================================================
  decision              0.840       0.861       ▲+0.021
  fact                  0.900       0.912       ▲+0.012
  goal                  0.740       0.778       ▲+0.038
  preference            0.840       0.855       ▲+0.015
  relationship          0.580       0.741       ▲+0.161
```

That `▲+0.161` on `relationship` is the payoff. You identified a specific weakness, targeted it with 200 rows of data, and moved recall from 0.58 to 0.74 — without touching the model size or the hyperparameters.

---

## When to scale model size instead of data

Targeted data fixes most problems. But not all of them. Here is how to tell which situation you are in:

| Symptom | Likely cause | Fix |
|---|---|---|
| One memory type has recall < 0.6 | Too few examples of that type | Add targeted data |
| All memory types improve with more data | Distribution is still sparse | Keep adding data |
| All types plateau near 0.8 despite 3000+ rows | Model capacity limit | Try a 7B or 8B base model |
| Model hallucinates even with hard negatives | Needs stronger reasoning | Try a 7B/8B model |
| Output JSON is often malformed | Instruction-following weakness | Try a 7B/8B model or more epochs |

The clearest signal that you have hit a model-size ceiling: adding 2x more data produces less than 0.02 improvement on all types simultaneously. At that point, upgrading from 4B to 7B or 8B parameters (see Ch10 for model options) is the right move. A 7B model typically adds 4–6 GB of VRAM and roughly doubles training time.

Do not jump to a bigger model prematurely. Three-to-five iteration cycles on a 4B model will usually get you further, faster, than one cycle on a 7B model — because each 4B iteration is cheaper and you learn more per dollar spent.

---

## A worked mini-iteration: end to end

Here is the complete cycle compressed into a single checklist you can follow for every iteration:

```
ITERATION CHECKLIST

[ ] 1. Run error_analysis.py on current model + test set
[ ] 2. Write down top 2-3 failure patterns with root causes
[ ] 3. Generate targeted data for each pattern (200-300 rows per pattern)
[ ] 4. Run quick LLM judge pass on targeted data (from Ch13) — use the same
        judge.py script as Ch13, no changes needed, just point it at each file:
          python judge.py --input data/targeted_relationship.jsonl
          python judge.py --input data/targeted_short.jsonl
          python judge.py --input data/targeted_hard_negatives.jsonl
[ ] 5. Merge into a new versioned dataset (v2, v3, ...)
[ ] 6. Update VERSIONS.txt with dataset details and date
[ ] 7. Retrain from the BASE model (not from a previous adapter — see common mistakes)
[ ] 8. Run evaluation on the frozen test set
[ ] 9. Run compare_versions.py — record delta per type
[ ] 10. Decide: good enough to ship, or run another cycle?
```

The "good enough to ship" bar depends on your application. For a personal memory tool, 0.80+ recall on all types is solid. For a production product where missing a memory is a real user problem, you might aim for 0.90+ on the high-confidence subset.

Most tasks see the biggest jump in cycles 1 and 2. By cycle 4 or 5, improvements become incremental and you are likely approaching the ceiling of your base model or your task's inherent ambiguity. That is when Ch23 (continual learning) becomes relevant — feeding the model fresh real-world data continuously rather than iterating in fixed cycles.

---

## Common mistakes

**Retraining from a previous adapter instead of the base model.** It feels efficient to take your `v1` adapter and train more on top of it. In practice this causes catastrophic forgetting in subtle ways — the model loses performance on things it previously handled well. Always start each training run from the frozen base model weights. The adapter is small; the base model is what you are learning on top of.

**Changing the test set between iterations.** If you add a difficult example to the test set when moving from v1 to v2, your scores are not comparable. The test set is sacred. Never change it. If you want to track something new, create a separate diagnostic set.

**Adding too much targeted data too fast.** Adding 1000 relationship rows to a 500-row base dataset doesn't fix the problem — it tilts the model toward always outputting relationship memories. Keep targeted additions to 20–40% of the total dataset size per iteration. If the imbalance persists, down-sample the original data slightly to compensate.

**Evaluating on the validation set and calling it done.** The validation set (from Ch14) is for tuning decisions during training. The test set is for final evaluation. Repeatedly evaluating on the test set and then making changes based on it is a form of data leakage (meaning you accidentally let your test answers influence your training or tuning decisions, making your reported score falsely optimistic) — you are implicitly fitting to the test set. Evaluate on validation during iteration; evaluate on test only to report a final number.

**Forgetting to shuffle the merged dataset.** If all the targeted relationship examples appear at the end of the JSONL file, the model's final training steps are dominated by relationship examples and it will overfit to them. Always shuffle before writing.

**Not labeling the targeted data's source.** Three months from now you will not remember which rows were original and which were targeted. The `meta.targeted` field in the examples above is not overhead — it is how you debug v4 when you are wondering why the model is strange on short conversations.

---

## Recap

- The iteration loop is: evaluate → error-analyse → generate targeted data → retrain → compare — not a one-shot process
- Error analysis must be specific: break down failures by memory type, conversation length, and failure mode
- Targeted synthetic data directly addresses known weaknesses: hard negatives reduce hallucination, type-focused prompts fix recall on underrepresented types, short-conversation examples fix length-based failure
- Version every dataset and every adapter with a clear naming convention; keep the validation and test sets completely frozen
- Most gains come from data, not hyperparameters — the data-centric mindset applies at every cycle
- Scale to a larger base model only when data additions consistently produce less than 0.02 improvement across all types
- Two to three iteration cycles on a 4B model will typically outperform one cycle on a 7B model in terms of cost and learning

---

## Next

**Ch21 - Saving, Merging, and Exporting Your Model** — once you are happy with your iterated adapter, you will merge it back into the base weights, export it to different formats (GGUF for local inference, vLLM-compatible for serving), and prepare it to ship.
