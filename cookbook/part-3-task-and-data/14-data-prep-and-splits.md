# Ch14 - Cleaning, Splitting, and Sanity-Checking Data

You have a JSONL file full of synthetic training examples. Before you hand it to the trainer, you need to know it's actually good. Garbage in, garbage out is not a metaphor in machine learning — it's a law. This chapter is about spending an hour now so you don't waste four hours of GPU time on broken data.

---

## What you'll learn

- How to load a JSONL dataset with the `datasets` library and validate every row against your memory schema
- How to detect and drop malformed rows, duplicates, and examples that are too long for your model
- How to split your data into train / validation / test sets — and why all three matter
- How to build a token-length histogram so you can pick the right `max_seq_length` setting
- How to render a few examples through the chat template and visually confirm the format is correct before training starts

---

## Concepts you need first

### What is a "split" and why do you need three of them?

Think of a student preparing for an exam. They study from a textbook (the **training set**). Occasionally they quiz themselves on practice problems from a separate workbook (the **validation set**) — not to study from, but to notice when they're getting confused and adjust. On exam day they sit a real test they've never seen (the **test set**). If they had used the exam problems to study, the score would be meaningless.

Your model works exactly the same way:

- **Training set** — what the model learns from. It sees this data many times.
- **Validation set** — examples the model never trains on. You check it periodically during training to measure whether the model is actually improving or just memorizing.
- **Test set** — examples the model has never touched, used only once at the very end to get an honest, unbiased accuracy number. If you peek at the test set earlier, you bias your decisions toward it and the number is no longer honest.

A common split for a few thousand rows is **80 / 10 / 10** (train / val / test). For very small datasets (under 500 rows) you might do 90 / 5 / 5 to give training as much data as possible.

### What is `max_seq_length`?

Every model has a maximum number of tokens it can process at once — its context window. When you train, you also set a `max_seq_length` that acts as a hard cap: any example longer than this gets truncated (cut off). Truncated examples are corrupted training data. You want to set `max_seq_length` high enough that almost no examples get cut, but not so high that you run out of GPU memory.

The way to choose it: count how long your actual examples are (in tokens) and look at the distribution. The 95th or 99th percentile is a sensible cap. You'll see this visualized as a histogram below.

### What is a chat template?

Models trained for conversation expect input in a specific format. A **chat template** is the model's personal formatting contract — it wraps your messages in special tokens and role labels that the model was taught to expect. Getting this wrong is a silent bug: the model will train on malformed input, produce worse results, and give you no error message. Chapter 5 ("Tokens, Context Windows, and Chat Templates") covers the mechanics. Here, you'll use it practically: render a few examples and read them like a human to confirm they look right.

---

## Step 1 — Load the data

Install the libraries you need if you haven't already (Chapter 9 covers the full setup):

```bash
pip install datasets transformers
```

The `datasets` library (by Hugging Face) handles loading, filtering, and splitting data efficiently even on large files. It stores data in Apache Arrow format under the hood, which is fast to iterate over without loading everything into RAM.

```python
# load_and_validate.py
# PURPOSE: Load raw JSONL, validate schema, drop bad rows, produce clean splits.

import json
from datasets import load_dataset

# ---------------------------------------------------------------------------
# 1. Load the raw JSONL produced by the synthetic data generator (Ch13).
#    datasets.load_dataset returns a DatasetDict or Dataset object — think of
#    it as a smart list that also knows about column names and types.
# ---------------------------------------------------------------------------
raw = load_dataset(
    "json",                           # format: one JSON object per line
    data_files="data/memories_raw.jsonl",  # path to your synthetic JSONL
    split="train",                    # load_dataset always needs a split name;
                                      # "train" is the default name when there
                                      # is only one file
)

print(f"Loaded {len(raw)} rows")
print("Column names:", raw.column_names)
# Expected output:
#   Loaded 2847 rows
#   Column names: ['messages', 'memories']
```

Each row of your JSONL should look like what you defined in Chapter 12:

```json
{
  "messages": [
    {"role": "user", "content": "Here is a conversation snippet:\n\nAlice: ..."},
    {"role": "assistant", "content": "[{\"text\": \"Alice prefers async standups\", \"type\": \"preference\", \"entities\": [\"Alice\"]}]"}
  ],
  "memories": [
    {"text": "Alice prefers async standups", "type": "preference", "entities": ["Alice"]}
  ]
}
```

---

## Step 2 — Validate every row against the schema

Synthetic data generation is not perfect. GPT-4 or Claude sometimes outputs:
- A memory list that isn't valid JSON (extra comma, unclosed bracket)
- A memory object missing a required field (`text`, `type`, or `entities`)
- The assistant message wrapped in markdown code fences (` ```json ... ``` `) instead of bare JSON

Write a validator that checks every row and collects the bad ones so you can inspect them.

```python
# validate_rows.py  (continue in the same file or import from it)

import re

# The required fields for each memory object in our schema (from Ch12).
REQUIRED_FIELDS = {"text", "type", "entities"}

# Valid values for the "type" field — keep this in sync with Ch12.
VALID_TYPES = {"preference", "fact", "relationship", "decision", "event"}


def extract_assistant_json(row):
    """
    Pull the assistant message content from a row and parse it as JSON.
    Returns the parsed list on success, raises ValueError on failure.
    """
    # Find the assistant turn in the messages list.
    assistant_content = None
    for msg in row["messages"]:
        if msg["role"] == "assistant":
            assistant_content = msg["content"]
            break

    if assistant_content is None:
        raise ValueError("No assistant message found")

    # Strip markdown code fences if the model wrapped the output.
    # e.g. ```json\n[...]\n``` → [...]
    cleaned = re.sub(r"^```(?:json)?\s*", "", assistant_content.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Attempt to parse as JSON.
    parsed = json.loads(cleaned)   # raises json.JSONDecodeError if malformed

    # Must be a list (could be an empty list for a conversation with no memories).
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list, got {type(parsed).__name__}")

    return parsed


def validate_memory(memory, row_index):
    """
    Check one memory object against the schema.
    Returns a list of error strings (empty list = valid).
    """
    errors = []

    # Check all required fields are present.
    missing = REQUIRED_FIELDS - set(memory.keys())
    if missing:
        errors.append(f"Row {row_index}: missing fields {missing}")

    # Check 'type' is one of the known categories.
    if "type" in memory and memory["type"] not in VALID_TYPES:
        errors.append(
            f"Row {row_index}: unknown type '{memory['type']}'"
            f" (expected one of {VALID_TYPES})"
        )

    # Check 'entities' is a list (not a string like "Alice, Bob").
    if "entities" in memory and not isinstance(memory["entities"], list):
        errors.append(f"Row {row_index}: 'entities' should be a list, got "
                      f"{type(memory['entities']).__name__}")

    # Check 'text' is a non-empty string.
    if "text" in memory and (
        not isinstance(memory["text"], str) or len(memory["text"].strip()) == 0
    ):
        errors.append(f"Row {row_index}: 'text' is empty or not a string")

    return errors


def validate_dataset(dataset):
    """
    Run validation over every row. Returns two lists:
      - valid_indices: row indices that passed
      - error_log: list of (index, error_message) tuples
    """
    valid_indices = []
    error_log = []

    for i, row in enumerate(dataset):
        try:
            memories = extract_assistant_json(row)
        except (json.JSONDecodeError, ValueError) as e:
            error_log.append((i, f"JSON parse error: {e}"))
            continue

        # Validate each individual memory object in the list.
        row_errors = []
        for memory in memories:
            row_errors.extend(validate_memory(memory, i))

        if row_errors:
            error_log.extend((i, err) for err in row_errors)
        else:
            valid_indices.append(i)

    return valid_indices, error_log


valid_indices, error_log = validate_dataset(raw)

print(f"Valid rows:   {len(valid_indices)}")
print(f"Invalid rows: {len(error_log)}")

# Print the first 10 errors so you can inspect and fix common patterns.
for idx, msg in error_log[:10]:
    print(f"  [{idx}] {msg}")
```

Typical output from a synthetic dataset of ~2800 rows:

```
Valid rows:   2761
Invalid rows: 86
  [14]  JSON parse error: Expecting ',' delimiter: line 3 column 5
  [89]  Row 89: unknown type 'opinion'
  [201] Row 201: 'entities' should be a list, got str
```

Keep the error log. If more than ~5% of rows are bad, go back to Chapter 13 and tighten your generation prompt — the current prompt is leaking an edge case.

---

## Step 3 — Drop duplicates and keep only valid rows

```python
# drop_bad_rows.py

# Keep only the rows that passed validation.
clean = raw.select(valid_indices)
print(f"After dropping invalid rows: {len(clean)} rows")

# ---------------------------------------------------------------------------
# Deduplicate on the user message content.
# Exact duplicates happen when the generator retries a failed call and the
# retry succeeds but both get written to the file.
# ---------------------------------------------------------------------------
seen = set()
unique_indices = []

for i, row in enumerate(clean):
    # Use the user message as the dedup key.
    user_content = next(
        (m["content"] for m in row["messages"] if m["role"] == "user"),
        ""
    )
    # Normalize whitespace before hashing to catch near-exact duplicates.
    key = " ".join(user_content.split())
    if key not in seen:
        seen.add(key)
        unique_indices.append(i)

clean = clean.select(unique_indices)
print(f"After deduplication:        {len(clean)} rows")
```

---

## Step 4 — Token-length histogram to choose `max_seq_length`

You need a tokenizer to count tokens. Load the same one you'll use for training — this is model-specific. Swap the model name for whichever you chose in Chapter 10.

```python
# token_histogram.py

from transformers import AutoTokenizer
from collections import Counter
import math

# Load the tokenizer for your chosen base model.
# This downloads a small config file (~1 MB) and does NOT load model weights.
# Use the same model ID you'll pass to Unsloth in Ch15.
MODEL_ID = "unsloth/Qwen3-8B"          # swap to unsloth/gemma-3-12b-it if using Gemma
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def count_tokens_for_row(row):
    """
    Render the full conversation as the model will see it during training,
    then count its tokens.
    """
    # apply_chat_template converts the messages list into the model's
    # native format string — the same string the trainer will tokenize.
    # tokenize=True returns token IDs directly; we just need the count.
    token_ids = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=True,
        add_generation_prompt=False,  # during training we include the full response
    )
    return len(token_ids)


# Count tokens for every row. This takes ~30 seconds for 2700 rows on CPU.
lengths = [count_tokens_for_row(row) for row in clean]

# ---------------------------------------------------------------------------
# Print a simple ASCII histogram so you can see the distribution at a glance.
# ---------------------------------------------------------------------------
bucket_size = 128   # group lengths into buckets of 128 tokens
buckets = Counter(math.floor(l / bucket_size) * bucket_size for l in lengths)

print(f"\nToken-length distribution ({len(lengths)} examples):\n")
max_count = max(buckets.values())
for bucket in sorted(buckets):
    bar_len = int(40 * buckets[bucket] / max_count)
    bar = "█" * bar_len
    print(f"  {bucket:>5}–{bucket + bucket_size - 1:<5}  {bar:<40}  {buckets[bucket]}")

# ---------------------------------------------------------------------------
# Print the percentiles you actually need for the decision.
# ---------------------------------------------------------------------------
lengths_sorted = sorted(lengths)
n = len(lengths_sorted)

def percentile(p):
    return lengths_sorted[int(p / 100 * n)]

print(f"\nPercentiles:")
print(f"  50th (median): {percentile(50)} tokens")
print(f"  90th:          {percentile(90)} tokens")
print(f"  95th:          {percentile(95)} tokens")
print(f"  99th:          {percentile(99)} tokens")
print(f"  max:           {max(lengths)} tokens")

# ---------------------------------------------------------------------------
# Suggest a max_seq_length. Rule of thumb: cover the 95th percentile,
# round up to the nearest power of 2 (models and GPUs prefer round numbers).
# ---------------------------------------------------------------------------
p95 = percentile(95)
suggested = 2 ** math.ceil(math.log2(p95))
print(f"\nSuggested max_seq_length: {suggested}  (covers 95th percentile of {p95} tokens)")
```

A typical memory-extraction dataset produces output like:

```
Token-length distribution (2741 examples):

    128–255    ████████████████████████████████████████  1203
    256–383    ██████████████████████████████            891
    384–511    ████████████████                          479
    512–639    ████                                      123
    640–767    █                                          35
    768–895                                               10

Percentiles:
  50th (median): 298 tokens
  90th:          521 tokens
  95th:          614 tokens
  99th:          742 tokens
  max:           891 tokens

Suggested max_seq_length: 1024  (covers 95th percentile of 614 tokens)
```

In this case, `max_seq_length = 1024` is the right setting for Chapter 15. It covers 95% of examples without truncation and is well within the 2048-token sweet spot for comfortable VRAM usage with QLoRA on an 8B model (~14 GB VRAM).

Now drop the rare examples that would be truncated anyway — they're noisy outliers:

```python
# Drop rows longer than your chosen max_seq_length.
MAX_SEQ_LENGTH = 1024

long_count = sum(1 for l in lengths if l > MAX_SEQ_LENGTH)
print(f"Dropping {long_count} rows that exceed {MAX_SEQ_LENGTH} tokens")

keep_indices = [i for i, l in enumerate(lengths) if l <= MAX_SEQ_LENGTH]
clean = clean.select(keep_indices)
print(f"After length filter: {len(clean)} rows")
```

---

## Step 5 — Train / validation / test split

```python
# split_data.py

# Shuffle first so the split isn't ordered by how the data was generated.
# seed=42 makes this reproducible — same seed = same shuffle every time.
clean = clean.shuffle(seed=42)

n = len(clean)
n_test  = max(50, int(n * 0.10))   # 10% for test, minimum 50 rows
n_val   = max(50, int(n * 0.10))   # 10% for validation, minimum 50 rows
n_train = n - n_test - n_val       # the rest for training

train_ds = clean.select(range(n_train))
val_ds   = clean.select(range(n_train, n_train + n_val))
test_ds  = clean.select(range(n_train + n_val, n))

print(f"Split sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
# e.g. Split sizes — train: 2160, val: 265, test: 265
```

**A note on the test set**: save it to disk right now and do not look at it again until Chapter 18. Do not use it to debug training, do not check a few rows to see if the model got them right, do not run evaluation on it mid-training. The moment you make any decision based on the test set, it becomes part of your validation set and your final accuracy number is no longer trustworthy.

---

## Step 6 — Inspect rendered chat-template examples

This step takes five minutes and has saved countless training runs. Load five random examples, render them through the chat template exactly as the trainer will, and read them out loud. You are looking for:

- The right model-specific special tokens (`<|im_start|>`, `<start_of_turn>`, etc.)
- The user message followed immediately by the assistant message with no garbage in between
- The assistant message containing valid JSON, not a markdown code fence
- No doubled separators or truncated endings

```python
# inspect_examples.py

import random

random.seed(7)
sample_indices = random.sample(range(len(train_ds)), 5)

print("=" * 72)
for idx in sample_indices:
    row = train_ds[idx]

    # apply_chat_template with tokenize=False returns the raw string the
    # model will see. This is the ground truth of what goes into training.
    rendered = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    token_count = len(tokenizer.encode(rendered))

    print(f"\n--- Example (dataset index {idx}, {token_count} tokens) ---\n")
    print(rendered)
    print("=" * 72)
```

Here is what a correct Qwen3 example looks like — note the `<|im_start|>` and `<|im_end|>` tokens wrapping each turn:

```
--- Example (dataset index 441, 312 tokens) ---

<|im_start|>system
You are a memory extraction assistant. Extract factual memories from the conversation below. Output a JSON array of memory objects. Each object must have: "text" (the atomic fact, written as a standalone sentence), "type" (one of: preference, fact, relationship, decision, event), "entities" (a list of named people, places, or products mentioned). Output only the JSON array, no explanation.<|im_end|>
<|im_start|>user
Extract memories from this conversation:

Jordan: I finally switched to Vim after years of resisting. Never going back.
Sam: Ha, I gave up on Vim after a week. Still on VS Code.
Jordan: To each their own. Did you see the Q3 roadmap Priya sent?
Sam: Yeah. Looks like the mobile launch moved to November.<|im_end|>
<|im_start|>assistant
[{"text": "Jordan uses Vim as their code editor.", "type": "preference", "entities": ["Jordan"]}, {"text": "Sam uses VS Code as their code editor.", "type": "preference", "entities": ["Sam"]}, {"text": "The mobile launch was moved to November.", "type": "decision", "entities": ["mobile launch"]}, {"text": "Priya sent the Q3 roadmap.", "type": "event", "entities": ["Priya"]}]<|im_end|>
```

If instead you see this, you have a template bug:

```
[WRONG] The assistant turn contains a markdown fence:
<|im_start|>assistant
```json
[{"text": "Jordan uses Vim...", ...}]
```
<|im_end|>
```

Fix: go back to your data generator (Chapter 13) and add an instruction to the generation prompt like: `Output only the raw JSON array with no code fences or extra text.` Then regenerate and re-run this pipeline.

---

## Step 7 — Save the clean splits to disk

```python
# save_splits.py

import os

OUTPUT_DIR = "data/splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save as JSONL — one JSON object per line. Easy to inspect with any text editor.
train_ds.to_json(f"{OUTPUT_DIR}/train.jsonl")
val_ds.to_json(f"{OUTPUT_DIR}/val.jsonl")
test_ds.to_json(f"{OUTPUT_DIR}/test.jsonl")

# Also save a small metadata file so you remember what choices were made.
meta = {
    "max_seq_length": MAX_SEQ_LENGTH,
    "model_id": MODEL_ID,
    "n_train": len(train_ds),
    "n_val": len(val_ds),
    "n_test": len(test_ds),
    "seed": 42,
}
with open(f"{OUTPUT_DIR}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"Saved to {OUTPUT_DIR}/")
print(f"  train.jsonl  ({len(train_ds)} rows)")
print(f"  val.jsonl    ({len(val_ds)} rows)")
print(f"  test.jsonl   ({len(test_ds)} rows)")
print(f"  meta.json")
```

Your `data/` directory now looks like:

```
data/
  memories_raw.jsonl      ← original synthetic output (keep this)
  splits/
    train.jsonl           ← what the trainer reads
    val.jsonl             ← what the trainer checks periodically
    test.jsonl            ← locked until Ch18 evaluation
    meta.json             ← provenance record
```

Keep `memories_raw.jsonl`. If you discover a systematic bug in the cleaning logic later, you want to be able to rerun this pipeline from the raw file without regenerating data.

---

## Putting it all together — one runnable script

```python
# data_prep.py — run this as a single script after generating your JSONL.
# Usage: python data_prep.py

import json, math, os, re, random
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────
RAW_FILE      = "data/memories_raw.jsonl"
OUTPUT_DIR    = "data/splits"
MODEL_ID      = "unsloth/Qwen3-8B"      # change to match your Ch10 choice
SPLIT_SEED    = 42
MAX_PCT       = 95                        # cover this percentile with max_seq_length

REQUIRED_FIELDS = {"text", "type", "entities"}
VALID_TYPES     = {"preference", "fact", "relationship", "decision", "event"}

# ── Load ──────────────────────────────────────────────────────────────────
raw = load_dataset("json", data_files=RAW_FILE, split="train")
print(f"Loaded {len(raw)} rows")

# ── Validate ──────────────────────────────────────────────────────────────
def extract_assistant_json(row):
    for msg in row["messages"]:
        if msg["role"] == "assistant":
            content = re.sub(r"^```(?:json)?\s*", "", msg["content"].strip())
            content = re.sub(r"\s*```$", "", content)
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                raise ValueError("Not a list")
            return parsed
    raise ValueError("No assistant message")

def row_is_valid(row, idx):
    try:
        memories = extract_assistant_json(row)
    except Exception:
        return False
    for m in memories:
        if REQUIRED_FIELDS - set(m.keys()):
            return False
        if m.get("type") not in VALID_TYPES:
            return False
        if not isinstance(m.get("entities"), list):
            return False
        if not isinstance(m.get("text"), str) or not m["text"].strip():
            return False
    return True

valid_idx = [i for i, row in enumerate(raw) if row_is_valid(row, i)]
clean = raw.select(valid_idx)
print(f"After validation: {len(clean)} rows ({len(raw)-len(clean)} dropped)")

# ── Deduplicate ───────────────────────────────────────────────────────────
seen, unique_idx = set(), []
for i, row in enumerate(clean):
    key = " ".join(
        next((m["content"] for m in row["messages"] if m["role"] == "user"), "").split()
    )
    if key not in seen:
        seen.add(key)
        unique_idx.append(i)
clean = clean.select(unique_idx)
print(f"After dedup:      {len(clean)} rows")

# ── Token lengths ─────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
lengths = [
    len(tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                      add_generation_prompt=False))
    for row in clean
]
sorted_lengths = sorted(lengths)
p = lambda pct: sorted_lengths[int(pct / 100 * len(sorted_lengths))]
max_seq = 2 ** math.ceil(math.log2(p(MAX_PCT)))
print(f"Token p95={p(95)}, p99={p(99)}, max={max(lengths)} → max_seq_length={max_seq}")

# ── Length filter ─────────────────────────────────────────────────────────
keep = [i for i, l in enumerate(lengths) if l <= max_seq]
clean = clean.select(keep)
print(f"After length filter: {len(clean)} rows")

# ── Split ─────────────────────────────────────────────────────────────────
clean = clean.shuffle(seed=SPLIT_SEED)
n = len(clean)
n_test = max(50, int(n * 0.10))
n_val  = max(50, int(n * 0.10))
train_ds = clean.select(range(n - n_test - n_val))
val_ds   = clean.select(range(n - n_test - n_val, n - n_test))
test_ds  = clean.select(range(n - n_test, n))

# ── Save ──────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
train_ds.to_json(f"{OUTPUT_DIR}/train.jsonl")
val_ds.to_json(f"{OUTPUT_DIR}/val.jsonl")
test_ds.to_json(f"{OUTPUT_DIR}/test.jsonl")
json.dump({"max_seq_length": max_seq, "model_id": MODEL_ID,
           "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds)},
          open(f"{OUTPUT_DIR}/meta.json", "w"), indent=2)

print(f"\nDone. train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
print(f"Use max_seq_length={max_seq} in Chapter 15.")
```

---

## Common mistakes

**1. Peeking at the test set before Chapter 18.**
The temptation is real: you have a model, you want to know if it's good. Don't. Run all interim checks on the validation set. One premature look at test turns it into a second validation set and your final accuracy number is biased.

*Fix*: save test.jsonl and don't open it. If you're discipline-challenged, rename it to `do-not-open-until-ch18.jsonl`.

**2. Not checking the chat template output.**
You assume the template is applied correctly, skip Step 6, and spend four hours training. The model produces blank output or repeats the input. Inspection would have caught that the template was using `add_generation_prompt=True` during training (which appends a partial assistant turn and confuses the loss calculation).

*Fix*: always render and read five examples before starting a training run.

**3. Deduplicating on the full row instead of just the input.**
Two rows with identical user messages but slightly different assistant outputs are both noise — the model sees the same input and gets conflicting right answers, which causes unstable training.

*Fix*: dedup on the user content only, as shown above.

**4. Setting `max_seq_length` based on the model's maximum context window instead of your data.**
Qwen3-8B supports 128k tokens. If you set `max_seq_length=128000` on a dataset whose examples are ~300 tokens, you waste enormous GPU memory padding every batch to that length.

*Fix*: use the histogram. Set `max_seq_length` based on your actual data's 95th percentile, rounded to a power of 2.

**5. Skipping validation and loading raw data directly into the trainer.**
The trainer will hit a JSON parse error mid-run, fail without a clear message, and you'll spend an hour debugging.

*Fix*: always run the validation step. It takes seconds and surfaces problems immediately.

**6. Losing track of which `max_seq_length` was used.**
You run the cleaning script, forget what value it produced, and set a different number in Chapter 15. Now some training examples are silently truncated.

*Fix*: `meta.json` records this automatically. Check it before training.

---

## Recap

- Load JSONL with `datasets.load_dataset("json", ...)` — it handles large files efficiently without loading everything into memory at once.
- Validate every row: parse the assistant message as JSON, check required fields (`text`, `type`, `entities`), check that `type` is a known value, check that `entities` is a list.
- Drop duplicates by hashing the user message content — same input with different outputs is noise.
- Build a token-length histogram with the exact tokenizer you'll use for training; set `max_seq_length` to cover the 95th percentile, rounded to a power of 2.
- Drop the rare examples that exceed `max_seq_length` — they would be silently truncated during training.
- Split 80 / 10 / 10: train, validation, test. Shuffle with a fixed seed for reproducibility.
- Render five examples through `tokenizer.apply_chat_template` and read them before training. Catch template bugs now, not after a four-hour GPU run.
- Save `meta.json` with the `max_seq_length` and split sizes so you have a provenance record.

## Next

**Ch15 — Your First Fine-Tune with Unsloth (Full Script)**: hand the cleaned `train.jsonl` and `val.jsonl` to Unsloth's `SFTTrainer` and run your first real training loop.
