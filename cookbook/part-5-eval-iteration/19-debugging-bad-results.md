# Ch19 - When It Goes Wrong: A Debugging Playbook

Something is broken. Your model is outputting garbled JSON, or hallucinating memories that were never in the conversation, or producing three-word sentences that go on forever. Training finished without errors. The loss curve looked reasonable. And yet — it doesn't work.

This chapter is your triage guide. For every failure mode that shows up repeatedly in real memory-extraction fine-tuning, we cover: how to recognize it, what is actually causing it, and exactly what to change. We end on the one principle that explains most failures: **the problem is almost always your data, not your training code**.

---

## What you'll learn

- How to recognize eight distinct failure modes by their symptoms
- A quick diagnosis script you can run against any model output
- Why most problems trace back to data quality, not hyperparameters
- Concrete fixes for each failure, from a one-line code change to a full data audit
- How to check for catastrophic forgetting without a benchmark suite

---

## Concepts you need first

### Why "training ran fine" doesn't mean "model is fine"

Training loss measures how well the model fits your training examples. It says nothing about whether those examples were correct, well-structured, or representative of real inputs. A model can reach near-zero training loss on bad data and be completely useless at inference time.

Think of it like a student who memorized wrong answers from a bad textbook. The exam practice scores look great. The real exam is a disaster.

This is why debugging starts with your data, not your training code.

### What "inference" means here

Throughout this chapter, "inference" means running your fine-tuned model on new inputs — calling it after training, the same way a user would. This is different from "evaluation," which is measuring quality with metrics. Inference is just: give the model a conversation, get output back.

When we say a bug "shows up at inference time," we mean: training succeeded, but the model produces wrong output when you actually use it.

### Overfitting in plain English

Overfitting happens when the model memorizes your training examples instead of learning the underlying pattern. A perfectly memorized training set scores well on training loss but fails on inputs the model hasn't seen before. The fix is nearly always: more diverse data, or fewer training steps.

### Catastrophic forgetting in plain English

Language models are pre-trained on hundreds of billions of words. That pre-training gives them grammar, reasoning, and world knowledge. Fine-tuning nudges the weights away from that starting point. If you nudge too hard — too many epochs, too high a learning rate — you can overwrite the pre-training. The model "forgets" how to write coherent sentences, follow basic instructions outside your task, or reason. This is called catastrophic forgetting.

---

## Complete inference helper — start here before using any diagnosis script

Every diagnosis script in this chapter needs to call the model and get output back. Here is the full, runnable pattern in one place. Copy this before you run any of the failure-specific snippets below.

```python
# inference_helper.py — paste this at the top of any diagnosis script.
# It gives you an `infer()` function you can call anywhere in this chapter.

import torch
from unsloth import FastLanguageModel
from config import SYSTEM_PROMPT, MODEL_PATH, MAX_NEW_TOKENS

# --- Step 1: Load the model and tokenizer (do this once, not inside the loop) ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,    # path to your saved fine-tune, e.g. "outputs/memory-extractor-v1"
    max_seq_length=2048,
    load_in_4bit=True,        # keeps VRAM usage low; must match how you originally trained
)

# --- Step 2: Enable fast inference kernels ---
# Required by Unsloth — enables fast inference kernels; omit this and generation
# may be slower or subtly different.
FastLanguageModel.for_inference(model)


def infer(conversation_text: str) -> str:
    """
    Run the fine-tuned model on a single conversation and return the raw output string.

    Args:
        conversation_text: The plain-text conversation to extract memories from.

    Returns:
        The raw string the model produced (may need JSON parsing by the caller).
    """
    # Step 3: Build the prompt using the same chat template as training.
    # apply_chat_template adds the model-specific special tokens so the model
    # sees exactly the structure it learned from.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": conversation_text},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # adds the assistant-turn opener so the model knows to start generating
    )

    # Step 4: Tokenize the prompt and move to GPU.
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

    # Step 5: Generate — no gradient tracking needed at inference time.
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,             # greedy decoding; more reliable for structured JSON output
            repetition_penalty=1.1,      # reduces repetition loops
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Step 6: Decode only the NEW tokens — not the prompt we fed in.
    # inputs["input_ids"].shape[1] is the length of the prompt in tokens.
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# Usage:
# raw = infer("Alex: I use Obsidian for notes. Sam: Nice, do you pay for it? Alex: No, free tier.")
# print(raw)   # should be a JSON array string
```

Keep this file open — the diagnosis scripts below call `infer()` by name.

---

## The symptom table

Scan down the "Symptom" column, find your failure, and jump to the section.

| # | Symptom | Most likely cause | Quick check |
|---|---------|-------------------|-------------|
| 1 | Output is not valid JSON | Wrong system prompt at inference / model undertrained | `json.loads(output)` raises an exception |
| 2 | Model ignores the system prompt | Template mismatch at inference | Print the raw tokenized prompt |
| 3 | Empty list `[]` on every input | Over-conservative training data | Count empty-list rows in your dataset |
| 4 | Every fact in the universe extracted | Over-extractive training data | Check average memories-per-example |
| 5 | Repetitive or runaway output | Generation parameters wrong | Look for looping text in raw output |
| 6 | Works on training examples, fails on new ones | Overfitting | Compare train vs. held-out accuracy |
| 7 | Lost general reasoning ability | Catastrophic forgetting | Ask the model an off-task question |
| 8 | Output was fine during eval, broken in the app | Template mismatch at inference | Compare eval code vs. app code character-by-character |

---

## Failure 1 — Invalid JSON output

### What it looks like

```
[{"text": "Alex uses Obsidian", "type": "fact",   ← truncated mid-object
```

or

```
Sure! Here are the memories I found:
[{"text": "Alex uses Obsidian", ...}]
```

or

```
[
  {"text": "Alex uses Obsidian", "type": "fact", "entities": ["Alex", "Obsidian"]},
  INVALID ENTRY HERE
]
```

### Diagnosis

Run this against a batch of outputs before anything else:

```python
import json

def audit_json_outputs(model_outputs: list[str]) -> dict:
    """
    Check a list of raw model output strings for JSON validity.
    Returns a summary of how many pass and what the failures look like.

    Args:
        model_outputs: List of raw strings the model produced.

    Returns:
        A dict with counts and a list of (index, error, raw_output) for failures.
    """
    results = {"valid": 0, "invalid": 0, "failures": []}

    for i, output in enumerate(model_outputs):
        # Strip leading/trailing whitespace — the model sometimes adds a newline
        cleaned = output.strip()

        # Some models wrap output in markdown fences despite instructions.
        # Strip those before trying to parse.
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Remove the opening fence (```json or ```) and closing fence (```)
            cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        try:
            parsed = json.loads(cleaned)
            # Also check it's a list, not just valid JSON
            if not isinstance(parsed, list):
                raise ValueError(f"Expected list, got {type(parsed).__name__}")
            results["valid"] += 1
        except (json.JSONDecodeError, ValueError) as e:
            results["invalid"] += 1
            results["failures"].append({
                "index": i,
                "error": str(e),
                # Show only the first 200 chars so the log stays readable
                "preview": output[:200]
            })

    results["validity_rate"] = results["valid"] / len(model_outputs) if model_outputs else 0
    return results


# Usage:
# outputs = [model.generate(prompt) for prompt in test_inputs]  # your inference loop
# report = audit_json_outputs(outputs)
# print(f"Valid: {report['valid']}/{report['valid'] + report['invalid']} ({report['validity_rate']:.1%})")
# for f in report['failures'][:5]:
#     print(f"  [{f['index']}] {f['error']}: {f['preview']}")
```

### Likely causes and fixes

**Cause A — The model adds prose before or after the JSON.**
The system prompt says "return ONLY a valid JSON array" but some training examples in your data include an explanation. One bad example is enough to teach the model that preamble is acceptable.

Fix: audit your training data. Search for any assistant-turn that does not start with `[`:

```python
import json, pathlib

def find_non_json_rows(jsonl_path: str) -> list[int]:
    """
    Return the line numbers of training rows where the assistant
    content does not begin with '[' (i.e., is not a JSON array literal).
    """
    bad_rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            assistant_content = row["messages"][2]["content"].strip()
            if not assistant_content.startswith("["):
                bad_rows.append(line_num)
    return bad_rows

# bad = find_non_json_rows("data/train/memory_extraction_train.jsonl")
# print(f"Found {len(bad)} rows with non-JSON assistant content: {bad[:10]}")
```

Fix every flagged row or delete it. Retrain.

**Cause B — Truncated output (the JSON cuts off mid-object).**
The model hit the `max_new_tokens` limit during inference. Raise it. For most memory-extraction outputs, 1024 tokens is enough; for long conversations, use 2048.

```python
from unsloth import FastLanguageModel

# Required by Unsloth — enables fast inference kernels; omit this and generation
# may be slower or subtly different.
FastLanguageModel.for_inference(model)

# In your inference call, increase max_new_tokens:
outputs = model.generate(
    **inputs,
    max_new_tokens=2048,   # was 512 — too short for multi-memory outputs
    do_sample=False,       # greedy decoding for structured output; more reliable than sampling
)
```

**Cause C — Model not trained long enough.**
If validity rate is below ~70%, the model may simply not have learned the output format yet. Check your training loss — if it had not converged by the end of training, run more steps (increase `max_steps` or `num_train_epochs` by 50% and retrain).

---

## Failure 2 — Model ignores instructions

### What it looks like

You ask for memory extraction. The model writes a friendly summary paragraph instead. Or it responds in a different language. Or it answers as if it's a general assistant with no task definition.

### Diagnosis

Print the exact prompt your inference code is sending — not the Python string, but the fully tokenized and decoded prompt:

```python
from transformers import AutoTokenizer

def print_inference_prompt(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_message: str
) -> None:
    """
    Apply the chat template and print the exact text the model sees.
    Compare this character-for-character against what training used.
    """
    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": user_message},
    ]
    # apply_chat_template adds the model-specific special tokens
    # tokenize=False returns the text version so we can read it
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # adds the assistant turn opener
    )
    print("=== FULL INFERENCE PROMPT ===")
    print(repr(prompt_text))  # repr() shows special characters like \n explicitly
    print("=== END PROMPT ===")
    print(f"Total characters: {len(prompt_text)}")

# print_inference_prompt(tokenizer, SYSTEM_PROMPT, user_conversation)
```

Compare what you see to how training formatted prompts. If they differ — different special tokens, missing system turn, different role labels — you have a template mismatch. See Failure 8 for the fix.

### Other cause — system prompt is missing entirely

Some inference code paths skip the system message by accident:

```python
# WRONG — no system message:
messages = [{"role": "user", "content": conversation_text}]

# RIGHT — system message first, always:
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": conversation_text},
]
```

The model learned to produce JSON only when it sees the system prompt. Without it, it falls back to generic behavior.

---

## Failure 3 — Empty list on every input

### What it looks like

Every single output is `[]`, even for conversations packed with facts.

### Diagnosis

Count how many rows in your training set have an empty list as the assistant output:

```python
import json

def count_empty_assistant_outputs(jsonl_path: str) -> tuple[int, int]:
    """
    Count rows where the assistant output is an empty JSON list [].

    Returns:
        (empty_count, total_count)
    """
    empty = 0
    total = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            assistant_content = row["messages"][2]["content"].strip()
            try:
                parsed = json.loads(assistant_content)
                if isinstance(parsed, list) and len(parsed) == 0:
                    empty += 1
            except json.JSONDecodeError:
                pass  # invalid JSON counted separately

    return empty, total

# empty, total = count_empty_assistant_outputs("data/train/memory_extraction_train.jsonl")
# print(f"Empty-list rows: {empty}/{total} ({empty/total:.1%})")
```

If more than ~15% of your training rows have `[]` as the output, the model learned that returning nothing is the safe, common response.

### Fix

Your training data needs more examples where memories are actually extracted. A rough target: fewer than 10% of rows should have empty output, and those should genuinely be conversations with no memorable content (pure small talk, filler).

If you used synthetic generation (covered in *Ch13 - Creating Your Training Data with Synthetic Generation*), re-run generation with a prompt that explicitly asks for information-dense conversations and filters out empty-output pairs before saving.

---

## Failure 4 — Over-extraction (everything becomes a memory)

### What it looks like

The model turns "It's raining today" into:

```json
[
  {"text": "It was raining at the time of the conversation.", "type": "fact", "entities": []},
  {"text": "The user mentioned weather.", "type": "fact", "entities": []},
  {"text": "Weather conditions were discussed.", "type": "fact", "entities": []}
]
```

Three memories for one throwaway comment. The model extracts every clause, every filler phrase.

### Diagnosis

Compute the average number of memories per output across your test set:

```python
import json, statistics

def memory_count_stats(model_outputs: list[str]) -> dict:
    """
    Given a list of raw model output strings, compute statistics
    on how many memories the model is extracting per conversation.
    """
    counts = []
    for output in model_outputs:
        try:
            parsed = json.loads(output.strip())
            if isinstance(parsed, list):
                counts.append(len(parsed))
        except json.JSONDecodeError:
            pass  # skip invalid outputs for this audit

    if not counts:
        return {"error": "no parseable outputs"}

    return {
        "mean": round(statistics.mean(counts), 1),
        "median": statistics.median(counts),
        "max": max(counts),
        "min": min(counts),
        "p90": sorted(counts)[int(len(counts) * 0.9)],  # 90th percentile
    }

# stats = memory_count_stats(model_outputs)
# print(stats)
# If mean is above ~6-7 for typical short conversations, you likely have over-extraction.
```

### Fix

Over-extraction is a data quality problem. Your training examples likely have too many memories labeled per conversation, or include memories for trivial statements. Go back to your training data and apply the "would this matter to someone a week later?" test to every memory. If the answer is no, remove it from the assistant output. Retrain with cleaner labels.

Also tighten your system prompt to add an explicit instruction:

```python
SYSTEM_PROMPT = """...
Rules:
- Extract only facts that would be worth remembering a week later.
- Omit small talk, filler, and generic statements with no named subject or lasting relevance.
- One fact per memory object. Do not bundle multiple facts into one.
- If there are no memorable facts, return an empty list: []
- Return ONLY a valid JSON array. No explanation, no markdown fences, no extra text.
"""
```

---

## Failure 5 — Repetition and runaway generation

### What it looks like

```json
[
  {"text": "Alex uses Obsidian.", "type": "fact", "entities": ["Alex"]},
  {"text": "Alex uses Obsidian.", "type": "fact", "entities": ["Alex"]},
  {"text": "Alex uses Obsidian.", "type": "fact", "entities": ["Alex"]},
  ...  (repeats for 2000 tokens)
```

Or the model closes one JSON object and then just... keeps generating characters without stopping.

### Diagnosis

This is almost always a generation parameter problem, not a data problem. Check how you're calling `generate`:

```python
from unsloth import FastLanguageModel

# Required by Unsloth — enables fast inference kernels; omit this and generation
# may be slower or subtly different.
FastLanguageModel.for_inference(model)

# Check your current generation call for these parameters:

outputs = model.generate(
    **inputs,
    max_new_tokens=2048,

    # REPETITION PENALTY — values above 1.0 penalize repeated tokens.
    # 1.0 means no penalty (default). Try 1.1–1.3 for structured output.
    repetition_penalty=1.1,

    # For structured JSON output, greedy decoding (do_sample=False) is
    # more reliable than sampling. Sampling can produce random JSON fragments.
    do_sample=False,

    # EOS token — make sure the model knows when to stop.
    # This is set automatically from the tokenizer, but verify it's present.
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,  # prevents pad_token warnings
)
```

Also add a stopping criterion on the closing bracket — this tells the model to stop as soon as it produces a complete JSON array:

```python
from transformers import StoppingCriteria, StoppingCriteriaList

class JSONArrayComplete(StoppingCriteria):
    """
    Stop generation once the output contains a complete JSON array
    (i.e., a closing bracket ']' appears after a complete object).
    This prevents runaway generation past the end of the JSON.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.found_close = False

    def __call__(self, input_ids, scores, **kwargs):
        # Decode what has been generated so far
        generated = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # Look for a closing bracket that follows at least one object
        stripped = generated.strip()
        if stripped.endswith("]") and stripped.startswith("["):
            return True  # signal: stop generating
        return False

# stopping_criteria = StoppingCriteriaList([JSONArrayComplete(tokenizer)])
# outputs = model.generate(**inputs, stopping_criteria=stopping_criteria, ...)
```

### Secondary cause — training data has runaway examples

If some training examples in your JSONL have extremely long assistant outputs (hundreds of memories, or malformed JSON that goes on for many lines), the model learned that long outputs are acceptable. Audit with:

```python
import json

def find_long_assistant_outputs(jsonl_path: str, token_threshold: int = 800) -> list[int]:
    """
    Find training rows where the assistant content is suspiciously long.
    Long outputs can teach the model that generating forever is OK.
    """
    long_rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = row["messages"][2]["content"]
            # Rough character count: ~4 chars per token is a common approximation
            approx_tokens = len(content) / 4
            if approx_tokens > token_threshold:
                long_rows.append((line_num, int(approx_tokens)))
    return long_rows

# long = find_long_assistant_outputs("data/train/memory_extraction_train.jsonl")
# print(f"Long rows: {long[:10]}")
```

Cap or remove rows that are significantly longer than your typical output.

---

## Failure 6 — Overfitting

### What it looks like

The model scores well on your training examples (or examples very similar to them) but fails on new conversations with different phrasings, different speakers, or different topics.

### Diagnosis

Split a sample of your training data into "seen" examples and "unseen" examples. Run the model on both and compare the validity rate and memory quality:

```python
import json, random

def split_for_overfit_check(jsonl_path: str, seen_n: int = 50, unseen_n: int = 50, seed: int = 42):
    """
    Load rows from the training file and split them into two groups:
    - 'seen': rows the model trained on (in-distribution)
    - 'unseen': rows held out before training (out-of-distribution)

    In practice, 'unseen' rows should be from your VALIDATION split,
    not the training file. This function illustrates the pattern.
    """
    random.seed(seed)
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    random.shuffle(rows)
    seen = rows[:seen_n]
    unseen = rows[seen_n:seen_n + unseen_n]
    return seen, unseen

# Then run inference on both groups and compare.
# `infer()` is defined in the "Complete inference helper" section at the top of this chapter.
# If you haven't set it up yet, go there first — it loads the model, calls
# FastLanguageModel.for_inference(model), and defines the full generate loop.
# Once you have it, run:
# seen_outputs = [infer(r["messages"][1]["content"]) for r in seen]
# unseen_outputs = [infer(r["messages"][1]["content"]) for r in unseen]
# seen_report = audit_json_outputs(seen_outputs)
# unseen_report = audit_json_outputs(unseen_outputs)
# print(f"Seen validity:   {seen_report['validity_rate']:.1%}")
# print(f"Unseen validity: {unseen_report['validity_rate']:.1%}")
# A big gap (>20%) signals overfitting.
```

Always hold out a validation split before training — *Ch14 - Cleaning, Splitting, and Sanity-Checking Data* covers this. If you skipped it, do it now.

### Fix

Overfitting is fixed with more data variety and fewer training steps, in that order:

1. **Add more diverse training examples.** Regenerate synthetic data with a wider range of conversation styles, topics, and speakers. If your dataset is all tech-related conversations, add some personal, travel, finance, and health conversations.

2. **Reduce epochs.** If you trained for 5 epochs, try 2–3. More passes over the same data increases overfitting risk. See *Ch16 - Hyperparameters: Which Knobs to Turn and When* for guidance on epoch count.

3. **Check LoRA rank.** A higher LoRA rank (e.g., `r=64`) means more trainable parameters, which means faster overfitting on small datasets. Try `r=16` or `r=8` if your dataset has fewer than ~1,000 examples.

---

## Failure 7 — Catastrophic forgetting

### What it looks like

The model extracts memories adequately, but something else is broken: it writes incoherent sentences, can't follow a simple if/then instruction, produces grammatical errors it would never have made before fine-tuning, or loses the ability to handle edge cases that require reasoning.

### Diagnosis

Ask the model something completely outside your task. No fine-tuning data touched these topics, so if the model is behaving strangely here, the pre-training knowledge has degraded:

```python
from unsloth import FastLanguageModel

def check_general_ability(model, tokenizer, device="cuda"):
    """
    Run a small battery of off-task prompts to check whether the model
    has retained general reasoning ability after fine-tuning.

    These prompts have nothing to do with memory extraction — that's the point.
    A healthy fine-tuned model should still handle them correctly.
    """
    # Required by Unsloth — enables fast inference kernels; omit this and generation
    # may be slower or subtly different.
    FastLanguageModel.for_inference(model)
    test_prompts = [
        # Basic arithmetic reasoning
        "What is 17 multiplied by 6? Think step by step.",
        # Simple logic
        "If all cats are mammals, and Whiskers is a cat, is Whiskers a mammal? Explain.",
        # Instruction following outside the task
        "List the first five prime numbers, separated by commas.",
        # Language coherence
        "Write one sentence explaining what a dictionary is in Python.",
    ]

    results = []
    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        # Apply chat template WITHOUT a system prompt — tests raw ability
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with __import__("torch").no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens (not the prompt)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append({"prompt": prompt, "response": response})
        print(f"\nQ: {prompt}\nA: {response}\n{'─'*60}")

    return results

# check_general_ability(model, tokenizer)
```

If the answers are nonsensical, the model has forgotten. Compare against the base model's answers to the same prompts (load the base model without your LoRA weights and run the same prompts).

### Fix

Catastrophic forgetting is caused by too many training steps at too high a learning rate. The fixes, roughly in order of how much they help:

1. **Reduce the learning rate.** The default Unsloth learning rate of `2e-4` is fine for most cases. If you went higher, drop back to `1e-4` or `2e-4`.

2. **Reduce epochs.** More epochs = more drift from pre-training. Two epochs is often sufficient. Three is usually a safe ceiling.

3. **Use a smaller LoRA rank.** `r=16` is a good default. `r=64` or higher trains more of the model and risks more forgetting on small datasets.

4. **Add regularization via weight decay.** In your `TrainingArguments`, set `weight_decay=0.01`. This penalizes large weight changes and keeps the fine-tuned model closer to its pre-trained starting point.

```python
from trl import SFTConfig

# Adjusted training config to reduce forgetting risk:
training_args = SFTConfig(
    output_dir="outputs/memory-extractor-v2",
    num_train_epochs=2,          # was 5 — reduce to limit drift
    learning_rate=2e-4,          # keep at or below 2e-4
    weight_decay=0.01,           # adds regularization
    warmup_ratio=0.05,           # gentle warm-up to avoid early instability
    lr_scheduler_type="cosine",  # cosine decay reaches a lower final LR
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
)
```

---

## Failure 8 — Template mismatch at inference

### What it looks like

The model worked during evaluation (you ran `eval_model.py` and results were good), but when you plug it into your app or API server, the output is wrong. You did not change the model. Only the calling code changed.

This is one of the most common "production" bugs in fine-tuned model deployments.

### Diagnosis

The root cause is always the same: the tokenizer's `apply_chat_template` is being called differently in two places, and the model sees a different prompt structure than it was trained on.

Run this comparison script to surface the mismatch:

```python
def compare_prompt_rendering(
    tokenizer,
    system_prompt: str,
    user_message: str,
    label_a: str = "Eval code",
    label_b: str = "App code",
):
    """
    Render the same conversation through two different prompt-building
    approaches and show exactly where they differ.

    Replace the two build_* functions below with your actual eval and app code.
    """

    # ── Approach A: how your eval script builds the prompt ──────────────────
    def build_prompt_eval(sys, usr):
        messages = [
            {"role": "system",  "content": sys},
            {"role": "user",    "content": usr},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ── Approach B: how your app builds the prompt ───────────────────────────
    # Common mistake: missing system role, wrong role name, manual string concat
    def build_prompt_app(sys, usr):
        # Example of a buggy app-side implementation:
        # Forgot to include the system message
        messages = [
            {"role": "user", "content": f"{sys}\n\n{usr}"}  # wrong structure
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    prompt_a = build_prompt_eval(system_prompt, user_message)
    prompt_b = build_prompt_app(system_prompt, user_message)

    print(f"=== {label_a} ===")
    print(repr(prompt_a[:300]))
    print(f"\n=== {label_b} ===")
    print(repr(prompt_b[:300]))
    print(f"\nMatch: {prompt_a == prompt_b}")

    if prompt_a != prompt_b:
        # Find first character that differs
        for i, (ca, cb) in enumerate(zip(prompt_a, prompt_b)):
            if ca != cb:
                print(f"First difference at character {i}:")
                print(f"  {label_a}: ...{repr(prompt_a[max(0,i-20):i+30])}...")
                print(f"  {label_b}: ...{repr(prompt_b[max(0,i-20):i+30])}...")
                break

# compare_prompt_rendering(tokenizer, SYSTEM_PROMPT, sample_conversation)
```

### Fix

Store your system prompt as a constant in a single shared module and import it everywhere. Never type it twice:

```python
# config.py — the single source of truth for everything inference-related
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
- Extract only facts that would be worth remembering a week later.
- If there are no memorable facts in the conversation, return an empty list: []
- Return ONLY a valid JSON array. No explanation, no markdown fences, no extra text.
"""

MODEL_PATH = "outputs/memory-extractor-v1"  # path to your saved fine-tune
MAX_NEW_TOKENS = 2048
```

```python
# eval_model.py
from config import SYSTEM_PROMPT, MODEL_PATH, MAX_NEW_TOKENS

# app_server.py
from config import SYSTEM_PROMPT, MODEL_PATH, MAX_NEW_TOKENS
```

One import. One source of truth. Template mismatches become impossible.

---

## The OOM (out-of-memory) crash

Out-of-memory errors are not a model quality problem — they are a resource problem. But they are common enough to include here.

### What it looks like

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X GiB.
```

This can happen during training or during inference.

### Fixes, roughly in order

```python
# ── During training: reduce memory usage ───────────────────────────────────

# 1. Reduce per-device batch size
per_device_train_batch_size = 1   # was 4

# 2. Increase gradient accumulation to compensate
# Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
# Keep the effective batch size the same (e.g., 1 * 16 = 16 instead of 4 * 4 = 16)
gradient_accumulation_steps = 16  # was 4

# 3. Reduce the maximum sequence length
# Unsloth's SFTTrainer accepts max_seq_length — shorter = less memory
max_seq_length = 1024  # was 2048

# 4. Make sure 4-bit quantization is active (Unsloth default)
# load_in_4bit=True in FastLanguageModel.from_pretrained()
# If you turned this off for any reason, turn it back on

# ── During inference: reduce memory usage ──────────────────────────────────

# 1. Enable fast inference kernels before generating
# Required by Unsloth — enables fast inference kernels; omit this and generation
# may be slower or subtly different.
from unsloth import FastLanguageModel
FastLanguageModel.for_inference(model)

# 2. Use torch.no_grad() — prevents gradient tracking during inference
import torch
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=2048)

# 2. Move inputs to the right device explicitly
inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

# 3. Free GPU memory between inference calls if processing a large batch
torch.cuda.empty_cache()
```

If you are still OOM after all of the above, your GPU simply does not have enough VRAM for the model size you chose. See *Ch8 - Hardware, GPUs, and Setting Up Your Environment* for VRAM requirements by model size, and *Ch10 - Choosing Your Base Model: Qwen vs Gemma* for smaller alternatives.

---

## The root principle: most problems are data problems

Step back and look at the failure table again. Of the eight failures:

- Failure 1 (invalid JSON): caused by training data that includes prose in the assistant turn
- Failure 3 (empty list): caused by training data with too many empty outputs
- Failure 4 (over-extraction): caused by training data with over-labeled memories
- Failure 5 (repetition): can be generation params, but often traces to long/malformed training rows
- Failure 6 (overfitting): caused by insufficient variety in training data
- Failure 7 (forgetting): caused by too many training steps — which itself is often compensating for noisy data that isn't converging cleanly

Only Failure 2, 8, and OOM are primarily code/config problems. Everything else traces to data.

Before you adjust a hyperparameter, audit your data. Before you retrain, fix your data. The fastest path to a working model is almost always: look at 50 random training examples by hand, identify the patterns in what's wrong, fix those patterns at the source.

```python
import json, random

def spot_check_training_data(jsonl_path: str, n: int = 20, seed: int = 42) -> None:
    """
    Print N randomly sampled training rows for manual inspection.
    Run this before every retraining run.

    For each row, shows:
    - The first 300 chars of the conversation (user message)
    - The full assistant output (the memories)
    - A basic validity check
    """
    random.seed(seed)
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    sample = random.sample(rows, min(n, len(rows)))

    for i, row in enumerate(sample):
        msgs = row["messages"]
        user_content = msgs[1]["content"]
        asst_content = msgs[2]["content"]

        print(f"\n{'='*60}")
        print(f"Example {i+1}/{len(sample)}")
        print(f"{'─'*60}")
        print(f"CONVERSATION (first 300 chars):\n{user_content[:300]}{'...' if len(user_content) > 300 else ''}")
        print(f"\nMEMORIES:")

        try:
            memories = json.loads(asst_content)
            if not memories:
                print("  (empty list)")
            for m in memories:
                print(f"  [{m.get('type','?')}] {m.get('text','?')} — entities: {m.get('entities','?')}")
        except json.JSONDecodeError as e:
            print(f"  INVALID JSON: {e}")
            print(f"  Raw: {asst_content[:200]}")

# spot_check_training_data("data/train/memory_extraction_train.jsonl", n=20)
```

Run this. Look at the output. You will almost certainly find something you did not expect.

---

## Common mistakes

**Debugging hyperparameters before auditing data.** This is the most expensive mistake you can make. Hyperparameter tuning takes time and compute. Data auditing takes minutes and a text editor. Always audit data first.

**Comparing outputs to your intuition rather than to a held-out set.** A model that "seems fine" on five hand-picked examples and fails on the 50th is overfitting. Use a validation split and measure systematically.

**Changing two things at once.** When debugging, change one variable per retraining run. If you change the learning rate and fix 30 training rows simultaneously, you will not know which one fixed the problem — or if a new problem was introduced.

**Assuming the base model is at fault.** Qwen3 and Gemma 3 are capable models. If a fine-tune is behaving strangely, the cause is almost never the base model. Look at your data and your inference code first.

**Not printing the raw prompt at inference time.** The tokenized prompt is the actual input to the model. If you have never printed it, you do not actually know what the model is seeing. Print it once. Check it against what training used.

---

## Recap

- Eight failure modes cover nearly all real memory-extraction bugs: invalid JSON, ignored instructions, empty outputs, over-extraction, repetition, overfitting, catastrophic forgetting, and template mismatch.
- Invalid JSON at inference usually means training data has non-JSON content in assistant turns — audit with `find_non_json_rows`.
- Empty list outputs mean too many empty-list training examples — target fewer than 10% empty in your dataset.
- Repetition and runaway generation are fixed with `repetition_penalty`, `do_sample=False`, and a JSON stopping criterion.
- Overfitting is fixed by adding more diverse training data and reducing epochs, in that order.
- Catastrophic forgetting is caused by too many epochs or too high a learning rate — reduce both and add `weight_decay=0.01`.
- Template mismatch is fixed by importing `SYSTEM_PROMPT` from a single `config.py` in all code paths.
- Most problems are data problems. Spot-check 20–50 random training examples before adjusting any hyperparameter.

## Next

*Ch20 - Iterating: From a Mediocre Model to a Good One* — now that you can diagnose what's wrong, we cover the full iteration loop: how to measure progress, when to add more data vs. more steps, and how to know when your model is good enough to ship.
