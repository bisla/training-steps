# Ch18 - Did It Actually Work? Evaluating Memory Extraction

You ran training. Loss went down. The script finished without errors. Congratulations — but that only means the model got better at predicting the tokens in your training set. It says nothing about whether the model actually extracts memories well on conversations it has never seen.

This chapter is about closing that gap. You will build a real evaluation pipeline: one that checks whether the model produces valid JSON, measures how accurately it finds the right memories, and uses a language model as a judge to score semantic quality. By the end you will have a single `evaluate.py` script that compares three systems side by side: your fine-tuned model, the base model you started from, and a prompted-only baseline.

---

## What you'll learn

- Why training loss is a poor proxy for real-world quality — and what you should measure instead
- How to compute parse-rate: the fraction of outputs that are valid, schema-conforming JSON
- How to define a "memory match" and compute precision, recall, and F1 over extracted memory sets
- How to use a language model as a judge to score semantic correctness when exact matching falls short
- How to compare your fine-tuned model against the base model and a prompt-only baseline with a single runnable script

---

## Concepts you need first

### Why training loss is not the finish line

Loss (introduced in *Ch7 - How Training Actually Works*) measures how surprised the model was by the tokens in your training set. A loss of 0.3 at the end of training means the model got very good at predicting those specific training examples.

The problem: your training examples are not your goal. Your goal is a model that extracts memories accurately from new conversations it has never seen. Loss does not measure that. Think of it like an essay exam. A student who memorized the practice essays scores zero on the real exam if the questions change. Loss is the practice-essay score. You need the real-exam score.

Real-world quality for our task lives in three layers:

1. **Format correctness** — did the model produce valid, schema-conforming JSON at all?
2. **Content accuracy** — did it find the right memories? Did it miss any? Did it invent ones that aren't there?
3. **Semantic quality** — even if the wording differs, is each extracted memory correct in meaning?

Each layer requires a different measurement tool.

### Precision, recall, and F1 — without the jargon

These three numbers come from information retrieval, but you can understand them in a minute with a concrete example.

Say the correct output for a conversation is 5 memories. Your model produces 4 memories. 3 of them are correct; 1 is hallucinated; 2 real memories were missed.

- **Precision** = how many of the things I returned are actually correct? Here: 3 out of 4 returned = 0.75. High precision means the model is not hallucinating much.
- **Recall** = how many of the correct things did I actually find? Here: 3 out of 5 real = 0.60. High recall means the model is not missing much.
- **F1** = a single number that balances the two. It is the harmonic mean of precision and recall: `2 * (precision * recall) / (precision + recall)`. Here: `2 * (0.75 * 0.60) / (0.75 + 0.60)` ≈ 0.67.

Why F1 and not just accuracy? Because "did I get it right" is ambiguous when the output is a *set* of items. F1 handles the asymmetry between what you returned and what you should have returned.

### Exact match vs. fuzzy match

Two memories can express the same fact with different words: `"Alex uses Obsidian for notes"` and `"Alex switched to Obsidian for note-taking"` both capture the same memory. Exact string matching would call these a miss. You need a softer notion of "match."

Two practical options:

- **Fuzzy string match** — normalize both strings (lowercase, strip punctuation) and compute a character-overlap ratio. A threshold around 0.75 works well for short factual sentences. The `difflib` standard library provides this with zero dependencies.
- **Embedding similarity** — encode both strings as vectors and compute cosine similarity. More powerful for paraphrases but requires loading a small embedding model (~22 MB for `sentence-transformers/all-MiniLM-L6-v2`).

This chapter implements both and lets you choose.

### LLM-as-judge

Sometimes you want a human-quality answer to "is this memory correct?" but you can't manually review thousands of outputs. The solution: ask a capable language model to rate each extracted memory on a simple rubric and assign a score.

This is called LLM-as-judge. It is widely used in production evaluation pipelines (including at companies like mem0). The key is a tightly constrained prompt with a numeric output format — no essays, no ambiguity, just a number.

---

## The evaluation framework

Here is how the full pipeline will work:

1. Load your held-out test set (the split you set aside in *Ch14 - Cleaning, Splitting, and Sanity-Checking Data*).
2. Run three models on every test example: your fine-tuned model, the raw base model, and the base model with just a good system prompt (the "prompted baseline").
3. For each model output, compute: parse-rate, precision/recall/F1, and LLM judge score.
4. Print a comparison table.

---

## The full `evaluate.py` script

Save this to `code/evaluate.py` in your project.

```python
# evaluate.py
#
# Evaluates three systems on the memory-extraction test set:
#   1. fine_tuned  — your LoRA-trained model
#   2. base_model  — the same base model with NO fine-tuning
#   3. prompted    — the base model with the full system prompt but no fine-tuning
#
# Usage:
#   python evaluate.py \
#       --test_file  data/splits/test.jsonl \
#       --finetuned  outputs/memory-extractor-merged \
#       --base_model unsloth/Qwen3-8B \
#       --judge_model gpt-4o-mini \
#       --max_examples 100
#
# Requirements:
#   pip install unsloth transformers datasets openai>=1.0 sentence-transformers
#   (difflib is part of the Python standard library — do NOT pip install it)

import argparse
import json
import re
import time
from pathlib import Path
from difflib import SequenceMatcher
from typing import Optional

# ── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate memory extraction models")
parser.add_argument("--test_file",    required=True,  help="Path to test.jsonl")
parser.add_argument("--finetuned",    required=True,  help="Path or HF ID of fine-tuned model")
parser.add_argument("--base_model",   required=True,  help="HF model ID of the base model")
parser.add_argument("--judge_model",  default="gpt-4o-mini", help="OpenAI model to use as judge")
parser.add_argument("--max_examples", type=int, default=100,  help="Cap test examples (faster during iteration)")
parser.add_argument("--match_mode",   default="fuzzy", choices=["exact", "fuzzy", "embedding"],
                    help="How to decide if two memories match")
parser.add_argument("--match_threshold", type=float, default=0.75,
                    help="Similarity threshold for fuzzy/embedding match (0-1)")
parser.add_argument("--skip_judge",   action="store_true",
                    help="Skip LLM-as-judge scoring (faster, no API cost)")
parser.add_argument("--output_file",  default="eval_results.json",
                    help="Where to save detailed results")
args = parser.parse_args()


# ── The system prompt — must be identical to what you used during training ────
# If you change this, copy it from your training script.
# A mismatch here is the single most common cause of surprisingly bad eval scores.
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

# The base model gets a minimal prompt — we want to see what it does without guidance.
BASE_MODEL_PROMPT = "Extract all memories from the following conversation as a JSON list:"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: Load test data
# ═════════════════════════════════════════════════════════════════════════════

def load_test_examples(path: str, max_n: int) -> list[dict]:
    """
    Load test.jsonl and return a list of dicts with keys:
      'conversation' — the input text (user message)
      'gold_memories' — the reference memory list (parsed from assistant message)
    """
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Each row must look exactly like this:
            #   {"messages": [
            #       {"role": "system",    "content": "..."},   # index 0
            #       {"role": "user",      "content": "..."},   # index 1 — the conversation
            #       {"role": "assistant", "content": "..."},   # index 2 — the gold JSON
            #   ]}
            # If your pipeline stores the system prompt separately and omits it from each
            # row, index 1 will be the assistant turn — swap indices and everything will
            # score wrong silently. Verify your file matches this layout before running.
            messages = row["messages"]
            assert messages[1]["role"] == "user",      \
                f"Expected messages[1] to be the user turn, got '{messages[1]['role']}'. " \
                f"Check your JSONL format — each row must have [system, user, assistant]."
            assert messages[2]["role"] == "assistant", \
                f"Expected messages[2] to be the assistant turn, got '{messages[2]['role']}'."
            conversation    = messages[1]["content"]   # the user turn
            assistant_text  = messages[2]["content"]   # the gold answer

            # Parse the gold answer into a Python list
            try:
                gold = json.loads(assistant_text)
            except json.JSONDecodeError:
                # Skip rows whose gold answer is malformed (shouldn't happen after Ch14)
                continue

            examples.append({
                "conversation":   conversation,
                "gold_memories":  gold,
            })

            if len(examples) >= max_n:
                break

    print(f"Loaded {len(examples)} test examples from {path}")
    return examples


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: Run inference on a model
# ═════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_path_or_id: str, use_4bit: bool = True):
    """
    Load a model and tokenizer using Unsloth's FastLanguageModel.
    Works for both fine-tuned local paths and raw HF model IDs.

    use_4bit=True loads in 4-bit quantization to save VRAM (~8-10 GB for a 7-8B model).
    """
    from unsloth import FastLanguageModel

    # NOTE for Qwen3 users: Unsloth enables "thinking mode" by default for Qwen3
    # models. This causes the model to emit a <think>...</think> reasoning block
    # before every answer. run_inference() strips that block automatically, so
    # evaluation still works. If you want to disable thinking mode entirely at
    # load time (slightly faster, no reasoning block), add:
    #   model_kwargs={"enable_thinking": False}
    # to the call below. For Gemma3 and other non-Qwen3 models this has no effect.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path_or_id,
        max_seq_length=4096,       # must match your training setting
        load_in_4bit=use_4bit,
        dtype=None,                # auto-detect (bfloat16 on Ampere+, float16 otherwise)
    )
    # Switch to inference mode — this disables dropout and other training-only layers
    FastLanguageModel.for_inference(model)
    print(f"Loaded model: {model_path_or_id}")
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    conversation: str,
    system_prompt: str,
    max_new_tokens: int = 1024,
) -> str:
    """
    Run one inference pass and return the raw model output string.

    We use the tokenizer's apply_chat_template to format the input exactly
    as it was formatted during training — same special tokens, same structure.
    """
    import torch

    # ── Qwen3 thinking-mode note ────────────────────────────────────────────
    # Qwen3 models emit a <think>...</think> block BEFORE the JSON answer when
    # thinking mode is enabled (the default when loaded with Unsloth).
    # That block starts with "<think>", which will cause parse_model_output to
    # fail for every single output — you will see 0% parse-rate and have no idea why.
    #
    # To suppress thinking mode at load time, pass enable_thinking=False in
    # load_model_and_tokenizer (see comment there).  Alternatively, if you want
    # to keep thinking mode on, strip the block before returning:
    #
    #   import re
    #   raw_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
    #
    # The strip approach is applied automatically below.

    # Format the prompt using the model's own chat template
    # This ensures special tokens like <|im_start|> are applied correctly for Qwen3,
    # or <start_of_turn> for Gemma — the tokenizer knows which to use.
    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": conversation},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   # adds the assistant turn opening token
    )

    # Tokenize and move to the model's device (GPU if available)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,           # low temperature for deterministic JSON output
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (not the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Strip any Qwen3 <think>...</think> block before returning.
    # Qwen3 thinking mode emits this block before the actual JSON answer.
    # This line is harmless for non-Qwen3 models (the regex finds nothing).
    raw_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)

    return raw_output.strip()


def batch_run_inference(
    model,
    tokenizer,
    examples: list[dict],
    system_prompt: str,
    label: str,
) -> list[str]:
    """
    Run inference on every example and return a list of raw output strings.
    Prints progress every 10 examples so you know it's working.
    """
    outputs = []
    for i, ex in enumerate(examples):
        raw = run_inference(model, tokenizer, ex["conversation"], system_prompt)
        outputs.append(raw)
        if (i + 1) % 10 == 0:
            print(f"  [{label}] {i+1}/{len(examples)} done")
    print(f"  [{label}] inference complete ({len(outputs)} outputs)")
    return outputs


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: Parsing and parse-rate
# ═════════════════════════════════════════════════════════════════════════════

VALID_TYPES = {"preference", "fact", "decision", "relationship"}

def parse_model_output(raw: str) -> tuple[Optional[list], str]:
    """
    Attempt to parse the model's raw string output into a list of memory dicts.

    Returns:
        (memories_list, error_string)
        On success: (list, "")
        On failure: (None, description_of_what_went_wrong)

    We try three strategies in order:
      1. Direct parse (model output is clean JSON)
      2. Extract from markdown code fence (model ignored the "no fences" rule)
      3. Find the first '[' and last ']' and try to parse that substring
    """
    raw = raw.strip()

    # Strategy 1: clean parse
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data, ""
        return None, f"Parsed JSON but got {type(data).__name__}, not a list"
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown code fence (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if fence_match:
        try:
            data = json.loads(fence_match.group(1))
            if isinstance(data, list):
                return data, ""
        except json.JSONDecodeError:
            pass

    # Strategy 3: find the outer JSON array by bracket
    start = raw.find("[")
    end   = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw[start:end+1])
            if isinstance(data, list):
                return data, ""
        except json.JSONDecodeError:
            pass

    return None, f"Could not extract a JSON array from output: {raw[:120]}..."


def validate_memory_schema(memories: list) -> tuple[bool, str]:
    """
    Check that every item in the list conforms to our memory schema.
    Returns (True, "") if valid, (False, reason) if not.
    """
    for i, m in enumerate(memories):
        if not isinstance(m, dict):
            return False, f"Item {i} is not a dict"
        for key in ("text", "type", "entities"):
            if key not in m:
                return False, f"Item {i} missing key '{key}'"
        if m["type"] not in VALID_TYPES:
            return False, f"Item {i}: invalid type '{m['type']}'"
        if not isinstance(m["entities"], list):
            return False, f"Item {i}: 'entities' must be a list"
    return True, ""


def compute_parse_rate(raw_outputs: list[str]) -> dict:
    """
    Compute how often the model produces:
      - valid JSON at all
      - valid JSON that also passes schema validation

    Returns a dict with counts and rates.
    """
    n = len(raw_outputs)
    n_parseable  = 0
    n_valid      = 0
    parse_errors = []
    schema_errors = []

    for raw in raw_outputs:
        memories, parse_err = parse_model_output(raw)
        if memories is None:
            parse_errors.append(parse_err)
            continue
        n_parseable += 1

        ok, schema_err = validate_memory_schema(memories)
        if ok:
            n_valid += 1
        else:
            schema_errors.append(schema_err)

    return {
        "total":            n,
        "parseable":        n_parseable,
        "schema_valid":     n_valid,
        "parse_rate":       round(n_parseable / n, 4) if n else 0,
        "schema_rate":      round(n_valid / n, 4)     if n else 0,
        "parse_errors":     parse_errors[:5],   # keep first 5 for debugging
        "schema_errors":    schema_errors[:5],
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: Precision / Recall / F1
# ═════════════════════════════════════════════════════════════════════════════

def normalize_text(s: str) -> str:
    """Lowercase and strip punctuation for fuzzy comparison."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)   # remove punctuation
    s = re.sub(r"\s+", " ", s)      # collapse whitespace
    return s.strip()


def fuzzy_match(a: str, b: str, threshold: float) -> bool:
    """Return True if the fuzzy similarity between a and b exceeds threshold."""
    ratio = SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()
    return ratio >= threshold


# We cache the embedding model globally so we only load it once.
_embed_model = None

def embedding_match(a: str, b: str, threshold: float) -> bool:
    """
    Return True if the cosine similarity between sentence embeddings of a and b
    exceeds threshold.

    Uses sentence-transformers/all-MiniLM-L6-v2 — about 22 MB, fast inference.
    Install with: pip install sentence-transformers
    """
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2: fast, small, good at semantic similarity
        _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    from sentence_transformers import util
    import torch
    emb_a = _embed_model.encode(a, convert_to_tensor=True)
    emb_b = _embed_model.encode(b, convert_to_tensor=True)
    sim = float(util.cos_sim(emb_a, emb_b))
    return sim >= threshold


def memories_match(pred_text: str, gold_text: str, mode: str, threshold: float) -> bool:
    """
    Decide if a predicted memory matches a gold memory.

    mode="exact"     — normalized string equality
    mode="fuzzy"     — character-level overlap via SequenceMatcher
    mode="embedding" — semantic cosine similarity
    """
    if mode == "exact":
        return normalize_text(pred_text) == normalize_text(gold_text)
    elif mode == "fuzzy":
        return fuzzy_match(pred_text, gold_text, threshold)
    elif mode == "embedding":
        return embedding_match(pred_text, gold_text, threshold)
    else:
        raise ValueError(f"Unknown match mode: {mode}")


def compute_set_f1(
    pred_memories: list[dict],
    gold_memories: list[dict],
    match_mode: str,
    match_threshold: float,
) -> dict:
    """
    Compute set-level precision, recall, and F1 for one example.

    We match by the 'text' field of each memory object.
    A gold memory is "found" if at least one predicted memory matches it.
    A predicted memory "counts" if it matches at least one gold memory.

    This is a greedy bipartite matching — good enough for short lists.
    """
    if not gold_memories and not pred_memories:
        # Both empty: perfect score
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}

    if not gold_memories:
        # Model produced output when there should be none
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0,
                "tp": 0, "fp": len(pred_memories), "fn": 0}

    if not pred_memories:
        # Model produced nothing when there should be output
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0,
                "tp": 0, "fp": 0, "fn": len(gold_memories)}

    pred_texts = [m.get("text", "") for m in pred_memories]
    gold_texts = [m.get("text", "") for m in gold_memories]

    # Track which gold memories have been matched (to avoid double-counting)
    gold_matched = [False] * len(gold_texts)
    tp = 0  # true positives: predicted memories that match a gold memory

    for pt in pred_texts:
        for j, gt in enumerate(gold_texts):
            if not gold_matched[j] and memories_match(pt, gt, match_mode, match_threshold):
                tp += 1
                gold_matched[j] = True
                break

    fp = len(pred_texts) - tp                      # predicted but no match in gold
    fn = sum(1 for m in gold_matched if not m)     # gold memories we missed

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def compute_corpus_f1(
    all_outputs: list[str],
    examples: list[dict],
    match_mode: str,
    match_threshold: float,
) -> dict:
    """
    Average precision, recall, and F1 across all parseable examples.
    Unparseable outputs count as zero for all three metrics.
    """
    precisions, recalls, f1s = [], [], []

    for raw, ex in zip(all_outputs, examples):
        gold = ex["gold_memories"]
        pred_list, _ = parse_model_output(raw)

        if pred_list is None:
            # Could not parse output — counts as producing nothing
            pred_list = []

        scores = compute_set_f1(pred_list, gold, match_mode, match_threshold)
        precisions.append(scores["precision"])
        recalls.append(scores["recall"])
        f1s.append(scores["f1"])

    n = len(f1s)
    return {
        "avg_precision": round(sum(precisions) / n, 4) if n else 0,
        "avg_recall":    round(sum(recalls)    / n, 4) if n else 0,
        "avg_f1":        round(sum(f1s)        / n, 4) if n else 0,
        "n_examples":    n,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: LLM-as-judge scoring
# ═════════════════════════════════════════════════════════════════════════════

# This prompt is the heart of the judge. Keep it tight — you want a single integer,
# not a paragraph. The rubric anchors the scale so different conversations get
# consistent scores.
JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for a memory extraction system.
A memory extraction model was given a conversation and asked to extract memorable facts as a list of atomic memory objects.

Your job: rate the quality of the extracted memories on a scale from 0 to 3.

SCORING RUBRIC:
  3 = Excellent. All important facts extracted, no hallucinations, correct types, entities accurate.
  2 = Good. Most important facts extracted, minor omissions or type errors, no hallucinations.
  1 = Partial. Some correct memories, but significant omissions OR at least one hallucination.
  0 = Failed. No valid memories, all hallucinated, or output is not valid JSON.

CONVERSATION:
{conversation}

EXTRACTED MEMORIES (the model's output):
{extracted}

REFERENCE MEMORIES (the correct answer):
{reference}

Respond with a single integer (0, 1, 2, or 3) and nothing else."""


def judge_single_output(
    client,
    judge_model: str,
    conversation: str,
    extracted_raw: str,
    reference_memories: list,
) -> Optional[int]:
    """
    Ask the judge model to score one extracted memory list.
    Returns an integer 0-3, or None if the judge's response can't be parsed.
    """
    # Format the extracted output — either as parsed JSON or the raw string if unparseable
    pred_list, _ = parse_model_output(extracted_raw)
    extracted_str = (json.dumps(pred_list, indent=2, ensure_ascii=False)
                     if pred_list is not None else extracted_raw[:500])

    reference_str = json.dumps(reference_memories, indent=2, ensure_ascii=False)

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        conversation=conversation[:1500],   # cap length to stay within judge's context
        extracted=extracted_str[:1500],
        reference=reference_str[:1500],
    )

    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,         # we only need a single digit
            temperature=0.0,      # deterministic scoring
        )
        raw_score = response.choices[0].message.content.strip()
        # Extract the first digit we see
        match = re.search(r"[0-3]", raw_score)
        return int(match.group()) if match else None
    except Exception as e:
        print(f"  [judge] API error: {e}")
        return None


def compute_judge_scores(
    outputs: list[str],
    examples: list[dict],
    judge_model: str,
    label: str,
) -> dict:
    """
    Score every model output with the LLM judge and return average score and distribution.
    Requires OPENAI_API_KEY in environment.
    """
    import openai
    import os

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  [judge] OPENAI_API_KEY not set — skipping judge scoring")
        return {"avg_judge_score": None, "distribution": {}}

    client = openai.OpenAI(api_key=api_key)
    scores = []

    for i, (raw, ex) in enumerate(zip(outputs, examples)):
        score = judge_single_output(
            client, judge_model,
            ex["conversation"], raw, ex["gold_memories"]
        )
        if score is not None:
            scores.append(score)
        # Throttle: the judge API has rate limits; 0.3 s between calls is safe
        time.sleep(0.3)

        if (i + 1) % 20 == 0:
            print(f"  [{label}] judge: {i+1}/{len(outputs)} scored, "
                  f"running avg={sum(scores)/len(scores):.2f}")

    if not scores:
        return {"avg_judge_score": None, "distribution": {}}

    distribution = {str(k): scores.count(k) for k in range(4)}
    return {
        "avg_judge_score": round(sum(scores) / len(scores), 3),
        "distribution":    distribution,
        "n_scored":        len(scores),
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: Main evaluation loop
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_system(
    outputs: list[str],
    examples: list[dict],
    label: str,
    match_mode: str,
    match_threshold: float,
    judge_model: str,
    skip_judge: bool,
) -> dict:
    """
    Run all three evaluation layers for one system and return a results dict.
    """
    print(f"\n--- Evaluating: {label} ---")

    # Layer 1: parse rate
    parse_stats = compute_parse_rate(outputs)
    print(f"  Parse rate:   {parse_stats['parse_rate']:.1%}  "
          f"({parse_stats['parseable']}/{parse_stats['total']} parseable)")
    print(f"  Schema valid: {parse_stats['schema_rate']:.1%}")

    # Layer 2: precision / recall / F1
    f1_stats = compute_corpus_f1(outputs, examples, match_mode, match_threshold)
    print(f"  Precision:    {f1_stats['avg_precision']:.3f}")
    print(f"  Recall:       {f1_stats['avg_recall']:.3f}")
    print(f"  F1:           {f1_stats['avg_f1']:.3f}  (match_mode={match_mode})")

    # Layer 3: LLM judge
    judge_stats = {}
    if not skip_judge:
        judge_stats = compute_judge_scores(outputs, examples, judge_model, label)
        avg = judge_stats.get("avg_judge_score")
        if avg is not None:
            print(f"  Judge score:  {avg:.2f}/3.0  "
                  f"(dist={judge_stats['distribution']})")
    else:
        print(f"  Judge score:  skipped (--skip_judge)")

    return {
        "label":   label,
        "parse":   parse_stats,
        "f1":      f1_stats,
        "judge":   judge_stats,
        "outputs": outputs,   # saved to file for manual inspection
    }


def print_comparison_table(results: list[dict]) -> None:
    """Print a side-by-side comparison table of all evaluated systems."""
    col = 22
    print("\n" + "═" * 80)
    print("EVALUATION SUMMARY")
    print("═" * 80)

    # Header
    header = f"{'Metric':<22}" + "".join(f"{r['label']:<22}" for r in results)
    print(header)
    print("─" * 80)

    def row(name, values):
        return f"{name:<22}" + "".join(f"{str(v):<22}" for v in values)

    print(row("Parse rate",
              [f"{r['parse']['parse_rate']:.1%}" for r in results]))
    print(row("Schema valid",
              [f"{r['parse']['schema_rate']:.1%}" for r in results]))
    print(row("Avg Precision",
              [f"{r['f1']['avg_precision']:.3f}" for r in results]))
    print(row("Avg Recall",
              [f"{r['f1']['avg_recall']:.3f}" for r in results]))
    print(row("Avg F1",
              [f"{r['f1']['avg_f1']:.3f}" for r in results]))

    judge_row = []
    for r in results:
        score = r["judge"].get("avg_judge_score")
        judge_row.append(f"{score:.2f}/3" if score is not None else "—")
    print(row("Judge score",  judge_row))
    print("═" * 80)


def main():
    # ── Load test data ──────────────────────────────────────────────────────
    examples = load_test_examples(args.test_file, args.max_examples)

    all_results = []

    # ── System 1: Fine-tuned model ──────────────────────────────────────────
    print("\n[1/3] Loading fine-tuned model...")
    ft_model, ft_tokenizer = load_model_and_tokenizer(args.finetuned)
    ft_outputs = batch_run_inference(
        ft_model, ft_tokenizer, examples,
        system_prompt=SYSTEM_PROMPT,
        label="fine_tuned",
    )
    del ft_model      # free VRAM before loading the next model
    del ft_tokenizer  # also free the tokenizer — it holds GPU-pinned buffers
    import torch; torch.cuda.empty_cache()   # tell PyTorch to release the freed memory now

    # ── System 2: Base model with full system prompt (prompted baseline) ────
    print("\n[2/3] Loading base model for prompted baseline...")
    base_model, base_tokenizer = load_model_and_tokenizer(args.base_model)
    prompted_outputs = batch_run_inference(
        base_model, base_tokenizer, examples,
        system_prompt=SYSTEM_PROMPT,        # same prompt, no training
        label="prompted_baseline",
    )

    # ── System 3: Base model with minimal prompt ────────────────────────────
    print("\n  Running base model with minimal prompt...")
    base_outputs = batch_run_inference(
        base_model, base_tokenizer, examples,
        system_prompt=BASE_MODEL_PROMPT,    # bare-bones prompt
        label="base_model",
    )
    del base_model      # free VRAM
    del base_tokenizer  # free tokenizer buffers too
    torch.cuda.empty_cache()

    # ── Evaluate all three ──────────────────────────────────────────────────
    for label, outputs in [
        ("fine_tuned",        ft_outputs),
        ("prompted_baseline", prompted_outputs),
        ("base_model",        base_outputs),
    ]:
        result = evaluate_system(
            outputs, examples,
            label=label,
            match_mode=args.match_mode,
            match_threshold=args.match_threshold,
            judge_model=args.judge_model,
            skip_judge=args.skip_judge,
        )
        all_results.append(result)

    # ── Print the comparison table ──────────────────────────────────────────
    print_comparison_table(all_results)

    # ── Save full results to JSON ───────────────────────────────────────────
    # Strip the raw outputs list from the saved file — they can be large.
    save_data = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "outputs"}
        # Save a small sample of raw outputs for manual inspection
        entry["output_samples"] = r["outputs"][:5]
        save_data.append(entry)

    Path(args.output_file).write_text(
        json.dumps(save_data, indent=2, ensure_ascii=False)
    )
    print(f"\nFull results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
```

---

## Running the evaluation

Once training is complete, the training script from Ch16/Ch17 saves a merged model to `outputs/memory-extractor-merged` by default — that is the path you pass to `--finetuned`. (Ch21 covers saving and exporting in detail, but you do not need to read it first; just use that default path.) Run:

```bash
# Fast pass — no judge scoring, fuzzy match, 50 examples
python code/evaluate.py \
    --test_file  data/splits/test.jsonl \
    --finetuned  outputs/memory-extractor-merged \
    --base_model unsloth/Qwen3-8B \
    --max_examples 50 \
    --skip_judge

# Full pass — with LLM judge (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... python code/evaluate.py \
    --test_file  data/splits/test.jsonl \
    --finetuned  outputs/memory-extractor-merged \
    --base_model unsloth/Qwen3-8B \
    --judge_model gpt-4o-mini \
    --max_examples 100
```

A healthy result looks like this (approximate ballpark — your numbers will vary based on dataset size and quality):

```
════════════════════════════════════════════════════════════════════════════════
EVALUATION SUMMARY
════════════════════════════════════════════════════════════════════════════════
Metric                 fine_tuned             prompted_baseline      base_model
────────────────────────────────────────────────────────────────────────────────
Parse rate             98.0%                  72.0%                  31.0%
Schema valid           96.0%                  68.0%                  19.0%
Avg Precision          0.821                  0.651                  0.183
Avg Recall             0.793                  0.602                  0.147
Avg F1                 0.807                  0.625                  0.163
Judge score            2.51/3                 1.89/3                 0.74/3
════════════════════════════════════════════════════════════════════════════════
```

What to read from this table:

- **Parse rate** should be near 100% for your fine-tuned model. If it is below 90%, the model learned the task but not the output format — see *Ch19 - When It Goes Wrong: A Debugging Playbook*.
- **F1** is the main quality number. A fine-tuned model typically lands 15-30 points higher than the prompted baseline on a task this structured, because training teaches both *what to extract* and *how to format it*.
- **The base model without a proper prompt** typically produces barely any structured output — parse rate under 40% is normal. This is the baseline you need to beat.
- **Judge score** provides a sanity check on the F1. If F1 is high but judge score is low, your match threshold may be too loose.

---

## Reading individual failures

Numbers alone do not tell you *why* a model is failing. After every evaluation run, open `eval_results.json` and look at the `output_samples` for each system. You are looking for patterns:

- **Parse failures in the fine-tuned model** — the model is wrapping output in markdown fences, or adding an explanation before the JSON array. Fix: add more training examples that have only raw JSON in the assistant turn. See *Ch19*.
- **Low recall, high precision** — the model is extracting fewer memories than it should. Common cause: the system prompt says "atomic" facts but the model is bundling two facts together. Fix: add training examples with more granular memories.
- **High recall, low precision** — the model is hallucinating memories that are not in the conversation. Common cause: not enough negative examples (conversations with zero extractable memories). Fix: add more empty-output examples to your training set.
- **Good F1, low judge score** — your match threshold is too permissive and fuzzy-matching strings that are not really the same fact. Lower `--match_threshold` or switch to `--match_mode embedding`.

---

## Common mistakes

**Using a different system prompt during evaluation than during training.**

This is the most damaging mistake and the hardest to notice. If you changed the system prompt between training runs (even slightly — adding a period, changing "atomic" to "standalone"), the fine-tuned model will score worse than it actually is. The fix is to define `SYSTEM_PROMPT` as a constant in a shared module and import it in both your training script and `evaluate.py`. Never copy-paste it by hand.

**Evaluating on the training set.**

If your test split somehow leaked into training (easy to do if you shuffled and split the wrong file), your fine-tuned model will score nearly perfectly — and your numbers are meaningless. Double-check that `test.jsonl` was never seen by the trainer. The split process from *Ch14* handles this, but verify it.

**Treating low parse-rate as a model quality problem before checking your prompt.**

A fine-tuned model that was never explicitly trained on your exact system prompt may produce valid memories in a slightly different format. Before concluding the model is bad, re-run evaluation with the exact system prompt you used at training time and check whether parse-rate jumps. It often does.

**Using only F1 and ignoring parse-rate.**

A model with a 60% parse-rate and 0.80 F1 (on the parseable outputs) is far worse than a model with a 98% parse-rate and 0.75 F1. The unparseable outputs are silent failures in production — your app would crash or return nothing. Always report both.

**Running the judge on 1000 examples at once.**

At $0.00015 per 1K tokens for `gpt-4o-mini`, 1000 judge calls with ~500-token prompts costs roughly $0.08 — negligible. But if you use a more expensive model, costs add up. Cap with `--max_examples 100` during development and run the full set only when you have a model you're serious about shipping.

**Getting 0% parse-rate on Qwen3 because of thinking mode.**

Qwen3 models loaded with Unsloth emit a `<think>...</think>` reasoning block before every answer by default. If you see parse-rate at 0% and the `output_samples` in `eval_results.json` all start with `<think>`, this is why. The `run_inference` function in this script already strips that block with a regex, so it is handled automatically. If you are adapting this code elsewhere and forget the strip, add this line after decoding the output:

```python
raw_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
```

Gemma3 and other non-Qwen3 models do not have this behavior.

**Forgetting to `del` both the model and the tokenizer between loads.**

Python does not free GPU memory automatically when a variable goes out of scope inside the same process. Call `del model`, `del tokenizer`, and `torch.cuda.empty_cache()` before loading the next model, or you will run out of VRAM when loading the second or third system. Deleting only the model and leaving the tokenizer alive is enough to cause a silent out-of-VRAM crash on a 24 GB card with two 8B models.

---

## Recap

- Training loss tells you the model improved on training data. It does not tell you whether the model works on new data. Use real evaluation metrics.
- The three layers of evaluation are: parse-rate (format correctness), precision/recall/F1 (content accuracy), and LLM-as-judge (semantic quality).
- Parse-rate should be near 100% for a properly trained model. Anything below 90% is a format failure that will cause problems in production.
- F1 is computed at the set level: precision asks "how many of my extractions are correct?", recall asks "how many of the real memories did I find?", F1 balances the two.
- Two memories "match" if their text is sufficiently similar — exact match is too strict; fuzzy or embedding match handles natural paraphrasing.
- The LLM-as-judge pattern uses a tightly constrained 0-3 rubric prompt to get a human-quality semantic score without manual review.
- Always compare your fine-tuned model against both the base model and a prompted-only baseline to prove the fine-tune added value beyond good prompting.
- After getting the numbers, read individual failures in `eval_results.json` to understand *why* the model is failing before trying to fix it.

## Next

*Ch19 - When It Goes Wrong: A Debugging Playbook* — now that you can measure failure, this chapter gives you a systematic process for diagnosing and fixing the most common failure modes: hallucinations, format breakdowns, low recall, and type misclassifications.
