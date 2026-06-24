# Ch13 - Creating Your Training Data with Synthetic Generation

You have a task. You have a model. You have almost no labeled examples.

This is the normal situation. Real conversation data is messy, private, or legally off-limits. Paid human labelers are slow and expensive. And yet your model needs hundreds or thousands of (input, output) pairs to learn from.

The solution: use a big, capable model — a "teacher" — to manufacture that data for you. Feed the teacher a prompt, get back synthetic conversations plus the gold-standard memory JSON you'd want your student model to produce. Repeat at scale. This is the core idea behind **knowledge distillation**, and it's how most domain fine-tuning happens in practice today.

This chapter is the heart of the whole book. Everything else — training, evaluation, iteration — is downstream of what you build here.

---

## What you'll learn

- Why synthetic data generation is the standard bootstrap for domain fine-tuning
- How to design prompts that produce diverse, realistic conversations
- How to call a teacher LLM (Claude via the Anthropic SDK) to generate labeled pairs
- How to run LLM-as-judge to filter out bad rows automatically
- How to deduplicate and quality-check your dataset, then write it to JSONL
- Rules of thumb for how much data you need and how to scale

---

## Concepts you need first

### Knowledge distillation — the 20% explanation

Imagine a master chef (the teacher model) and an apprentice (your student model). The apprentice can't afford to eat at every restaurant in the world the way the master has. But the master *can* watch a dish come in and narrate exactly what's happening: "this is a Maillard reaction, here's the structure, here's what went wrong." The apprentice learns from those narrated examples, not from raw experience.

In ML terms: a large, expensive model generates high-quality (input → output) pairs for your specific task. A smaller, cheaper model is then trained on those pairs. The small model learns to imitate the large model's behavior on your task — without needing the large model's size or training budget.

Why it matters for memory extraction: you can't hire people to label ten thousand conversations. But you *can* call Claude 10,000 times and get labeled pairs that are nearly as good.

### JSONL — the data format for training

A JSONL file (JSON Lines) is just a text file where each line is a self-contained JSON object. No surrounding array, no commas between lines — just one JSON blob per line.

```
{"input": "...", "output": "..."}
{"input": "...", "output": "..."}
```

Training libraries (Hugging Face `datasets`, TRL) expect this format. It's easy to stream, easy to append to, and easy to inspect with any text editor or `grep`.

### Teacher vs. student model

**Teacher**: the strong model you use to generate labels. You call it via API; you never train it. Claude, GPT-4o, Gemini Ultra — any capable model works. You pay per token, but this is a one-time cost.

**Student**: the small open model you actually fine-tune and own. Qwen3-4B, Gemma 3-4B — see Ch10 for the comparison. This is what gets deployed.

The teacher creates the dataset. The student trains on it. They never interact directly.

---

## The pipeline end to end

Here's what you're building:

```
seed topics + personas
        ↓
generate synthetic conversations (teacher LLM)
        ↓
generate gold memory JSON for each conversation (same or second call)
        ↓
self-verify with LLM-as-judge (filter bad rows)
        ↓
dedup + quality filters
        ↓
write to data/memories_train.jsonl
```

Each step is a Python function. You'll run the whole pipeline from a single script.

---

## Step 1 — Seed topics and personas for diversity

The biggest trap in synthetic data: all your generated conversations feel the same. Same topics, same sentence structure, same length. A model trained on homogeneous data will overfit to that narrow style and fail on anything slightly different. (Overfit means the model memorises the training examples so narrowly that it fails on anything slightly different — it learns your dataset's quirks, not the actual task.)

The fix is **seeds** — a list of topics, speaker personas, and conversation styles that you shuffle and sample from. Every generation call draws a different combination.

```python
# seeds.py
# These seeds drive diversity in the generated conversations.
# Add more rows to any list to widen the coverage of your dataset.

TOPICS = [
    "planning a weekend hiking trip",
    "discussing a recent job interview",
    "deciding what to cook for a dinner party",
    "catching up after months apart",
    "troubleshooting a slow laptop",
    "talking through a disagreement with a coworker",
    "planning a birthday surprise for a friend",
    "discussing a movie they just watched",
    "figuring out a travel itinerary",
    "venting about a difficult client",
    "planning a home renovation project",
    "discussing a new fitness routine",
    "catching up on family news",
    "deciding whether to adopt a pet",
    "talking through a career change decision",
]

PERSONAS = [
    "two close friends in their late 20s",
    "a parent and their adult child",
    "two colleagues at a tech startup",
    "a couple planning a vacation",
    "two roommates",
    "a manager and a direct report",
    "two siblings",
    "old college friends reconnecting",
]

STYLES = [
    "casual and full of abbreviations",
    "warm and detailed",
    "quick back-and-forth, short messages",
    "one person is clearly more talkative",
    "mix of practical planning and personal reflection",
]

TURN_COUNTS = [6, 8, 10, 12, 14]  # how many messages in the conversation
```

---

## Step 2 — The generation prompt

You need two things from the teacher: a realistic conversation, and the gold memory JSON for it. You can get both in one API call by asking for them together in a structured format.

Here's the prompt template. Notice how it embeds the schema from Ch12 — the teacher needs to know exactly what fields you expect. We extend the base three-field schema from Ch12 (`text`, `type`, `entities`) with a fourth field, `confidence`, here so the training data captures how certain each extracted memory is.

```python
# prompts.py
import random
from seeds import TOPICS, PERSONAS, STYLES, TURN_COUNTS

# This is the schema your student model will learn to produce.
# Keep it in sync with what you defined in Ch12.
MEMORY_SCHEMA = """
Each memory object must have these fields:
  - "text": a single, standalone fact written as a complete sentence
  - "type": one of "fact", "preference", "decision", "relationship", "goal"
  - "entities": a list of the key people, places, or things mentioned
  - "confidence": one of "high", "medium", or "low"

Only include memories that are durable and meaningful — skip small talk,
filler phrases, and things said only in passing.
""".strip()


def build_generation_prompt(topic, persona, style, turn_count):
    """
    Returns the full prompt string sent to the teacher LLM.
    It asks for BOTH the conversation and the memory JSON in one shot.
    """
    return f"""You are generating training data for a memory-extraction system.

Your job has two parts:

PART 1 — Write a realistic chat conversation.
- Topic: {topic}
- Who is talking: {persona}
- Conversational style: {style}
- Number of messages: {turn_count} (alternate between Speaker A and Speaker B)
- Format each message as: "A: ..." or "B: ..."
- Make it feel like a real chat log. Include names, specific details, opinions.

PART 2 — Extract the memories from that conversation.
Given the conversation you just wrote, produce a JSON array of memory objects.

{MEMORY_SCHEMA}

Output format (strictly follow this, no extra text outside the JSON block):

<conversation>
[the full conversation here]
</conversation>

<memories>
[
  {{"text": "...", "type": "...", "entities": [...], "confidence": "..."}}
]
</memories>
"""
```

---

## Step 3 — Calling the teacher LLM (Anthropic Claude)

You'll use the official `anthropic` Python SDK. Install it with:

```bash
pip install anthropic
```

The current model to use is `claude-opus-4-5` for the highest quality labels, or `claude-sonnet-4-5` if you want a 3–4x cost reduction with only a slight quality dip on this task. Either works; start with Sonnet for prototyping.

```python
# generate.py
import os
import re
import json
import random
import time
from anthropic import Anthropic

from seeds import TOPICS, PERSONAS, STYLES, TURN_COUNTS
from prompts import build_generation_prompt

# Initialize the client. It reads ANTHROPIC_API_KEY from the environment.
# Set it with: export ANTHROPIC_API_KEY="sk-ant-..."
client = Anthropic()

# If you prefer OpenAI instead of Anthropic, swap this block:
#   from openai import OpenAI
#   client = OpenAI()  # reads OPENAI_API_KEY
#   then call client.chat.completions.create(model="gpt-4o", messages=[...])
# The prompt template works identically.

# If you prefer a local model (e.g., via Ollama), you can use the openai-compatible
# endpoint: client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# Local models are slower to generate good JSON and may need stricter prompting.


def call_teacher(prompt: str, max_retries: int = 3) -> str:
    """
    Sends the prompt to Claude and returns the raw text response.
    Retries on transient API errors with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-5",   # swap to claude-opus-4-5 for higher quality
                max_tokens=2048,             # conversations + JSON fit in ~800-1200 tokens
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            # The response content is a list of blocks; the first is the text.
            return message.content[0].text
        except Exception as e:
            if attempt == max_retries - 1:
                raise  # give up after max retries
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"  API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
            time.sleep(wait)


def parse_response(raw: str) -> dict | None:
    """
    Extracts the conversation and memories from the teacher's response.
    Returns a dict with keys 'conversation' and 'memories', or None if parsing fails.
    """
    # Use regex to pull out the tagged sections.
    conv_match = re.search(r"<conversation>(.*?)</conversation>", raw, re.DOTALL)
    mem_match  = re.search(r"<memories>(.*?)</memories>",      raw, re.DOTALL)

    if not conv_match or not mem_match:
        return None  # malformed — caller will discard this row

    conversation = conv_match.group(1).strip()

    # Parse the JSON block of memories.
    try:
        memories = json.loads(mem_match.group(1).strip())
    except json.JSONDecodeError:
        return None  # bad JSON — discard

    # Basic sanity: at least one memory, all required fields present.
    required_fields = {"text", "type", "entities", "confidence"}
    for mem in memories:
        if not required_fields.issubset(mem.keys()):
            return None

    return {"conversation": conversation, "memories": memories}
```

---

## Step 4 — LLM-as-judge: filtering bad rows

Not every generated row is good. The teacher sometimes produces memories that are too vague ("they talked about plans"), duplicates of each other, or factually wrong relative to the conversation. Rather than reviewing thousands of rows by hand, you use a *second* LLM call as a quality judge.

This is called **LLM-as-judge**. You show the judge the conversation and the proposed memories, and ask: are these memories accurate, atomic, and non-redundant?

```python
# judge.py
import json
from anthropic import Anthropic

client = Anthropic()

JUDGE_PROMPT = """You are a quality-control reviewer for a memory-extraction dataset.

Given a conversation and a list of extracted memories, evaluate whether the memories are:
1. ACCURATE — each memory is actually supported by the conversation
2. ATOMIC — each memory is a single, standalone fact (not a bundle of facts)
3. NON-REDUNDANT — no two memories say the same thing
4. APPROPRIATE TYPE — the "type" label fits (fact/preference/decision/relationship/goal)

Reply with ONLY a JSON object in this exact format:
{{
  "verdict": "pass" or "fail",
  "reason": "one sentence explaining the verdict"
}}

Conversation:
{conversation}

Memories:
{memories}
"""


def judge_row(conversation: str, memories: list) -> bool:
    """
    Returns True if the row passes quality review, False otherwise.
    Uses a cheaper/faster model for judging since it's a binary classification.
    """
    prompt = JUDGE_PROMPT.format(
        conversation=conversation,
        memories=json.dumps(memories, indent=2)
    )

    try:
        message = client.messages.create(
            model="claude-haiku-4-5",  # fast and cheap; good enough for binary judgment
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text.strip()
        result = json.loads(raw)
        return result.get("verdict") == "pass"
    except Exception:
        # If the judge call fails, conservatively reject the row.
        return False
```

Typical pass rate is 75–90% with a good generation prompt. If you're seeing below 60%, your generation prompt is the problem — tighten the schema description or add few-shot examples.

---

## Step 5 — Dedup and quality filters

Even after the judge, you can end up with near-duplicate conversations (same topic, same persona, nearly identical phrasing). A quick fingerprint check handles this.

```python
# dedup.py
import hashlib


def fingerprint(conversation: str) -> str:
    """
    Creates a rough fingerprint of a conversation for dedup.
    We take the first 200 characters, strip whitespace, lowercase — fast and good enough.
    """
    normalized = conversation[:200].lower().replace(" ", "").replace("\n", "")
    return hashlib.md5(normalized.encode()).hexdigest()


def is_duplicate(fp: str, seen: set) -> bool:
    """Returns True if we've already seen this fingerprint."""
    if fp in seen:
        return True
    seen.add(fp)
    return False


def passes_length_filter(conversation: str, memories: list) -> bool:
    """
    Reject rows that are trivially short or implausibly long.
    These numbers are loose guards — adjust for your domain.
    """
    words_in_conv = len(conversation.split())
    if words_in_conv < 60 or words_in_conv > 1500:
        return False
    # Also reject if there are zero or suspiciously many memories.
    if len(memories) < 1 or len(memories) > 20:
        return False
    return True
```

---

## Step 6 — The full pipeline script

Now assemble everything. This script generates N rows and writes them to a JSONL file.

```python
# pipeline.py
"""
Full synthetic data generation pipeline for the memory-extraction task.

Usage:
    python pipeline.py --target 1000 --output data/memories_train.jsonl

Cost ballpark (as of mid-2025):
    claude-sonnet-4-5: ~$3 per 1000 rows (generation + judging)
    claude-opus-4-5:   ~$15 per 1000 rows
    Time: ~30-60 minutes for 1000 rows depending on API concurrency limits.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from generate import call_teacher, parse_response
from judge import judge_row
from dedup import fingerprint, is_duplicate, passes_length_filter
from prompts import build_generation_prompt
from seeds import TOPICS, PERSONAS, STYLES, TURN_COUNTS


def generate_one_row() -> dict | None:
    """
    Picks a random seed combination, calls the teacher, parses the response.
    Returns a clean row dict or None if anything went wrong.
    """
    topic      = random.choice(TOPICS)
    persona    = random.choice(PERSONAS)
    style      = random.choice(STYLES)
    turn_count = random.choice(TURN_COUNTS)

    prompt = build_generation_prompt(topic, persona, style, turn_count)

    # call_teacher raises on final retry failure, so we catch it here and
    # return None so run_pipeline skips the row gracefully instead of crashing.
    try:
        raw = call_teacher(prompt)
    except Exception as e:
        print(f"  Teacher call failed: {e}. Skipping row.")
        return None

    parsed = parse_response(raw)
    if parsed is None:
        return None

    # Attach the metadata — useful for debugging and diversity analysis later.
    parsed["meta"] = {
        "topic": topic,
        "persona": persona,
        "style": style,
        "turn_count": turn_count,
    }
    return parsed


def run_pipeline(target: int, output_path: str):
    """
    Main loop. Generates rows until we hit `target` accepted rows.
    Writes each accepted row to JSONL immediately (so crashes don't lose progress).
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    seen_fingerprints: set = set()
    accepted = 0
    attempted = 0

    print(f"Generating {target} rows → {output}")
    print("This will take a while. Progress is saved after each accepted row.\n")

    with open(output, "a", encoding="utf-8") as f:
        while accepted < target:
            attempted += 1

            row = generate_one_row()

            # --- filter 1: parsing ---
            if row is None:
                print(f"  [{attempted}] SKIP: parse failed")
                continue

            conv = row["conversation"]
            mems = row["memories"]

            # --- filter 2: length / sanity ---
            if not passes_length_filter(conv, mems):
                print(f"  [{attempted}] SKIP: length filter")
                continue

            # --- filter 3: dedup ---
            fp = fingerprint(conv)
            if is_duplicate(fp, seen_fingerprints):
                print(f"  [{attempted}] SKIP: duplicate")
                continue

            # --- filter 4: LLM judge ---
            if not judge_row(conv, mems):
                print(f"  [{attempted}] SKIP: judge rejected")
                continue

            # All filters passed — write to disk immediately.
            # The training row format matches what Ch15 expects:
            # {"conversation": "...", "memories": [...]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()  # don't lose data if the script crashes

            accepted += 1
            pass_rate = accepted / attempted * 100
            print(f"  [{attempted}] ACCEPTED ({accepted}/{target}, pass rate {pass_rate:.0f}%)")

            # Polite rate limiting — avoid hammering the API.
            # Remove or reduce this if you're using a high-tier API plan.
            time.sleep(0.5)

    print(f"\nDone. {accepted} rows written to {output}")
    print(f"Total attempts: {attempted}, pass rate: {accepted/attempted*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=500,
                        help="Number of accepted rows to generate")
    parser.add_argument("--output", type=str, default="data/memories_train.jsonl",
                        help="Output file path")
    args = parser.parse_args()

    run_pipeline(args.target, args.output)
```

Run it:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python pipeline.py --target 500 --output data/memories_train.jsonl
```

A 500-row run costs roughly $1.50–$2 and takes about 20–30 minutes on a standard API plan. **Start with 500 rows.** You'll iterate on your prompts and filters before investing in a full run.

---

## Step 7 — Inspecting and validating your output

Before you train anything, spend 10 minutes reading 20–30 rows by hand. This is your last quality gate.

```python
# inspect.py
"""Quick script to eyeball a sample of your generated dataset."""
import json
import random

def inspect_jsonl(path: str, n: int = 20):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))

    sample = random.sample(rows, min(n, len(rows)))

    for i, row in enumerate(sample):
        print(f"\n{'='*60}")
        print(f"Row {i+1} | Topic: {row['meta']['topic']}")
        print(f"Persona: {row['meta']['persona']}")
        print(f"\nCONVERSATION:\n{row['conversation']}")
        print(f"\nMEMORIES ({len(row['memories'])}):")
        for mem in row['memories']:
            print(f"  [{mem['type']}] {mem['text']}")
            print(f"    entities: {mem['entities']} | confidence: {mem['confidence']}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/memories_train.jsonl"
    inspect_jsonl(path)
```

What to look for:
- Do the memories feel useful, not obvious?
- Are entities correctly identified?
- Are type labels sensible?
- Is there variety in topics, styles, lengths?

If something looks systematically wrong, fix the generation prompt and regenerate. It's much cheaper to fix the data now than to retrain.

---

## How much data do you actually need?

This is the question everyone asks first, and the honest answer is: less than you fear, and the right amount is whatever your eval tells you. This section is the canonical home for sizing your *first* dataset. (Once you're running this as a living system across many retraining rounds, the question changes shape — how much *new* data per round, and how often — and that lives in *Ch32 - How Much Data, and How Often to Retrain*. Don't size your continual loop from this table.)

Think of it like learning to parallel park. You don't need a thousand attempts to get the basic motion. You need a handful to stop crashing, a few dozen to be reliable, and after that each extra attempt teaches you less than the last. A narrow, well-defined task like memory extraction behaves the same way: the model picks up the *format* almost immediately, the *judgment* over a thousand-ish examples, and then the curve flattens.

Here are the ranges for a ~4B student model on this task, with the reasoning behind each — not magic numbers:

| Stage | Rows | What you get, and why |
|---|---|---|
| **Proof of life** | 200–500 | Enough for the model to reliably emit valid JSON in the pinned schema and stop hallucinating fields. The *shape* of the task gets learned fast — this is exactly the Ch0 speedrun's lower bound. Don't expect production accuracy yet; expect "it clearly understands the job." |
| **Solid baseline** | 1,000–3,000 | Where most projects should live. By here the model has seen enough variety to handle topics and phrasings it wasn't trained on verbatim. Accuracy on a held-out eval set climbs steeply through this band. If you only ever build one dataset, build one here. |
| **Strong** | 3,000–10,000 | Squeezes out the long tail — unusual conversation shapes, rare entity types, tricky `decision` vs. `fact` calls. Note the lever: across this band, *diversity and label quality* move the needle far more than raw row count. 5,000 varied, clean rows beat 10,000 near-duplicates every time. |
| **Diminishing returns** | >~10,000 | For a task this narrow on a model this size, you're mostly paying for marginal gains. Your effort is better spent on eval coverage, harder edge-case data, or the preference-tuning of Parts 7–8 than on generating row 10,001. |

**Why quality and diversity beat raw count.** A model trained on 800 genuinely different conversations generalizes better than one trained on 4,000 that are slight rewordings of the same fifty. Duplicate-ish data doesn't teach the model anything new; it just teaches it to be *more confident* about the narrow slice it already knows — which is overfitting wearing a bigger number. This is why the next section ("Making synthetic data diverse and not slop") matters more than your row target does.

**The tokens rule of thumb.** Another way to sanity-check size: count training *tokens*, not rows. For a narrow LoRA/QLoRA fine-tune, somewhere around **1–5 million training tokens** is ample. A memory-extraction row (a multi-turn conversation plus its JSON output) runs roughly 600–1,200 tokens, so ~2,000–4,000 rows already lands you squarely in that window. If you're well past 5M tokens and your eval has stopped improving, that's the curve flattening — believe it.

**The actual recipe: start small, measure, grow only if eval says so.** Don't pick a final number up front. Generate ~500 rows, fine-tune (*Ch15*), and run the held-out eval from *Ch18 - Did It Actually Work?*. Read where it fails — wrong `type` labels? missed entities? whole categories of conversation it fumbles? Then generate *targeted* data for those specific failures and retrain. *Ch20 - Iterating: From a Mediocre Model to a Good One* is the full playbook for this loop. A growing eval score earns the next batch of data; a flat one tells you to fix quality or coverage instead of adding volume. Iteration beats volume, almost always.

**What this costs.** The good news is that the proof-of-life and baseline stages are cheap enough that there's no excuse not to start. Generating ~1,000 rows with a teacher LLM — generation plus the LLM-as-judge pass — costs only a few dollars, roughly **$1–5** depending on whether you run Sonnet or Opus as the teacher (see the cost ballpark in the pipeline script above). That's the synthetic-data slice of the Ch0 speedrun's **~$5–30 all-in budget**, where the rest goes to an hour or three of rented GPU time for the actual fine-tune. In other words: the experiment that tells you whether this whole approach works for your task costs about the same as lunch. Run it before you agonize over the perfect dataset size.

---

## Scaling up

Once your pipeline is working at 500 rows, scaling is mostly an infrastructure question:

**Parallel generation** — the `call_teacher` function is I/O-bound (waiting on the API). You can parallelize it with Python's `concurrent.futures.ThreadPoolExecutor`:

```python
# Rough sketch — add this to pipeline.py if you need speed
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dedup import fingerprint, is_duplicate, passes_length_filter
from judge import judge_row

# seen_fingerprints must be shared across threads; use a set protected by a lock
# for production use, but for simplicity here we accept minor duplicate risk.
_seen_fingerprints: set = set()

def passes_all_filters(row: dict) -> bool:
    """
    Combines the same four filters used in run_pipeline:
    length check, dedup, and LLM judge. Returns True only if all pass.
    (Parsing is already done by generate_one_row before this is called.)
    """
    conv = row["conversation"]
    mems = row["memories"]
    if not passes_length_filter(conv, mems):
        return False
    fp = fingerprint(conv)
    if is_duplicate(fp, _seen_fingerprints):
        return False
    if not judge_row(conv, mems):
        return False
    return True

def run_parallel(n_workers=5, target=2000, output_path="data/train.jsonl"):
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(generate_one_row) for _ in range(target * 2)]
        accepted = 0
        with open(output_path, "a") as f:
            for future in as_completed(futures):
                if accepted >= target:
                    break
                row = future.result()
                if row and passes_all_filters(row):
                    f.write(json.dumps(row) + "\n")
                    accepted += 1
```

5 workers gives roughly a 4x speedup on typical API rate limits.

**Expanding seeds** — the single best lever for quality. Add 20 more topics, 10 more personas. The model will be exposed to more variety, which directly improves how well it generalizes.

**Domain-specific seeds** — if you're building this for a specific product (e.g., a project management tool), add topics like "discussing sprint planning", "assigning tasks after a meeting". The more your seeds match your real-world input distribution, the better.

---

## Avoiding teacher-model bias

A subtle risk: if your teacher model has quirks (unusual phrasing, a preference for certain memory types), your student will learn those quirks rather than the underlying task. A few guards:

1. **Use a strong, general teacher.** Claude Opus or Sonnet are good choices. Avoid distilling from a model weaker than your student.
2. **Vary the generation temperature.** Temperature controls how random vs. predictable the model's output is — higher values produce more varied text, lower values produce more consistent formatting. Claude defaults to around 1.0. You can explicitly set `temperature=0.9` in the API call for more variety, or `temperature=0.7` for more reliable JSON formatting.
3. **Human spot-check 5% of rows.** Even 25 rows out of 500 gives you a calibration signal. If you're seeing systematic bias, fix the prompt.
4. **Don't mix teachers mid-dataset.** If you switch from Sonnet to Opus halfway through, the dataset will have two subtly different labeling styles. Use one teacher for one dataset version.

---

## Making synthetic data diverse and not slop

"Slop" is the word for synthetic data that *looks* fine row by row but is secretly the same five conversations wearing different hats. It's the single most common way a synthetic dataset quietly fails: every row parses, the judge passes them, the file hits your target row count — and the trained model is brittle because it only ever saw a narrow slice of the world. The seeds in Step 1 are your first defense. This section is about pushing diversity further and catching mode collapse *before* it reaches training.

Think of a teacher LLM as a slightly lazy storyteller. Ask it for "a conversation" a thousand times with the same prompt and it will drift toward its favorite groove — the same opening ("Hey! How's it going?"), the same comfortable topics, the same tidy three-memory output. Your job is to keep nudging it off that groove.

### Seed every axis, not just the topic

The Step 1 seeds already vary topic, persona, style, and turn count. The reason that works is that you're sampling a *combination* each call, so the space of possible conversations is the product of all four lists — fifteen topics × eight personas × five styles × five lengths is already 3,000 distinct setups before the teacher adds its own variation. The practical move when your data feels samey is to widen the *thin* axis. Most people over-invest in topics and under-invest in style and length. A conversation that's "quick back-and-forth, short messages, 6 turns" produces structurally different training signal than "warm and detailed, 14 turns" even on the *same* topic — and your model needs both.

### Control the distribution across the four memory types

The pinned schema has exactly four memory types — `preference | fact | decision | relationship` — and left to its own devices, a teacher will lopsidedly favor `fact`, because facts are the easiest thing to pull out of a transcript. A student trained on that skew gets good at facts and weak at spotting a `decision` ("we're going with the September launch") or a `relationship` ("Maya is Tom's manager"). The fix is to *ask* for balance and then *measure* it.

First, steer generation by occasionally focusing a call on an underrepresented type. Add an optional emphasis to your prompt builder:

```python
# Add to prompts.py — biases a fraction of calls toward a target memory type.
import random

# The four pinned memory types. Keep this list in sync with the schema and
# the verbatim SYSTEM_PROMPT (preference | fact | decision | relationship).
MEMORY_TYPES = ["preference", "fact", "decision", "relationship"]

def type_emphasis(target_type: str | None) -> str:
    """
    Returns an extra instruction nudging the teacher toward a memory type,
    or an empty string for an unbiased call. Used to rebalance the dataset
    when one type is underrepresented.
    """
    if not target_type:
        return ""
    return (
        f"\nIMPORTANT: write the conversation so it naturally contains at least "
        f"two '{target_type}' memories, without forcing it or sounding unnatural."
    )

def pick_emphasis(p_biased: float = 0.4) -> str | None:
    """With probability p_biased, focus this call on a random memory type."""
    if random.random() < p_biased:
        return random.choice(MEMORY_TYPES)
    return None
```

You append `type_emphasis(pick_emphasis())` to the prompt inside `generate_one_row()`. Leaving ~60% of calls unbiased keeps conversations natural; the biased ~40% backfill the thin types.

Second, *measure* the resulting distribution so you're steering with data, not vibes:

```python
# audit_types.py — count memory types across the generated dataset.
import json
from collections import Counter

def type_distribution(path: str) -> Counter:
    counts = Counter()
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            for mem in row["memories"]:
                counts[mem["type"]] += 1
    return counts

if __name__ == "__main__":
    import sys
    dist = type_distribution(sys.argv[1] if len(sys.argv) > 1 else "data/memories_train.jsonl")
    total = sum(dist.values())
    for t, n in dist.most_common():
        print(f"{t:14s} {n:5d}  ({n/total*100:4.1f}%)")
    # Healthy-ish target: no single type above ~50%, none below ~10%.
    # An exact 25/25/25/25 split is NOT the goal — real conversations
    # genuinely contain more facts than decisions. You're avoiding a
    # pathological skew, not enforcing a quota.
```

If `fact` is eating 70% of your memories, raise `p_biased` or add more decision/relationship-rich topics to your seeds, then regenerate the thin slice. You don't need a perfect quota — real conversations skew toward facts — you just need every type *well represented* so the student learns all four.

### Avoid teacher repetition and mode collapse

Beyond type skew, watch for the teacher repeating itself at the *phrasing* level — the same names (every conversation stars a "Sarah" and a "Mike"), the same opening line, the same memory wording. A few cheap habits keep it honest:

- **Run the teacher with temperature variety**, as the previous section describes — a higher `temperature` (≈0.9–1.0) buys more lexical variety; drop it only if JSON formatting gets unreliable.
- **Inject entropy into the prompt.** Pass a couple of random seed names and a random "detail to include" so two calls with the same topic/persona still diverge. A one-line addition like `f"Use names such as {random.choice(NAMES)} and {random.choice(NAMES)}."` measurably broadens the entity vocabulary your student sees.
- **Don't crank the biased fraction too high.** If you force a target type on *every* call, you trade one kind of slop (all facts) for another (every conversation conspicuously engineered to contain a decision). Keep the unbiased majority.

### A light validation and dedup pass before it goes downstream

You already have the right *primitives* — `fingerprint`/`is_duplicate` and `passes_length_filter` from Step 5, plus the LLM judge. The point here is to run a final, cheap, deterministic sweep over the *finished* file to catch the diversity failures that slip past row-by-row checks, and to confirm the data is structurally sound before training touches it:

```python
# validate.py — a fast, no-API gate to run once on the finished JSONL.
import json
from collections import Counter

VALID_TYPES = {"preference", "fact", "decision", "relationship"}

def validate_dataset(path: str) -> None:
    rows, fingerprints, bad = [], Counter(), 0
    with open(path) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            for mem in row["memories"]:
                # Schema check against the four pinned types.
                if mem["type"] not in VALID_TYPES:
                    print(f"  row {i}: invalid type {mem['type']!r}")
                    bad += 1
                if not mem["text"].strip().endswith((".", "!", "?")):
                    # The schema wants standalone *sentences*, not fragments.
                    print(f"  row {i}: non-sentence text: {mem['text']!r}")
            # Near-duplicate detection on the first 200 chars (same as Step 5).
            fp = row["conversation"][:200].lower().replace(" ", "")
            fingerprints[fp] += 1
            rows.append(row)

    dupes = sum(c - 1 for c in fingerprints.values() if c > 1)
    print(f"\n{len(rows)} rows | {bad} schema issues | ~{dupes} near-duplicates")
    print("If duplicates are >2-3% of rows, widen your seeds and regenerate.")

if __name__ == "__main__":
    import sys
    validate_dataset(sys.argv[1] if len(sys.argv) > 1 else "data/memories_train.jsonl")
```

This is deliberately a *light* pass — schema sanity, sentence-shape, and a duplicate count, all without spending another API token. It's a tripwire, not a deep clean. The full train/validation/test split and rigorous schema validation belong in *Ch14 - Cleaning, Splitting, and Sanity-Checking Data*, which runs right after this chapter. And once you're in the continual-learning loop and need to actively *select* which data earns a place in the next round, the advanced curation lives in *Ch31 - Selecting and Curating Data That Actually Helps*. For now, the goal is simply: don't hand obviously sloppy data to the trainer.

---

## Common mistakes

**Generating all the same topic.** You seeded 15 topics but the model gravitated toward 3 of them. Fix: use `random.sample` instead of `random.choice` to cycle through topics more evenly, or track topic frequency and weight against it.

**JSON in memories that doesn't parse.** The teacher occasionally wraps JSON in markdown code fences (` ```json ... ``` `). Fix: strip code fences before `json.loads`. Add this to `parse_response`:
```python
raw_json = mem_match.group(1).strip().lstrip("```json").rstrip("```").strip()
```

**LLM judge approving everything.** If your judge prompt is too loose, it'll pass 99% of rows. Test it on 10 rows you know are bad. If it passes them, tighten the criteria or add examples of what "fail" looks like.

**Running the full pipeline before testing the prompt.** Always generate 10 rows manually and inspect them before launching a 1000-row run. One bad parameter costs you dollars and time.

**Not flushing to disk.** If you buffer writes and the script crashes at row 800, you lose everything. The pipeline above calls `f.flush()` after every write — keep that.

**Forgetting to set the API key.** The script will fail immediately with an `AuthenticationError`. Set `ANTHROPIC_API_KEY` in your shell before running, or load it from a `.env` file with `python-dotenv`.

---

## Recap

- You rarely have labeled data for a custom task — synthetic generation is the standard solution
- A teacher LLM generates (conversation, memory JSON) pairs; your student model trains on those pairs
- Diversity in seeds (topics, personas, styles) directly determines how well your model generalizes
- LLM-as-judge is a cheap, automatic way to filter low-quality rows before training
- Deduplication and length filters catch edge cases the judge misses
- Start with 500 rows; iterate on your prompts and filters before scaling
- Always inspect a sample by hand before training — bugs in data are cheaper to fix now than after training

---

## Next

**Ch14 - Cleaning, Splitting, and Sanity-Checking Data** — now that you have a raw JSONL file, you'll split it into train/validation/test sets, run final schema validation, and prepare it in the exact format that Ch15's training script expects.
