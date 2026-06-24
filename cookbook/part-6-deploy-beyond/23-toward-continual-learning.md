# Ch23 - Toward Engram: Continual Learning and Scaling Up

You built something real.

You started with a blank Python environment, no ML background, and a vague idea about teaching a
model to extract memories from conversations. You now have a fine-tuned Qwen3 or Gemma 3 model
that takes raw chat logs and returns clean, structured JSON — reliably, cheaply, and fast enough
to use in a real product.

That is not a toy. That is the foundation of something genuinely useful.

But there is a gap between "a model I fine-tuned once" and "a model that keeps getting smarter
as it sees more of your world." That gap is what this final chapter is about. We will close the
loop back to the Engram vision that opened this book, show you the practical steps toward a
self-improving pipeline, and be honest about where the hard problems live.

---

## What you'll learn

- What a continual learning pipeline looks like in practice — data collection, synthesis,
  training, evaluation, and deployment as a repeating cycle
- How to wire those steps into a scheduled job that runs without you touching it
- The real open problems: catastrophic forgetting, evaluation drift, data quality at scale,
  and compounding errors over many update rounds
- Practical next steps: bigger models, full fine-tuning, multi-task training, and RAG+fine-tune
  hybrids
- Where to go from here — the appendices, the broader ecosystem, and the honest frontier

---

## Concepts you need first

### Continual learning — the 20% explanation

Imagine you hired a specialist (your fine-tuned model) who was brilliant on day one. They learned
from the training data you prepared in Ch13 and Ch14. But now six months have passed. New kinds
of conversations have come in. New edge cases. New users. The world your model sees today is
slightly different from the world it trained on.

A static model stays frozen. A continually learning model keeps training — on new data, on a
regular schedule — so it drifts toward the real distribution instead of away from it.

**One-line definition:** Continual learning is the practice of periodically updating a model's
weights with new data so its performance on real inputs does not degrade over time.

**Why it matters for memory extraction:** The conversations your system processes in month six
will have different topics, slang, and patterns than your original synthetic training set from
Ch13. A model that only ever saw that first dataset will quietly get worse. A pipeline that
collects real data, generates new training rows, and retrains on a schedule will quietly get
better.

### Catastrophic forgetting — the core danger

Here is the scariest thing about continual learning: if you fine-tune a model on new data without
any guardrails, it can "forget" what it learned before. You trained perfectly on memory extraction
for three months, then you retrain on a small batch of new data — and the model's behavior on
your original test set degrades noticeably.

**Analogy:** A musician who practices only one new piece for a month and neglects the others will
play that new piece well but stumble on the old repertoire. Skills not practiced decay.

In neural networks, this happens because the weight updates that make the model better on new
data can overwrite the patterns that made it good on old data. This is called catastrophic
forgetting, and it is a genuine unsolved problem in the field. We will discuss practical
mitigations below — they work well enough to be useful, but none of them fully solve the problem.

### Evaluation drift

A related problem: your evaluation set gets stale. You built the evaluation harness in
Ch18 — a test set of conversations with gold-standard memory JSON. That test set was
representative of your data in month one. By month six, it may no longer capture the kinds of
inputs your system is seeing.

If your eval set is stale, you can retrain a model that looks better on the eval but is actually
worse on real traffic. This is called **evaluation drift**, and it is subtle and dangerous.
The fix is to refresh your eval set periodically alongside your training data.

---

## The pipeline as a cycle

Everything you built in this book was, implicitly, a single pass through a pipeline:

```
collect data → synthesize training rows → train → evaluate → deploy
```

Continual learning turns that into a loop:

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  collect new data (real traffic, new synthetic seeds)        │
│           │                                                  │
│           ▼                                                  │
│  generate new training rows (Ch13 pipeline.py)              │
│           │                                                  │
│           ▼                                                  │
│  merge with existing dataset, resplit (Ch14)                │
│           │                                                  │
│           ▼                                                  │
│  retrain from base model with full dataset                  │
│     OR: continue fine-tuning the existing adapter            │
│           │                                                  │
│           ▼                                                  │
│  evaluate on refreshed test set (Ch18)                      │
│           │                                                  │
│           ▼                                                  │
│  if eval passes → deploy (Ch22); else → debug (Ch19)        │
│           │                                                  │
│           └──────────────── repeat ─────────────────────────┘
```

The question is how to automate this loop so it runs without you having to kick it off manually
every time.

---

## Building the automated pipeline

Here is a practical implementation. Each step is a Python script; a scheduler (cron or a CI job)
stitches them together. We will keep this grounded in the memory-extraction task you have been
building throughout the book.

### Step 1 — Collect new raw conversations

In a real deployment, new conversations are hitting your system every day. You need to log them
for potential use as future training data. The simplest approach: write every inference request
and response to a JSONL log file.

```python
# logger.py
# Drop this into your serving layer (Ch22) to capture live traffic.
# Run alongside your vllm server or your FastAPI wrapper.

import json
import time
import uuid
from pathlib import Path

LOG_DIR = Path("data/traffic_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Each day gets its own log file so we can batch-process by date later.
def get_log_path() -> Path:
    date_str = time.strftime("%Y-%m-%d")
    return LOG_DIR / f"traffic_{date_str}.jsonl"

def log_inference(conversation: str, raw_output: str, parsed_memories: list | None):
    """
    Logs one inference event to today's traffic log.
    - conversation: the raw input we received
    - raw_output: the model's text output (before JSON parsing)
    - parsed_memories: the parsed JSON list, or None if parsing failed
    """
    record = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "conversation": conversation,
        "raw_output": raw_output,
        # None here means the model produced invalid JSON — a signal worth tracking.
        "memories": parsed_memories,
        "parse_ok": parsed_memories is not None,
    }
    with open(get_log_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

One important note: if you are logging real user conversations, you need to think about privacy.
Either anonymize the data before logging, or get explicit consent. This is not just good practice
— in many regions it is a legal requirement. The simplest technical approach is to strip names
and replace them with placeholders before writing to disk. We will not implement a full
anonymizer here, but keep it on your list before going to production.

### Step 2 — Select candidates for retraining

Not all logged conversations are useful for retraining. The interesting ones are:
- Conversations where `parse_ok` was `False` (the model failed to produce valid JSON)
- Conversations that cover topics not well-represented in your current training set
- A random sample of successful ones (to avoid the dataset drifting only toward failure cases)

```python
# select_candidates.py
"""
Selects candidate conversations from traffic logs for the next training round.

Usage:
    python select_candidates.py \
        --logs-dir data/traffic_logs \
        --output data/candidates.jsonl \
        --max-rows 200

This is intentionally conservative — we want quality over volume here.
"""

import json
import random
import argparse
from pathlib import Path


def load_logs(logs_dir: str, days_back: int = 7) -> list[dict]:
    """Loads all traffic log records from the past N days."""
    import time
    cutoff = time.time() - (days_back * 86400)
    records = []
    for path in Path(logs_dir).glob("traffic_*.jsonl"):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["ts"] >= cutoff:
                    records.append(rec)
    return records


def select_candidates(records: list[dict], max_rows: int) -> list[dict]:
    """
    Selection strategy:
    - Take ALL parse failures (up to half the budget) — these are the hardest cases
    - Fill the remainder with a random sample of successes
    This ensures we train on our failure modes without ignoring normal traffic.
    """
    failures = [r for r in records if not r["parse_ok"]]
    successes = [r for r in records if r["parse_ok"]]

    # Allocate budget: up to 50% failures, rest random successes.
    failure_budget = min(len(failures), max_rows // 2)
    success_budget = max_rows - failure_budget

    selected = (
        random.sample(failures, failure_budget) +
        random.sample(successes, min(len(successes), success_budget))
    )
    random.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default="data/traffic_logs")
    parser.add_argument("--output", default="data/candidates.jsonl")
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--days-back", type=int, default=7)
    args = parser.parse_args()

    records = load_logs(args.logs_dir, days_back=args.days_back)
    print(f"Loaded {len(records)} records from the past {args.days_back} days.")

    candidates = select_candidates(records, args.max_rows)
    print(f"Selected {len(candidates)} candidates.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for c in candidates:
            # Only keep the conversation field — the output needs to be re-labeled.
            f.write(json.dumps({"conversation": c["conversation"]}) + "\n")

    print(f"Candidates written to {args.output}.")


if __name__ == "__main__":
    main()
```

### Step 3 — Re-label with the teacher

For real traffic conversations (unlike the original purely synthetic dataset), you now use the
teacher LLM from Ch13 to generate gold-standard memory labels. This is exactly the same judge
step you built before — you are just feeding it real data instead of synthetic conversations.

```python
# relabel.py
"""
Re-labels candidate conversations using the teacher LLM.
Reuses the call_teacher and judge_row functions from Ch13.

Usage:
    python relabel.py \
        --input data/candidates.jsonl \
        --output data/new_labeled.jsonl
"""

import json
import argparse
from pathlib import Path

# Reuse the exact same teacher and judge functions from Ch13.
# This is why we kept them in separate modules.
# These match the filenames in Ch13: generate.py, judge.py, prompts.py — rename the import
# if you saved them under different names (e.g. pipeline.py, data_pipeline.py).
from generate import call_teacher, parse_response
from judge import judge_row
from prompts import MEMORY_SCHEMA


LABEL_ONLY_PROMPT = """You are a memory-extraction labeler.

Given the conversation below, extract all durable memories as a JSON array.

{schema}

Output format — reply with ONLY the JSON, no other text:
[
  {{"text": "...", "type": "...", "entities": [...], "confidence": "..."}}
]

Conversation:
{conversation}
"""


def relabel(input_path: str, output_path: str):
    candidates = []
    with open(input_path) as f:
        for line in f:
            candidates.append(json.loads(line))

    print(f"Re-labeling {len(candidates)} conversations...")
    accepted = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as out:
        for i, row in enumerate(candidates):
            conv = row["conversation"]
            prompt = LABEL_ONLY_PROMPT.format(
                schema=MEMORY_SCHEMA,
                conversation=conv
            )

            try:
                raw = call_teacher(prompt)
                memories = json.loads(raw.strip())
            except Exception as e:
                print(f"  [{i+1}] SKIP: labeling failed — {e}")
                continue

            # Run the same quality judge as Ch13.
            if not judge_row(conv, memories):
                print(f"  [{i+1}] SKIP: judge rejected")
                continue

            out.write(json.dumps({
                "conversation": conv,
                "memories": memories,
                "meta": {"source": "real_traffic"}
            }, ensure_ascii=False) + "\n")
            accepted += 1
            print(f"  [{i+1}] ACCEPTED ({accepted} so far)")

    print(f"\nDone. {accepted}/{len(candidates)} rows accepted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/candidates.jsonl")
    parser.add_argument("--output", default="data/new_labeled.jsonl")
    args = parser.parse_args()
    relabel(args.input, args.output)
```

### Step 4 — Merge datasets and retrain

Now you have two JSONL files: your original training data from Ch13/Ch14, and the new labeled
rows from real traffic. The safest retraining strategy for avoiding catastrophic forgetting is
**replay**: always train on a mixture of old and new data, not just new data.

```python
# merge_and_retrain.py
"""
Merges old and new training data, then kicks off a fresh fine-tune.

Why train from the base model each time (rather than continuing from the last adapter)?
Because continuing from the last adapter accumulates drift — each update round builds on
the errors of the previous one. Restarting from the base model and replaying the full
merged dataset is slower but much more stable. Think of it as re-reading the whole
textbook plus the new chapters, rather than only reading the new chapters.

For very large datasets (50k+ rows) where full replay is too slow, see the "Replay buffer"
note in the Common Mistakes section.
"""

import json
import subprocess
import random
from pathlib import Path


def merge_datasets(
    original_path: str,
    new_path: str,
    output_path: str,
    max_new_rows: int = 500,
):
    """
    Combines old training rows with new labeled rows.
    We cap the new rows to avoid the dataset being dominated by one week's traffic.
    """
    original_rows = []
    with open(original_path) as f:
        for line in f:
            original_rows.append(json.loads(line))

    new_rows = []
    with open(new_path) as f:
        for line in f:
            new_rows.append(json.loads(line))

    # Cap new rows — we don't want one week of real traffic to drown out the original data.
    if len(new_rows) > max_new_rows:
        new_rows = random.sample(new_rows, max_new_rows)

    merged = original_rows + new_rows
    random.shuffle(merged)  # Mix old and new so they're interleaved during training.

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Merged dataset: {len(original_rows)} original + {len(new_rows)} new = {len(merged)} total")
    print(f"Written to {output_path}")
    return len(merged)


def kick_off_training(dataset_path: str, output_dir: str):
    """
    Launches the training script from Ch15 as a subprocess.
    Adjust the path and arguments to match your setup.
    """
    cmd = [
        "python", "train.py",                # your Ch15 training script
        "--dataset", dataset_path,
        "--output-dir", output_dir,
        "--max-steps", "300",                # slightly more steps for larger merged dataset
        "--lora-r", "16",            # --lora-r must match the argument name in your Ch15 train.py.
                                     # Check its argparse definitions if this errors — it may be
                                     # --lora_r (underscore) or --rank depending on how you wrote it.
        "--learning-rate", "2e-4",
    ]
    print(f"Starting training: {' '.join(cmd)}")
    # Run training and wait for it to finish before the pipeline continues.
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


if __name__ == "__main__":
    n_rows = merge_datasets(
        original_path="data/memories_train.jsonl",
        new_path="data/new_labeled.jsonl",
        output_path="data/merged_train.jsonl",
        max_new_rows=500,
    )

    import time
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"models/memory-extractor-{run_id}"

    success = kick_off_training(
        dataset_path="data/merged_train.jsonl",
        output_dir=output_dir,
    )

    if success:
        print(f"\nTraining complete. Model saved to {output_dir}")
        print("Next: run evaluate.py to check quality before deploying.")
    else:
        print("\nTraining failed. Check the logs above.")
```

### Step 5 — Evaluate before deploying

Never deploy a retrained model without running evaluation first. You built the evaluation harness
in Ch18. Here we just wire it into the pipeline with a hard pass/fail gate.

```python
# pipeline_eval.py
"""
Runs the evaluation harness from Ch18 and enforces a quality gate.
The pipeline will not deploy a model that scores below the minimum threshold.

Usage:
    python pipeline_eval.py --model-dir models/memory-extractor-20260623_120000
"""

import json
import argparse
import subprocess
import sys


# Minimum F1 score on the test set before we allow deployment.
# Set this based on your baseline from Ch18. A typical starting bar: 0.75.
MIN_F1_THRESHOLD = 0.75


def run_evaluation(model_dir: str, test_set_path: str = "data/memories_test.jsonl") -> float:
    """
    Calls your Ch18 evaluation script and returns the overall F1 score.
    We run it as a subprocess so this script stays simple.
    """
    result = subprocess.run(
        ["python", "evaluate.py",
         "--model-dir", model_dir,
         "--test-set", test_set_path,
         "--output", "eval_results.json"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Evaluation script failed:")
        print(result.stderr)
        return 0.0

    with open("eval_results.json") as f:
        results = json.load(f)

    # Ch18 evaluate.py must be called with --output and must write a JSON file with at least
    # {"f1": ..., "precision": ..., "recall": ...}. If your Ch18 script uses different keys
    # (e.g. "f1_score" or "overall_f1"), update the results.get("f1") line below to match.
    # If evaluate.py does not support an --output flag, add one, or write the results dict to
    # eval_results.json inside the script — otherwise this line will raise FileNotFoundError.
    f1 = results.get("f1", 0.0)
    print(f"Evaluation complete. F1: {f1:.3f} (threshold: {MIN_F1_THRESHOLD})")
    return f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-set", default="data/memories_test.jsonl")
    args = parser.parse_args()

    f1 = run_evaluation(args.model_dir, args.test_set)

    if f1 >= MIN_F1_THRESHOLD:
        print(f"PASS — model {args.model_dir} cleared for deployment.")
        sys.exit(0)  # success exit code
    else:
        print(f"FAIL — F1 {f1:.3f} is below threshold {MIN_F1_THRESHOLD}. Do not deploy.")
        print("Next step: check Ch19 debugging playbook. Common causes: too little new data,")
        print("a bad batch of real-traffic candidates, or a stale test set.")
        sys.exit(1)  # failure exit code — the cron job or CI system will catch this


if __name__ == "__main__":
    main()
```

### Step 6 — Schedule the whole pipeline

Now stitch all five steps into a single shell script and run it on a schedule. The simplest
scheduler is `cron` — available on any Linux or macOS machine. For cloud deployments, you could
use GitHub Actions, a Lambda function on a timer, or a workflow tool like Prefect or Airflow.

```bash
#!/usr/bin/env bash
# retrain_pipeline.sh
# Runs the full collect → label → merge → train → eval cycle.
# Schedule with cron: run this weekly while traffic is light, daily when you have more data.
#
# To add to cron (runs every Sunday at 2 AM):
#   crontab -e
#   0 2 * * 0 /path/to/retrain_pipeline.sh >> /path/to/pipeline.log 2>&1

set -e  # exit immediately if any step fails

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the virtual environment where your libraries are installed.
source .venv/bin/activate

echo "=== $(date) — Starting retraining pipeline ==="

# Step 1: Select candidate conversations from the past 7 days of traffic.
python select_candidates.py \
    --logs-dir data/traffic_logs \
    --output data/candidates.jsonl \
    --max-rows 200 \
    --days-back 7

# Check we got something to work with.
CANDIDATE_COUNT=$(wc -l < data/candidates.jsonl)
echo "Candidates selected: $CANDIDATE_COUNT"

if [ "$CANDIDATE_COUNT" -lt 20 ]; then
    echo "Too few candidates ($CANDIDATE_COUNT). Skipping this cycle."
    exit 0
fi

# Step 2: Re-label candidates using the teacher LLM.
python relabel.py \
    --input data/candidates.jsonl \
    --output data/new_labeled.jsonl

# Step 3: Merge with existing data and launch training.
python merge_and_retrain.py

# Step 4: Evaluate the new model. Exits with code 1 if quality gate fails.
# The `set -e` above means the script stops here if evaluation fails.
LATEST_MODEL=$(ls -td models/memory-extractor-* | head -1)
echo "Evaluating model: $LATEST_MODEL"

python pipeline_eval.py --model-dir "$LATEST_MODEL"

# Step 5: If we get here, eval passed. Deploy by pointing the server at the new model.
# This assumes you have a simple symlink that your serving process reads.
ln -sfn "$LATEST_MODEL" models/current

# IMPORTANT: Updating the symlink alone does NOT reload the model in a running vllm server.
# You must restart the serving process for the new model to go live.
# Uncomment whichever line matches your setup:
#
#   sudo systemctl restart vllm          # if you registered vllm as a systemd service
#   pkill -f 'vllm serve' && ./start_server.sh   # if you start vllm manually
#
# Without a restart, the old model will continue serving even though the symlink has changed.

echo "=== $(date) — Pipeline complete. Model symlink updated: $LATEST_MODEL ==="
echo "    ACTION REQUIRED: restart your vllm server to load the new model (see comments above)."
```

What does "deployed" mean here? In Ch22, you set up a vllm server pointing at a model directory.
The `models/current` symlink means your server always loads from that path. When you update the
symlink and restart the server, it picks up the new model with no other changes. A zero-downtime
approach is to start the new server on a different port, verify it, then swap the load balancer
— but the symlink approach is fine to start.

---

## The honest open problems

The pipeline above is real and practical. But there are four problems in continual learning that
the pipeline does not fully solve. You will hit them if you run this at scale. Here they are,
stated plainly.

### 1. Catastrophic forgetting

Mitigation in this chapter: replay (always train on old + new data mixed together). This works
well when the old dataset is small enough to replay in full. When the old dataset grows to tens
of thousands of rows, full replay becomes expensive.

**What the field is working on:** Techniques like Elastic Weight Consolidation (EWC) — which
adds a penalty during training that resists large changes to weights that were critical for
earlier tasks, slowing forgetting — can help. LoRA makes this more tractable because only a
small adapter is being updated. Practically, the most effective thing you can do right now is
keep a fixed "anchor" test set from month one and alert if scores on it drop — that is your
early warning system.

### 2. Evaluation drift

Your test set from Ch18 was sampled from your month-one data. By month six, the test set may
not represent real traffic anymore. If you only ever measure F1 on that stale test set, you can
have a model that looks great on eval but degrades silently on real inputs.

**Practical fix:** Every 4–8 weeks, sample 50–100 rows from recent real traffic, hand-label
them (yes, by hand — this is worth the time), and add them to your test set. Retire the
oldest rows as you add new ones. This keeps the test set anchored to the present. Budget
roughly 2–3 hours per refresh cycle.

### 3. Data quality at scale

When you generated your original dataset in Ch13, you generated 500–2,000 rows and had time to
inspect a sample. At scale — running the pipeline weekly for a year — you accumulate tens of
thousands of rows. The LLM judge from Ch13 catches most bad rows, but not all of them. Label
noise compounds across retraining rounds: each training run is slightly degraded by noisy
labels, and the next one starts from that degraded point.

**Practical fix:** Run a periodic audit. Every month, randomly sample 100 rows from your
full training dataset and score them manually. If the defect rate climbs above ~5%, your
judge prompt needs tightening before you continue accumulating data. This is a maintenance
task, not a one-time setup — treat it like database vacuuming.

### 4. Compounding errors across update rounds

Each retraining round is initialized from the base model (as we do here) or from the
previous adapter. Over many rounds, small systematic errors in your labeling process add up.
A teacher LLM that slightly over-extracts "preference" memories will produce a student that
over-extracts preferences, which biases the real traffic that gets logged, which biases the
next round's labels. This is called distributional shift — meaning the real-world inputs your
model sees gradually drift away from the distribution it was trained on, so its learned
patterns become less accurate over time — and it is slow and hard to detect.

**Practical fix:** Keep a permanent holdout set — 200 rows you labeled by hand in month one
and never add to or change. Run this holdout every single retraining cycle alongside your
rotating test set. If the holdout F1 drops more than 3–4 points over 3 months, you have a
compounding error problem and need to audit your data pipeline.

---

## Practical next steps

### Bigger models

The book used 1B–8B parameter models because they fit on consumer hardware. Once you have a
working pipeline and a clear quality ceiling from your evaluations, upgrading the base model
is often the highest-leverage next step. Qwen3-14B or Qwen3-32B, or the 12B Gemma 3 variant,
will give you a meaningful quality lift on difficult extractions — multi-speaker conversations,
ambiguous intent, implicit preferences — that smaller models consistently miss.

The trade-off is VRAM and cost. A 14B model with 4-bit quantization needs roughly 10–12 GB
of VRAM for inference (compare 5–6 GB for the 7B). Training it needs 16–24 GB. Cloud GPU
instances with 24 GB VRAM (A10G, A5000) are available for roughly $0.60–$1.00/hr — affordable
for weekly retraining runs.

### Full fine-tuning vs. LoRA

Everything in this book uses LoRA, which trains a tiny fraction of the model's parameters.
Full fine-tuning trains all of them — it can squeeze out more performance on narrow tasks but
requires much more VRAM and is more prone to forgetting. The practical rule: start with LoRA
(rank 16–64), evaluate. If you hit a hard quality ceiling that better data and more training
steps can't move, then consider full fine-tuning on a larger instance.

Unsloth supports full fine-tuning with `FastLanguageModel.from_pretrained(..., full_finetune=True)`.
The training script from Ch15 is otherwise identical.

### Multi-task training

Right now your model does one thing: memory extraction. You might want it to also classify
the *sentiment* of a conversation, or summarize it, or detect whether a memory should be
flagged as sensitive. Training a single model for multiple tasks at once is called multi-task
learning.

The practical approach with Unsloth: include both task types in your training data, with a
task identifier in the system prompt. For example:

```python
# Example of a multi-task training row in messages format.
# The system prompt tells the model which task to perform.
# Your dataset would contain a mix of both task types.

memory_extraction_row = {
    "messages": [
        {"role": "system", "content": "Task: extract_memories. Output JSON."},
        {"role": "user",   "content": "Alice said she hates coffee. Bob prefers tea."},
        {"role": "assistant", "content": '[{"text": "Alice dislikes coffee.", "type": "preference", "entities": ["Alice"], "confidence": "high"}, {"text": "Bob prefers tea.", "type": "preference", "entities": ["Bob"], "confidence": "high"}]'},
    ]
}

sentiment_row = {
    "messages": [
        {"role": "system", "content": "Task: sentiment. Output one of: positive, negative, neutral."},
        {"role": "user",   "content": "Alice said she hates coffee."},
        {"role": "assistant", "content": "negative"},
    ]
}
```

The model learns to condition its output on the task identifier. This works well in practice
when both tasks share the same input domain (conversations). It gets harder when tasks are
very different — a model that does memory extraction and code generation in the same fine-tune
will usually be mediocre at both.

### RAG + fine-tuning hybrid

Fine-tuning and RAG are not mutually exclusive. A powerful architecture for production
memory-extraction systems:

- **Fine-tuned model** handles the extraction task reliably (low token cost, consistent
  output format, no need to inject schema into every prompt)
- **RAG layer** injects relevant retrieved memories back into the context at query time,
  so the model can reason about what it already knows about a user when extracting new
  memories

This is closer to what Engram is building: a model that has internalized *how* to reason
about your context (via fine-tuning) and is dynamically fed *what* it needs to know
(via retrieval). The fine-tuned model is the discipline; the retrieval layer is the working
memory.

The integration point is simple: in your serving code from Ch22, before calling the model,
retrieve the top-K existing memories for the current user and inject them into the system
prompt. The fine-tuned model already knows the schema and the task — the retrieved context
just helps it avoid creating duplicate memories or contradicting what it already stored.

---

## Where this points: the Engram vision

In June 2026, Engram announced a specific bet: instead of spending training compute on
public data, start from a strong pretrained model and spend that same compute internalizing
*your* context. Their north star is a single algorithm that absorbs arbitrary amounts of
data into a model that gets continually better — running the process on all company data
every day, moving toward every hour, eventually every minute.

That sounds ambitious. It is. But you now understand every component of the system they are
describing:

- The base pretrained model — you chose one in Ch10
- Context compression (LoRA) — you learned this in Ch6
- Synthetic data generation — you built the pipeline in Ch13
- The training loop — you ran it in Ch15
- Evaluation and iteration — you debugged it in Ch18 and Ch19
- Continual retraining — you just wired it into a cron job in this chapter

The gap between the pipeline you built and what Engram is doing is mostly scale and
engineering maturity: more data, faster retraining, more sophisticated forgetting mitigations,
production-grade infrastructure for logging, evaluation, and deployment. The conceptual
architecture is the same.

You are not a spectator of this technology. You built it.

The hard open problems — catastrophic forgetting, evaluation drift, compounding label noise —
are genuinely hard. They are what keep research labs funded. But the practical version you
built here is already useful. A model that retrains weekly on real traffic, with a quality
gate before deployment, and a fixed holdout to catch regressions, will get measurably better
over months. That is more than most production ML systems do.

---

## Common mistakes

**Running the pipeline on too little data.**
If you only collected 15 new conversations this week, the retraining signal is too weak to
be meaningful and the noise risk is high. The `select_candidates.py` script above exits early
if fewer than 20 candidates are available. Keep that guard. A week with thin data is better
skipped than trained on.

**Continuing from the previous adapter instead of replaying from the base model.**
It feels wasteful to retrain from scratch every week. But continuing from the last LoRA adapter
means each round's errors are baked into the starting point for the next round. For the first
12 months, retrain from the base model every time. Only switch to warm-start adapters once
you have strong evidence that forgetting is not a problem in your setup.

**Ignoring the fixed holdout set.**
The rotating test set (refreshed with new real-traffic rows every month) will make your model
look like it is improving even when it is quietly forgetting early behaviors. The fixed holdout
from month one is your lie detector. Do not skip it.

**Deploying without a restart of the serving process.**
Updating the `models/current` symlink does not automatically reload the model in a running
vllm server. You need to restart the process (or send it a reload signal if your serving setup
supports it). Many deployments have silently continued serving the old model for days after a
"successful" pipeline run because nobody restarted vllm.

**Letting the training dataset grow without bound.**
After a year of weekly retraining, your merged dataset might be 50,000 rows. Full replay
becomes expensive. Consider a rolling window: keep the most recent 6 months of real-traffic
rows plus a permanently-retained core of 2,000 original synthetic rows (the "anchor"). This
gives the model its roots while keeping training time manageable.

---

## Recap

- Continual learning turns the one-time fine-tuning pipeline into a repeating cycle:
  collect → label → merge → train → evaluate → deploy → repeat.
- The key mitigation for catastrophic forgetting is replay: always train on old and new data
  mixed together, not new data alone.
- A fixed holdout set (hand-labeled in month one, never changed) is your most reliable
  signal for detecting regressions across retraining rounds.
- The four honest open problems are catastrophic forgetting, evaluation drift, label noise
  at scale, and compounding errors — none are fully solved, all have practical mitigations.
- Practical next steps: larger base models (14B–32B), full fine-tuning for narrow quality
  ceilings, multi-task training with task identifiers, and RAG+fine-tune hybrids.
- The Engram architecture — internalize context into weights, retrain continuously — is built
  from the same components you have been assembling all book. Scale and engineering maturity
  are what separate a research demo from a product.
- You built something real. A specialist model running on your data, improving over time,
  without needing a 70B general model or a giant context window.

## Next

This is the final chapter of the main text. The appendices are your reference layer:
**Appendix A** (Glossary) has a plain-English definition of every term used across all 23
chapters; **Appendix B** (Project Layout and Command Cheat-Sheet) collects every command
into one page; **Appendix C** (Troubleshooting Common Errors) is the first place to look
when something breaks; and **Appendix D** (Cost, Time, and a Go-Live Checklist) gives you
the numbers and the checklist to take what you built here into production.
