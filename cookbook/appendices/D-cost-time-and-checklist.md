# Appendix D - Cost, Time, and a Go-Live Checklist

You have read the book. You have run the scripts. Now comes the real question: *how much will this actually cost me, and how long will it take?*

This appendix answers that concretely — not with vague ranges, but with the specific ballparks you need to plan your project. Then it ends with a final pre-flight checklist: the things you should verify before trusting your memory-extraction model with real data.

---

## What you'll learn

- Realistic time and GPU cost estimates for each phase: data generation, training, evaluation, and serving
- A side-by-side comparison of the free Colab path vs. paid cloud GPU paths
- A final pre-flight checklist you can run through before deploying your model
- Where to go next without leaving this ecosystem mid-task

---

## Concepts you need first

### GPU-hours and dollars

Cloud GPU providers charge by the hour for GPU access. The price depends on which GPU you rent. The unit is simple: **one A10G GPU for one hour costs roughly $0.50–$0.80** depending on the provider (Lambda Labs, RunPod, Vast.ai). An A100 80GB costs roughly $2.00–$3.50/hour. A T4 is the cheapest at roughly $0.30–$0.50/hour but is slower and has less memory.

When this book says "2 GPU-hours on an A10G," that means your wall-clock time will be 2 hours and your bill will be roughly $1.00–$1.60.

### Colab free tier

Google Colab gives you free access to a T4 or occasionally an L4 GPU, but with limits: sessions time out after roughly 12 hours of idle, and Colab Pro ($10/month) gets you priority access and longer sessions. The free tier is enough to run training once and evaluate it. It is not reliable enough for multi-day iteration work. Budget $10–$20 for Colab Pro if you plan to use it seriously.

### Tokens vs. rows in data generation

When you generate synthetic training data with an API (as covered in *Ch13 - Creating Your Training Data with Synthetic Generation*), you pay per token. The OpenAI API, Anthropic API, or Gemini API all charge per million tokens of input and output. For our task — generating conversations and their memory JSON — each row costs roughly 500–1,500 tokens total (input + output combined). At GPT-4o's current pricing (roughly $5/million input tokens, $15/million output tokens as of mid-2026), generating 1,000 training rows costs roughly $2–$5 depending on conversation length.

---

## Phase-by-phase time and cost breakdown

All numbers below assume the memory-extraction task on a 7B model (Qwen3-8B or Gemma-3-4B), which is the main recommended size throughout this book. Numbers in parentheses are the 1.7B/3B variant which runs faster and cheaper but with lower ceiling quality.

### Phase 1 — Synthetic data generation

This is the cheapest phase, but it depends heavily on how much data you generate and which API you use.

| Rows generated | Tokens used (approx.) | GPT-4o cost | Gemini 1.5 Flash cost | Time |
|---|---|---|---|---|
| 200 rows | ~250K tokens | ~$1.50 | ~$0.10 | 5–15 min |
| 500 rows | ~600K tokens | ~$4.00 | ~$0.25 | 15–40 min |
| 1,000 rows | ~1.2M tokens | ~$8.00 | ~$0.50 | 30–90 min |
| 2,000 rows | ~2.5M tokens | ~$16.00 | ~$1.00 | 1–3 hours |

**Key takeaway:** If cost is a concern, use Gemini 1.5 Flash for data generation — it is roughly 10x cheaper and fast enough for this task. The quality of synthetic conversations is good enough for training data purposes. Save the more expensive frontier models for your evaluation set, where quality of the reference answers matters more.

**Minimum viable dataset:** As discussed in *Ch14 - Cleaning, Splitting, and Sanity-Checking Data*, 300–500 diverse training rows is enough to get a working model. You do not need thousands of rows for your first iteration. Start small, evaluate, then expand only where the model is failing.

```python
# estimate_data_cost.py
# Run this before generating data to estimate your API bill.
# Adjust the numbers to match your actual plan.

def estimate_generation_cost(
    num_rows: int,
    avg_tokens_per_row: int = 1000,
    input_fraction: float = 0.4,         # ~40% of tokens are in the prompt
    output_fraction: float = 0.6,         # ~60% are in the generated response
    input_price_per_million: float = 5.0, # GPT-4o input price, USD
    output_price_per_million: float = 15.0,# GPT-4o output price, USD
) -> dict:
    """
    Estimate the cost of generating a synthetic training dataset via an LLM API.

    Returns a dict with token counts and estimated cost in USD.
    These are rough ballparks — actual costs depend on your prompt length
    and the verbosity of the model's output.
    """
    total_tokens = num_rows * avg_tokens_per_row
    input_tokens = total_tokens * input_fraction
    output_tokens = total_tokens * output_fraction

    input_cost = (input_tokens / 1_000_000) * input_price_per_million
    output_cost = (output_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "num_rows": num_rows,
        "estimated_total_tokens": int(total_tokens),
        "estimated_input_tokens":  int(input_tokens),
        "estimated_output_tokens": int(output_tokens),
        "estimated_cost_usd":      round(total_cost, 2),
    }


# Example: estimate for three dataset sizes
for n in [300, 500, 1000]:
    est = estimate_generation_cost(n)
    print(
        f"{est['num_rows']:>5} rows │ "
        f"~{est['estimated_total_tokens']:>7,} tokens │ "
        f"~${est['estimated_cost_usd']:.2f}"
    )
```

Running this prints something like:

```
  300 rows │ ~  300,000 tokens │ ~$2.10
  500 rows │ ~  500,000 tokens │ ~$3.50
 1000 rows │ ~ 1,000,000 tokens │ ~$7.00
```

Pass `input_price_per_million=0.075` and `output_price_per_million=0.30` for Gemini 1.5 Flash pricing, and the numbers drop dramatically.

---

### Phase 2 — Training

Training time depends on three things: GPU type, model size, and dataset size. Here are measured ballparks for the Unsloth fine-tuning setup from *Ch15 - Your First Fine-Tune with Unsloth (Full Script)*:

**Qwen3-8B or Gemma-3-4B (the main book model)**

| Hardware | 500 rows, 3 epochs | 1,000 rows, 3 epochs | GPU cost (approx.) |
|---|---|---|---|
| Colab T4 (free) | ~45–60 min | ~90–120 min | Free (or ~$0.50 Colab Pro) |
| RunPod / Lambda A10G (24 GB) | ~20–35 min | ~40–70 min | $0.20–$0.50 |
| RunPod A100 (80 GB) | ~10–20 min | ~20–40 min | $0.50–$1.50 |

**Qwen3-1.7B or Gemma-3-3B (smaller, faster)**

| Hardware | 500 rows, 3 epochs | 1,000 rows, 3 epochs | GPU cost (approx.) |
|---|---|---|---|
| Colab T4 (free) | ~15–25 min | ~30–50 min | Free |
| A10G | ~8–15 min | ~15–25 min | $0.10–$0.20 |

**Important:** These times assume QLoRA with 4-bit quantization (as configured in Ch15's training script). Full 16-bit training would be 2–4x slower and require more VRAM. Always use Unsloth's 4-bit loading for training — it is how the book's scripts are written and tested.

**Multiple training runs.** Expect to train at least 2–4 times before your model is good. The first run establishes a baseline. Subsequent runs tune hyperparameters, expand the dataset, or fix data quality issues identified in evaluation. Budget the training cost accordingly: if one training run costs $0.40 on an A10G, budget $1.50–$2.00 for a realistic iteration cycle.

---

### Phase 3 — Evaluation

Evaluation (from *Ch18 - Did It Actually Work? Evaluating Memory Extraction*) is cheap. It is just inference — no gradient computation, no optimizer state. You run your model on the held-out test set and compare output to reference answers.

| What you're doing | Time | Cost |
|---|---|---|
| Running the model on 50-row test set (in-process, A10G) | 5–10 min | ~$0.05 |
| Running the model on 100-row test set (in-process, A10G) | 10–20 min | ~$0.10 |
| LLM-as-judge scoring 100 outputs via API (GPT-4o) | 10–20 min | ~$0.50–$1.00 |
| LLM-as-judge scoring 100 outputs via API (Gemini Flash) | 10–20 min | ~$0.05 |

Evaluation is fast enough that you can and should run it after every training run. Do not skip it. A training run that looks like it converged might still produce worse outputs if you changed something subtle in the data or hyperparameters.

---

### Phase 4 — Serving

Serving cost depends on how much traffic you handle and which option you choose. The three serving paths from *Ch22 - Serving Your Model and Using It in an App* have very different cost profiles:

**Ollama on a laptop (local, no cloud cost)**
- No GPU rental cost. You are using your own hardware.
- Latency: 3–8 seconds per request on a laptop GPU.
- Good for: development, demos, personal use.
- Not good for: anything needing sub-second latency or concurrent users.

**vLLM on a cloud GPU**
- A10G (24 GB) at ~$0.60/hour handles roughly 5–15 concurrent requests.
- At ~1 second per request and 10 concurrent users, that is roughly 36,000 requests per hour.
- Cost per request: ~$0.0000167 — under two thousandths of a cent. For most internal tools, this is negligible.
- A T4 at $0.35/hour handles 2–5 concurrent users, half the throughput.

**In-process inference for batch jobs**
- You only pay while the script is running.
- 1,000 conversations on an A10G at ~3 conversations/second: roughly 6 minutes, about $0.06.
- 10,000 conversations: roughly 60 minutes, about $0.60.

```python
# estimate_serving_cost.py
# Estimate monthly serving cost for a vLLM deployment.

def estimate_monthly_serving_cost(
    requests_per_day: int,
    gpu_hourly_rate: float = 0.60,     # A10G on RunPod, approx.
    seconds_per_request: float = 1.0,  # vLLM on A10G, short conversations
    concurrent_users: int = 10,        # how many requests vLLM processes in parallel
) -> dict:
    """
    Rough estimate of monthly cloud GPU cost for a vLLM serving setup.

    Assumes the GPU is always rented (24/7). If traffic is bursty, consider
    spot instances that can be shut down when idle — most cloud providers
    support this.

    Args:
        requests_per_day:   Total extraction requests per day.
        gpu_hourly_rate:    Cost of the GPU per hour in USD.
        seconds_per_request: Median latency per request at the chosen concurrency.
        concurrent_users:   Max simultaneous requests vLLM can handle on this GPU.

    Returns a cost estimate dict.
    """
    # Compute total GPU seconds needed per day
    # (requests / concurrent throughput × seconds each)
    requests_per_second_capacity = concurrent_users / seconds_per_request
    seconds_of_gpu_needed_per_day = requests_per_day / requests_per_second_capacity

    # Convert to hours — the GPU is billed by the hour
    hours_per_day = seconds_of_gpu_needed_per_day / 3600

    # If the workload needs less than 1 hour of GPU time per day,
    # you still pay for at least 1 hour minimum per session.
    # Here we assume the GPU is kept warm (always-on).
    hours_per_month_always_on = 24 * 30
    cost_always_on = hours_per_month_always_on * gpu_hourly_rate

    # If you shut down between traffic: only pay for active hours
    hours_per_month_on_demand = hours_per_day * 30
    cost_on_demand = hours_per_month_on_demand * gpu_hourly_rate

    return {
        "requests_per_day":             requests_per_day,
        "gpu_hours_needed_per_day":     round(hours_per_day, 2),
        "cost_always_on_usd_per_month": round(cost_always_on, 2),
        "cost_on_demand_usd_per_month": round(cost_on_demand, 2),
    }


# Three typical usage tiers
tiers = [
    ("Light internal tool", 500),
    ("Small team / beta product", 5_000),
    ("Production app", 50_000),
]

print(f"{'Tier':<30} {'Reqs/day':>10} {'Always-on $/mo':>16} {'On-demand $/mo':>16}")
print("-" * 76)

for label, rpd in tiers:
    est = estimate_monthly_serving_cost(rpd)
    print(
        f"{label:<30} "
        f"{est['requests_per_day']:>10,} "
        f"${est['cost_always_on_usd_per_month']:>14.2f} "
        f"${est['cost_on_demand_usd_per_month']:>14.2f}"
    )
```

Expected output:

```
Tier                           Reqs/day  Always-on $/mo  On-demand $/mo
----------------------------------------------------------------------------
Light internal tool                 500          $432.00            $0.45
Small team / beta product         5,000          $432.00            $4.50
Production app                   50,000          $432.00           $45.00
```

The "always-on" cost dominates at low volumes. At 500 requests/day, it's not worth keeping a GPU warm all month — spin one up when you need it and shut it down when you don't. At 50,000 requests/day, always-on at $432/month becomes reasonable compared to the alternative (calling a frontier model API, which at GPT-4o's pricing would cost roughly $150–500/month for the same volume, plus higher latency).

---

## Total project budget: first iteration end-to-end

Here is what you should budget for a complete first iteration of the memory-extraction fine-tune — from zero to a deployed model you've tested on real conversations:

| Phase | Free (Colab T4) | Paid (A10G cloud) |
|---|---|---|
| Data gen: 500 rows via Gemini Flash | ~$0.25 | ~$0.25 |
| Training: 500 rows, 3 epochs | Free / ~$1 Colab Pro | ~$0.40 |
| Evaluation: 100-row test set | Free | ~$0.10 |
| LLM-as-judge: 100 outputs (Gemini Flash) | ~$0.05 | ~$0.05 |
| Export + serving setup | Free | ~$0.20 |
| **Total** | **~$0.30–$1.30** | **~$1.00** |

The bottom line: your first working model costs roughly **$1–$5** depending on choices. Iteration 2 (better data, adjusted hyperparameters) adds another $1–$3. This is not an expensive project.

The main cost risk is **not planning iteration.** If you assume one training run will produce a perfect model, you will underestimate by 3–5x. Budget for 3–5 training runs per task. Plan for data quality improvements between runs.

---

## Pre-flight checklist before going live

This checklist is for the moment before you attach your fine-tuned model to real user data or a production system. Work through it top to bottom. Do not skip items because they seem obvious — the obvious ones are often the ones that bite you.

### Data and schema

```python
# preflight_checks.py
# Run this script before deploying. It checks the most common failure modes.
# It assumes you have the model serving at http://localhost:8000/v1
# (vLLM) or can load it in-process.

import json
import re
from openai import OpenAI

# ── The shared system prompt from memory_prompt.py ────────────────────────────
# Import this — never copy-paste it or retype it here.
from memory_prompt import SYSTEM_PROMPT

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
MODEL  = "memory-extractor"


def run_one(conversation: str, label: str) -> list | None:
    """
    Send one test conversation to the model and return parsed memories.
    Prints PASS/FAIL with the label so you can scan results quickly.
    Returns None on parse failure.
    """
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": conversation},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        raw = resp.choices[0].message.content.strip()
        memories = json.loads(raw)
        assert isinstance(memories, list), "Output is not a JSON array"
        print(f"  PASS  [{label}]  →  {len(memories)} memories")
        return memories
    except Exception as e:
        print(f"  FAIL  [{label}]  →  {e}")
        return None


print("=== Pre-flight checks ===\n")

# ── CHECK 1: Basic extraction ──────────────────────────────────────────────────
# The model should extract at least one memory from a clear factual conversation.
print("1. Basic extraction:")
result = run_one(
    "Alex: I moved to Berlin two years ago. I work as a data engineer at a logistics startup.",
    "basic facts"
)
assert result and len(result) >= 1, "Expected at least 1 memory — extraction is broken"


# ── CHECK 2: Empty-conversation handling ───────────────────────────────────────
# Pure small-talk with no facts should return [].
# If it returns invented memories, the model is hallucinating.
print("\n2. Empty/small-talk input:")
result = run_one(
    "User: haha yeah\nAssistant: totally\nUser: k bye",
    "empty input → should return []"
)
assert result is not None, "Parse error on empty input"
assert len(result) == 0, f"Model hallucinated {len(result)} memories from small-talk"


# ── CHECK 3: Schema compliance ─────────────────────────────────────────────────
# Every returned memory must have the three required fields with the right types.
print("\n3. Schema compliance:")
result = run_one(
    "Jordan: I'm allergic to shellfish. Been that way my whole life.",
    "schema check"
)
if result:
    VALID_TYPES = {"preference", "fact", "decision", "relationship"}
    for i, mem in enumerate(result):
        assert "text"     in mem,                 f"Memory {i} missing 'text'"
        assert "type"     in mem,                 f"Memory {i} missing 'type'"
        assert "entities" in mem,                 f"Memory {i} missing 'entities'"
        assert isinstance(mem["text"], str),      f"Memory {i} 'text' is not a string"
        assert mem["type"] in VALID_TYPES,        f"Memory {i} has invalid type '{mem['type']}'"
        assert isinstance(mem["entities"], list), f"Memory {i} 'entities' is not a list"
    print(f"  PASS  schema validated on {len(result)} memories")


# ── CHECK 4: No markdown fences in output ──────────────────────────────────────
# The model should return raw JSON, not ```json ... ``` fences.
# Fences break json.loads() — if they appear, the post-processing layer will fail.
print("\n4. No markdown fences:")
resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "Sam: I prefer dark mode in all my apps."},
    ],
    temperature=0.1,
    max_tokens=256,
)
raw_output = resp.choices[0].message.content.strip()
has_fences = raw_output.startswith("```") or "```json" in raw_output
if has_fences:
    print(f"  FAIL  model is wrapping output in markdown fences")
    print(f"        Raw: {raw_output[:100]}")
    print("        Fix: add more training examples WITHOUT fences, or strip fences in post-processing")
else:
    print("  PASS  no markdown fences in output")


# ── CHECK 5: Atomicity — no bundled facts ─────────────────────────────────────
# Each memory should contain one fact. If a memory text contains " and " connecting
# two separate facts, the model may be bundling, which reduces downstream usefulness.
print("\n5. Atomicity check (heuristic):")
result = run_one(
    "Casey: I have a dog named Pepper. I also have a cat named Biscuit. I live in Austin.",
    "three distinct facts"
)
if result:
    bundled = [m for m in result if " and " in m["text"] and m["text"].count(".") == 0]
    if bundled:
        print(f"  WARN  {len(bundled)} memories may be bundled (contain ' and '):")
        for m in bundled:
            print(f"        {m['text']}")
    else:
        print(f"  PASS  {len(result)} memories, no obvious bundling")


# ── CHECK 6: System prompt is unchanged ───────────────────────────────────────
# This one you check manually — not via the model.
# Print the system prompt so you can visually confirm it matches what was used in training.
print("\n6. System prompt verification:")
print("   The system prompt your code is sending:")
print("   " + "-" * 60)
for line in SYSTEM_PROMPT.strip().split("\n")[:5]:
    print(f"   {line}")
print("   ...")
print("   Confirm this matches the prompt from your training data (Ch12).")


print("\n=== Pre-flight complete ===")
```

Run this before you go live. Fix any `FAIL` or unexpected `WARN` lines before proceeding.

---

### Manual verification checklist

Run through these manually — they cannot be automated easily but catch real problems:

**Model behavior**
- [ ] The model returns valid JSON on every test input (no parse errors in 10 manual tests)
- [ ] Small-talk with no facts returns `[]`, not hallucinated memories
- [ ] Each memory is one fact, not two facts joined with "and"
- [ ] The `type` field uses only values from your schema: `preference`, `fact`, `decision`, `relationship`
- [ ] The `text` field is a complete standalone sentence — not a fragment that requires context to understand
- [ ] Entities are proper nouns (names, places, products), not generic words like "user" or "conversation"

**Data and training**
- [ ] Evaluation F1 score (from *Ch18 - Did It Actually Work? Evaluating Memory Extraction*) is above your minimum threshold (0.65 is a reasonable floor for a first model; iterate toward 0.80+)
- [ ] You evaluated on the held-out test set, not the training set
- [ ] You checked at least one example from each memory type: preference, fact, decision, relationship
- [ ] You tested on at least one conversation longer than 10 turns

**Infrastructure**
- [ ] The system prompt in your serving code is imported from a shared constant, not retyped
- [ ] `temperature` is set to `0.1` or lower in all serving calls
- [ ] Your serving code handles `[]` returns gracefully (no crash on empty list)
- [ ] Your serving code handles `json.JSONDecodeError` and logs the raw output when it occurs
- [ ] The model format matches the serving tool: merged 16-bit for vLLM, GGUF for Ollama (see *Ch21 - Saving, Merging, and Exporting Your Model*)
- [ ] You tested the serving endpoint end-to-end with a real `curl` or Python call

**For production use**
- [ ] You have a fallback strategy if the model produces invalid JSON (retry with temperature 0, or fall back to a rules-based extraction)
- [ ] You have logging in place to capture raw model outputs for a monitoring sample
- [ ] You know which version of the model is deployed (checkpoint name, commit hash, or Hub model ID)
- [ ] You have tested with at least 20 real conversations from your actual use case, not just the toy examples from this book

---

## Where to go next (without leaving the ecosystem)

These are the official documentation sources for every library used in this book. Bookmark them — not to read front-to-back, but to consult when something breaks or you need a capability we did not cover.

**Unsloth** — `https://docs.unsloth.ai`
The source of truth for `FastLanguageModel`, training arguments, GGUF export, and the latest supported model list. When a new Qwen or Gemma variant comes out, check here first to see if Unsloth supports it yet.

**Hugging Face Transformers** — `https://huggingface.co/docs/transformers`
Reference for `AutoModelForCausalLM`, `AutoTokenizer`, `apply_chat_template`, and model loading. The section on chat templates is particularly useful as you adapt the system prompt format to new base models.

**TRL (Transformer Reinforcement Learning)** — `https://huggingface.co/docs/trl`
Documents `SFTTrainer` — the trainer we used throughout this book — and every training argument. When you want to try DPO or RLHF as your next step, TRL is where those tools live.

**PEFT** — `https://huggingface.co/docs/peft`
Documents LoRA configuration (`LoraConfig`), adapter loading and saving, and multi-adapter workflows. Useful when you want to manage multiple fine-tunes on the same base model.

**Datasets** — `https://huggingface.co/docs/datasets`
Reference for loading, filtering, splitting, and formatting datasets. Covers the JSONL format we used in *Ch12 - Data Format: Turning the Task into Training Rows* and *Ch14 - Cleaning, Splitting, and Sanity-Checking Data*.

**vLLM** — `https://docs.vllm.ai`
Documents the serving arguments, quantization options, and OpenAI-compatible API format. Check here when you need to tune concurrency, add GPU tensor parallelism across multiple cards, or configure structured JSON output enforcement.

**Weights & Biases** — `https://docs.wandb.ai`
If you want richer loss curve tracking than the console logging shown in *Ch17 - Watching Training: Loss Curves and When to Stop*, W&B integrates with TRL in two lines and gives you a browser dashboard.

---

## Common mistakes

**Mistake: treating the cost estimate as a ceiling, not a floor.**

The estimates in this appendix assume things go reasonably well. In practice, the first training run often reveals a data quality problem (see *Ch19 - When It Goes Wrong: A Debugging Playbook*), which means you fix the data and train again. Plan for 3–5 training runs, not 1.

**Mistake: running the pre-flight checklist only once.**

Every time you retrain or update your model, run the pre-flight script again. A dataset change that improves F1 on one category can quietly degrade another. The checklist takes 2 minutes. A production incident takes much longer.

**Mistake: leaving the vLLM server running all night on a rented GPU.**

Cloud GPUs are billed to the second. A T4 running overnight costs roughly $3.50. Not much, but it adds up across iteration cycles. Shut down the instance when you are done. Download your model files to your local machine or push them to the Hub before terminating the instance — cloud instance storage is ephemeral.

**Mistake: forgetting that Colab sessions expire.**

Colab free sessions time out after roughly 12 hours, and anything written to `/content/` is gone when the session ends. Before starting a training run on Colab, set your output directories to point to your mounted Google Drive. A training run that completes but whose output is lost is wasted compute.

```python
# colab_save_to_drive.py
# Mount Drive and redirect all output there before training.
# Run this at the TOP of your Colab notebook, before any training code.

from google.colab import drive  # only available in Colab
drive.mount("/content/drive")

# Set all output paths to Drive so they survive session expiry.
OUTPUT_DIR = "/content/drive/MyDrive/memory-extractor/outputs"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Then in your training script, use OUTPUT_DIR as the base for all saves:
# trainer.save_model(f"{OUTPUT_DIR}/adapter")
# model.save_pretrained_merged(f"{OUTPUT_DIR}/merged-16bit", ...)
```

**Mistake: skipping the empty-input test.**

If your model was trained only on examples where there is always something to extract, it may not have learned to return `[]` for small-talk or uninformative text. Test this explicitly. If it fails, add 20–30 examples of empty-output conversations to your training set and retrain (see *Ch20 - Iterating: From a Mediocre Model to a Good One*).

**Mistake: using the training set for the final evaluation.**

Your training loss goes down because the model memorized the training examples. Evaluation on the training set will show inflated scores. Always hold out at least 10% of your data as a test set that the model never sees during training (covered in *Ch14 - Cleaning, Splitting, and Sanity-Checking Data*). If your test set F1 diverges badly from your training loss, you have an overfitting problem.

---

## Recap

- Synthetic data generation costs roughly $0.25–$8 for 300–1,000 rows, depending on which API you use. Gemini Flash is 10x cheaper than GPT-4o for this purpose.
- A 7B model trains in 20–70 minutes on an A10G at $0.30–$0.70 per run. A 1.7B model trains in 8–25 minutes at $0.10–$0.25 per run.
- Evaluation and LLM-as-judge scoring add $0.05–$1.00 per round, depending on the judge model.
- Total cost for a complete first iteration: roughly $1–$5. Budget for 3–5 training runs across your whole project.
- At production scale, a vLLM deployment on an A10G costs roughly $45/month for 50,000 requests/day — cheaper than comparable frontier model API calls.
- Run the pre-flight checklist after every model update, not just before the first deployment.
- The official docs for Unsloth, Transformers, TRL, PEFT, Datasets, and vLLM are your primary reference from here on.

## Next

This is the end of the book — you now have everything you need to build, train, evaluate, and deploy a domain-specific fine-tune. For the path forward, see *Ch23 - Toward Engram: Continual Learning and Scaling Up*, which sketches how to close the loop: collecting production signal, running continual fine-tuning, and building toward models that genuinely internalize your users' world over time.
