# Chapter 0 - The Speedrun: A Working Fine-Tune This Afternoon for Under $30

Here is the deal. In the next few hours you are going to teach a small open model a single, specific skill — reading a conversation and extracting structured memories from it — and you are going to do it cheaply enough that the whole thing fits on a coffee budget. You will rent (or borrow for free) a GPU, generate your own training data with a bigger "teacher" model, run a real fine-tune, eyeball the results, and serve the model behind a tiny API. By dinner you will have a model that does something a base model cannot reliably do, and it will be *yours* — small enough to run on hardware you can afford, fast enough to call in a loop, and worth the twenty-odd dollars because it replaces an expensive general-purpose model on a narrow task you actually need.

This is the fast path. It is deliberately light on theory. Every step here works, but this chapter does not stop to explain *why* each step works, or how to make any of it production-grade — that is the entire rest of the book. When we make a choice without justifying it ("use rank 16", "2 epochs", "Qwen3-4B"), trust it for now; the justification is coming. Treat this as the trailer, not the movie.

---

## What you'll learn

- How to get a GPU for this afternoon — free on Colab, or a few dimes an hour on RunPod/Vast.ai
- How to generate ~500–1,000 synthetic training examples with a teacher LLM, in the exact format the trainer wants
- How to run a QLoRA fine-tune of a 4-bit small model with Unsloth + TRL, with hyperparameters that just work
- How to sanity-check the result: is the output valid JSON, and does it actually capture the facts?
- How to serve the model the simplest way possible (Ollama, or a ~10-line vLLM + FastAPI call)
- Roughly what it all costs, and where to go next to do each step *properly*

## Concepts you need first

You need almost nothing to follow this chapter. Here is the whole mental model in four sentences.

**Fine-tuning is teaching by example.** You do not write rules; you show the model a few hundred examples of a task done correctly, and it learns the pattern. Think of onboarding a sharp new hire: you do not hand them a rulebook, you show them a stack of correctly-handled tickets and they pick up the shape of the job. That stack of examples *is* your training data, and most of the work in this chapter is making a good stack.

**We are not retraining the whole model — just bolting on a small adapter.** The base model has billions of frozen weights; we attach a tiny set of new, trainable weights (a "LoRA adapter") and only those move during training. This is what makes it cheap: you are training ~1% of the parameters, so it fits on a small GPU and finishes in under an hour. (*Ch6 - LoRA and QLoRA Without the Math Headache* explains the trick; here we just use it.)

**The task is the book's running example: memory extraction.** Given a chat transcript, the model emits a JSON array of atomic facts. It is one example of *domain fine-tuning* — teaching a small model a narrow, structured skill specific to your world (extracting decisions from meeting notes, fields from contracts, or symptoms from clinical notes — same technique, different schema).

**The teacher–student pattern gets you data for free-ish.** You do not have hundreds of hand-labeled examples lying around. So you pay a big, smart model (the "teacher") a few dollars to *generate* them, then train your small model (the "student") to imitate that output at a fraction of the cost and latency. The student ends up specialized and cheap; the teacher was just the bootstrap.

That is everything. Let's build.

---

## The one thing that must never drift: the schema and the system prompt

Before any code, pin these two constants in your head, because every single step — data generation, training, evaluation, serving — uses them *verbatim*. The number-one cause of a "it trained fine but does nothing in my app" bug is a system prompt that changed by one word between training and inference.

A single memory object looks exactly like this:

```python
{
    "text": "Sarah prefers dark roast coffee in the morning",   # the fact, as a complete sentence
    "type": "preference",                                        # one of: preference | fact | decision | relationship
    "entities": ["Sarah"]                                        # named people, places, or things involved
}
```

The model's output is a JSON array of zero or more such objects — and an empty array `[]` is a perfectly valid answer when there is nothing to remember.

And here is the system prompt. Copy it once into a file you import everywhere; do not retype it.

```python
# memory_prompt.py  — import SYSTEM_PROMPT from here in EVERY script below.
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

That's the contract. Now we honor it the whole way down.

---

## Step 1 — Get a GPU (cheap or free)

Fine-tuning needs a CUDA GPU. You have a CPU; that is roughly ten-thousand times too slow for this. You do not need to *own* a GPU — you rent one for an hour or two, or use a free one.

**Option A — Free: Google Colab (T4, 15 GB VRAM).** This is the zero-dollar path and it is enough to finish this chapter.

1. Go to [https://colab.research.google.com](https://colab.research.google.com) and start a new notebook.
2. *Runtime → Change runtime type → T4 GPU → Save.*
3. Run this cell to confirm the GPU is live:

```python
import torch
print("GPU available:", torch.cuda.is_available())
print("GPU name:     ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
# Expected on Colab free tier:
#   GPU available: True
#   GPU name:      Tesla T4
```

The catch with free Colab: sessions time out after a few hours and the T4 is slow-ish, so training takes ~45–90 minutes instead of ~15–30. For a first run, that is fine.

**Option B — Cheap: rent by the hour on RunPod or Vast.ai (L4 or A100, ~$0.30–$0.80/hr).** When you want it faster, or you keep getting kicked off Colab, rent a real GPU:

- **[RunPod](https://www.runpod.io)** — pick a "GPU Pod" with a PyTorch template (CUDA is preinstalled), e.g. an **L4 (24 GB, ~$0.40/hr)** or an **A100 (40/80 GB, ~$0.70–$1.80/hr)**. Click Deploy, open the web terminal or a Jupyter session, and you have a box.
- **[Vast.ai](https://vast.ai)** — a marketplace of rented GPUs, often cheaper; filter for a PyTorch image and an L4/A100, rent, SSH in.

An L4 at ~$0.40/hr finishes this chapter's training in ~15–25 minutes, so the GPU bill for the whole afternoon is well under a dollar. An A100 is faster still if you are impatient.

Whichever you pick, run the two-line `torch.cuda.is_available()` check above first. If it prints `False`, stop — nothing downstream will work until you have a GPU.

---

## Step 2 — Install the libraries

All the code in this book is pinned to one tested snapshot of versions. Mismatched versions are the second-most-common source of mysterious errors (drifting APIs in the fine-tuning ecosystem are brutal), so we pin hard. The book ships a `code/requirements.txt`; the lines that matter for this chapter are:

```
# code/requirements.txt  (excerpt — the full file is in the repo)
unsloth==2026.6.9      # fast QLoRA training + the FastLanguageModel loader
trl==1.6.0             # SFTTrainer: the supervised fine-tuning loop
transformers           # compatible with trl 1.6.0
peft                   # LoRA adapter machinery (Unsloth uses it under the hood)
datasets               # loads our JSONL into the trainer
accelerate
bitsandbytes           # 4-bit quantization (the "Q" in QLoRA)
anthropic>=0.40        # teacher model for synthetic data (Step 3)
# openai>=1.60         # uncomment if you'd rather use OpenAI as the teacher
```

Install it. On a fresh Colab/RunPod/Vast box:

```bash
pip install -r code/requirements.txt
# or, if you just want the speedrun deps without the file:
pip install "unsloth==2026.6.9" "trl==1.6.0" transformers peft datasets accelerate bitsandbytes "anthropic>=0.40"
```

> **Unsloth's install can vary by CUDA version.** The command above works on most cloud GPU images and on Colab. If you hit a CUDA-mismatch error on a local machine, follow the per-environment instructions at [https://docs.unsloth.ai/get-started/installing-unsloth](https://docs.unsloth.ai/get-started/installing-unsloth). *Ch15 - Your First Fine-Tune with Unsloth* covers the install gotchas in full.

---

## Step 3 — Generate ~500–1,000 training examples with a teacher LLM

We need a stack of correct examples: conversations paired with the memory JSON a great model would extract from them. We will have a teacher model produce both halves — the conversation *and* its extraction — so we never hand-label anything.

**How much data?** For a narrow structured task on a ~4B model, **200–500 rows** is enough to prove the format is learned and **1,000–3,000** is where most real projects live. We will target **~600** for the speedrun — enough to work, cheap enough to generate in a few minutes. (*Ch13 - Generating Synthetic Training Data* does this properly: deduplication, difficulty tiers, quality filtering, validation against the schema. This version is the stripped-down cousin.)

**Cost:** generating ~600 rows with a strong teacher runs about **$1–$5** depending on model and verbosity. That is the bulk of your spend in this chapter.

The script below uses Anthropic's Claude as the teacher. Set `ANTHROPIC_API_KEY` in your environment first (`export ANTHROPIC_API_KEY=sk-ant-...`). If you prefer OpenAI, the structure is identical — swap the client (noted in a comment).

```python
# generate_data.py
# Ask a teacher LLM to invent realistic conversations AND extract memories from them,
# then write training rows in TRL's {"messages": [...]} conversational format.
#
# Cost: ~$1-5 for ~600 rows. Time: a few minutes.
# Output: data/train.jsonl  and  data/val.jsonl

import json
import os
import random

import anthropic                       # pip install "anthropic>=0.40"
from memory_prompt import SYSTEM_PROMPT  # the pinned constant from earlier

# ── Config ───────────────────────────────────────────────────────────────────
N_ROWS        = 600                    # ~600 conversations -> ~600 training rows
VAL_FRACTION  = 0.1                    # hold 10% out for evaluation
TEACHER_MODEL = "claude-opus-4-8"      # the big, smart "teacher"
OUT_DIR       = "data"

# A few seed topics so the generated conversations aren't all about the same thing.
# Diversity here matters more than raw count — varied data teaches a more robust skill.
TOPICS = [
    "planning a trip", "discussing a new job", "a doctor's appointment",
    "choosing software at work", "a conversation about hobbies",
    "moving to a new city", "dietary preferences and cooking",
    "a project kickoff meeting", "catching up with an old friend",
    "shopping for a big purchase", "a disagreement about scheduling",
    "talking about family", "a customer support chat", "weekend plans",
]

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment

# ── The instruction we give the TEACHER (not the same as SYSTEM_PROMPT) ───────
# We ask it to fabricate a short conversation, then extract memories from it using
# the EXACT schema our student will be trained on. We ask for both in one shot so
# the conversation and its labels are guaranteed to be consistent.
GENERATION_INSTRUCTION = """Invent a short, realistic multi-turn conversation (4-8 turns) between a User and an Assistant on the topic: "{topic}".

Then extract the memorable facts from it, following this schema exactly:
{{
  "text": "<the fact, written as a complete, standalone sentence>",
  "type": "<one of: preference | fact | decision | relationship>",
  "entities": ["<list of named people, places, or things involved>"]
}}

Roughly 1 in 6 conversations should contain NOTHING worth remembering (small talk only) — for those, the memories list must be exactly [].

Return ONLY valid JSON in this shape, no markdown:
{{"conversation": "<the conversation as plain text, with 'User:' and 'Assistant:' labels>",
  "memories": [ ...array of memory objects, possibly empty... ]}}"""


def generate_one(topic: str) -> dict | None:
    """Ask the teacher for one (conversation, memories) pair. Returns None on a bad row."""
    resp = client.messages.create(
        model=TEACHER_MODEL,
        max_tokens=2000,
        messages=[
            {"role": "user", "content": GENERATION_INSTRUCTION.format(topic=topic)},
        ],
    )
    # response.content is a list of blocks; grab the first text block.
    raw = next((b.text for b in resp.content if b.type == "text"), "").strip()
    try:
        obj = json.loads(raw)
        conversation = obj["conversation"]
        memories = obj["memories"]
        assert isinstance(memories, list)
    except (json.JSONDecodeError, KeyError, AssertionError):
        # The teacher occasionally returns malformed JSON. We just skip and move on
        # rather than crash a 600-call run. Ch13 shows how to validate properly.
        return None

    # Assemble ONE training row in TRL's conversational format. The assistant turn
    # is the memories serialized as a compact JSON array — exactly what we want the
    # student to learn to produce. Note the SYSTEM_PROMPT is identical to inference.
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": conversation},
            {"role": "assistant", "content": json.dumps(memories, ensure_ascii=False)},
        ]
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for i in range(N_ROWS):
        topic = random.choice(TOPICS)
        row = generate_one(topic)
        if row is not None:
            rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  generated {len(rows)} good rows out of {i + 1} attempts")

    random.shuffle(rows)
    n_val = int(len(rows) * VAL_FRACTION)
    val, train = rows[:n_val], rows[n_val:]

    for name, split in [("train", train), ("val", val)]:
        path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(path, "w") as f:
            for row in split:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(split):4d} rows -> {path}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
python generate_data.py
```

Sample console output:

```
  generated 24 good rows out of 25 attempts
  generated 49 good rows out of 50 attempts
  ...
wrote  540 rows -> data/train.jsonl
wrote   60 rows -> data/val.jsonl
```

And one line of `data/train.jsonl`, pretty-printed so you can see the shape (it is one JSON object per line in the file):

```json
{"messages": [
  {"role": "system",    "content": "You are a memory extraction assistant. ..."},
  {"role": "user",      "content": "User: I just accepted the offer at Northwind. Starting in March.\nAssistant: Congratulations! ..."},
  {"role": "assistant", "content": "[{\"text\": \"The user accepted a job offer at Northwind.\", \"type\": \"decision\", \"entities\": [\"Northwind\"]}, {\"text\": \"The user starts the new job in March.\", \"type\": \"fact\", \"entities\": []}]"}
]}
```

That `{"messages": [...]}` shape is exactly what TRL's trainer eats in the next step. No conversion needed.

> Using OpenAI instead? Replace the client with `from openai import OpenAI; client = OpenAI()` and the call with `client.chat.completions.create(model="gpt-...", messages=[{"role":"user","content": GENERATION_INSTRUCTION.format(topic=topic)}])`, then read `resp.choices[0].message.content`. Everything else is identical.

---

## Step 4 — Fine-tune with Unsloth + TRL (QLoRA)

Now the main event. We load a small model in 4-bit, attach a LoRA adapter, and train it on the rows from Step 3. One tight script. The hyperparameters are sane defaults for this task — *Ch16 - Hyperparameters: Which Knobs to Turn and When* explains every number, and *Ch15* is the full annotated walkthrough.

```python
# train.py
# QLoRA fine-tune of a 4-bit small model on the memory-extraction task.
#
# Expected VRAM: ~5-6 GB (fits a free Colab T4).
# Expected time: ~15-25 min on an L4/A100, ~45-90 min on a T4.
# Output: data/adapter/  — a LoRA adapter (~100-300 MB).

import torch
from unsloth import FastLanguageModel          # import unsloth FIRST — it patches transformers
from trl import SFTTrainer, SFTConfig, train_on_responses_only
from datasets import load_dataset

from memory_prompt import SYSTEM_PROMPT          # pinned constant (kept identical to data + inference)

# ── Config — everything tunable in one place ─────────────────────────────────
MODEL_NAME     = "unsloth/Qwen3-4B-bnb-4bit"   # small, strong, 4-bit. ~4 GB download.
# Alternative: "unsloth/gemma-3-4b-it-bnb-4bit"  — Gemma 3 4B, also fine here.
MAX_SEQ_LENGTH = 2048                            # plenty for our short conversations
LORA_RANK      = 16                              # adapter "width"; 16 is a solid default
LORA_ALPHA     = 32                              # scaling; rule of thumb alpha = 2 x rank
OUTPUT_DIR     = "data/adapter"

# ── 1. Load the base model in 4-bit ──────────────────────────────────────────
print("Loading base model in 4-bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit   = True,    # this is the "Q" in QLoRA — base weights kept in 4-bit
    dtype          = None,    # None = auto (bfloat16 on A100/L4, float16 on T4)
)

# ── 2. Attach the LoRA adapter (the only weights that will train) ─────────────
model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_RANK,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = 0.0,            # 0 is fine for a small dataset
    target_modules = "all-linear",  # attach adapters to every linear layer (the safe default)
    bias           = "none",
    use_gradient_checkpointing = "unsloth",   # saves VRAM
    random_state   = 42,
)
trainable, total = model.get_nb_trainable_parameters()
print(f"Training {trainable:,} of {total:,} params ({100*trainable/total:.2f}%)")
# Typical: ~0.5-1% of params are trainable. The other 99% never move.

# ── 3. Load the JSONL we generated in Step 3 ─────────────────────────────────
dataset = load_dataset(
    "json",
    data_files={"train": "data/train.jsonl", "validation": "data/val.jsonl"},
)
print(f"train rows: {len(dataset['train'])}, val rows: {len(dataset['validation'])}")

# ── 4. Configure training ────────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = 2,        # 2-3 epochs for a small dataset; 2 is a safe start
    per_device_train_batch_size = 2,        # small batch (T4-friendly)
    gradient_accumulation_steps = 4,        # effective batch = 2 x 4 = 8
    learning_rate               = 2e-4,     # standard LoRA learning rate
    warmup_ratio                = 0.05,
    lr_scheduler_type           = "cosine",
    logging_steps               = 10,
    optim                       = "adamw_8bit",   # memory-efficient optimizer
    bf16                        = torch.cuda.is_bf16_supported(),   # picks bf16/fp16 for your GPU
    fp16                        = not torch.cuda.is_bf16_supported(),
    seed                        = 42,
    max_seq_length              = MAX_SEQ_LENGTH,
    dataset_text_field          = "messages",   # our rows store the chat under "messages"
    packing                     = False,
)

# Note: the Unsloth SFTTrainer path takes `tokenizer=` (the newer TRL RL trainers
# use `processing_class=` instead — but for this SFT path, `tokenizer=` is correct).
trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset  = dataset["validation"],
    args          = training_args,
)

# ── 5. Train only on the assistant's answer, not the prompt ──────────────────
# This masks the system+user tokens out of the loss, so the model is only graded
# on producing the right JSON — not on re-predicting the prompt it already saw.
# Qwen3 marks turns with <|im_start|>user / <|im_start|>assistant. (Gemma 3 uses
# <start_of_turn>user / <start_of_turn>model — swap these two strings if you chose Gemma.)
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n",
)

# ── 6. Go ────────────────────────────────────────────────────────────────────
print("Training...")
stats = trainer.train()
print(f"Done in {stats.metrics['train_runtime']/60:.1f} min, "
      f"final loss {stats.training_loss:.3f}")

# ── 7. Save the adapter (just the ~100-300 MB of trained weights) ────────────
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}/")
```

Run it:

```bash
python train.py
```

You will see a progress bar and a loss number ticking down. Sample output:

```
Training 9,830,400 of 4,031,000,000 params (0.24%)
train rows: 540, val rows: 60
{'loss': 1.284, 'grad_norm': 0.71, 'learning_rate': 8e-05, 'epoch': 0.15}
{'loss': 0.402, 'grad_norm': 0.55, 'learning_rate': 1.9e-04, 'epoch': 0.59}
{'loss': 0.231, 'grad_norm': 0.40, 'learning_rate': 9e-05, 'epoch': 1.48}
...
Done in 18.3 min, final loss 0.178
Adapter saved to data/adapter/
```

Loss should fall steadily into the **0.15–0.35** range for this task. If it never drops below ~1.5, something is off — usually too little data or a system prompt that doesn't match between your data and the `SYSTEM_PROMPT` constant. *Ch19 - When It Goes Wrong* is the debugging playbook.

---

## Step 5 — Quick eval: is it actually any good?

You have an adapter. Does it work? At speedrun pace we do two cheap things: **eyeball one held-out conversation**, and compute **two rough metrics** — what fraction of outputs are valid JSON, and roughly how many of the expected facts get captured. (*Ch18 - Did It Actually Work? Evaluating Memory Extraction* builds a real eval harness; this is the back-of-the-envelope version.)

```python
# evaluate.py
# Load the adapter, run it on the held-out val set, and report two cheap metrics
# plus one human-readable example.

import json
from unsloth import FastLanguageModel
from memory_prompt import SYSTEM_PROMPT

ADAPTER_PATH = "data/adapter"
VALID_TYPES  = {"preference", "fact", "decision", "relationship"}

# ── Load adapter + base together, switch to fast inference ───────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = ADAPTER_PATH,   # reads adapter_config.json, pulls the right base model
    max_seq_length = 2048,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(model)   # enables Unsloth's fast generation kernels


def extract(conversation: str) -> str:
    """Run the model on one conversation, return its raw string output."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": conversation},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to("cuda")
    out = model.generate(
        inputs, max_new_tokens=512, do_sample=False,    # greedy = deterministic JSON
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


def fact_words(memories: list) -> set:
    """Crude bag-of-words over the 'text' fields — used for a rough recall estimate."""
    words = set()
    for m in memories:
        words |= set(str(m.get("text", "")).lower().split())
    return words


# ── Run over the val set ──────────────────────────────────────────────────────
val = [json.loads(line) for line in open("data/val.jsonl")]

json_ok = 0          # how many outputs parse as a valid JSON list of well-formed objects
recall_scores = []   # per-example fraction of expected fact-words recovered

for row in val:
    conversation = row["messages"][1]["content"]            # the user turn
    gold = json.loads(row["messages"][2]["content"])        # the teacher's answer
    raw = extract(conversation)

    try:
        pred = json.loads(raw)
        assert isinstance(pred, list)
        assert all(set(m) >= {"text", "type", "entities"} and m["type"] in VALID_TYPES
                   for m in pred)
        json_ok += 1
    except (json.JSONDecodeError, AssertionError, TypeError):
        recall_scores.append(0.0)
        continue

    gold_words = fact_words(gold)
    if not gold_words:                       # gold was [] (nothing to remember)
        recall_scores.append(1.0 if pred == [] else 0.0)
    else:
        recovered = gold_words & fact_words(pred)
        recall_scores.append(len(recovered) / len(gold_words))

n = len(val)
print(f"JSON-valid rate : {json_ok}/{n} = {100*json_ok/n:.0f}%")
print(f"Rough fact recall: {100*sum(recall_scores)/n:.0f}%")

# ── Eyeball one example ───────────────────────────────────────────────────────
print("\n── Sample (held-out conversation) ─────────────────────────────")
sample = val[0]["messages"][1]["content"]
print(sample[:400])
print("\n── Model output ──────────────────────────────────────────────")
print(extract(sample))
```

Run it:

```bash
python evaluate.py
```

Sample output:

```
JSON-valid rate : 60/60 = 100%
Rough fact recall: 81%

── Sample (held-out conversation) ─────────────────────────────
User: I finally switched my notes over to Obsidian. Notion was getting too slow for me.
Assistant: Nice — do you use it for work or personal?
User: Both. My teammate Priya actually recommended it.

── Model output ──────────────────────────────────────────────
[
  {"text": "The user switched their notes from Notion to Obsidian.", "type": "decision", "entities": ["Obsidian", "Notion"]},
  {"text": "The user finds Notion too slow.", "type": "preference", "entities": ["Notion"]},
  {"text": "The user's teammate Priya recommended Obsidian.", "type": "relationship", "entities": ["Priya", "Obsidian"]}
]
```

A **JSON-valid rate near 100%** means the model reliably learned the *format* — the single biggest win of fine-tuning a structured task. **Rough fact recall in the 70–90% range** on a 600-row speedrun is a healthy result; the bag-of-words metric is crude (it counts word overlap, not meaning), so read it as a smoke signal, not a grade. If the output is clean JSON capturing the obvious facts, the fine-tune worked. Pushing recall higher is a data-quality and quantity problem — exactly what Parts 3–5 are about.

---

## Step 6 — Serve it

Last step: make the model answer requests. The simplest path is **Ollama** after a GGUF export — a couple of commands and you are calling it with `curl`. If you would rather stay in Python and need real throughput, a **~10-line vLLM + FastAPI** server is below too. (*Ch21 - Saving, Merging, and Exporting Your Model* and *Ch22 - Serving Your Model and Using It in an App* cover both paths and the production concerns in depth.)

### Simplest: export to GGUF and run with Ollama

First, merge the adapter into the base model and export a GGUF file (a compact, quantized single-file format). Unsloth does this in one call — add it to the end of `train.py` or run as a snippet after loading the adapter:

```python
# export_gguf.py — produce a single quantized .gguf file for Ollama/llama.cpp.
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="data/adapter", max_seq_length=2048, load_in_4bit=True,
)
# Merges the LoRA adapter into the base and writes a 4-bit GGUF under ./gguf/
model.save_pretrained_gguf("gguf", tokenizer, quantization_method="q4_k_m")
print("Wrote gguf/  — look for a *.gguf file inside.")
```

Then point Ollama at it. Create a `Modelfile` (bake in the same system prompt, verbatim):

```
# Modelfile  — set FROM to the actual .gguf filename inside ./gguf/ (run: ls gguf/)
FROM ./gguf/unsloth.Q4_K_M.gguf

SYSTEM """You are a memory extraction assistant. Your job is to read a conversation and extract every memorable piece of information as a list of atomic memory objects.

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

PARAMETER num_ctx 4096
```

Build and call it ([install Ollama](https://ollama.com) first):

```bash
ollama create memory-extractor -f Modelfile

curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "memory-extractor",
    "messages": [
      {"role": "user", "content": "User: I switched to Obsidian for notes. My teammate Priya recommended it."}
    ],
    "temperature": 0.1
  }'
```

The response's `choices[0].message.content` is your JSON array of memories. Done — the model is serving.

### Alternative: ~10 lines of vLLM + FastAPI

If you want a Python server with real throughput, vLLM serves the *merged* model (not the adapter). Export the merged weights, then:

```python
# serve.py — run with:  python serve.py    (needs: pip install vllm fastapi uvicorn)
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from memory_prompt import SYSTEM_PROMPT

llm = LLM(model="data/merged")   # the merged 16-bit model dir (save_pretrained_merged in Ch21)
app = FastAPI()

class Req(BaseModel):
    conversation: str

@app.post("/extract")
def extract(req: Req):
    prompt = llm.get_tokenizer().apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user",   "content": req.conversation}],
        tokenize=False, add_generation_prompt=True,
    )
    out = llm.generate(prompt, SamplingParams(temperature=0.1, max_tokens=512))
    return {"memories": out[0].outputs[0].text}

# then:  uvicorn serve:app --port 8000
#   curl -X POST localhost:8000/extract -H 'Content-Type: application/json' \
#        -d '{"conversation": "User: I moved to Lisbon last month."}'
```

To produce the merged directory, add one line where you load the adapter: `model.save_pretrained_merged("data/merged", tokenizer, save_method="merged_16bit")`. *Ch22* explains why vLLM needs merged weights (not the adapter) and how to handle concurrency.

---

## Cost + time budget

Here is the whole afternoon, itemized. Numbers are ranges with the *reason* behind them — your actual spend depends on which GPU and teacher you pick.

| Item | Free path (Colab T4) | Cheap path (rented L4) | Why |
|---|---|---|---|
| GPU | **$0** (free tier) | **~$0.40/hr × ~0.5 hr ≈ $0.20** | Training fits in ~5–6 GB; an L4 finishes in ~15–25 min |
| Teacher API (~600 rows) | **~$1–5** | **~$1–5** | The one unavoidable cash cost; bulk of the budget |
| Serving (local Ollama) | **$0** | **$0** (or pennies if you keep a GPU box up) | Inference is cheap; runs on the same box |
| **Total** | **≈ $1–5** | **≈ $5–30** | Comfortably under the $30 promise either way |
| **Wall-clock** | ~2–4 hours | ~1.5–3 hours | Data gen (mins) + training (15–90 min) + setup/eval |

The free path costs about the price of the teacher API calls and nothing else. Even the paid path — renting a real GPU and being generous with data generation — lands well under $30. That is the whole pitch: a working, specialized model for less than a team lunch.

---

## Where to go next

You now have a working fine-tuned model. That is the entire point of this chapter — proof, in your hands, that the technique works and is cheap. But you got here by trusting a stack of unexplained choices, and a speedrun model is not a production model. Here is the arc the rest of the book walks, and it is a deliberate progression:

- **You have a working model →** that is Chapter 0, done.
- **Parts 1–2 explain *why* it worked** — the mental models (what fine-tuning actually changes, why LoRA is cheap, what a loss curve is telling you) and a real environment/tooling setup. The intuition behind every magic number you just used.
- **Parts 3–6 do it *properly*** — *Part 3* defines the task and builds real datasets (the grown-up version of Step 3, with quality control); *Part 4* is full SFT training and hyperparameter tuning (*Ch15*, *Ch16*); *Part 5* is serious evaluation and iteration (*Ch18*, *Ch19*); *Part 6* is robust deployment (*Ch21*, *Ch22*).
- **Part 7 — Preference & RL** goes beyond imitation. So far your model only *copies* the teacher. Preference methods (reward models, DPO, GRPO) teach it to *prefer* better answers over worse ones — to get good at the task, not just mimic examples of it.
- **Part 8 — Continuous Learning** runs the whole thing as a living system: collect real usage, fold it back in, retrain in rounds, and guard against forgetting — a model that gets better the longer it's in production.

Go run the speedrun. Then come back to Part 1 and learn what you just did.

## Next

*Ch1 - Why Teach a Model Your Own World* — now that you've seen one work, the case for *when* a fine-tune beats a prompt, a bigger model, or RAG — and when it doesn't.
