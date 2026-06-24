# Ch15 - Your First Fine-Tune with Unsloth (Full Script)

This is the chapter you have been building toward. Everything so far — the task spec from *Ch11*, the data format from *Ch12*, the synthetic dataset from *Ch13*, the cleaned splits from *Ch14*, the LoRA theory from *Ch6*, and the training mechanics from *Ch7* — all of it feeds into one complete, top-to-bottom Python script that actually fine-tunes a model.

By the end of this chapter you will have run real training, saved a real adapter, and tested your model by asking it to extract memories from a conversation it has never seen.

---

## What you'll learn

- How to load a quantized base model and tokenizer with Unsloth's `FastLanguageModel`
- How to attach LoRA adapters to the model with `get_peft_model`
- How to load your JSONL dataset and feed it to TRL's `SFTTrainer`
- What `train_on_responses_only` is and why it matters (hint: it stops the model from wasting effort memorizing your prompts)
- How to set sane training hyperparameters for a first run
- How to save the trained adapter and run a quick generation test to confirm the model now speaks memory JSON

---

## Concepts you need first

### What 4-bit loading means for you practically

In *Ch6 - LoRA and QLoRA Without the Math Headache* we explained that QLoRA works by storing the frozen base model weights in 4-bit precision — roughly a quarter of the memory of normal 32-bit floats. Here is what that means practically for this script:

- A 7B-parameter model in 4-bit takes roughly **5–6 GB of VRAM** to load.
- The LoRA adapter weights you actually train are tiny — typically **50–150 MB** — and live in full 16-bit precision alongside the frozen base.
- Unsloth's `FastLanguageModel` handles all the quantization automatically when you pass `load_in_4bit=True`. You do not touch bitsandbytes configuration directly.

On an A100 (40 GB), a 7B model trains comfortably at batch size 4–8. On a free Colab T4 (15 GB), stick to batch size 1–2 with gradient accumulation steps of 8.

### What `train_on_responses_only` does

When you train a language model, the loss (the error signal — see *Ch7 - How Training Actually Works*) is computed over every token in the example. Without any masking, the model spends training effort trying to predict the tokens in the system prompt and user message. That is wasted effort — you already know what those tokens are, you wrote them.

`train_on_responses_only` is a helper in TRL that masks the prompt tokens out of the loss computation. The model is only penalized for getting the *assistant* turn wrong. For our task that means: the model is only taught to produce correct memory JSON. It does not try to predict "You are a memory extraction assistant…" for the ten-thousandth time.

In practice this improves training efficiency noticeably, especially when your system prompt is long relative to the response.

### Epochs vs. steps

You will see both terms in the training arguments. An **epoch** is one full pass through the entire training dataset. A **step** is one batch update. If you have 1,000 training examples and use batch size 4 (with gradient accumulation of 2), one epoch = 1,000 / (4 × 2) = 125 steps.

For fine-tuning on a structured output task like ours, 2–3 epochs is usually enough. More than that and you risk **overfitting** — the model memorizes your training examples rather than learning the underlying skill.

---

## Before you run

### Step 1 — Install the libraries

Unsloth, TRL, and the datasets library are not part of the Python standard library. You need to install them before the first import will work. Run this in your terminal (or a notebook cell):

```bash
pip install unsloth trl datasets transformers accelerate bitsandbytes
```

> **Unsloth installation varies by CUDA version and environment.** The command above works for most cloud GPU setups (RunPod, Vast.ai) and for Google Colab. If you are on a local machine with a custom CUDA version, or if the above fails with a CUDA mismatch error, follow the environment-specific instructions at **https://docs.unsloth.ai/get-started/installing-unsloth** — the page has copy-paste commands for Colab, local pip, and conda.

### Step 2 — Check that you have a GPU

This script requires a CUDA-capable GPU. Paste this two-liner into a Python shell or notebook cell to check:

```python
import torch
print("GPU available:", torch.cuda.is_available())
print("GPU name:     ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
```

If `GPU available: False`, the training script will either crash immediately or run thousands of times slower than intended on CPU. You need a GPU before proceeding.

**Where to get a GPU if you do not have one locally:**

- **Google Colab** (free tier — T4 GPU, 15 GB VRAM): go to https://colab.research.google.com, open a new notebook, then go to *Runtime → Change runtime type → T4 GPU*. Free tier sessions time out after a few hours; Colab Pro gives longer sessions and access to A100s.
- **Vast.ai** or **RunPod** (paid, ~$0.30–$0.80/hr for an A100): rent a GPU by the hour, SSH in, and run the script there. Both services offer pre-built PyTorch images so the CUDA environment is already set up.

A free Colab T4 is enough to complete this chapter. The training will take 60–90 minutes instead of 20–40 minutes on an A100, but the result is identical.

### Step 3 — Verify GPU availability from the terminal (optional)

If you are on a Linux server or inside a Docker container, `nvidia-smi` in the terminal shows your GPU model and current memory usage:

```bash
nvidia-smi
```

If `nvidia-smi` is not found, the NVIDIA driver is not installed — follow the driver install for your OS before continuing.

---

## The full script

The script below is organized into labeled blocks. Each block has a plain-English explanation above it. You can run the whole file as-is, or paste blocks into a notebook cell by cell.

Save this file as `train.py`. Save the inference test in the next section as `inference_test.py` in the same directory. All paths assume you are running from the root of your project directory, with your training data at `data/splits/train.jsonl` and `data/splits/val.jsonl` — the output of *Ch14 - Cleaning, Splitting, and Sanity-Checking Data*.

> **Running the scripts:** Once both files are saved, open a terminal in your project root and run them in order:
> ```bash
> python train.py          # fine-tunes the model; takes 20–90 minutes depending on GPU
> python inference_test.py # loads the adapter and tests it on a new conversation
> ```
> You should see a progress bar with a loss number during training, then a JSON array of memories printed at the end of inference. That is the complete flow.

```python
# train.py
# Fine-tune a memory extraction model with Unsloth + TRL.
#
# Expected VRAM:   ~8 GB for a 7B model in 4-bit (Qwen3-8B or Gemma-3-4B)
# Expected time:   ~20–40 minutes on an A100 for 1,000 training rows × 3 epochs
#                  ~60–90 minutes on a T4 (free Colab GPU)
# Output:          data/adapter/  — a LoRA adapter you can load or merge later

import json
import os
import torch   # needed for torch.cuda.is_bf16_supported() used in SFTConfig below

# ── 1. Imports ──────────────────────────────────────────────────────────────
# unsloth must be imported before transformers. It patches the model classes
# internally and the patch only works if it runs first.
from unsloth import FastLanguageModel

# SFTTrainer is TRL's supervised fine-tuning wrapper. It handles data loading,
# tokenization, and the training loop so you don't have to.
from trl import SFTTrainer, SFTConfig

# train_on_responses_only is the masking helper described in the concepts section.
# It modifies the data collator so the loss is only computed on assistant tokens.
from trl import train_on_responses_only

# Hugging Face datasets — the library that loads our JSONL files into a format
# the trainer can iterate over.
from datasets import load_dataset

# ── 2. Configuration — all tuneable values in one place ─────────────────────
# Putting these at the top means you can adjust a run without hunting through
# the script for the relevant numbers.

MODEL_NAME = "unsloth/Qwen3-8B-bnb-4bit"
# Alternatives:
#   "unsloth/Qwen3-8B-bnb-4bit"          — Qwen3 7B, 4-bit. ~5.5 GB VRAM.
#   "unsloth/gemma-3-4b-it-bnb-4bit"     — Gemma 3 7B instruct, 4-bit. ~5.5 GB VRAM.
#   "unsloth/Qwen3-14B-bnb-4bit"         — 14B. Needs ~10 GB VRAM minimum.
# We discussed the Qwen vs Gemma trade-offs in Ch10. Either 7B works here.

MAX_SEQ_LENGTH = 2048
# The maximum number of tokens in one training example (system + user + assistant combined).
# Most of our examples are well under 1,000 tokens. 2048 gives comfortable headroom.
# Increase to 4096 if you have long conversations, but VRAM usage rises.

LORA_RANK = 16
# The "width" of the LoRA adapter (explained in Ch6).
# 16 is a solid default for a task with clear structure like ours.
# Lower (8) = fewer trainable params, faster, slightly less capacity.
# Higher (32) = more capacity, more VRAM, useful if quality plateaus.

LORA_ALPHA = 32
# Scaling factor for the LoRA updates. A common rule of thumb: set alpha = 2 × rank.
# This controls how strongly the adapter's updates influence the base model.

LORA_DROPOUT = 0.05
# Randomly zero out 5% of LoRA connections during training. Mild regularization —
# helps generalize slightly. 0.0 is also fine for small datasets.

TRAIN_DATA_PATH = "data/splits/train.jsonl"
VAL_DATA_PATH   = "data/splits/val.jsonl"
OUTPUT_DIR      = "data/adapter"   # where the trained adapter will be saved

BATCH_SIZE      = 2     # examples per GPU per step
GRAD_ACCUM      = 4     # accumulate gradients over this many steps before updating
                        # effective batch size = BATCH_SIZE × GRAD_ACCUM = 8
NUM_EPOCHS      = 3     # full passes through the training data
LEARNING_RATE   = 2e-4  # how fast the adapter weights move each step (see Ch7)
WARMUP_RATIO    = 0.05  # fraction of steps spent ramping the learning rate up from zero
SAVE_STEPS      = 100   # save a checkpoint every 100 steps (in case of crashes)
LOGGING_STEPS   = 10    # print training loss to the console every 10 steps
MAX_GRAD_NORM   = 1.0   # clip gradients to prevent any single step from being huge

# ── 3. System prompt — must match exactly what you used to build training data ─
# This is the same constant defined in Ch12 and Ch13.
# Storing it here (rather than importing from a shared module) keeps this script
# self-contained. In production code, import it from a shared constants file.
SYSTEM_PROMPT = """You are a memory extraction assistant. Your job is to read a conversation and extract every memorable piece of information as a list of atomic memory objects.

Each memory object must follow this exact JSON schema:
{
  "text": "<the fact, written as a complete, standalone sentence>",
  "type": "<one of: preference | fact | decision | relationship | event>",
  "entities": ["<list of named people, places, or things involved>"]
}

Rules:
- One fact per memory object. Do not bundle multiple facts into one.
- Write "text" as a sentence someone could read without any surrounding context.
- If there are no memorable facts in the conversation, return an empty list: []
- Return ONLY a valid JSON array. No explanation, no markdown fences, no extra text."""


# ── 4. Load the model and tokenizer ─────────────────────────────────────────
# FastLanguageModel.from_pretrained downloads the model from the Hugging Face Hub
# (or loads from local cache) and applies Unsloth's speed optimisations.
#
# First run: downloads ~4 GB. Subsequent runs: loads from ~/.cache/huggingface/
print("Loading base model…")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = MODEL_NAME,
    max_seq_length= MAX_SEQ_LENGTH,
    load_in_4bit  = True,   # QLoRA — keeps the base frozen in 4-bit (see Ch6)
    dtype         = None,   # None = auto-detect; Unsloth picks bfloat16 on Ampere GPUs
    # token       = "hf_...",  # only needed for gated models (not required for these)
)
print("Model loaded.")


# ── 5. Attach LoRA adapters ──────────────────────────────────────────────────
# get_peft_model adds small trainable weight matrices (the LoRA adapters) to
# the model's attention and MLP layers. The base model weights are frozen —
# only the adapter weights will change during training.
#
# target_modules: which layers to attach adapters to.
# "all-linear" tells Unsloth to attach to every linear projection in the model.
# This is almost always the right choice — it gives the adapter maximum reach
# without the complexity of picking individual layer names.
model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_RANK,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    target_modules = "all-linear",   # attach to all linear layers
    bias           = "none",         # do not train bias terms — rarely helps, wastes params
    use_gradient_checkpointing = "unsloth",  # Unsloth's memory-efficient checkpointing
    random_state   = 42,             # reproducibility
)

# Handy: print how many parameters are actually trainable
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable params: {trainable:,}  /  Total params: {total:,}")
print(f"Trainable %: {100 * trainable / total:.2f}%")
# Typical output for Qwen3-8B + rank 16 all-linear:
#   Trainable params: ~83,000,000  /  Total: ~7,600,000,000
#   Trainable %: ~1.09%
# You are training roughly 1% of the model. The rest never moves.


# ── 6. Load the dataset ──────────────────────────────────────────────────────
# load_dataset from Hugging Face reads JSONL files and returns a DatasetDict.
# Each row is the {"messages": [...]} structure we built in Ch12.
print("Loading dataset…")
dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_DATA_PATH,
        "validation": VAL_DATA_PATH,
    },
    # split=None means we get a DatasetDict with both "train" and "validation" keys.
)
print(f"Train rows: {len(dataset['train'])}")
print(f"Val rows:   {len(dataset['validation'])}")


# ── 7. Apply the chat template to format tokens correctly ────────────────────
# The tokenizer knows the model's specific special tokens — the markers that say
# "here is where the system turn starts", "here is the user turn", etc.
# (This is what Ch5 calls the "chat template".)
#
# We need to tell Unsloth which token marks the start of the assistant turn so
# train_on_responses_only knows where to begin masking.
#
# For Qwen3: the assistant turn begins with the token "<|im_start|>assistant"
# For Gemma 3: it begins with "<start_of_turn>model"
#
# Check which model you are using and set these accordingly.

if "qwen" in MODEL_NAME.lower():
    instruction_part  = "<|im_start|>user\n"       # token that opens the user turn
    response_part     = "<|im_start|>assistant\n"  # token that opens the assistant turn
elif "gemma" in MODEL_NAME.lower():
    instruction_part  = "<start_of_turn>user\n"
    response_part     = "<start_of_turn>model\n"
else:
    # Fallback: inspect the tokenizer to find the correct tokens for your model.
    # Print tokenizer.chat_template to see the raw template string.
    raise ValueError(
        f"Unknown model family: {MODEL_NAME}. "
        "Set instruction_part and response_part manually for your model."
    )


# ── 8. Define the trainer ────────────────────────────────────────────────────
# SFTConfig is the configuration object for SFTTrainer. It wraps Hugging Face's
# TrainingArguments with sensible SFT-specific defaults.

training_args = SFTConfig(
    output_dir            = OUTPUT_DIR,        # where checkpoints are saved
    num_train_epochs      = NUM_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    warmup_ratio          = WARMUP_RATIO,
    learning_rate         = LEARNING_RATE,
    bf16                  = torch.cuda.is_bf16_supported(),
    fp16                  = not torch.cuda.is_bf16_supported(),
    # bf16 (bfloat16) and fp16 (float16) are both 16-bit "half precision" formats.
    # They use half the memory of 32-bit floats and run faster on modern GPUs.
    # The difference: bf16 handles a wider range of numbers and is less likely to
    # produce NaN (not-a-number) errors during training. It is supported on Ampere
    # GPUs and newer (A100, A10, RTX 30xx/40xx series). Older GPUs like the T4 and
    # V100 only support fp16. torch.cuda.is_bf16_supported() detects this automatically
    # so you do not need to know your GPU model — the right choice is picked for you.
    # If you use the wrong one, training will either crash early or produce NaN losses.
    logging_steps         = LOGGING_STEPS,
    save_steps            = SAVE_STEPS,
    save_total_limit      = 2,         # keep only the 2 most recent checkpoints
    eval_strategy         = "steps",   # run validation every eval_steps
    eval_steps            = SAVE_STEPS,
    load_best_model_at_end= True,      # restore the best checkpoint when training ends
    metric_for_best_model = "eval_loss",
    greater_is_better     = False,     # lower loss = better
    max_grad_norm         = MAX_GRAD_NORM,
    weight_decay          = 0.01,      # mild L2 regularization on adapter weights
    optim                 = "adamw_8bit",
    # adamw_8bit is Unsloth's memory-efficient version of the AdamW optimizer.
    # It stores optimizer state in 8-bit, saving ~2 GB of VRAM on a 7B model
    # with no meaningful impact on quality. Use "adamw_torch" if you hit issues.
    lr_scheduler_type     = "cosine",
    # cosine: the learning rate follows a cosine curve — high at start, smoothly
    # decaying to near-zero by the end of training. Usually outperforms a flat
    # rate or linear decay for fine-tuning.
    seed                  = 42,
    max_seq_length        = MAX_SEQ_LENGTH,
    dataset_text_field    = "messages",
    # SFTTrainer needs to know which field in each dataset row holds the conversation.
    # Our rows have {"messages": [...]} — this tells it to use that field.
    packing               = False,
    # packing=True would concatenate short examples to fill the context window,
    # improving GPU utilisation. We leave it off so each example stays isolated —
    # simpler to reason about for a first run.
)

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = dataset["train"],
    eval_dataset    = dataset["validation"],
    args            = training_args,
)

# Apply response-only masking.
# This modifies the data collator inside trainer so the loss is only computed
# on the assistant tokens (the JSON output), not on the system prompt or user message.
# Pass the exact token strings that mark where the user turn and assistant turn begin.
trainer = train_on_responses_only(
    trainer,
    instruction_part = instruction_part,
    response_part    = response_part,
)

print("Trainer ready.")
print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"Steps per epoch: {len(dataset['train']) // (BATCH_SIZE * GRAD_ACCUM)}")


# ── 9. Run training ──────────────────────────────────────────────────────────
# This is the one line that actually trains the model.
# You will see a progress bar and periodic loss logs.
#
# Sample output (first few steps):
#   {'loss': 1.432, 'grad_norm': 0.812, 'learning_rate': 4e-05, 'epoch': 0.04}
#   {'loss': 1.201, 'grad_norm': 0.654, 'learning_rate': 1.2e-04, 'epoch': 0.08}
#   ...
# Loss should drop steadily in the first epoch, then more slowly.
# Typical final training loss for this task: 0.15–0.35.
# If loss does not move at all in the first 50 steps, something is wrong —
# see Ch19 for the debugging playbook.

print("Starting training…")
trainer_stats = trainer.train()

print("\nTraining complete.")
print(f"Total steps:    {trainer_stats.global_step}")
print(f"Final train loss: {trainer_stats.training_loss:.4f}")
print(f"Total time:     {trainer_stats.metrics['train_runtime'] / 60:.1f} minutes")


# ── 10. Save the adapter ─────────────────────────────────────────────────────
# model.save_pretrained saves ONLY the LoRA adapter weights — a few hundred MB.
# The base model is not saved here (it lives in your Hugging Face cache).
# To produce a fully self-contained model you can merge them — see Ch21.
print(f"\nSaving adapter to {OUTPUT_DIR}…")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
# This writes:
#   data/adapter/adapter_config.json   — LoRA config (rank, alpha, target modules)
#   data/adapter/adapter_model.safetensors  — the trained weights
#   data/adapter/tokenizer.json        — tokenizer files (for convenience at load time)
print("Adapter saved.")
```

---

## Quick inference test

Training is done. Now confirm the model actually works. The block below loads the adapter back from disk and runs a generation call on a fresh conversation the model never saw during training.

```python
# inference_test.py
# Load the trained adapter and test it on a new conversation.
# Run this after train.py completes.

import json
from unsloth import FastLanguageModel

ADAPTER_PATH   = "data/adapter"
MAX_NEW_TOKENS = 512   # upper bound on how many tokens the model can generate
                        # our JSON outputs are typically 100–300 tokens

# ── Load the adapter ─────────────────────────────────────────────────────────
# from_pretrained detects the adapter_config.json and loads the base model +
# adapter together. If the base model is already in cache, this takes ~10 seconds.
print("Loading model + adapter…")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = ADAPTER_PATH,
    max_seq_length= 2048,
    load_in_4bit  = True,
    dtype         = None,
)

# Switch to inference mode — Unsloth applies additional speed optimisations
# (fused attention, faster kernels) that are only valid during generation.
# Always call this before running generate().
FastLanguageModel.for_inference(model)
print("Ready.")


# ── The system prompt (identical to training) ─────────────────────────────────
SYSTEM_PROMPT = """You are a memory extraction assistant. Your job is to read a conversation and extract every memorable piece of information as a list of atomic memory objects.

Each memory object must follow this exact JSON schema:
{
  "text": "<the fact, written as a complete, standalone sentence>",
  "type": "<one of: preference | fact | decision | relationship | event>",
  "entities": ["<list of named people, places, or things involved>"]
}

Rules:
- One fact per memory object. Do not bundle multiple facts into one.
- Write "text" as a sentence someone could read without any surrounding context.
- If there are no memorable facts in the conversation, return an empty list: []
- Return ONLY a valid JSON array. No explanation, no markdown fences, no extra text."""


# ── A test conversation the model has never seen ──────────────────────────────
TEST_CONVERSATION = """User: I just got back from a three-week trip to Japan. Absolutely loved Kyoto.
Assistant: That sounds amazing. Did you have a favorite experience?
User: Honestly the food. I'm gluten-free so I was nervous, but Japanese food works really well for that.
Assistant: A lot of rice-based dishes naturally avoid gluten.
User: Exactly. My partner Maya is a chef, so she was in heaven the whole trip.
User: We're actually thinking about moving there within the next two years.
"""

# ── Build the prompt and tokenize it ─────────────────────────────────────────
# apply_chat_template takes the messages list and wraps it in the model's
# special tokens (e.g. <|im_start|>system ... <|im_start|>user ... etc.)
# add_generation_prompt=True appends the opening of the assistant turn,
# which tells the model "your turn to speak now."
messages = [
    {"role": "system",  "content": SYSTEM_PROMPT},
    {"role": "user",    "content": TEST_CONVERSATION},
]

# Tokenize and move tensors to the GPU
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize            = True,
    add_generation_prompt = True,
    return_tensors      = "pt",
).to("cuda")

print(f"Prompt token length: {inputs.shape[1]}")

# ── Run generation ────────────────────────────────────────────────────────────
# temperature=0.0 → deterministic output (greedy decoding). Best for structured
# output tasks where you want the most likely JSON, not creative variation.
# do_sample=False is required when temperature=0.
output_ids = model.generate(
    inputs,
    max_new_tokens = MAX_NEW_TOKENS,
    temperature    = 0.0,
    do_sample      = False,
    pad_token_id   = tokenizer.eos_token_id,  # suppress a warning about padding
)

# Decode only the newly generated tokens (strip the prompt)
generated_ids = output_ids[0][inputs.shape[1]:]
raw_output    = tokenizer.decode(generated_ids, skip_special_tokens=True)

print("\n── Raw model output ──────────────────────────────────────────────────")
print(raw_output)

# ── Validate the output against our schema ────────────────────────────────────
# We check three things inline here:
#   1. The output is valid JSON (json.loads does not raise).
#   2. It is a list (not a dict or string).
#   3. Every item has the three required fields: text, type, entities.
# This is the same logic from Ch11's validate_memory_output — reproduced here
# so inference_test.py is fully self-contained and you do not need to track
# down that file.

VALID_TYPES = {"preference", "fact", "decision", "relationship", "event"}

def validate_memory_output(raw: str) -> list:
    """Parse and validate a model's raw string output against our memory schema.
    Returns the list of memory dicts if valid. Raises ValueError with a clear
    message if anything is wrong."""
    try:
        memories = json.loads(raw.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Output is not valid JSON: {e}\nRaw output was:\n{raw}") from e

    if not isinstance(memories, list):
        raise ValueError(f"Expected a JSON array, got {type(memories).__name__}: {raw}")

    for i, m in enumerate(memories):
        for field in ("text", "type", "entities"):
            if field not in m:
                raise ValueError(f"Memory #{i} is missing required field '{field}': {m}")
        if m["type"] not in VALID_TYPES:
            raise ValueError(
                f"Memory #{i} has invalid type '{m['type']}'. "
                f"Must be one of: {sorted(VALID_TYPES)}"
            )
        if not isinstance(m["entities"], list):
            raise ValueError(f"Memory #{i} 'entities' must be a list, got: {m['entities']}")

    return memories

try:
    memories = validate_memory_output(raw_output)
    print(f"\n── Parsed memories ({len(memories)} total) ─────────────────────────────")
    for m in memories:
        print(f"  [{m['type']:12s}] {m['text']}")
        if m.get("entities"):
            print(f"               entities: {m['entities']}")
except ValueError as e:
    print(f"\nWARNING: {e}")
    print("This can happen if training was too short or the model is hallucinating.")
    print("See Ch19 for the debugging playbook.")
```

### What good output looks like

Running `inference_test.py` on a properly trained model should produce something like:

```json
[
  {
    "text": "The user recently returned from a three-week trip to Japan.",
    "type": "event",
    "entities": ["Japan"]
  },
  {
    "text": "The user loved Kyoto.",
    "type": "preference",
    "entities": ["Kyoto"]
  },
  {
    "text": "The user is gluten-free.",
    "type": "preference",
    "entities": []
  },
  {
    "text": "The user's partner is named Maya and is a chef.",
    "type": "relationship",
    "entities": ["Maya"]
  },
  {
    "text": "The user and their partner Maya are considering moving to Japan within the next two years.",
    "type": "decision",
    "entities": ["Maya", "Japan"]
  }
]
```

Five clean, atomic, standalone memories. No prose. No markdown fences. Valid JSON. If your output looks like this, the fine-tune worked.

---

## Understanding the output directory

After training, `data/adapter/` will contain:

```
data/adapter/
├── adapter_config.json          ← LoRA metadata (rank, alpha, target layers)
├── adapter_model.safetensors    ← the trained adapter weights (~150–300 MB)
├── tokenizer.json               ← tokenizer vocabulary
├── tokenizer_config.json        ← tokenizer settings (chat template lives here)
├── special_tokens_map.json      ← special token definitions
└── checkpoint-100/              ← mid-training checkpoint (auto-saved at step 100)
    └── ...
```

The most important thing to understand: the adapter is **not** a standalone model. It is a diff on top of the base model. To use it, you always load the base model first, then apply the adapter. The Unsloth `from_pretrained` call with the adapter path handles this automatically — it reads `adapter_config.json` to know which base model to load.

In *Ch21 - Saving, Merging, and Exporting Your Model* you will learn how to merge the adapter into the base weights to produce a single self-contained model file you can share or deploy without needing the original base.

---

## Common mistakes

**Mistake: not calling `FastLanguageModel.for_inference(model)` before generating.**

Without this call, Unsloth does not enable its fast inference kernels. The model will still generate — just slower, and in some versions it may produce slightly different token probabilities. Always call `for_inference` after loading an adapter for generation.

**Mistake: a different system prompt at inference time.**

If your inference script uses even a slightly different system prompt than training, you are effectively speaking a different dialect than the one the model learned. Keep the prompt in one shared constant. This was covered in *Ch12*, and it is worth repeating here because it is the most common cause of a "it works on the training data but not in my app" complaint.

**Mistake: `packing=True` with response-only masking.**

`train_on_responses_only` finds the boundary between user and assistant turns by looking for specific token strings. When `packing=True`, multiple examples are concatenated, and the token boundaries from different examples can confuse the masker — you end up training on parts of the wrong turn. For this task, leave `packing=False`.

**Mistake: forgetting to set `add_generation_prompt=True`.**

`apply_chat_template` with `add_generation_prompt=False` (or unset) does not append the opening assistant turn token. The model does not receive the signal that it should start generating. You will get either silence or the model continuing the user turn. Always pass `add_generation_prompt=True` at inference time.

**Mistake: large `max_new_tokens` with `temperature=0.0` and a model that has not learned to stop.**

If the model never learned to emit an EOS (end-of-sequence) token cleanly, it will fill `max_new_tokens` with repetitive output. The fix is to ensure your training data had clean, consistent endings (the JSON array closes, nothing follows). `Ch14 - Cleaning, Splitting, and Sanity-Checking Data` covers this — specifically, the assistant content field must end with `]` and nothing else.

**Mistake: training on a full-quality 4-bit run and then trying to merge on CPU.**

The merge step (in *Ch21*) dequantizes the base model and adds the adapter. It needs to fit in RAM. A 7B model in full float32 takes ~28 GB of RAM. If your machine does not have that, the merge will crash. Use `save_pretrained_merged` with `save_method="merged_16bit"` instead, which uses half-precision and requires ~14 GB of RAM.

**Mistake: loss never drops below 1.5.**

This usually means one of three things: (1) the dataset is too small (under ~200 examples), (2) the learning rate is too low (try `5e-4`), or (3) the system prompt in training data does not match the one in the SYSTEM_PROMPT constant and the model is confused. The full debugging playbook is in *Ch19 - When It Goes Wrong*.

---

## Recap

- Load a 4-bit quantized model with `FastLanguageModel.from_pretrained`. Roughly 5–6 GB VRAM for a 7B model.
- Attach LoRA adapters with `FastLanguageModel.get_peft_model`. You train ~1% of total parameters; the base model is completely frozen.
- `train_on_responses_only` masks the loss so the model only learns from the assistant turn (the JSON output), not from re-predicting the prompt.
- `SFTTrainer` with `SFTConfig` handles tokenization, batching, evaluation, and checkpointing.
- Save only the adapter with `model.save_pretrained`. It is a few hundred MB, not the full 5+ GB model.
- At inference time: load the adapter with `from_pretrained`, call `FastLanguageModel.for_inference`, then use `apply_chat_template` with `add_generation_prompt=True` before generating.
- A successful first run produces valid JSON with atomic, standalone memories on conversations the model never saw.
- Typical training time: 20–40 minutes on an A100, 60–90 minutes on a T4.

## Next

*Ch16 - Hyperparameters: Which Knobs to Turn and When* — the script above uses reasonable defaults; this chapter explains what each number actually does and gives you a systematic approach to improving results when the first run is not good enough.
