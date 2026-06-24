# Ch9 - The Toolbox: Unsloth, Transformers, TRL, PEFT, and Friends

Fine-tuning a model involves more than one library. But before you feel overwhelmed, here is the
good news: **Unsloth wraps most of them for you**. You will mostly write Unsloth code, and the
other libraries will quietly do their jobs in the background.

This chapter introduces each tool in the stack, explains what problem it solves, and shows you
exactly where it shows up in your memory-extraction pipeline. By the end you will have a complete
mental map of the stack — and a short installation script you can run right now.

---

## What you'll learn

- What each library in the fine-tuning stack actually does, in plain terms.
- Why Unsloth is the right starting point instead of building on raw Hugging Face code.
- How all the pieces connect into a single pipeline for our memory-extraction task.
- Which libraries you interact with directly, and which run silently underneath.
- How to install the full stack and verify it works in under five minutes.

---

## Concepts you need first

### What is a library "wrapper"?

A wrapper is a library that calls other libraries on your behalf. Instead of writing 200 lines of
Hugging Face boilerplate to load a model with 4-bit compression and LoRA, Unsloth lets you write
10 lines. Under the hood it is still calling `transformers`, `peft`, and `bitsandbytes` — you just
do not have to orchestrate them yourself.

Think of it like a travel agent. You could call the airline, the hotel, and the car rental company
separately. Or you could call one person who handles all three. Unsloth is that travel agent for
model fine-tuning.

### What does "the training loop" mean?

Every fine-tuning run follows a cycle: feed the model a batch of examples, measure how wrong the
output is (the loss), adjust the model weights to be less wrong, repeat. This cycle — forward pass,
loss, backward pass, weight update — is the training loop. In Ch7 (How Training Actually Works) we
covered this in depth. The library TRL gives you a pre-built, high-quality training loop so you do
not have to code the cycle yourself.

---

## The stack, piece by piece

Here is the full picture of how the libraries fit together for our pipeline. Read this diagram
top-to-bottom — it shows data flow from raw conversations to a finished, serving model.

```
┌──────────────────────────────────────────────────────────────────┐
│                        YOUR TRAINING SCRIPT                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   datasets          ──► loads your JSONL training rows           │
│       │                                                           │
│       ▼                                                           │
│   Unsloth           ──► loads model + tokenizer (fast, low VRAM) │
│       │  wraps:                                                   │
│       │    transformers  (model architecture + tokenizer)         │
│       │    peft          (adds LoRA adapter layers)               │
│       │    bitsandbytes  (4-bit quantization)                     │
│       │    accelerate    (GPU memory + mixed precision)           │
│       │                                                           │
│       ▼                                                           │
│   TRL / SFTTrainer  ──► runs the training loop                   │
│       │                                                           │
│       ▼                                                           │
│   wandb (optional)  ──► streams loss curves to your browser      │
│                                                                   │
│   OUTPUT: saved LoRA adapter weights                             │
│       │                                                           │
│       ▼                                                           │
│   Unsloth merge     ──► merges adapter into full model           │
│       │                                                           │
│       ▼                                                           │
│   vllm / llama.cpp  ──► serves the model for inference           │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

Now let's meet each player individually.

---

### 1. Unsloth — your primary tool

**What it does:** Speeds up fine-tuning by 2–5x and cuts VRAM usage by roughly 40–60% compared to
training with raw Hugging Face code. It also provides the cleanest API for loading models,
attaching LoRA adapters, and saving results.

**Why Unsloth over vanilla Hugging Face?** Without Unsloth, loading a 7B model with 4-bit
quantization and LoRA requires you to correctly combine three separate libraries — `transformers`,
`peft`, and `bitsandbytes` — with the right settings for each. One wrong flag and you are either
training in float32 (wastes VRAM), not actually applying LoRA (too slow), or crashing on a memory
error. Unsloth's `FastLanguageModel` handles all of that in two function calls and adds its own
kernel optimizations on top.

**When you touch it:** In almost every step — loading the base model, attaching LoRA, and saving
the final weights.

**Typical VRAM usage:** A 7B model with Unsloth + 4-bit quantization fits in roughly 6–8 GB of
VRAM. Without Unsloth and quantization, the same model needs 14–16 GB.

```python
# unsloth is our primary entry point. install: pip install unsloth
from unsloth import FastLanguageModel

# Load a quantized model ready for LoRA fine-tuning.
# max_seq_length: the maximum number of tokens in one training example.
# We set 2048 — enough for a conversation chunk + its memory JSON output.
# dtype=None: let Unsloth detect the right float type for your GPU.
# load_in_4bit=True: use bitsandbytes 4-bit quantization (saves ~half the VRAM).
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B-bnb-4bit",  # pre-quantized checkpoint on HF Hub
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Attach LoRA adapter layers to the model.
# This is where we choose WHICH parts of the model get new trainable parameters.
# r=16: the LoRA rank (see Ch6). A good starting value.
# target_modules: the attention layers we will train. These are standard for Qwen/Gemma.
# lora_alpha=16: a scaling factor — keep it equal to r to start.
# lora_dropout=0: disabling dropout is fine here; Unsloth recommends it.
# bias="none": we do not train bias terms. Standard practice with LoRA.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's own checkpointing, saves more VRAM
    random_state=42,
)
```

---

### 2. transformers — the model and tokenizer engine

**What it does:** The Hugging Face `transformers` library contains the actual architecture code
for thousands of models — Qwen3, Gemma 3, Llama, Mistral, and so on. It also provides the
tokenizer: the code that turns a string like `"extract memories from this chat"` into a list of
integer IDs the model can process.

**Why it exists:** Without `transformers`, you would need to implement the Transformer
architecture yourself from scratch. The library gives you battle-tested implementations.

**When you touch it:** Mostly indirectly through Unsloth. You will use the tokenizer directly
when formatting data (Ch12) and when applying the chat template (Ch5).

```python
# You usually get the tokenizer from Unsloth's loader (shown above).
# But you can also use it directly for formatting checks:

# Example: see how many tokens a sample training row uses.
# This is important — if your examples are longer than max_seq_length, they get truncated.
sample = {
    "role": "user",
    "content": "Alice told Bob she prefers morning meetings. Bob said he hates Mondays."
}
tokens = tokenizer.encode(sample["content"])
print(f"Token count: {len(tokens)}")  # expect ~20 for this short example
```

---

### 3. TRL (SFTTrainer) — the training loop

**What it does:** TRL stands for "Transformer Reinforcement Learning," but for our purposes we
only use one component: `SFTTrainer` (Supervised Fine-Tuning Trainer). It runs the training loop —
iterates over your dataset in batches, computes the loss, and updates the model weights.

**Why TRL over writing a training loop yourself?** A custom training loop requires handling batch
collation, gradient accumulation, mixed-precision math, saving checkpoints at the right time, and
more. `SFTTrainer` handles all of that. It is the standard choice in the Hugging Face ecosystem
for supervised fine-tuning and integrates seamlessly with Unsloth.

**When you touch it:** One `SFTTrainer(...)` call in your training script (Ch15).

```python
# trl gives us SFTTrainer. install: pip install trl
# NOTE: trl >= 0.8 introduced SFTConfig (replacing the old dataset_text_field argument
# and renaming tokenizer= to processing_class=). The snippet below uses the current API.
# Ch15 has the full, version-matched training script — use that for your actual run.
from trl import SFTTrainer, SFTConfig

# SFTConfig is the new settings panel for SFTTrainer (replaces TrainingArguments for SFT).
# It accepts all the same fields as TrainingArguments, plus SFT-specific ones.
training_args = SFTConfig(
    output_dir="./memory-extractor-checkpoints",  # where to save model snapshots
    per_device_train_batch_size=2,   # examples per GPU per step (tune to VRAM)
    gradient_accumulation_steps=4,   # simulate batch_size=8 without needing the VRAM
    warmup_steps=10,                 # slowly ramp up learning rate at the start
    max_steps=200,                   # total training steps (use this OR num_train_epochs)
    learning_rate=2e-4,              # how fast to adjust weights; 2e-4 is a good LoRA default
    fp16=True,                       # use 16-bit math to save VRAM (or bf16 on newer GPUs)
    logging_steps=10,                # print a loss update every N steps
    save_steps=50,                   # save a checkpoint every N steps
    optim="adamw_8bit",              # 8-bit optimizer from bitsandbytes — saves ~2 GB VRAM
    dataset_text_field="text",       # the column with our formatted prompt+response (new home)
    max_seq_length=2048,
    seed=42,
)

# SFTTrainer wires together: model, tokenizer, dataset, and training config.
# processing_class= replaced tokenizer= in trl >= 0.8.
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,        # trl >= 0.8 name; older versions used tokenizer=
    train_dataset=train_dataset,       # a Hugging Face Dataset object (see datasets below)
    args=training_args,
)

trainer.train()  # starts the loop; prints loss every logging_steps
```

We cover all these hyperparameters in Ch16 (Hyperparameters: Which Knobs to Turn and When).

---

### 4. PEFT — LoRA adapter management

**What it does:** PEFT stands for "Parameter-Efficient Fine-Tuning." It is the library that
actually implements LoRA — injecting the small adapter matrices into the model's attention layers
and making sure only those matrices get updated during training.

**When you touch it:** Almost never directly. Unsloth's `get_peft_model()` calls PEFT under the
hood. You might interact with PEFT when loading a saved adapter or merging it into the base model.

**Why it matters:** Without PEFT, LoRA would not exist as a concept in the Hugging Face ecosystem.
Every LoRA fine-tune you have ever read about uses PEFT behind the scenes.

```python
# You usually do not call peft directly — Unsloth wraps it.
# But here is what Unsloth is doing internally, so you can read the docs confidently:
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",  # we are fine-tuning a language model
)
# Unsloth calls something equivalent to this under the hood:
# peft_model = get_peft_model(base_model, config)
```

---

### 5. datasets — loading and formatting your training data

**What it does:** The Hugging Face `datasets` library is the standard way to load, inspect, and
process structured data for model training. It stores data efficiently (in Arrow format), supports
streaming for large files, and plugs directly into `SFTTrainer`.

**When you touch it:** When loading your JSONL file of memory-extraction examples, when splitting
into train/validation sets, and when applying a formatting function to turn raw rows into the
prompt format your model expects.

```python
# datasets gives us a clean data pipeline. install: pip install datasets
from datasets import load_dataset

# Load a local JSONL file. Each line is one training example:
# {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
dataset = load_dataset("json", data_files="data/memory_extraction_train.jsonl", split="train")

print(f"Loaded {len(dataset)} training examples")
# your number will vary — Ch13 covers how many examples you need and how to generate them

# Inspect one row to make sure the format looks right before training.
print(dataset[0])

# Apply a formatting function to each row.
# This converts our messages format into a single string the SFTTrainer expects.
# The tokenizer's apply_chat_template method handles the special tokens.
def format_row(row):
    # apply_chat_template turns a list of {role, content} messages
    # into the exact string format the model was pre-trained on (e.g., <|im_start|>user...)
    text = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,          # return a string, not token IDs
        add_generation_prompt=False,  # we want the full conversation including the answer
    )
    return {"text": text}

dataset = dataset.map(format_row)
```

We cover dataset format in detail in Ch12 (Data Format: Turning the Task into Training Rows).

---

### 6. bitsandbytes — 4-bit quantization

**What it does:** bitsandbytes provides the math that compresses a model's weights from 32-bit
or 16-bit floating-point numbers down to 4-bit integers, cutting VRAM usage roughly in half.
Think of it like compressing a large image file — you lose a tiny bit of quality, but the file
size drops dramatically and it is still clearly the same image.

**When you touch it:** Almost never directly. You pass `load_in_4bit=True` to Unsloth, and
bitsandbytes does the rest. The only time you might touch it directly is if you are debugging a
quantization error.

**Why it matters for us:** Without 4-bit quantization, a 7B model occupies roughly 14 GB of VRAM
in float16. With bitsandbytes 4-bit, it drops to around 5–6 GB. That is the difference between
needing a high-end A100 and being able to train on a consumer RTX 3090 or a rented T4.

```python
# bitsandbytes works invisibly via the load_in_4bit=True flag in Unsloth.
# You do not import it directly in day-to-day use.

# To verify it is installed correctly:
import bitsandbytes as bnb
print(f"bitsandbytes version: {bnb.__version__}")
# The 8-bit Adam optimizer also comes from here:
# optim="adamw_8bit" in TrainingArguments uses bnb.optim.AdamW8bit under the hood.
```

---

### 7. accelerate — GPU orchestration

**What it does:** `accelerate` is a Hugging Face library that abstracts the low-level details of
running PyTorch on one GPU, multiple GPUs, or mixed-precision (using float16 instead of float32
for some operations to save VRAM and run faster).

**When you touch it:** Almost never directly for single-GPU training. Unsloth and TRL configure
`accelerate` automatically. If you ever want to train across multiple GPUs, you will interact with
it through an `accelerate config` command in the terminal.

**Why it matters:** Mixed-precision training — the `fp16=True` or `bf16=True` flag you will see
in `TrainingArguments` — is powered by `accelerate`. Without it, every weight update would be
computed in float32, which is slower and uses more memory.

```python
# accelerate is installed automatically by Unsloth. You rarely import it directly.
# To confirm it is present and check your version:
import accelerate
print(f"accelerate: {accelerate.__version__}")

# Quick GPU-type guide for the flag you set in TrainingArguments:
#   fp16=True  — use on older GPUs (GTX 10xx/20xx series, T4, V100)
#   bf16=True  — use on Ampere or newer (RTX 30xx/40xx, A100, H100)
# Not sure? Run: python -c "import torch; print(torch.cuda.get_device_name(0))"
# Ampere = RTX 3080, 3090, A100, etc. If in doubt, use fp16 — it works everywhere.
```

---

### 8. vllm / llama.cpp — serving your finished model

**What it does:** Once training is done, you need something to actually run the model for
inference — that is, to take a new conversation and produce a memory-extraction JSON response.

- **vllm** is a high-performance Python server. It is fast, supports streaming, and has an
  OpenAI-compatible API. Best for GPU servers where you want throughput. Install: `pip install vllm`
- **llama.cpp** compiles the model down to a highly optimized binary. It can run on CPU (slowly)
  or GPU and is the best choice for running locally on a Mac or a machine without a Python
  environment. The key output format here is GGUF, which Unsloth can export to directly.
  Install the Python bindings with: `pip install llama-cpp-python`

**When you touch it:** In Ch22 (Serving Your Model and Using It in an App), which covers both
options in full. If you are on a Mac or a CPU-only machine, llama.cpp is your path — jump to
Ch22 after finishing this chapter for a working example.

**Quick llama.cpp preview** (after exporting your model to GGUF in Ch21):

```bash
# Install the Python bindings (CPU build; for GPU add CMAKE_ARGS flags — see Ch22).
pip install llama-cpp-python

# Run your exported GGUF model from the command line:
# llama-cli --model ./memory-extractor.gguf \
#           --prompt "Extract memories: Alice prefers morning meetings." \
#           --n-predict 200
```

```python
# vllm serving example — runs after you have merged and saved your model.
# This starts a local OpenAI-compatible API server on port 8000.
# Run this in your terminal (not in a training script):
# python -m vllm.entrypoints.openai.api_server \
#     --model ./memory-extractor-merged \
#     --host 0.0.0.0 \
#     --port 8000

# Then call it from Python like any OpenAI-compatible client:
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

response = client.chat.completions.create(
    model="memory-extractor-merged",
    messages=[
        {"role": "system", "content": "Extract memories as JSON."},
        {"role": "user", "content": "Alice told Bob she prefers morning meetings."},
    ],
)
print(response.choices[0].message.content)
# expect: [{"text": "Alice prefers morning meetings.", "type": "preference", "entities": ["Alice"]}]
```

---

### 9. wandb — training visibility

**What it does:** Weights & Biases (`wandb`) is an experiment tracker. It captures your loss
curve, learning rate schedule, and any metrics you log, and streams them to a web dashboard at
wandb.ai so you can watch training progress from your phone.

**Why wandb over just reading terminal output?** Loss numbers printed to the terminal disappear.
wandb keeps a permanent, searchable record of every run — which hyperparameters you used, how
the loss moved, and how different runs compare. When you are iterating toward a better model
(Ch20), this history is essential.

**When you touch it:** Optionally, in your training script. One line enables it.

```python
# install: pip install wandb
# First time: run `wandb login` in the terminal and paste your API key from wandb.ai.

import wandb

# Initialize a run before calling trainer.train().
# project: groups all your memory-extractor experiments together.
# name: a human-readable label for this specific run.
wandb.init(
    project="memory-extractor",
    name="qwen3-8b-lora-r16-run1",
    config={
        "model": "Qwen3-8B",
        "lora_r": 16,
        "max_steps": 200,
        "learning_rate": 2e-4,
    },
)

# After this, SFTTrainer automatically logs to wandb — no extra code needed.
# wandb is free for individuals (unlimited runs, with storage limits on the free tier;
# see wandb.ai/pricing for current caps). You can also set WANDB_MODE=offline to log
# locally only — no account needed, logs saved to ./wandb/ in your project directory.
```

---

## Putting it all together — the full installation

Here is a single script that installs the complete stack and verifies everything is working. Run
this once on your machine or cloud instance before you start training.

```bash
# Run in your terminal (not in Python).
# We recommend a virtual environment first:
#   python -m venv .venv && source .venv/bin/activate

# Step 1: Install Unsloth. It pulls in transformers, peft, accelerate, and bitsandbytes.
# IMPORTANT: the install command varies by environment and CUDA version.
# The command below works on Linux with CUDA 12.x (e.g., a RunPod or Lambda Labs instance).
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# On CUDA 11.x (older cloud GPUs like some T4 instances), replace [colab-new] with [cu118].
# On Google Colab, use: pip install unsloth  (Colab sets up CUDA automatically).
# On macOS: Unsloth does not support macOS GPUs. Use a cloud GPU instance (see Ch8).
# On Windows: install WSL2 first, then treat the environment as Linux inside WSL2.
# When in doubt: check https://github.com/unslothai/unsloth#installation for the
# current one-liner matched to your exact CUDA version (run `nvidia-smi` to find it).

# Step 2: Install TRL and datasets.
pip install trl datasets

# Step 3: Install optional tools.
pip install wandb          # experiment tracking
pip install vllm           # serving (GPU required; skip on CPU-only machines)
```

```python
# verify_stack.py — run this after installation to confirm everything loaded correctly.
import torch

# Check GPU availability. Fine-tuning on CPU is possible but extremely slow.
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    # Get VRAM in GB. You need at least 8 GB for a 7B model with 4-bit quantization.
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram_gb:.1f} GB")

# Check each library can be imported.
import transformers
import peft
import trl
import datasets
import bitsandbytes
import accelerate
import unsloth

print(f"transformers:  {transformers.__version__}")
print(f"peft:          {peft.__version__}")
print(f"trl:           {trl.__version__}")
print(f"datasets:      {datasets.__version__}")
print(f"bitsandbytes:  {bitsandbytes.__version__}")
print(f"accelerate:    {accelerate.__version__}")
print(f"unsloth:       {unsloth.__version__}")

print("\nAll libraries imported successfully. Ready to fine-tune.")
```

When this script prints "All libraries imported successfully," your environment is ready.

---

## Common mistakes

**Mistake 1: Importing from the wrong library.**
After reading this chapter you know bitsandbytes, peft, and transformers all exist. It is tempting
to import directly from them. For the most part, use Unsloth's API. Mixing Unsloth's optimized
model with raw PEFT calls can break the custom kernels Unsloth adds.

*Fix:* Start all model loading with `from unsloth import FastLanguageModel`. Only reach for
the underlying libraries when the Unsloth docs explicitly say to.

**Mistake 2: Installing the wrong Unsloth variant.**
Unsloth has different install commands for Colab, local CUDA 11.x, local CUDA 12.x, and CPU. The
wrong variant will install but then fail silently or throw confusing CUDA errors at runtime.

*Fix:* Check `nvidia-smi` for your CUDA version, then use the matching install command from the
official Unsloth README.

**Mistake 3: Skipping wandb and then not knowing why the model went wrong.**
Loss curves look boring until you have a bad run and need to diagnose it. Without tracking, you
have no record of what you tried.

*Fix:* Set up wandb before your first training run. It takes two minutes (`wandb login`). If you
do not want cloud tracking, set `WANDB_MODE=offline` in your environment to log locally.

**Mistake 4: Not checking VRAM before starting training.**
Starting a run that requires 12 GB on an 8 GB GPU will run for a few minutes, then crash with an
out-of-memory error and give you nothing.

*Fix:* Run `verify_stack.py` and check the printed VRAM number. If you are under 12 GB, use
`per_device_train_batch_size=1` and `gradient_accumulation_steps=8` to keep memory usage low.

**Mistake 5: Confusing vllm for a training tool.**
vllm is for inference only. You cannot use it to continue training a model.

*Fix:* Train with Unsloth + TRL, export the merged model, then hand it off to vllm. These are
two separate phases: training phase and serving phase.

---

## Recap

- **Unsloth** is your primary tool — it wraps the lower-level libraries and makes the whole stack
  faster and easier with less VRAM.
- **transformers** provides the model architecture and tokenizer. You mostly use the tokenizer
  directly when formatting data.
- **TRL's SFTTrainer** runs the actual training loop. One call, many settings — explored fully
  in Ch16.
- **PEFT** implements LoRA under the hood. Unsloth calls it for you.
- **bitsandbytes** provides 4-bit quantization, cutting VRAM roughly in half. Activated by
  `load_in_4bit=True`.
- **datasets** loads and formats your JSONL training data cleanly.
- **accelerate** handles GPU setup and mixed-precision — mostly invisible in single-GPU training.
- **vllm / llama.cpp** serve your finished model for real inference after training is done.
- **wandb** tracks your loss curves and experiment history — not required but strongly recommended
  from run one.

## Next

**Ch10 - Choosing Your Base Model: Qwen vs Gemma** — now that you know the tools, we pick the
starting model your fine-tune will build on top of.
