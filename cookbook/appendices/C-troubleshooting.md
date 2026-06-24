# Appendix C - Troubleshooting Common Errors

This appendix is a field guide. You hit an error, you look it up here, you fix it and move on.
Each entry quotes a realistic slice of the actual error text, explains the one real cause, and
gives you a concrete fix — often with a runnable code snippet to verify the repair.

All examples assume you are working through the memory-extraction fine-tune built across this
book. The JSON schema used in the snippets is the same one established in Chapter 12 — Data
Format: Turning the Task into Training Rows: each memory is a dict with `text`, `type`, and
`entities` fields.

---

## What you'll learn

- How to diagnose and fix CUDA out-of-memory crashes — the most common blocker
- How to resolve `bitsandbytes` / CUDA version mismatches without reinstalling everything
- What to do when Colab says "no GPU found" and why restarting the session matters
- How to fix tokenizer and chat-template errors that silently corrupt training
- How to repair JSON parse failures in your evaluation loop with a robust fallback parser
- How to handle pad-token and EOS-token warnings before they cause `nan` loss
- How to fix slow training and Unsloth import errors

---

## Concepts you need first

**What "CUDA" is.**
CUDA is NVIDIA's programming layer that lets Python code talk directly to a GPU. Think of it as
a translator: your Python says "multiply these matrices," CUDA converts that to instructions the
GPU chip can execute. When an error message mentions CUDA, it almost always means one of two
things: a version mismatch (library compiled for CUDA 11 but your driver only speaks CUDA 12, or
vice-versa), or running out of GPU memory. Both are fixable without reinstalling your OS or
buying a new GPU.

**What "bitsandbytes" does.**
`bitsandbytes` is the library that performs 4-bit and 8-bit quantization — compressing the
model's weights so they fit in less VRAM (see Chapter 6 — LoRA and QLoRA Without the Math
Headache). It ships as a pre-compiled C extension that must match your CUDA version exactly.
A mismatch is the single most common install error in QLoRA workflows, and the fix is almost
always a targeted reinstall, not a fresh environment.

**What "pad token" and "EOS token" mean.**
A tokenizer (see Chapter 5 — Tokens, Context Windows, and Chat Templates) converts text into
integer IDs. Two special IDs matter for training. The EOS token ("end of sequence") signals
that the model's turn is finished. The PAD token fills empty space in a batch so every row is
exactly the same length — the GPU requires rectangular arrays. Some base models ship without a
dedicated pad token, which forces the trainer to reuse EOS for both purposes. This creates a
subtle bug: the model learns that EOS sometimes means "stop generating" and sometimes means
"this is just filler," and the training signal becomes confused.

---

## Error entries

---

### E-01 — CUDA out of memory

**Error text you will see**

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 0 has a total capacity of 15.78 GiB of which 1.23 GiB is free.
```

or, from Unsloth specifically:

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB.
```

**Cause**

The batch of sequences you asked the GPU to process at one time is larger than the free VRAM.
This is almost always `per_device_train_batch_size` being too high, or `max_seq_length` being
set much larger than your actual sequences. Both waste VRAM on empty padding.

**Fix — step by step**

1. Lower `per_device_train_batch_size` to `1`.
2. Raise `gradient_accumulation_steps` by the same factor you dropped the batch size, so the
   effective batch seen during optimization stays the same. For example: batch 4 → batch 1 plus
   accumulation 4.
3. If that is still not enough, lower `max_seq_length`. For the memory-extraction task, 1 024
   tokens covers the vast majority of inputs; going to 2 048 rarely adds value and doubles the
   VRAM cost per token.
4. On a 15–16 GB card (T4, RTX 3080, RTX 4080), confirm `load_in_4bit=True` is set.

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Step 1 — load the model in 4-bit to cut VRAM roughly in half.
# On a 16 GB card (T4, RTX 3080), this is not optional — it is required.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Qwen3-1.7B",
    max_seq_length = 1024,    # keep this as small as your data allows
    load_in_4bit   = True,    # compresses weights from 16-bit to 4-bit in VRAM
    dtype          = None,    # let Unsloth choose (bfloat16 on Ampere+, float16 older)
)

# Step 2 — use batch size 1 and accumulate gradients over 4 steps.
# From the optimizer's perspective, this is identical to batch size 4,
# but only 1 example occupies VRAM at any moment.
trainer = SFTTrainer(
    model = model,
    args  = TrainingArguments(
        per_device_train_batch_size = 1,  # was 4, dropped to 1
        gradient_accumulation_steps = 4,  # compensates: effective batch = 1 * 4 = 4
        # ... the rest of your arguments stay the same
    ),
    # ...
)
```

**Quick sanity-check after the fix**

```python
import torch

# Run this after loading the model and before starting training.
# A positive free-VRAM number means you have headroom. Near-zero means OOM is still likely.
free_bytes, total_bytes = torch.cuda.mem_get_info()
print(f"Free VRAM : {free_bytes  / 1e9:.1f} GB")
print(f"Total VRAM: {total_bytes / 1e9:.1f} GB")
print(f"Used      : {(total_bytes - free_bytes) / 1e9:.1f} GB")
# For a 1.7B model in 4-bit on a 16 GB card, you should see roughly 12-13 GB free here.
```

---

### E-02 — bitsandbytes / CUDA version mismatch

**Error text you will see**

```
CUDA Setup failed despite /usr/local/cuda/lib64/libcudart.so.11.0 being available.
...
The currently running CUDA version 11.8 does not match bitsandbytes CUDA version 12.1.
```

or:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

or a quieter variant that just prints:

```
WARNING: bitsandbytes is currently running on CPU. Most slow operations will be on CPU.
```

**Cause**

`bitsandbytes` ships separate compiled binaries for each CUDA version. The version it was
compiled against and the version your system driver exposes must match. They diverge most often
after a `pip upgrade`, after a Colab runtime update, or when you move a project from one machine
to another.

**Fix**

First, figure out your actual CUDA version. There are three places to check, and they can
disagree — you need the one that PyTorch sees:

```bash
# Run each of these in a terminal or Colab cell (prefix with ! in a notebook cell)

nvcc --version
# Shows the CUDA toolkit version installed on disk.

nvidia-smi
# Shows the maximum CUDA version the GPU driver supports.
# "CUDA Version: 12.2" in the top-right corner is the ceiling — you can use any version <= this.

python -c "import torch; print('PyTorch CUDA:', torch.version.cuda)"
# This is the version that matters for bitsandbytes. Match to this one.
```

Once you know the version PyTorch sees, reinstall `bitsandbytes` to match:

```bash
# Try the auto-detect upgrade first — works in most cases with bitsandbytes >= 0.43
pip install --upgrade bitsandbytes

# If auto-detect still fails, pin to your exact CUDA version:

# CUDA 11.8
pip install "bitsandbytes>=0.43.3" --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install "bitsandbytes>=0.43.3" --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install "bitsandbytes>=0.43.3" --index-url https://download.pytorch.org/whl/cu124
```

**Verify the fix**

```python
import bitsandbytes as bnb

# If this prints without an error or warning, the CUDA binding loaded correctly.
print("bitsandbytes version:", bnb.__version__)

# This should print a small positive integer, not raise an exception.
# It confirms that 4-bit matrix multiplication is available on your GPU.
test = bnb.nn.Linear4bit(16, 16)
print("4-bit layer created successfully:", test)
```

---

### E-03 — "No GPU found" in Google Colab

**Error text you will see**

```
Failed to detect a default GPU. Please set the runtime to use a GPU.
```

or from Unsloth:

```
Unsloth: No GPU detected. Training will be very slow or may fail.
```

or from PyTorch:

```
AssertionError: No CUDA-capable device found.
```

**Cause**

Colab defaults to a CPU-only runtime. You must manually switch the hardware accelerator before
importing any GPU library. The critical detail that trips people up: if you import `torch` or
`unsloth` before switching runtimes, those libraries have already initialized against the CPU.
Switching the runtime type after the fact does not retroactively fix the running kernel — you
must restart the session.

**Fix**

In Colab: **Runtime → Change runtime type → Hardware accelerator → T4 GPU** (free tier) or
A100/L4 (Colab Pro). Then: **Runtime → Restart session**. Do not click "Reconnect" — restart.

After restart, put this check at the very top of your first cell, before any other imports:

```python
import torch

# This assertion will fail fast and loudly if the GPU is still not available,
# rather than letting you run 10 cells before the real crash.
assert torch.cuda.is_available(), (
    "No GPU detected. Steps to fix:\n"
    "1. Runtime → Change runtime type → Hardware accelerator → T4 GPU\n"
    "2. Runtime → Restart session (NOT just reconnect)\n"
    "3. Run this cell again before doing anything else."
)

# Print GPU details so you know exactly what hardware you have.
print(f"GPU model : {torch.cuda.get_device_name(0)}")
print(f"VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# T4 → 15.0 GB, L4 → 22.5 GB, A100 → 40.0 or 80.0 GB
```

---

### E-04 — Tokenizer / chat-template errors

**Error text you will see**

```
jinja2.exceptions.TemplateError: 'messages' variable is missing
```

or:

```
KeyError: 'role'
```

or a subtler failure: training completes without any error, but at evaluation time the model
outputs repeated tokens, empty strings, or raw prose instead of the expected JSON array.

**Cause**

Hugging Face chat templates (see Chapter 5 — Tokens, Context Windows, and Chat Templates) expect
the input to be a list of dicts, each with exactly `role` and `content` as string keys. The role
values must be lowercase and must match what the specific model's template expects (typically
`"system"`, `"user"`, `"assistant"`). A wrong key name, a capitalized role value, or passing a
plain string instead of a list breaks the template — sometimes loudly, sometimes silently with
a corrupted prompt that poisons every training example.

**Fix**

Always validate your formatted prompt on a single example before running any training loop:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Qwen3-1.7B",
    max_seq_length = 1024,
    load_in_4bit   = True,
)

# This is the correct structure for a memory-extraction training example.
# Every key name and role value must be exactly as shown here.
messages = [
    {
        "role": "system",                          # lowercase, not "System"
        "content": (
            "Extract memories from the conversation below. "
            "Return a JSON array only — no other text. "
            "Each memory is an object with keys: text, type, entities."
        )
    },
    {
        "role": "user",                            # lowercase, key is "content" not "text"
        "content": "Alice told me she hates cilantro and always skips it in recipes."
    },
    {
        "role": "assistant",
        "content": (
            '[{"text": "Alice hates cilantro.", '
            '"type": "preference", "entities": ["Alice"]}, '
            '{"text": "Alice skips cilantro in recipes.", '
            '"type": "behavior", "entities": ["Alice"]}]'
        )
    },
]

# apply_chat_template with tokenize=False returns the raw string so you can read it.
# Inspect this carefully: you should see the system prompt, user turn, and assistant
# turn clearly delimited by the model's special tokens.
rendered = tokenizer.apply_chat_template(
    messages,
    tokenize              = False,   # string output, not token IDs, so we can read it
    add_generation_prompt = False,   # True at inference; False at training
)
print(rendered)
print(f"\nToken count: {len(tokenizer.encode(rendered))}")
# If you see garbled output, empty output, or a Jinja exception, the dict structure is wrong.
```

Common structural mistakes and their fixes:

```python
# WRONG — the content key is called "text" instead of "content"
{"role": "user", "text": "Alice mentioned she hates cilantro."}

# WRONG — role is capitalized; Qwen3 and Gemma3 templates use lowercase
{"role": "User", "content": "Alice mentioned she hates cilantro."}

# WRONG — passing a plain string instead of a list of dicts
tokenizer.apply_chat_template("Alice mentioned she hates cilantro.", ...)

# WRONG — the messages list contains strings, not dicts
messages = ["Alice mentioned she hates cilantro.", "Here are the memories: ..."]
```

---

### E-05 — JSON parse failures in the evaluation loop

**Error text you will see**

```
json.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

or:

```
json.JSONDecodeError: Extra data: line 3 column 1 (char 47)
```

or you get no exception at all, but your eval loop silently counts zero extracted memories
for every example.

**Cause**

Your model's output is not clean JSON. This happens in three situations, all common during
early training or with an under-trained model:

1. The model prepends prose before the JSON: `"Here are the memories I found: [...]"`
2. The model wraps the JSON in a markdown code fence: ` ```json\n[...]\n``` `
3. The model partially generates output and cuts off mid-array (usually a `max_new_tokens`
   limit being too low)

**Fix**

Write a robust parser that strips the common noise before attempting `json.loads`. This parser
should be part of your evaluation pipeline from day one — you want it to degrade gracefully
and log bad outputs rather than crash the whole eval run:

```python
import json
import re
from typing import Any

def parse_memory_output(raw: str) -> list[dict[str, Any]]:
    """
    Extract a JSON list of memories from raw model output.

    Handles the three most common failure modes:
      - Markdown code fences (```json ... ```)
      - Leading prose before the JSON array
      - Trailing text after the closing bracket

    Returns an empty list if nothing can be parsed — never raises.
    This is intentional: a failed parse counts as zero memories extracted,
    which correctly penalizes the model in your F1 calculation.
    """

    # Step 1: strip markdown code fences if present.
    # The pattern matches ```json ... ``` or just ``` ... ```.
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        # Step 2: find the outermost JSON array by locating the first '[' and last ']'.
        # This handles cases where the model says "Here are your memories: [...]".
        start = raw.find("[")
        end   = raw.rfind("]")
        if start == -1 or end == -1 or end < start:
            return []   # no array-like structure found at all
        candidate = raw[start : end + 1]

    # Step 3: attempt to parse the isolated candidate.
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError:
        return []

    # Step 4: validate that the result is a list of dicts with the required "text" key.
    # A model might return a dict instead of a list, or a list of strings — reject those.
    if not isinstance(result, list):
        return []

    cleaned = []
    for item in result:
        if isinstance(item, dict) and "text" in item:
            # Fill in optional fields with defaults so downstream code can rely on them.
            cleaned.append({
                "text"    : item.get("text", ""),
                "type"    : item.get("type", "unknown"),
                "entities": item.get("entities", []),
            })

    return cleaned


# Usage inside your evaluation loop:
raw_output = "<whatever your model.generate() returns>"
memories   = parse_memory_output(raw_output)

if not memories:
    # Log the bad output so you can review it after the eval run.
    # Do not raise — let the loop continue.
    print(f"[WARN] Parse failed. Raw output (first 300 chars):\n{raw_output[:300]!r}")
```

If the model consistently outputs prose before the JSON even after fine-tuning, revisit your
system prompt. The prompt in Chapter 11 (Defining the Task: What "Memory Extraction" Means)
contains the phrase "Return a JSON array only — no other text." Make sure that line is present
verbatim and that the assistant turn in every training example starts directly with `[`, not with
any preamble.

---

### E-06 — Pad-token / EOS-token warnings and NaN loss

**Error text you will see**

During setup or the first training step:

```
Setting `pad_token_id` to `eos_token_id`: 151643 for open-end generation.
```

or:

```
UserWarning: `use_cache=True` is incompatible with gradient checkpointing. Setting
`use_cache=False`.
```

or the most alarming version — loss is `nan` from the very first step:

```
{'loss': nan, 'grad_norm': nan, 'learning_rate': 1e-4, 'epoch': 0.01}
```

**Cause**

The pad-token warning itself is usually harmless for inference. For training it can become a
real problem: when EOS and PAD share the same token ID, the model receives contradictory
training signal. Some positions that should mean "stop" are masked as padding, and some that
should be ignored are accidentally included in the loss. On a small dataset this can cause the
loss to spike or go `nan` immediately.

The `use_cache` warning is a separate issue: gradient checkpointing (a memory-saving technique
Unsloth enables automatically) and the KV-cache are mutually exclusive during training. This is
just a warning — Unsloth suppresses it correctly — but if you are building a custom training
loop without Unsloth, you must set `model.config.use_cache = False` manually.

The `nan` loss with a correct pad-token setup almost always means the data itself has a bad row:
an empty assistant turn, a row missing the `content` key, or a JSON field that Python serialized
as `NaN` (a float that is not a number).

**Fix**

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Qwen3-1.7B",
    max_seq_length = 1024,
    load_in_4bit   = True,
)

# Fix 1: assign a dedicated pad token if the tokenizer does not have one.
# Qwen3 ships with a pad token; older Llama-family models often do not.
# This block is safe to run unconditionally — it is a no-op if pad_token already exists.
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model.resize_token_embeddings(len(tokenizer))
    print(f"Added pad token: {tokenizer.pad_token!r} (id {tokenizer.pad_token_id})")
else:
    print(f"Pad token already set: {tokenizer.pad_token!r} (id {tokenizer.pad_token_id})")

# Fix 2: disable the KV-cache during training.
# Unsloth does this automatically; set it explicitly if you see the warning.
model.config.use_cache = False

# Fix 3: if loss is nan, find the bad row before training starts.
# Run this on your dataset after formatting but before passing it to the trainer.
from datasets import load_dataset

dataset = load_dataset("json", data_files="data/train.jsonl", split="train")

bad_rows = []
for i, row in enumerate(dataset):
    messages = row.get("messages", [])
    # Check for empty assistant turns — the most common cause of nan loss.
    for msg in messages:
        if msg.get("role") == "assistant" and not msg.get("content", "").strip():
            bad_rows.append((i, "empty assistant turn"))
            break

if bad_rows:
    print(f"Found {len(bad_rows)} bad rows. First 5:")
    for idx, reason in bad_rows[:5]:
        print(f"  Row {idx}: {reason}")
    print("Fix these rows before training, or filter them out with dataset.filter().")
else:
    print("All rows look valid. NaN loss is likely a learning-rate issue — try 5e-5.")
```

---

### E-07 — Training is very slow (steps per second too low)

**Symptom**

The first step completes in a reasonable time, but subsequent steps take 3–5 minutes each. Or
`steps/sec` reported by the trainer stays below 0.1 throughout. On a T4 GPU with a 1.7B model
and `max_seq_length=1024`, you should expect roughly 2–4 steps per second.

**Cause**

Almost always one of four things:

1. You are training on CPU, not GPU (see E-03).
2. `max_seq_length` is set to something large (4 096 or 8 192) but your actual sequences are
   short — the GPU processes a padded rectangle, and 70% of each rectangle is empty.
3. The data-loading worker count is `0` (the default), so the GPU stalls waiting for the CPU to
   prepare each batch.
4. `gradient_checkpointing` is off, so the model stores all intermediate activations in VRAM,
   leaving less room and forcing the GPU to spill to system RAM.

**Fix**

```python
import torch
import statistics
from datasets import load_dataset

# Step 1: confirm you are on GPU, not CPU.
device = "GPU" if torch.cuda.is_available() else "CPU"
print(f"Training on: {device}")
if device == "CPU":
    print("This is the problem. See E-03 to fix the GPU runtime.")

# Step 2: measure your actual sequence lengths before choosing max_seq_length.
dataset   = load_dataset("json", data_files="data/train.jsonl", split="train")
# Sample up to 500 rows so this check is fast even on large datasets.
sample    = dataset.select(range(min(500, len(dataset))))

lengths = []
for row in sample:
    ids = tokenizer.apply_chat_template(
        row["messages"],
        tokenize              = True,   # return token IDs so we can count them
        add_generation_prompt = False,
    )
    lengths.append(len(ids))

p95 = sorted(lengths)[int(len(lengths) * 0.95)]
print(f"Median token length : {statistics.median(lengths):.0f}")
print(f"95th percentile     : {p95:.0f}")
print(f"Max in sample       : {max(lengths)}")

# Set max_seq_length to the 95th-percentile length rounded up to the next power of 2.
# Examples: p95=680 → use 1024; p95=350 → use 512; p95=1100 → use 2048.
# This eliminates wasted padding from oversized batches.
import math
recommended = 2 ** math.ceil(math.log2(p95))
print(f"\nRecommended max_seq_length: {recommended}")
```

In your `TrainingArguments`, add the workers and enable gradient checkpointing:

```python
from transformers import TrainingArguments

args = TrainingArguments(
    # ... your existing args ...
    dataloader_num_workers      = 4,     # default is 0; 4 keeps the GPU fed
    gradient_checkpointing      = True,  # trades compute for VRAM; Unsloth enables this
    # Unsloth handles gradient_checkpointing internally; set it in FastLanguageModel.get_peft_model
)
```

---

### E-08 — Unsloth import errors

**Error text you will see**

```
ModuleNotFoundError: No module named 'unsloth'
```

or after an apparently successful install:

```
ImportError: cannot import name 'FastLanguageModel' from 'unsloth' (/path/to/unsloth/__init__.py)
```

or on non-NVIDIA hardware:

```
RuntimeError: Unsloth currently only supports NVIDIA GPUs. Please use a CUDA-capable device.
```

**Cause**

`unsloth` is not a simple PyPI package. The version on PyPI is a thin stub that does not include
the compiled CUDA extensions. The real package must be installed from the GitHub source using an
extras specifier that encodes your CUDA version and PyTorch version. Installing `pip install
unsloth` without extras installs the stub, which is why `import unsloth` may succeed but
`from unsloth import FastLanguageModel` fails.

**Fix**

Uninstall the stub first, then install the correct variant:

```bash
pip uninstall unsloth -y

# Google Colab with a T4, L4, or A100 (CUDA 12.x) — use this in a notebook cell:
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Local machine with CUDA 12.1 and PyTorch 2.3:
pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"

# Local machine with CUDA 12.4 and PyTorch 2.4+:
pip install "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git"

# Local machine with CUDA 11.8:
pip install "unsloth[cu118-torch220] @ git+https://github.com/unslothai/unsloth.git"
```

To find your CUDA version to pick the right bracket, run:

```bash
python -c "import torch; print('CUDA:', torch.version.cuda, '| PyTorch:', torch.__version__)"
```

After install, verify:

```python
import unsloth
print("Unsloth version:", unsloth.__version__)

# This is the import used throughout this book. If this line succeeds, you are good.
from unsloth import FastLanguageModel
print("FastLanguageModel imported successfully.")
```

The "only supports NVIDIA GPUs" error means you are on Apple Silicon (M-series) or an AMD GPU.
Unsloth does not support those as of mid-2026. On Apple Silicon, use `mlx-lm` from the `mlx-lm`
package as a replacement for the training loop. The data format, schema, and concepts in this
book are identical — only the `FastLanguageModel` class name and some argument names differ.
Refer to the `mlx-lm` documentation for the equivalent of `from_pretrained` and the trainer
setup.

---

## Common mistakes

**Using `float16` on Ampere GPUs.**
RTX 30xx / 40xx cards and A100s handle `bfloat16` natively. `float16` on these cards causes
intermittent loss spikes and occasional `nan`. Leave `dtype=None` in `from_pretrained` — Unsloth
auto-detects the best precision. If you must set it explicitly, use `torch.bfloat16`.

**Forgetting `FastLanguageModel.get_peft_model(model, ...)`.**
Calling `from_pretrained` loads the base model. Calling `get_peft_model` activates the LoRA
adapters that are actually trained. If you skip `get_peft_model`, training runs on the frozen
base model: loss will decrease (the model is memorizing), but the adapters are never updated and
the fine-tune does nothing useful.

**Evaluating before calling `FastLanguageModel.for_inference(model)`.**
Unsloth uses different internal code paths for training and inference. After training completes,
call `FastLanguageModel.for_inference(model)` before running `model.generate()`. Without it,
generation works but runs 2–3x slower than it should.

**Checking loss but not output.**
A loss of 0.9 after training looks like progress. But the model might still be outputting prose
instead of JSON, or generating memories in the wrong schema. Always run at least 10 evaluation
examples through the full generation pipeline (as covered in Chapter 18 — Did It Actually Work?
Evaluating Memory Extraction) before declaring the fine-tune a success.

**Installing packages in the wrong order.**
PyTorch must be installed before `bitsandbytes`, and both must be installed before `unsloth`. If
you install them in reverse order, CUDA binding detection can silently fail and fall back to CPU,
giving you no error but very slow training. The safe order is: PyTorch → bitsandbytes →
transformers, peft, trl → unsloth.

**Setting `max_new_tokens` too low at evaluation time.**
The memory-extraction output is a JSON array. On a conversation with five memories, the output
can be 300–400 tokens. If `max_new_tokens=128`, the array gets cut off mid-element and
`json.loads` fails. Set `max_new_tokens` to at least 512 for evaluation; 1 024 is safer.

---

## Recap

- **CUDA OOM**: lower `per_device_train_batch_size` to 1, raise `gradient_accumulation_steps`
  to compensate, confirm `load_in_4bit=True`, and keep `max_seq_length` at or below the 95th
  percentile of your actual token lengths.
- **bitsandbytes mismatch**: run `python -c "import torch; print(torch.version.cuda)"` to find
  the version PyTorch sees, then reinstall `bitsandbytes` with the matching `--index-url`.
- **No GPU in Colab**: change the runtime type to GPU, then **restart the session** — a simple
  reconnect is not enough, because libraries are already initialized against CPU.
- **Chat-template errors**: always call `apply_chat_template` with a list of `{"role": ...,
  "content": ...}` dicts using lowercase role values; print the rendered string for one example
  before training to confirm it looks sane.
- **JSON parse failures**: use a defensive parser that strips markdown fences and leading prose,
  returns `[]` on failure rather than raising, and logs bad output for post-hoc review.
- **Pad-token / NaN loss**: add a dedicated pad token if the tokenizer lacks one, set
  `model.config.use_cache = False`, and scan the dataset for empty assistant turns before
  training.
- **Slow training**: confirm GPU is active, measure real sequence lengths and set
  `max_seq_length` to the 95th-percentile value rounded to the next power of 2, and add
  `dataloader_num_workers=4` to `TrainingArguments`.
- **Unsloth import errors**: uninstall the PyPI stub, then reinstall from GitHub using the extras
  bracket that matches your CUDA version and PyTorch version.

## Next

See **Appendix D — Cost, Time, and a Go-Live Checklist** for rough GPU-hour estimates and a
pre-deployment checklist before you serve your fine-tuned memory-extraction model.
