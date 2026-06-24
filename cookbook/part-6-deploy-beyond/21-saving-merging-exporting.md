# Ch21 - Saving, Merging, and Exporting Your Model

Training is done. The loss curve flattened, evaluation looked good, and you have a LoRA adapter sitting on disk. Now comes the step most tutorials skip: actually turning that adapter into something you can use in the real world.

This chapter covers all four exit paths — keep the adapter lightweight, merge it into a full model, export it for llama.cpp or Ollama, or prepare it for a high-throughput server. By the end, you will know exactly which format to reach for and why, with runnable code for each.

---

## What you'll learn

- The difference between a LoRA adapter and a merged full model, and why it matters
- How to merge your adapter into the base model weights and save the result
- How to push your model to the Hugging Face Hub so it's accessible from any machine
- How to export a GGUF file so you can run the model locally with llama.cpp or Ollama
- How to save a 16-bit merged model for high-throughput serving with vLLM
- What disk size to expect for each format and when to choose each one

---

## Concepts you need first

### The adapter vs. the merged model

When you fine-tuned with Unsloth and LoRA (as covered in *Ch6 - LoRA and QLoRA Without the Math Headache* and *Ch15 - Your First Fine-Tune with Unsloth (Full Script)*), you did not touch the original base model. You trained two small matrices per layer — the "sticky notes." Those sticky notes are stored separately as the **LoRA adapter**.

The adapter is small. For a 1.7B parameter model with `r=16`, the adapter might be **40–80 MB**. The base model it sits on is a separate download, typically **1–4 GB** at 4-bit.

At inference time, Unsloth or PEFT loads the base model and the adapter together and applies the adapter math on the fly. This works fine but has one catch: the inference library has to know about LoRA. Not all serving tools do.

**Merging** takes the adapter math and bakes it permanently into the base model weights, producing one self-contained model file. The merged model is larger — it contains the full weights — but it looks like any ordinary model to any tool that can load a Hugging Face model. No adapter awareness required.

Think of it as printing a revised edition of the textbook instead of handing someone the textbook plus a packet of sticky notes.

### The four output formats

| Format | File size (1.7B model) | Best for |
|---|---|---|
| LoRA adapter only | ~40–80 MB | Fast iteration, keeping multiple fine-tunes on one base model |
| Merged 16-bit (`bfloat16`) | ~3.4 GB | vLLM serving, Hugging Face Hub, highest quality |
| Merged 4-bit (QLoRA weights) | ~1.1 GB | Low-VRAM inference with the Unsloth/transformers stack |
| GGUF (quantized) | ~0.6–1.8 GB depending on quant level | llama.cpp, Ollama, CPU inference, Mac deployment |

**What is `bfloat16`?** It is a 16-bit floating-point number format used by modern GPUs (the "b" stands for "Brain" — it was developed at Google Brain). Every weight in a neural network is stored as a number; the format you choose controls how precisely that number is stored and how much memory it takes up. `float32` (the default on most computers) uses 4 bytes per weight and is very precise. `bfloat16` uses 2 bytes per weight — half the space — while keeping the same *range* of values as float32, just with less decimal precision. That is an acceptable tradeoff for inference: the model behaves almost identically but uses half the disk and half the VRAM. This is why the merged 16-bit model is ~3.4 GB instead of ~6.8 GB (what a float32 version of the same model would be).

---

## Before you export: what you should already have

By the end of *Ch15 - Your First Fine-Tune with Unsloth (Full Script)*, your training script called `trainer.save_model("outputs/memory-extractor-adapter")`. That folder contains the LoRA adapter — not a runnable model, just the delta. Let's verify it's all there before we do anything with it.

```python
# verify_adapter.py
# Run this after training to confirm the adapter files are present.
# This is a sanity check, not a required step.

import os
import pathlib

ADAPTER_DIR = "outputs/memory-extractor-adapter"

# These are the files Unsloth saves for a LoRA adapter
EXPECTED_FILES = [
    "adapter_config.json",   # describes the LoRA architecture (r, alpha, target modules)
    "adapter_model.safetensors",  # the actual trained weights
    "tokenizer.json",        # the tokenizer (saved alongside the adapter for convenience)
    "tokenizer_config.json",
    "special_tokens_map.json",
]

path = pathlib.Path(ADAPTER_DIR)

if not path.exists():
    print(f"ERROR: adapter directory '{ADAPTER_DIR}' does not exist.")
    print("Make sure training completed and trainer.save_model() ran successfully.")
else:
    print(f"Adapter directory found: {ADAPTER_DIR}")
    for fname in EXPECTED_FILES:
        fpath = path / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1_048_576
            print(f"  OK  {fname}  ({size_mb:.1f} MB)")
        else:
            print(f"  MISSING  {fname}")
```

Run this before any export step. If `adapter_model.safetensors` is missing, training did not finish cleanly — go back and re-run the training script.

---

## Path 1 — Keep just the adapter (no merge)

If you plan to keep iterating — maybe training a second version tomorrow, or running multiple fine-tunes on the same base model — you do not need to merge anything yet. The adapter folder from `trainer.save_model()` is the artifact. Load it like this at inference time:

```python
# load_adapter_inference.py
# Use when you want to run the fine-tuned model WITHOUT merging the adapter.
# Requires Unsloth (or PEFT) at inference time.

from unsloth import FastLanguageModel   # handles both base model + adapter math
import json

# Unsloth's FastLanguageModel.from_pretrained can load a saved adapter directly.
# It re-downloads the base model from the Hub if it's not cached locally.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/memory-extractor-adapter",  # path to adapter folder
    max_seq_length=2048,
    load_in_4bit=True,   # keeps VRAM low at inference time
    dtype=None,          # auto-detect
)

# Switch to inference mode (disables dropout, speeds up generation)
FastLanguageModel.for_inference(model)

# The system prompt must match what you used during training exactly.
# See Ch12 for the canonical SYSTEM_PROMPT string.
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

conversation = """
Dana: I've decided to go with PostgreSQL instead of MongoDB for the new service.
Chris: Makes sense. Are you handling the migration yourself?
Dana: Yeah, I'll start next Monday. The data team needs a week's heads-up first.
Chris: Noted — I'll tell them today.
"""

# Build the prompt in the same three-turn format used during training
messages = [
    {"role": "system",    "content": SYSTEM_PROMPT},
    {"role": "user",      "content": conversation.strip()},
]

# Apply the model's chat template to get the properly formatted input string
input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,  # appends the assistant turn opener so the model knows to continue
)

# Tokenize and move to GPU
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Generate — low temperature for deterministic structured output
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.1,    # near-zero = more deterministic, better for JSON
    do_sample=True,     # required when temperature < 1.0
)

# Decode only the new tokens (not the prompt we fed in)
generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
result_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

# Parse the JSON output
try:
    memories = json.loads(result_text.strip())
    for m in memories:
        print(f"[{m['type'].upper()}] {m['text']}")
        print(f"  entities: {', '.join(m['entities'])}\n")
except json.JSONDecodeError:
    # If parsing fails, see Ch19 - When It Goes Wrong for the debugging playbook
    print("Raw output (failed to parse as JSON):")
    print(result_text)
```

**When to use this:** during development, evaluation (Ch18), and debugging (Ch19). Keep the adapter format until you are satisfied with quality.

---

## Path 2 — Merge the adapter into the base model

Once you are happy with the model, merge the adapter weights permanently into the base model. This produces a standard Hugging Face model that any tool can load without knowing about LoRA.

Unsloth provides `save_pretrained_merged()` specifically for this. It handles the merge math and saves the result in one call.

```python
# merge_and_save.py
# Merges the LoRA adapter into the base model weights and saves the result.
# Run this once when you're ready to ship. Takes ~2-5 minutes on a T4 GPU.

from unsloth import FastLanguageModel

# Load the adapter (same call as for inference — Unsloth knows it's an adapter)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/memory-extractor-adapter",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)

# ── Merge + save as 16-bit (bfloat16) ───────────────────────────────────────
# This is the highest-quality format. Use this for:
#   - Uploading to the Hugging Face Hub
#   - vLLM serving (Ch22 covers this)
#   - Situations where you want the sharpest possible output
#
# Disk size: ~3.4 GB for a 1.7B model, ~8 GB for a 4B model
# VRAM needed to load: ~3.4 GB (1.7B) or ~8 GB (4B) — no quantization

model.save_pretrained_merged(
    "outputs/memory-extractor-merged-16bit",  # directory to write to
    tokenizer,                                 # save tokenizer alongside the model
    save_method="merged_16bit",                # bfloat16, full precision
)

print("16-bit merge saved to outputs/memory-extractor-merged-16bit")
print("Directory contents:")
import os
for f in sorted(os.listdir("outputs/memory-extractor-merged-16bit")):
    size = os.path.getsize(f"outputs/memory-extractor-merged-16bit/{f}") / 1_048_576
    print(f"  {f}  ({size:.1f} MB)")
```

After this runs, `outputs/memory-extractor-merged-16bit/` is a complete Hugging Face model directory. You can load it with any standard `transformers` or vLLM call — no Unsloth required.

### What if you need to keep VRAM low on the merged model?

You can also merge and save in 4-bit, which keeps the merged model small enough to load on a 4 GB GPU:

```python
# Merge + save as 4-bit (LoRA weights baked into quantized base)
# Use this when you will load and run the model with Unsloth/transformers,
# but don't need the full 16-bit quality or are VRAM-constrained.
#
# Disk size: ~1.1 GB for a 1.7B model
# Quality: slightly lower than 16-bit, but usually negligible for our task

model.save_pretrained_merged(
    "outputs/memory-extractor-merged-4bit",
    tokenizer,
    save_method="merged_4bit_forced",   # forces 4-bit quantization on the merged weights
)

print("4-bit merge saved to outputs/memory-extractor-merged-4bit")
```

---

## Path 3 — Push to the Hugging Face Hub

The Hugging Face Hub is the standard place to store and share models. Even if you're not sharing publicly, it's useful as versioned cloud storage that any machine can pull from with a one-liner.

```bash
# Install the Hub CLI if you haven't already
pip install huggingface_hub

# Log in with your Hugging Face token
# Get your token at https://huggingface.co/settings/tokens
huggingface-cli login
```

Then push directly from Python:

```python
# push_to_hub.py
# Pushes the merged 16-bit model to your Hugging Face Hub repository.
# Requires: huggingface-cli login (see above)

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the merged model we just saved
# We use standard transformers here — no Unsloth needed once it's merged
model = AutoModelForCausalLM.from_pretrained(
    "outputs/memory-extractor-merged-16bit",
    torch_dtype="auto",   # loads in the dtype it was saved in (bfloat16)
)
tokenizer = AutoTokenizer.from_pretrained(
    "outputs/memory-extractor-merged-16bit"
)

# Replace "your-username" with your actual Hugging Face username
HUB_REPO = "your-username/memory-extractor-qwen3-1.7b"

# push_to_hub uploads all model files and creates the repo if it doesn't exist
# private=True keeps it invisible to the public — set False to share openly
model.push_to_hub(HUB_REPO, private=True)
tokenizer.push_to_hub(HUB_REPO, private=True)

print(f"Model pushed to: https://huggingface.co/{HUB_REPO}")
```

Upload time varies by connection speed. A 3.4 GB model typically takes 5–15 minutes on a cloud instance with fast egress. Once it's on the Hub, any machine can load it with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/memory-extractor-qwen3-1.7b")
tokenizer = AutoTokenizer.from_pretrained("your-username/memory-extractor-qwen3-1.7b")
```

### Pushing the adapter only (smaller upload)

If you want to upload just the adapter instead of the full merged model, you can push from the adapter directory directly. This is only ~80 MB:

```python
# push_adapter_to_hub.py
# Pushes only the LoRA adapter — much smaller upload.
# Anyone who downloads this also needs to download the base model separately.

from transformers import AutoTokenizer

# The adapter folder from trainer.save_model() is already Hub-compatible
tokenizer = AutoTokenizer.from_pretrained("outputs/memory-extractor-adapter")
tokenizer.push_to_hub("your-username/memory-extractor-adapter", private=True)

# Push the adapter weights and config
import shutil
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="outputs/memory-extractor-adapter",
    repo_id="your-username/memory-extractor-adapter",
    repo_type="model",
    private=True,
)

print("Adapter pushed to Hub (base model must be downloaded separately to use).")
```

---

## Path 4 — Export to GGUF for llama.cpp and Ollama

GGUF is the file format used by [llama.cpp](https://github.com/ggerganov/llama.cpp) and [Ollama](https://ollama.com). These tools can run a model using only a CPU (though a GPU helps), which makes them popular for Mac users, edge devices, and anyone who can't or won't pay for cloud GPU inference.

Unsloth's `save_pretrained_gguf()` handles the conversion. It calls llama.cpp's conversion toolchain under the hood — you don't need to install llama.cpp yourself.

### Understanding GGUF quantization levels

GGUF files are always quantized (compressed). The quantization level trades file size and memory against output quality:

| Quantization | Disk size (1.7B model) | Quality notes | Recommended for |
|---|---|---|---|
| `q2_k` | ~0.6 GB | Noticeably lower quality | Extremely constrained storage |
| `q4_k_m` | ~1.0 GB | Good quality, 4-bit | Most use cases — the sweet spot |
| `q5_k_m` | ~1.2 GB | Very good, 5-bit | When you want to push quality |
| `q8_0` | ~1.8 GB | Near-lossless | When quality matters most and space doesn't |
| `f16` | ~3.4 GB | Full 16-bit, no compression | Rare — only if you need exact match to merged model |

For our memory-extraction task, `q4_k_m` is the right default. JSON extraction is not sensitive enough to quality drops to justify the extra size of q8.

```python
# export_gguf.py
# Exports the fine-tuned model to GGUF format for llama.cpp and Ollama.
# Run this after training, from the same session or by reloading the adapter.

from unsloth import FastLanguageModel

# Load the adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/memory-extractor-adapter",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)

# ── Export q4_k_m GGUF ──────────────────────────────────────────────────────
# This is the recommended default: good quality, ~1 GB, runs on most hardware.
# The conversion takes 5–15 minutes on a T4 GPU.

# WARNING: GGUF conversion temporarily holds a 16-bit copy of the model in
# memory/on disk during conversion. For a 1.7B model you need ~8 GB of free
# disk. Check before starting:
import shutil
print(shutil.disk_usage('.').free // 1_073_741_824, 'GB free')
# If this prints less than 8, free up space before continuing.

model.save_pretrained_gguf(
    "outputs/memory-extractor-gguf",    # output directory
    tokenizer,
    quantization_method="q4_k_m",       # quantization level (see table above)
)

# This produces a single file:
#   outputs/memory-extractor-gguf/memory-extractor-adapter-unsloth.Q4_K_M.gguf
# (filename is auto-generated from the adapter directory name)

print("GGUF export complete.")
print("File(s) in outputs/memory-extractor-gguf/:")
import os
for f in os.listdir("outputs/memory-extractor-gguf"):
    size_mb = os.path.getsize(f"outputs/memory-extractor-gguf/{f}") / 1_048_576
    print(f"  {f}  ({size_mb:.1f} MB)")
```

### Export multiple quantization levels in one pass

If you want to offer your model at multiple quality tiers (say, both q4 and q8), you can export them back-to-back in the same script:

```python
# export_gguf_multiple.py
# Exports two GGUF quantization levels in one pass.
# Useful for benchmarking quality vs. size before deciding which to ship.

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/memory-extractor-adapter",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)

# Export the recommended quality tier
model.save_pretrained_gguf(
    "outputs/memory-extractor-gguf-q4",
    tokenizer,
    quantization_method="q4_k_m",
)

# Export the near-lossless quality tier for comparison
# Note: the model object can be reused for multiple exports in the same session
model.save_pretrained_gguf(
    "outputs/memory-extractor-gguf-q8",
    tokenizer,
    quantization_method="q8_0",
)

print("Both GGUF exports complete.")
```

### You can also push GGUF directly to the Hub

```python
# push_gguf_to_hub.py
# Pushes GGUF files to Hugging Face Hub.
# Hub hosts GGUF models and Ollama can pull from Hub directly.
#
# NOTE: Run this in the SAME Python session immediately after the GGUF export
# above — model and tokenizer are still in memory. If you start a fresh
# session, reload them first:
#   from unsloth import FastLanguageModel
#   model, tokenizer = FastLanguageModel.from_pretrained(
#       "outputs/memory-extractor-adapter", max_seq_length=2048,
#       load_in_4bit=True, dtype=None)

model.push_to_hub_gguf(
    "your-username/memory-extractor-gguf",   # Hub repo name
    tokenizer,
    quantization_method="q4_k_m",
    private=True,
)

print("GGUF pushed to Hub.")
```

### Running the GGUF locally with Ollama

**Install Ollama first** — download the one-click installer for your OS at https://ollama.com/download. On Mac you can also use `brew install ollama`. Confirm it is running before continuing:

```bash
ollama list
# If Ollama is running, this prints a (possibly empty) list of local models.
# If it fails with "connection refused", start the Ollama app or run: ollama serve
```

Once Ollama is running and you have a GGUF file, running it requires two steps. First, create a `Modelfile` that wraps the GGUF:

```
# Modelfile
# Save this as "Modelfile" (no extension) in the same directory as the GGUF.

FROM ./memory-extractor-adapter-unsloth.Q4_K_M.gguf

# The system prompt from training — must match exactly (see Ch12 for the canonical version)
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
- Return ONLY a valid JSON array. No explanation, no markdown fences, no extra text."""

# Keep temperature low for structured output
PARAMETER temperature 0.1
PARAMETER num_predict 512
```

Then create and run the model:

```bash
# In the directory containing the GGUF and the Modelfile:

# Register the model with Ollama
ollama create memory-extractor -f Modelfile

# Run it interactively to test
ollama run memory-extractor

# Or query it via Ollama's REST API (runs on port 11434 by default)
curl http://localhost:11434/api/generate \
  -d '{"model": "memory-extractor", "prompt": "Alex: I drink two coffees every morning before 9am.", "stream": false}'
```

---

## Path 5 — 16-bit merge for vLLM

vLLM (covered in *Ch22 - Serving Your Model and Using It in an App*) is a high-throughput inference server that batches requests efficiently. It needs a clean 16-bit Hugging Face model — not an adapter, not a GGUF. You already created this in Path 2:

```python
# The 16-bit merge from Path 2 is exactly what vLLM needs.
# No additional export step required.

# Verify the model loads correctly with standard transformers before handing it to vLLM:
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "outputs/memory-extractor-merged-16bit",
    torch_dtype=torch.bfloat16,   # load in bfloat16 to match what we saved
    device_map="auto",            # spread across available GPUs if you have more than one
)
tokenizer = AutoTokenizer.from_pretrained("outputs/memory-extractor-merged-16bit")

print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("Ready to hand to vLLM.")
```

If this loads cleanly, the directory is valid for vLLM. See Ch22 for the full server setup.

---

## Choosing the right export path

Here is the decision tree in plain terms:

**Still iterating on the model?** Keep the adapter. Don't merge yet. Merging does not delete your adapter files — they stay in `outputs/memory-extractor-adapter/`. But the merged model itself cannot be split back into adapter + base. Keep the adapter folder until you are done iterating, then merge once.

**Want to share on the Hub or use with vLLM?** Use the 16-bit merge (`save_pretrained_merged` with `"merged_16bit"`). This is the most universally compatible format.

**Running locally on a Mac, a laptop, or a CPU-only machine?** Export to GGUF with `q4_k_m`. Use Ollama for the simplest experience.

**VRAM is tight (under 4 GB) and you are staying in the Unsloth/transformers stack?** Use the 4-bit merge (`"merged_4bit_forced"`).

**Need to run thousands of requests per minute?** Use the 16-bit merge, push to Hub, and set up vLLM (Ch22).

---

## Common mistakes

**Mistake: Merging before you're done evaluating.**

Merging is not destructive — the adapter files are still there — but it adds 5–10 minutes of work and ~3 GB of disk per export. Do not merge until evaluation (Ch18) and debugging (Ch19) are complete.

**Mistake: Forgetting to save the system prompt alongside the model.**

The merged model has no memory of the training system prompt. It's just weights. If you don't write the system prompt down somewhere (a `README.md` in the model directory, a config file, a constant in your code), you will eventually forget what it was, and the model's output quality will mysteriously drop when you use a different prompt. Save it explicitly:

```python
# save_system_prompt.py
# Save the system prompt alongside the model so it's never lost.

import pathlib, json

SYSTEM_PROMPT = """You are a memory extraction assistant..."""  # your full prompt

output_dir = pathlib.Path("outputs/memory-extractor-merged-16bit")

# Save as a JSON file next to the model weights
config = {"system_prompt": SYSTEM_PROMPT, "model_task": "memory-extraction"}
(output_dir / "inference_config.json").write_text(
    json.dumps(config, indent=2, ensure_ascii=False)
)

print("System prompt saved to inference_config.json")
```

**Mistake: Using `q2_k` GGUF for JSON tasks.**

`q2_k` is the most aggressive quantization level. On prose tasks it's survivable. On structured JSON output, it often produces broken output — missing closing brackets, wrong field names. Stick with `q4_k_m` or higher for any task that requires precise output format.

**Mistake: Pushing large model files to a regular Git repo.**

The Hub uses Git LFS (Large File Storage) for model weights. `push_to_hub()` handles this automatically. But if you try to commit a 3.4 GB `.safetensors` file to a plain Git repo (not the Hub), it will fail or corrupt the repo. Always use the Hub API or `push_to_hub()` for model files.

**Mistake: Calling `save_pretrained_gguf` without enough disk space.**

The GGUF export temporarily holds the model in memory in 16-bit form during conversion, then writes the quantized GGUF. For a 1.7B model you need about 8 GB of free disk. For a 4B model, plan for ~20 GB of headroom. Check with `df -h .` before starting.

**Mistake: Loading the merged model with `load_in_4bit=True` and wondering why it's different from the adapter.**

If you merged to 16-bit and then load with `load_in_4bit=True`, you're re-quantizing the merged weights at load time. The result is not the same as the original QLoRA training quantization — it's a second round of precision loss. For production inference, either load the 16-bit merged model in bfloat16, or use the explicit 4-bit merged model you saved with `"merged_4bit_forced"`.

---

## Recap

- Training with LoRA produces a small **adapter** (40–80 MB) that sits on top of the unchanged base model. The adapter alone is enough for continued development and evaluation.
- **Merging** bakes the adapter math permanently into the base model weights, producing a standalone model compatible with any tool. Use `save_pretrained_merged("...", save_method="merged_16bit")` for the highest-quality result (~3.4 GB for a 1.7B model).
- Push to the **Hugging Face Hub** with `push_to_hub()` for versioned cloud storage and easy cross-machine access.
- **GGUF** export (via `save_pretrained_gguf`) produces a file for llama.cpp and Ollama. The `q4_k_m` quantization is the recommended default — good quality at ~1 GB, runs on CPUs and Macs.
- A **16-bit merged model** is what vLLM needs for high-throughput serving.
- Always save the system prompt alongside the model — the weights carry the behavior, but not the instructions.
- Choose your format based on what you are doing next: adapter for iteration, 16-bit for vLLM and Hub, GGUF for local and CPU inference.

## Next

*Ch22 - Serving Your Model and Using It in an App* — now that the model is exported, we spin up a local inference server with vLLM, call it from Python, and integrate memory extraction into a small end-to-end app.
