# Appendix B - Project Layout and Command Cheat-Sheet

This appendix is a practical reference. Dog-ear it, print it out, keep it open in a second tab. It answers three questions: where does every file live, in what order do I run things, and what is the exact command for each step?

Nothing in this appendix introduces new concepts. It assembles everything taught across the book into one place you can scan without jumping between chapters.

---

## What you'll learn

- The recommended directory layout for your memory-extraction project
- What every file and folder is for, in one sentence
- The order you run scripts from first install through to a live model
- A copy-pasteable cheat-sheet of every significant command in the book
- A "do the whole thing" sequence you can paste into a fresh terminal

---

## Concepts you need first

No new concepts here. This appendix assumes you have read at least through *Ch15 - Your First Fine-Tune with Unsloth (Full Script)*. The terms used — adapter, JSONL, GGUF, LoRA, system prompt — are all defined in the Glossary (*Appendix A - Glossary of Every Term Used*).

---

## The recommended project layout

Below is the full directory tree for the memory-extraction project as built throughout this book. Every directory and file is named exactly as it is referenced in the chapter code. If you follow this layout, every script runs without path changes.

```
memory-extractor/                ← your project root; cd here before running anything
│
├── data/
│   ├── raw/                     ← unprocessed output from the generation pipeline
│   │   └── memories_raw.jsonl   ← written by scripts/generate.py (Ch13)
│   │
│   ├── splits/                  ← cleaned, validated, split data ready for training
│   │   ├── train.jsonl          ← written by scripts/prepare.py (Ch14)
│   │   ├── val.jsonl            ← written by scripts/prepare.py (Ch14)
│   │   └── test.jsonl           ← written by scripts/prepare.py (Ch14)
│   │
│   └── adapter/                 ← LoRA adapter checkpoints saved during training
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── outputs/                     ← exported model artifacts (created by export scripts)
│   ├── memory-extractor-adapter/        ← best adapter, copied here after training
│   ├── memory-extractor-merged-16bit/   ← merged bfloat16 model (for vLLM / Hub)
│   ├── memory-extractor-merged-4bit/    ← merged 4-bit model (for low-VRAM inference)
│   └── memory-extractor-gguf/          ← GGUF file for Ollama / llama.cpp
│
├── scripts/
│   ├── seeds.py                 ← topic / persona / style lists (Ch13)
│   ├── prompts.py               ← generation prompt builder (Ch13)
│   ├── generate.py              ← teacher-LLM call + response parser (Ch13)
│   ├── judge.py                 ← LLM-as-judge quality filter (Ch13)
│   ├── dedup.py                 ← fingerprint deduplication (Ch13)
│   ├── pipeline.py              ← full data generation pipeline (Ch13)
│   ├── inspect.py               ← eyeball a random sample from a JSONL file (Ch13)
│   ├── prepare.py               ← clean, validate, split data (Ch14)
│   ├── train.py                 ← Unsloth fine-tune script (Ch15)
│   ├── evaluate.py              ← evaluate the fine-tuned model on test.jsonl (Ch18)
│   ├── merge.py                 ← merge adapter → 16-bit or 4-bit model (Ch21)
│   └── export_gguf.py           ← export GGUF for Ollama / llama.cpp (Ch21)
│
├── configs/
│   └── train_config.yaml        ← optional: externalize training hyperparameters (Ch16)
│
├── memory_prompt.py             ← the canonical SYSTEM_PROMPT constant (Ch12, Ch22)
│                                   import this everywhere — never re-type the prompt
│
├── requirements.txt             ← pinned dependencies for the whole project
├── .env                         ← API keys (never commit this file to git)
└── .gitignore                   ← ignores .env, outputs/, __pycache__, *.gguf
```

### Why this layout?

`data/` and `outputs/` are kept strictly separate. `data/` is for training artifacts — things you generate, clean, and split. `outputs/` is for trained model artifacts — things the training script writes. This separation makes it much easier to re-train from scratch without accidentally overwriting your best model.

`scripts/` holds every runnable Python file. All scripts are designed to be run from the project root (the `memory-extractor/` directory), which is why paths like `data/splits/train.jsonl` work without absolute paths.

`memory_prompt.py` sits at the root because it is imported by both training code (`scripts/train.py`) and serving code. It is the single source of truth for the system prompt. See *Ch12 - Data Format: Turning the Task into Training Rows* for why this matters.

---

## The run order

Here is the complete sequence, step by step, from a fresh machine to a working model. Each step names the chapter where it is explained in full.

```
Step 1   Set up environment (Ch8)
         Install Python, CUDA, and the Unsloth stack.

Step 2   Set your API key (Ch13)
         export ANTHROPIC_API_KEY="sk-ant-..."

Step 3   Generate synthetic training data (Ch13)
         python scripts/pipeline.py --target 1000 --output data/raw/memories_raw.jsonl

Step 4   Inspect a sample of generated data (Ch13)
         python scripts/inspect.py data/raw/memories_raw.jsonl

Step 5   Clean, validate, and split the data (Ch14)
         python scripts/prepare.py \
             --input  data/raw/memories_raw.jsonl \
             --outdir data/splits/

Step 6   Run fine-tuning (Ch15)
         python scripts/train.py

Step 7   Evaluate the fine-tuned model (Ch18)
         python scripts/evaluate.py \
             --adapter data/adapter \
             --test    data/splits/test.jsonl

Step 8   Debug if needed (Ch19, Ch20)
         Review evaluation output. Fix data or hyperparameters. Repeat from Step 3 or 6.

Step 9   Merge and export (Ch21)
         python scripts/merge.py          # merge adapter → 16-bit
         python scripts/export_gguf.py    # export GGUF for local use

Step 10  Serve and integrate (Ch22)
         ollama create memory-extractor -f Modelfile
         ollama run memory-extractor
         # OR spin up a vLLM server (see Ch22 for the full server command)
```

---

## The full `memory_prompt.py` (canonical system prompt)

Every script in the book imports the system prompt from this module. Create it once at the project root and never copy-paste the prompt string inline.

```python
# memory_prompt.py
# The canonical system prompt for the memory-extraction task.
# Import SYSTEM_PROMPT from here in every script that calls the model —
# training, evaluation, and serving alike.
# Even a single changed word will degrade output quality; keep this as
# the single source of truth.

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
- Return ONLY a valid JSON array. No explanation, no markdown fences, no extra text."""
```

---

## `requirements.txt`

Pin your dependencies so the environment is reproducible. The versions below are the ones used throughout this book. Check the Unsloth GitHub releases page for a newer pin if you are starting fresh.

```
# requirements.txt
# Install with: pip install -r requirements.txt
# Unsloth must be installed separately first — see Step 1 below.

# Core training stack
transformers>=4.45.0
trl>=0.12.0
peft>=0.13.0
datasets>=3.0.0
bitsandbytes>=0.44.0
accelerate>=1.0.0

# Data generation
anthropic>=0.34.0        # teacher LLM (Claude)

# Optional: environment variable loading
python-dotenv>=1.0.0

# Optional: vLLM serving (Ch22)
# vllm>=0.6.0            # uncomment when you reach Ch22
```

Unsloth is installed separately because it requires a CUDA-version-specific wheel. See Step 1 for the exact command.

---

## `.gitignore`

Put this at your project root before your first `git init`. It keeps large binary files and secrets out of version control.

```
# .gitignore

# API keys — never commit these
.env

# Large model outputs — too big for git
outputs/
data/adapter/

# Python build artifacts
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
venv/

# Jupyter checkpoints
.ipynb_checkpoints/

# GGUF and safetensors are large binary files
*.gguf
*.safetensors

# macOS
.DS_Store
```

---

## The command cheat-sheet

Every command used in the book, grouped by phase. All commands are run from the project root unless noted otherwise.

---

### Phase 0 — Environment setup (Ch8, Ch9)

```bash
# --- Verify Python and CUDA versions BEFORE installing anything ---
# Requires Python 3.10 or 3.11 and CUDA 12.1.
# Unsloth wheels are built for specific Python/CUDA pairs and will fail with a
# "no matching distribution found" error if the wrong versions are installed.
python --version   # must print Python 3.10.x or 3.11.x
nvcc --version     # must show release 12.1
# If either is wrong, install the correct version first, then continue.

# --- Install Unsloth (CUDA 12.1 + PyTorch 2.3) ---
# Check https://github.com/unslothai/unsloth for the latest install command.
# This is the version used in this book:
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

# --- Install the rest of the stack ---
pip install -r requirements.txt

# --- Smoke test: confirm Unsloth + GPU are working ---
python -c "
from unsloth import FastLanguageModel
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('Unsloth import OK')
"
# Expected output: CUDA available: True / GPU: NVIDIA ... / Unsloth import OK

# --- Log into Hugging Face (only needed to push a model to the Hub in Ch21) ---
pip install huggingface_hub
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens
```

---

### Phase 1 — Generate synthetic training data (Ch13)

```bash
# Set your Anthropic API key (or put it in .env and load with python-dotenv)
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# Generate 500 rows (proof-of-concept run, ~$1.50, ~20 minutes)
python scripts/pipeline.py \
    --target 500 \
    --output data/raw/memories_raw.jsonl

# Generate 1,000 rows (a useful first real run, ~$3, ~40 minutes)
python scripts/pipeline.py \
    --target 1000 \
    --output data/raw/memories_raw.jsonl

# Eyeball a random sample of 20 rows before doing anything else
python scripts/inspect.py data/raw/memories_raw.jsonl

# Count how many rows are in the file
wc -l data/raw/memories_raw.jsonl
```

---

### Phase 2 — Clean and split the data (Ch14)

```bash
# Clean, validate schema, and split into train/val/test (80/10/10 by default)
python scripts/prepare.py \
    --input  data/raw/memories_raw.jsonl \
    --outdir data/splits/

# Confirm the split sizes
wc -l data/splits/train.jsonl data/splits/val.jsonl data/splits/test.jsonl

# Eyeball a few rows from the train split to double-check the format
python scripts/inspect.py data/splits/train.jsonl
```

---

### Phase 3 — Fine-tune the model (Ch15, Ch16)

```bash
# Run training (all config is in scripts/train.py at the top of the file)
python scripts/train.py

# Expected output during training (printed every LOGGING_STEPS=10 steps):
#   {'loss': 1.432, 'grad_norm': 0.812, 'learning_rate': 1.8e-04, 'epoch': 0.12}
#   {'loss': 0.783, 'grad_norm': 0.541, 'learning_rate': 1.5e-04, 'epoch': 0.48}
#   ...
# Loss should drop from ~1.4 toward ~0.3-0.6 over 3 epochs.
# If loss is stuck above 1.0 after epoch 1, see Ch19.

# Check that the adapter was saved correctly
python -c "
import pathlib
adapter = pathlib.Path('data/adapter')
for f in sorted(adapter.iterdir()):
    print(f.name, f'({f.stat().st_size / 1_048_576:.1f} MB)')
"
# You should see adapter_config.json, adapter_model.safetensors (~80 MB), tokenizer files.
```

---

### Phase 4 — Evaluate (Ch18)

```bash
# Run evaluation on the held-out test split
python scripts/evaluate.py \
    --adapter data/adapter \
    --test    data/splits/test.jsonl

# Quick manual test: ask the model to extract memories from a single conversation
python - <<'EOF'
from unsloth import FastLanguageModel
from memory_prompt import SYSTEM_PROMPT
import json, torch

model, tokenizer = FastLanguageModel.from_pretrained(
    "data/adapter", max_seq_length=2048, load_in_4bit=True, dtype=None
)
FastLanguageModel.for_inference(model)

conversation = "Jordan: I'm switching from VSCode to Neovim this month. Sam: Bold move — why now? Jordan: Wanted something I control completely."

messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": conversation}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True)
new_ids = outputs[0][inputs["input_ids"].shape[1]:]
result = tokenizer.decode(new_ids, skip_special_tokens=True)
print(json.dumps(json.loads(result), indent=2))
EOF
```

---

### Phase 5 — Export the model (Ch21)

```bash
# --- Merge adapter into a 16-bit model (for vLLM and Hugging Face Hub) ---
python scripts/merge.py
# Writes to: outputs/memory-extractor-merged-16bit/
# Disk size: ~3.4 GB for a 1.7B model, ~8 GB for a 4B model
# Time: ~3-5 minutes on a T4

# --- Export GGUF for Ollama / llama.cpp ---
python scripts/export_gguf.py
# Writes to: outputs/memory-extractor-gguf/
# Disk size: ~1 GB (q4_k_m quantization)
# Time: ~5-15 minutes

# --- Verify the GGUF file exists ---
ls -lh outputs/memory-extractor-gguf/

# --- Push 16-bit model to Hugging Face Hub (optional) ---
python - <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("outputs/memory-extractor-merged-16bit", torch_dtype="auto")
tok   = AutoTokenizer.from_pretrained("outputs/memory-extractor-merged-16bit")
model.push_to_hub("your-username/memory-extractor", private=True)
tok.push_to_hub("your-username/memory-extractor", private=True)
print("Done.")
EOF
```

---

### Phase 6 — Serve and use the model (Ch22)

```bash
# --- Option A: Ollama (easiest, local, CPU or GPU) ---

# Install Ollama: https://ollama.com/download  (macOS: brew install ollama)
# Place your GGUF and Modelfile in the same directory, then:

# IMPORTANT: ollama create must be run from INSIDE outputs/memory-extractor-gguf/
# because the Modelfile uses `FROM ./...` (a relative path).
# Running it from the project root will produce a "file not found" error.
cd outputs/memory-extractor-gguf/
ollama create memory-extractor -f Modelfile
cd ../..   # return to project root

# Test it interactively
ollama run memory-extractor

# Query it via HTTP (Ollama runs on port 11434 by default)
curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "memory-extractor",
    "prompt": "Alex: I drink oat milk, not regular milk. Sam: Noted!",
    "stream": false
  }'

# --- Option B: vLLM server (higher throughput, requires GPU) ---

pip install vllm   # do this once

# Start the vLLM server (serves the OpenAI-compatible API on port 8000)
python -m vllm.entrypoints.openai.api_server \
    --model outputs/memory-extractor-merged-16bit \
    --dtype bfloat16 \
    --port 8000 \
    --served-model-name memory-extractor

# Query the vLLM server with curl
# NOTE: curl cannot import Python modules, so you must paste the full SYSTEM_PROMPT
# string from memory_prompt.py into the "content" field below.
# For anything beyond a one-off test, use the Python snippet further down —
# it imports SYSTEM_PROMPT correctly and is the recommended approach.
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "memory-extractor",
    "messages": [
      {"role": "system", "content": "<paste the full SYSTEM_PROMPT string from memory_prompt.py here>"},
      {"role": "user",   "content": "Alex: I drink oat milk, not regular milk."}
    ],
    "temperature": 0.1,
    "max_tokens": 256
  }'

# Query the vLLM server from Python using the openai library
python - <<'EOF'
from openai import OpenAI
from memory_prompt import SYSTEM_PROMPT
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

def extract_memories(conversation: str) -> list:
    response = client.chat.completions.create(
        model="memory-extractor",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": conversation},
        ],
        temperature=0.1,
        max_tokens=512,
    )
    raw = response.choices[0].message.content
    return json.loads(raw)

memories = extract_memories("Jordan: I always use dark mode. Sam: Same, bright screens hurt my eyes.")
for m in memories:
    print(f"[{m['type']}] {m['text']}")
EOF
```

---

### Utility commands (useful throughout)

```bash
# Count rows in any JSONL file
wc -l data/splits/train.jsonl

# Pretty-print the first row of a JSONL file
head -1 data/splits/train.jsonl | python -m json.tool

# Check free disk space (important before GGUF export — you need ~20 GB headroom)
df -h .

# Check GPU VRAM available
python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB total VRAM')"

# Kill a vLLM or Ollama process running on a port
lsof -ti:8000 | xargs kill -9   # vLLM on 8000
lsof -ti:11434 | xargs kill -9  # Ollama on 11434

# Show the Hugging Face model cache location and size
du -sh ~/.cache/huggingface/

# Delete the HF cache for a specific model (to free disk space)
# Replace "unsloth/Qwen3-8B-bnb-4bit" with the model you want to remove
python -c "
from huggingface_hub import scan_cache_dir
cache = scan_cache_dir()
for repo in cache.repos:
    print(repo.repo_id, f'{repo.size_on_disk_str}')
"
```

---

## "Do the whole thing" — a copy-paste sequence

The sequence below assumes you are starting on a fresh CUDA machine (Colab, RunPod, or local NVIDIA GPU) with nothing installed. It runs end to end: install, generate 500 rows, train, and launch an Ollama server.

Adjust `--target 500` to a larger number once you have confirmed the pipeline works.

```bash
# ── 0. Clone or create your project ─────────────────────────────────────────
mkdir memory-extractor && cd memory-extractor

# ── 1. Install Unsloth + stack ───────────────────────────────────────────────
# Requires Python 3.10 or 3.11 and CUDA 12.1.
# Run these two commands to confirm before proceeding:
#   python --version        # must print 3.10.x or 3.11.x
#   nvcc --version          # must show release 12.1
# If either is wrong, install the correct version before continuing —
# Unsloth wheels are built for specific Python/CUDA pairs and will fail silently otherwise.

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
# NOTE: the line below skips the pinned versions in requirements.txt.
# It is a quick-start shortcut — if you hit import errors or version conflicts,
# copy requirements.txt into the project root and run `pip install -r requirements.txt` instead.
pip install transformers datasets anthropic python-dotenv huggingface_hub

# Smoke test — checks Unsloth import AND GPU availability cleanly
python -c "
from unsloth import FastLanguageModel
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU only — training will not work)')
print('Unsloth import OK')
"

# ── 2. Set your API key ──────────────────────────────────────────────────────
export ANTHROPIC_API_KEY="sk-ant-..."

# ── 3. Generate 500 training rows ────────────────────────────────────────────
# Copy seeds.py, prompts.py, generate.py, judge.py, dedup.py, pipeline.py
# from the book's code listings (Ch13) into scripts/, then:
mkdir -p data/raw data/splits data/adapter outputs
python scripts/pipeline.py --target 500 --output data/raw/memories_raw.jsonl

# Quick sanity check
python scripts/inspect.py data/raw/memories_raw.jsonl

# ── 4. Prepare splits ────────────────────────────────────────────────────────
# Copy prepare.py from Ch14 into scripts/, then:
python scripts/prepare.py \
    --input  data/raw/memories_raw.jsonl \
    --outdir data/splits/
wc -l data/splits/*.jsonl

# ── 5. Fine-tune ─────────────────────────────────────────────────────────────
# Copy train.py from Ch15 into scripts/, then:
python scripts/train.py
# Watch for loss dropping from ~1.4 toward ~0.4 over 3 epochs.

# ── 6. Quick evaluation ──────────────────────────────────────────────────────
python -c "
from unsloth import FastLanguageModel
from memory_prompt import SYSTEM_PROMPT
import json

model, tok = FastLanguageModel.from_pretrained(
    'data/adapter', max_seq_length=2048, load_in_4bit=True, dtype=None)
FastLanguageModel.for_inference(model)

conv = 'Priya: I switched to standing desk three months ago and my back pain is gone. Raj: That is great — which brand?'
msgs = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': conv}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors='pt').to('cuda')
out = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True)
result = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(json.dumps(json.loads(result), indent=2))
"

# ── 7. Export GGUF ────────────────────────────────────────────────────────────
# Copy export_gguf.py from Ch21 into scripts/, then:
python scripts/export_gguf.py
ls -lh outputs/memory-extractor-gguf/

# ── 8. Serve with Ollama ──────────────────────────────────────────────────────
# Install Ollama first: https://ollama.com/download

# First, find the exact GGUF filename that export_gguf.py wrote.
# The name depends on the base model; it always ends in .Q4_K_M.gguf.
ls outputs/memory-extractor-gguf/
# Example output:  memory-extractor-adapter-unsloth.Q4_K_M.gguf
# Copy the filename you see — you will paste it into FROM below.

# Create a Modelfile next to the GGUF.
# IMPORTANT: replace <your-gguf-filename.gguf> with the filename printed by `ls` above.
cat > outputs/memory-extractor-gguf/Modelfile <<'MODELFILE'
FROM ./<your-gguf-filename.gguf>
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
PARAMETER temperature 0.1
PARAMETER num_predict 512
MODELFILE

# Run ollama create from INSIDE outputs/memory-extractor-gguf/.
# The Modelfile uses `FROM ./...` (a relative path), so Ollama must be run
# from the same directory as the GGUF — otherwise it cannot find the file.
cd outputs/memory-extractor-gguf/
ollama create memory-extractor -f Modelfile
cd ../..   # return to project root

ollama run memory-extractor

# You should now be talking to your fine-tuned model.
# Type a conversation and it returns a JSON list of extracted memories.
```

---

## Common mistakes

**Running scripts from the wrong directory.**

All paths in the book assume you are in the project root (`memory-extractor/`). If you `cd scripts/` and run `python train.py`, the relative paths like `data/splits/train.jsonl` will not resolve. Fix: always stay at the root and run scripts as `python scripts/train.py`, not `cd scripts && python train.py`.

**Committing model weights to git.**

A `.safetensors` file for a 4B model is around 8 GB. Git will hang, fail, or corrupt your repository trying to handle it. The `.gitignore` above marks `outputs/` and `data/adapter/` as ignored. If you accidentally `git add`-ed a large file before setting up `.gitignore`, remove it with `git rm --cached path/to/file` before committing.

**Losing the system prompt.**

The system prompt is the invisible connector between training and inference. Losing it — or using a slightly different version — is the most common cause of a trained model appearing broken in production. Keep `memory_prompt.py` at the project root and always import from it. Never re-type the prompt string anywhere else.

**Not running the smoke test before fine-tuning.**

Running `train.py` on a machine where Unsloth is not correctly installed produces cryptic import errors deep into the script. Run the 5-line smoke test in Phase 0 first. It fails fast and clearly if something is wrong with the environment.

**Filling the disk during GGUF export.**

Unsloth's `save_pretrained_gguf()` temporarily holds the full 16-bit model in memory and on disk during conversion. For a 4B model that is roughly 20 GB of scratch space. Check `df -h .` before exporting. If you are on Colab, clear the Colab disk cache first — Colab instances typically have 80–120 GB, but HuggingFace caches can eat most of it.

**Re-quantizing a merged 16-bit model by loading it with `load_in_4bit=True`.**

If you merged the adapter to 16-bit (Path 2 in Ch21) and then load the merged model with `load_in_4bit=True`, you apply a second round of quantization on top of the merge. The result is lower quality than intended. Load the 16-bit merged model with `torch_dtype=torch.bfloat16` and `device_map="auto"` instead, or use the explicit 4-bit merged model you saved with `save_method="merged_4bit_forced"`.

---

## Recap

- Keep `data/` for training artifacts (raw, splits) and `outputs/` for model artifacts (adapter, merged, GGUF) — never mix them.
- `memory_prompt.py` at the project root is the single source of truth for the system prompt. Import it everywhere; never re-type it.
- The run order is always: generate → inspect → prepare → train → evaluate → (iterate) → export → serve.
- The cheat-sheet above has the exact command for every phase. Copy-paste from it; don't reconstruct commands from memory.
- Check disk space before exporting GGUF (need ~20 GB headroom for a 4B model).
- The "do the whole thing" sequence at the end of this appendix is a complete, copy-pasteable path from zero to a running Ollama server.
- When something breaks, the command for each phase is the right starting point for debugging — run just that phase in isolation to narrow the fault.
- *Appendix C - Troubleshooting Common Errors* has environment-specific fixes; *Appendix D - Cost, Time, and a Go-Live Checklist* has the pre-launch checklist.

## Next

*Appendix C - Troubleshooting Common Errors* — a reference for the most common crashes, error messages, and failure modes you will encounter, with concrete fixes for each.
