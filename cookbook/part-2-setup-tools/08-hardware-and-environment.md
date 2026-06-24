# Ch8 - Hardware, GPUs, and Setting Up Your Environment

You've spent the last few chapters learning *what* fine-tuning is and *why* it works. Now it's time to answer the question every developer asks at this point: "OK, but what computer do I actually need?"

The honest answer is: probably not the one you own — but also not anything exotic. This chapter gives you three concrete paths to get a working GPU environment, then walks you through an exact setup you can copy line-for-line.

---

## What you'll learn

- What GPU VRAM is, and exactly how much you need for 4-bit fine-tuning
- Three ready-to-use paths: free (Colab/Kaggle), rented cloud GPU (RunPod), and local NVIDIA
- Why this book targets CUDA, and what to do if you're on Apple Silicon
- A pinned, copy-pasteable install command for the full Unsloth stack
- A 5-line smoke test that confirms everything works before you touch training code

---

## Concepts you need first

### VRAM: the one number that controls everything

VRAM stands for Video RAM — the memory built into a GPU. It is completely separate from your laptop's regular RAM. When you load a model onto a GPU, it lives in VRAM. When you train, the gradients and optimizer states also live in VRAM. If you run out, the process crashes with a cryptic `CUDA out of memory` error.

**The Pareto version:** Think of VRAM like a workbench. The model is your project — it has to fit on the bench entirely. Larger models need a wider bench. With 4-bit quantization (explained in Ch6 — *LoRA and QLoRA Without the Math Headache*), you compress the model to roughly one-quarter of its original size, which lets you use a much narrower bench.

### What we are training for

> **Running project anchor:** Throughout this book, the hands-on example is a *memory-extraction model* — a fine-tuned LLM that reads a piece of text and outputs structured JSON identifying what is worth remembering. Every extracted memory has three fields: `text` (the remembered fact as a string), `type` (a category such as `"preference"` or `"relationship"`), and `entities` (a list of people, places, or things the fact is about). The shared schema looks like `{"text": "...", "type": "...", "entities": [...]}`. This chapter gets your GPU environment ready so that, by the end of Part 3, you can build the training data, run the fine-tune, and evaluate a model that produces exactly this output reliably.

Here is a practical table for 4-bit fine-tuning with LoRA, which is exactly what this book does:

| VRAM | What you can fine-tune |
|------|------------------------|
| 6 GB | Qwen3-1.7B or Gemma-3-1B (tight, but works) |
| 12 GB | Qwen3-4B, Gemma-3-4B (comfortable) |
| 16 GB | Qwen3-8B, Gemma-3-12B (recommended sweet spot) |
| 24 GB | Qwen3-14B, Gemma-3-27B (generous headroom) |
| 40 GB+ | Large models, bigger batches, fewer compromises |

The models we'll actually use in Ch15 (*Your First Fine-Tune with Unsloth*) are in the 4B–8B range, so **16 GB VRAM is the comfortable target.** You can get by on 12 GB with small batch sizes.

### CUDA: one paragraph and done

CUDA is NVIDIA's software layer that lets programs talk to NVIDIA GPUs. Every deep-learning library — PyTorch, Unsloth, everything — calls CUDA under the hood. You don't write CUDA yourself; you just need the right version installed. When people say "CUDA 12.1" or "CUDA 12.4" they mean the version of this software layer. The install commands below handle this automatically in each path.

### Why not Apple Silicon (M1/M2/M3/M4)?

Apple Silicon chips have a technology called MLX that can run and even fine-tune some models. It is genuinely impressive hardware. However, the entire Unsloth ecosystem — which gives us the 2–3x training speed and the memory optimizations this book relies on — is built on CUDA and does not run on Apple Silicon. AMD GPUs have similar friction. **This book targets NVIDIA GPUs with CUDA.** If you have a Mac, use Path 1 (Colab) or Path 2 (RunPod) below. Your Mac is a perfectly good terminal for SSH-ing into a cloud machine.

---

## Three paths to a GPU

### Path 1 — Free: Google Colab or Kaggle (recommended for beginners)

Both platforms give you a free NVIDIA GPU with no setup at all. You just open a notebook in your browser.

**Google Colab** (colab.research.google.com)
- Free tier: T4 GPU, ~16 GB VRAM, sessions up to ~4 hours
- Colab Pro ($10/month): more GPU time, V100/A100 access
- Best for: getting started quickly, shorter training runs

**Kaggle Notebooks** (kaggle.com/code)
- Free tier: T4 or P100, 30 GPU hours per week, sessions up to 12 hours
- No cost tier required
- Best for: longer runs on the free tier since 12-hour sessions beat Colab's 4-hour limit

**Enabling the GPU:**

- **Google Colab:** Click **Runtime → Change runtime type → Hardware accelerator → T4 GPU**, then save. (Older Colab UI shows this as just "Runtime → Change runtime type → GPU (T4)" — same destination either way.)
- **Kaggle:** Click the **Settings** panel on the right side of the notebook editor → **Accelerator → GPU T4**. (Kaggle does not have a Runtime menu — look for the Settings cog/panel on the right.) Save the setting and the session will restart with a GPU.

Once the GPU is enabled, skip ahead to *Installing the stack* below; the pip install command is the same. You can skip the virtual-environment step (Step 1) — Colab and Kaggle sessions are already isolated containers.

> **Approximate cost:** $0. Approximate setup time: 2 minutes.

---

### Path 2 — Rented cloud GPU: RunPod (recommended for serious training)

RunPod (runpod.io) is a GPU marketplace where you rent a machine by the hour, pay only when it's running, and delete it when you're done. It is the most cost-effective path for runs longer than a few hours.

**Step-by-step to a running instance:**

1. Create a free account at runpod.io.

2. Click **Deploy** → **GPU Pods** → **+ Deploy**.

3. In the GPU picker, filter to **RTX 3090** (24 GB, ~$0.44/hr) or **RTX 4090** (24 GB, ~$0.74/hr). Both have plenty of VRAM for our work.

4. For the **container image**, use:
   ```
   runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
   ```
   This image comes with Python 3.11, PyTorch 2.4, and CUDA 12.4 already installed — the right base for Unsloth.

5. Set **Container Disk** to at least 30 GB (models are large). Leave everything else at defaults.

6. Click **Deploy On-Demand**. The pod starts in about 60–90 seconds.

7. Click **Connect → Start Web Terminal** (or use the SSH command RunPod shows you). You land at a bash prompt inside the machine.

**Once you are in the shell:** You are logged in as root inside a Linux container — no `sudo` needed for any command. The image already has Python 3.11 installed at `/usr/bin/python3`. Unlike Colab/Kaggle, you should still create the virtual environment in Step 1 below so your packages stay isolated from any system libraries that come with the image. Run all commands from the bash prompt you landed in — both the web terminal and SSH drop you in the same environment, so either works. Then continue with *Installing the stack* below.

> **Approximate cost:** RTX 3090 at ~$0.44/hr. A typical training run for our memory extraction model takes 30–90 minutes, so a full session including data prep is roughly **$0.50–$2.00**. Stop the pod when not in use; delete it when you're done with the chapter.

---

### Path 3 — Local NVIDIA GPU

If your desktop or workstation has an NVIDIA GPU with 12 GB or more VRAM (RTX 3060 12 GB, 3080, 3090, 4070, 4080, 4090, or any datacenter card), you can train locally.

**Prerequisites:**
- NVIDIA driver version 525 or newer (check with `nvidia-smi`)
- Ubuntu 20.04 / 22.04, or Windows with WSL2 (WSL2 is preferred — Unsloth works best on Linux)
- Python 3.10 or 3.11

The install steps below are identical to RunPod. The only difference is that you are already on your own machine.

> **Approximate cost:** $0 in cash, but your electricity bill. A typical training run draws 200–300 W for ~1 hour, which is pennies.

---

## Installing the stack

These commands work on all three paths. Run them once per environment.

### Step 1 — Create an isolated environment

A virtual environment keeps the Unsloth packages from colliding with anything else on the machine.

```bash
# Create a fresh Python 3.11 environment called "finetune"
python3 -m venv finetune

# Activate it (do this every time you open a new terminal)
source finetune/bin/activate

# Verify you're inside it — the prompt should show (finetune)
which python  # should print something ending in finetune/bin/python
```

> **On Colab or Kaggle:** Skip the venv — you're already in an isolated container. Just run the pip install in the next step directly in a notebook cell.

### Step 2 — Install the full stack (one command)

This single line installs Unsloth, PyTorch, the Hugging Face ecosystem, and bitsandbytes (the library that handles 4-bit quantization). The version pins ensure these packages work together; mixing random versions is the most common source of cryptic errors.

```bash
# Install everything. This takes 3–8 minutes on a fresh machine.
# The --no-deps flag on unsloth is intentional: we're managing deps ourselves
# to avoid version conflicts.
pip install \
  "unsloth @ git+https://github.com/unslothai/unsloth.git@2025.1.6" \
  "torch==2.4.0" \
  "transformers==4.46.3" \
  "trl==0.12.1" \
  "peft==0.13.2" \
  "datasets==3.1.0" \
  "bitsandbytes==0.44.1" \
  "accelerate==1.1.1" \
  "huggingface_hub>=0.26.0" \
  "sentencepiece" \
  "protobuf"
```

Why these libraries?

| Library | What it does | Why not alternatives |
|---------|-------------|----------------------|
| **unsloth** | Speeds up training 2–3x, patches memory use | Nothing else does this automatically for Qwen3/Gemma |
| **torch** | The math engine everything runs on | PyTorch dominates research/production; JAX is the main alternative but has a steeper API |
| **transformers** | Loads models, tokenizers, chat templates | The universal standard; used by every model on Hugging Face |
| **trl** | Training loop for instruction/chat fine-tuning | SFT stands for *Supervised Fine-Tuning* — the standard technique of training a model on labeled input/output pairs, which is exactly what we do when we show it example memory extractions. `SFTTrainer` is TRL's ready-made class that handles this; it is the easiest correct way to do SFT without writing a training loop from scratch |
| **peft** | Implements LoRA adapters | The reference implementation; works with TRL out of the box |
| **datasets** | Loads and preprocesses training data | Handles streaming, sharding, arrow caching efficiently |
| **bitsandbytes** | Runs 4-bit/8-bit quantization on CUDA | The only mature CUDA quantization library for training (not just inference) |
| **accelerate** | Handles device placement and mixed precision | Required internally by TRL; you rarely call it directly |

### Step 3 — Smoke test

Run this script. It should complete in under 30 seconds and print your GPU information plus a confirmation that a tiny model loaded correctly. If it crashes, you'll know before wasting time on training.

```python
# smoke_test.py
# Run this after installing. It checks GPU visibility and model loading.
# Expected output: your GPU name, VRAM amount, and "Smoke test passed."

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. Check that CUDA is available ---
# torch.cuda.is_available() returns True if PyTorch can see an NVIDIA GPU.
# If this prints False, your CUDA install is broken — see Common Mistakes below.
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("No GPU found. Check your environment setup.")

# --- 2. Print GPU name and total VRAM ---
# device_name gives the GPU model (e.g. "NVIDIA GeForce RTX 4090")
# get_device_properties gives detailed specs including total memory
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu_name}")
print(f"VRAM: {vram_gb:.1f} GB")

# --- 3. Load a tiny model to confirm transformers + bitsandbytes work ---
# We use a small 135M-parameter model just to test loading — not our real model.
# load_in_4bit=True exercises the bitsandbytes quantization path.
print("\nLoading a tiny model to test quantization...")

model_name = "Qwen/Qwen3-0.6B"  # ~0.6B params, smallest Qwen3 model — same family we train in Ch15

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,          # test that bitsandbytes 4-bit works
    device_map="auto",          # let the library pick GPU vs CPU
    torch_dtype=torch.float16,  # half-precision weights on GPU
)

# --- 4. Run one forward pass to confirm the model is live ---
# We're not training here — just passing text through to verify nothing is broken.
inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
with torch.no_grad():  # no_grad means don't compute gradients — saves memory
    outputs = model(**inputs)

# If we got here without an exception, everything works.
print(f"\nOutput logits shape: {outputs.logits.shape}")  # should be [1, N, vocab_size]
print("Smoke test passed.")
```

Run it with:
```bash
python smoke_test.py
```

Expected output (GPU names will vary):
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
VRAM: 24.6 GB

Loading a tiny model to test quantization...
Output logits shape: torch.Size([1, 4, 151936])
Smoke test passed.
```

> **Note:** Qwen3-0.6B is the smallest model in the Qwen3 family — the same model family used throughout this book. The vocab size in the output (151936) is the number of tokens the model knows. Your exact number may differ slightly across model revisions, and the GPU name will match whatever hardware you are on. The only line that must match exactly is `Smoke test passed.`

If you see `Smoke test passed.`, you are ready to work through the rest of this book.

---

## Cost and time reference table

| Path | GPU | VRAM | Approx cost | Session limit | Setup time |
|------|-----|------|-------------|---------------|------------|
| Colab free | T4 | 16 GB | $0 | ~4 hrs | 2 min |
| Colab Pro | V100/A100 | 16–40 GB | ~$10/mo | longer | 2 min |
| Kaggle free | T4 / P100 | 16 GB | $0 | 12 hrs / 30 hr wk | 5 min |
| RunPod RTX 3090 | RTX 3090 | 24 GB | ~$0.44/hr | unlimited | 10 min |
| RunPod RTX 4090 | RTX 4090 | 24 GB | ~$0.74/hr | unlimited | 10 min |
| Local RTX 3060 | RTX 3060 | 12 GB | electricity | unlimited | 20 min |
| Local RTX 4090 | RTX 4090 | 24 GB | electricity | unlimited | 20 min |

> All costs are approximate ballparks as of mid-2025 and will vary. Check current RunPod pricing on their site before budgeting.

---

## A note on Hugging Face tokens

Several chapters (starting in Ch10 — *Choosing Your Base Model: Qwen vs Gemma*) will download gated models that require a Hugging Face account.

1. Create a free account at huggingface.co
2. Go to **Settings → Access Tokens → New token** (read permission is enough)
3. Run this once in your environment:

```bash
huggingface-cli login
# Paste your token when prompted. It gets saved to ~/.cache/huggingface/token
```

Alternatively, set it as an environment variable:
```bash
export HF_TOKEN="hf_your_token_here"
```

You only need to do this once per machine.

---

## Common mistakes

**`CUDA out of memory` on the smoke test**

The tiny model in the smoke test should use under 1 GB of VRAM. If you hit OOM here, another process is holding your GPU. On RunPod: you may have a stale session. Run `nvidia-smi` and look for a process using VRAM; kill it with `kill <PID>`. On Colab: restart the runtime (Runtime → Restart session).

**`CUDA available: False` even though you have an NVIDIA GPU**

The NVIDIA driver and the CUDA toolkit version need to match. On RunPod, use the image specified above — it ships matching versions. Locally, run `nvidia-smi` first; if that command is not found, install the driver from nvidia.com before anything else. Then reinstall PyTorch with the CUDA 12.4 index URL:
```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

**Mixed version installs cause mysterious import errors**

If you ran pip installs incrementally without the pinned versions above, you may have incompatible versions of transformers and trl. The symptom is usually `ImportError` or an attribute not found on a class. Fix: nuke the environment and reinstall from scratch using the single pinned command above.

```bash
deactivate
rm -rf finetune
python3 -m venv finetune
source finetune/bin/activate
# paste the full pip install command again
```

**Colab disconnects mid-session**

Colab free tier disconnects after ~4 hours of inactivity or ~12 hours total. Save model checkpoints to Google Drive to avoid losing work. In Ch15, we'll configure `save_strategy` in the training script to checkpoint automatically every N steps.

**`bitsandbytes` loads but prints warnings about CPU fallback**

If you see `bitsandbytes: ERROR: 8-bit operations on CPU are not supported`, bitsandbytes found no GPU. This is the same root cause as CUDA not being available — check `torch.cuda.is_available()` first.

**Running on Windows without WSL2**

Unsloth has limited Windows-native support. If you are on Windows, use WSL2 (Windows Subsystem for Linux 2) with Ubuntu 22.04. The install steps above work identically inside WSL2, and NVIDIA's CUDA driver automatically bridges to WSL2 from the Windows side — no separate driver install inside WSL is needed.

---

## Recap

- VRAM is the hard constraint; 4-bit quantization (LoRA/QLoRA) cuts model size roughly 4x, making 16 GB VRAM sufficient for 4B–8B models
- Three workable paths: Colab/Kaggle (free, zero setup), RunPod (flexible, ~$0.50–$2 per run), local NVIDIA GPU (free after hardware cost)
- This book targets CUDA; Apple Silicon and AMD GPUs require a different toolchain (MLX, ROCm) not covered here
- One pinned pip install command gets you the full stack: unsloth, torch, transformers, trl, peft, datasets, bitsandbytes, accelerate
- The 5-line smoke test confirms GPU visibility, VRAM, and 4-bit model loading before any real training
- CUDA version mismatches and stale GPU processes are the two most common setup failures
- Log into Hugging Face with `huggingface-cli login` before the next chapter

## Next

Ch9 — *The Toolbox: Unsloth, Transformers, TRL, PEFT, and Friends* — goes deeper into each library you just installed, showing exactly which piece does what during a training run and how they hand off to each other.
