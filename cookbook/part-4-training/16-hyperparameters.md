# Ch16 - Hyperparameters: Which Knobs to Turn and When

You just ran your first fine-tune in *Ch15 - Your First Fine-Tune with Unsloth (Full Script)*. The script worked. Now you want to make the model *better* — or you hit a problem (loss won't drop, GPU runs out of memory, model starts repeating itself) and you need to fix it.

That's what this chapter is for. A **hyperparameter** is any setting you choose before training starts — the knobs on the outside of the machine that determine how the machine behaves inside. The model's weights are what training changes; hyperparameters are what *you* change before hitting run.

There are roughly a dozen knobs worth knowing. This chapter goes through each one, tells you the safe default, and gives you a clear rule: "if X happens, turn this knob in this direction."

---

## What you'll learn

- What every major hyperparameter controls, in plain English
- A safe starting recipe for memory-extraction fine-tuning you can copy immediately
- When to turn each knob up or down — and by how much
- How batch size and gradient accumulation work together as "effective batch size"
- The three most common failure modes (loss stuck, overfitting, out-of-memory) and exactly which knobs fix each

---

## Concepts you need first

### Learning rate: the most important knob

We introduced learning rate in *Ch7 - How Training Actually Works*: it controls how big a step the optimizer takes downhill toward lower loss after each batch.

Think of it like adjusting the sensitivity on a thermostat. Too high: the heating overshoots, the room swings from hot to cold and never settles. Too low: the room barely warms up in a reasonable time. You want it just sensitive enough to reach the target without oscillating.

**One-line definition:** learning rate is a small multiplier (usually between `1e-5` and `5e-4`) that scales how aggressively the optimizer updates weights after each batch.

For LoRA fine-tuning, the safe starting zone is `2e-4`. Full fine-tuning would use something closer to `1e-5` — LoRA adapters are small and can absorb larger updates without breaking the base model's knowledge.

### Effective batch size: the product of two settings

You have two batch-related knobs: `per_device_train_batch_size` (how many examples fit on the GPU at once) and `gradient_accumulation_steps` (how many mini-batches to accumulate before doing one weight update).

**Effective batch size = per_device_train_batch_size × gradient_accumulation_steps**

If you set batch size to 2 and accumulation to 8, the optimizer sees updates as if you had a batch of 16, even though only 2 examples are in GPU memory at any moment. This lets you simulate large batches on small GPUs. The gradient math is identical; only the memory footprint changes.

For memory extraction — where each training row is a conversation plus a JSON list — a good effective batch size target is **16–32**. Anything lower and the gradient estimates get noisy; anything higher and you're probably using VRAM you don't have.

### The learning rate scheduler: how the rate changes over time

A fixed learning rate is rarely optimal. The **scheduler** is a function that adjusts the learning rate automatically as training progresses. The most common and best-default choice for fine-tuning is **cosine decay**: the rate starts at your chosen value, holds steady through the warmup phase, then gradually decays — following a cosine curve — down to near zero by the end of training.

Why cosine? Early in training, the model is making large, useful updates and benefits from a full learning rate. Late in training, it's making fine-grained adjustments, and a smaller rate prevents overshooting the optimum.

You don't need to implement this yourself — set `lr_scheduler_type="cosine"` and it's handled.

### Warmup: the on-ramp

Introduced in *Ch7*, warmup is a brief phase at the very start of training where the learning rate ramps up from near-zero to its target value. Without warmup, the first few batches — when the model's predictions are most chaotic — get full-size gradient updates that can damage the base model's knowledge before it has a chance to learn your task.

The `warmup_ratio` parameter is the fraction of total training steps to spend warming up. A value of `0.05` means the first 5% of steps are warmup. For most runs, this translates to 20–100 steps — enough to stabilize things without spending too long in the slow ramp-up phase.

---

## The knobs, one by one

### 1. Learning rate (`learning_rate`)

| Default | Range to explore | Units |
|---|---|---|
| `2e-4` | `5e-5` to `5e-4` | dimensionless multiplier |

**If loss drops quickly then plateaus high (above ~0.6):** try raising to `3e-4` or `4e-4`.  
**If loss oscillates wildly and won't settle:** cut in half — try `1e-4`.  
**If loss barely moves at all:** something else is wrong (check your data format first); if data looks correct, try raising to `3e-4`.

Do not go above `5e-4` for LoRA on small datasets. The adapters are powerful enough that a high learning rate will overfit fast.

---

### 2. Number of epochs (`num_train_epochs`)

| Default | Typical range | Notes |
|---|---|---|
| `3` | `1` to `5` | For datasets under ~5,000 rows |

An **epoch** is one full pass through your training data. Fine-tuning a small task-specific dataset rarely benefits from more than 3 epochs — beyond that, the model starts memorizing specific examples rather than learning the skill.

**If your dataset is small (under ~500 rows):** stay at 2 epochs maximum.  
**If your dataset is large (5,000+ rows):** 1–2 epochs is often enough; the model sees enough variety that more passes aren't needed.  
**If validation loss is rising while training loss keeps dropping:** you've overfit — stop early, use fewer epochs next time.

As an alternative to epochs, you can set `max_steps` directly. This is useful when you want predictable training time regardless of dataset size. A typical starting point for memory extraction with ~2,000 rows: `max_steps=500`.

---

### 3. Batch size and gradient accumulation (`per_device_train_batch_size`, `gradient_accumulation_steps`)

| Setting | Default | Notes |
|---|---|---|
| `per_device_train_batch_size` | `2` | Keep low on 16 GB GPU; raise to 4 on 24+ GB |
| `gradient_accumulation_steps` | `8` | Gives effective batch of 16 with batch_size=2 |

The only reason to change `per_device_train_batch_size` is VRAM. Start at 2. If you get an out-of-memory (OOM) error, drop to 1 and double `gradient_accumulation_steps` to compensate.

**To increase effective batch size without touching VRAM:** raise `gradient_accumulation_steps`.  
**Effective batch sizes to try:** 16 (default), 32 (smoother gradients, slower per-step), 8 (if you're in a hurry).

> **Heads up:** larger effective batch size tends to need a slightly higher learning rate to stay effective. A rough rule: if you double the effective batch size, scale learning rate up by ~40% (`× sqrt(2)`). This is a heuristic, not a law.

---

### 4. Maximum sequence length (`max_seq_length`)

| Default | Range | Notes |
|---|---|---|
| `2048` | `512` to `8192` | In tokens, not characters |

This is the longest input+output sequence the model will process in a single training example. Sequences longer than this are **truncated** — silently cut off. Sequences shorter are padded.

For memory extraction, your inputs are conversation chunks (typically 200–600 tokens) and your outputs are JSON lists (typically 100–400 tokens). A total of 2048 tokens is almost always more than enough. To be concrete: a typical assistant turn — a JSON list of 3–5 memory objects each with `text`, `type`, and `entities` fields — runs only 80–250 tokens, well within the 2048 default.

**If you're seeing truncated training examples** (check with `tokenizer(example, return_length=True)` — if length hits `max_seq_length` exactly, it was truncated): raise to 4096.  
**If you're running out of VRAM and lowering batch size didn't help:** try lowering `max_seq_length` to 1024. This directly reduces the memory cost of each training step.

> **Rule:** VRAM cost scales roughly linearly with sequence length. Halving `max_seq_length` from 2048 to 1024 can free up 1–3 GB depending on model size.

> **Where to set it:** `max_seq_length` is passed to `FastLanguageModel.from_pretrained(...)`, not to `SFTConfig`. Setting it in `SFTConfig` would be silently ignored or raise a `TypeError`. See the recipe below.

---

### 5. LoRA rank (`r`) and alpha (`lora_alpha`)

These were covered in detail in *Ch6 - LoRA and QLoRA Without the Math Headache*. Quick summary for reference here:

| Parameter | Default | Notes |
|---|---|---|
| `r` | `16` | Raise to `32` if quality is poor after full training |
| `lora_alpha` | `16` | Keep equal to `r`, or set to `2×r` for a stronger adapter |

**The key interaction with other hyperparameters:** higher `r` means more trainable parameters, which means you may want to lower the learning rate slightly to avoid overshooting. If you raise `r` to `32`, consider dropping learning rate to `1.5e-4`.

---

### 6. LoRA dropout (`lora_dropout`)

| Default | Range | Notes |
|---|---|---|
| `0.05` | `0.0` to `0.1` | Regularization — prevents overfitting |

Dropout randomly zeros out some fraction of the adapter's activations during each training step. This prevents the adapter from over-relying on specific patterns in your training data.

**If validation loss diverges from training loss** (overfitting signal): raise dropout to `0.1`.  
**If your dataset is large (10,000+ rows):** set to `0.0` — enough data diversity handles regularization naturally.  
**If training loss is stubbornly high and you suspect the adapter can't learn enough**: set to `0.0` — dropout might be suppressing useful learning.

---

### 7. Warmup (`warmup_ratio` or `warmup_steps`)

| Default | Notes |
|---|---|
| `warmup_ratio=0.05` | Use ratio for epoch-based runs; use `warmup_steps` for step-based runs |

Use one or the other — not both.

`warmup_ratio=0.05` means: "use 5% of total steps as warmup." On a 500-step run, that's 25 warmup steps. On a 200-step run, it's 10. This scales gracefully.

`warmup_steps=50` means: exactly 50 steps, regardless of total run length. Use this when you want explicit control.

**If training loss spikes dramatically in the first few steps:** increase warmup — try `warmup_ratio=0.1`.  
**If training feels slow to get started and you have a very short run** (under 100 steps): drop to `warmup_ratio=0.02`.

---

### 8. Weight decay (`weight_decay`)

| Default | Range | Notes |
|---|---|---|
| `0.01` | `0.0` to `0.1` | Regularization on optimizer, separate from dropout |

Weight decay is a penalty added to the loss that nudges weights toward zero, discouraging the model from relying on extreme parameter values. Think of it as a gentle force that keeps the adapter's numbers from getting too large.

In practice, `0.01` works well and you rarely need to change it. It is a weaker effect than learning rate, epochs, or dropout.

**If you're seeing very fast overfitting even with dropout:** try raising to `0.05`.  
**For most runs:** leave it at `0.01` and focus your tuning energy elsewhere.

---

### 9. LR scheduler type (`lr_scheduler_type`)

| Default | Alternatives |
|---|---|
| `"cosine"` | `"linear"`, `"constant"`, `"cosine_with_restarts"` |

- `"cosine"` — best general default. Rate decays smoothly to near-zero. Rarely needs changing.
- `"linear"` — simpler. Decays linearly from start to zero. Slightly more aggressive early decay. Fine if cosine isn't available.
- `"constant"` — learning rate never changes. Only useful for debugging or very short runs.
- `"cosine_with_restarts"` — rate decays, then resets multiple times. Useful for very long training runs or continual learning (discussed in *Ch23 - Continual Learning and Scaling Up*). Not useful for standard fine-tuning.

**For memory extraction fine-tuning: always use `"cosine"`.**

---

### 10. Optimizer (`optim`)

| Default | Notes |
|---|---|
| `"adamw_8bit"` | 8-bit AdamW via bitsandbytes. Same behavior, ~30% less VRAM. |

AdamW is the standard optimizer for transformer fine-tuning. You do not need to understand its internals — just know it's battle-tested for this class of problem.

The `_8bit` suffix means the optimizer's internal state is stored in 8-bit precision rather than 32-bit. This is a free 3–4× memory saving on optimizer state with negligible quality loss. Always use `"adamw_8bit"` unless you have a specific reason not to.

**Do not change the optimizer.** It is the one knob where the default is almost universally correct for this use case.

---

## The recommended starting recipe

Copy this block into your training script. It is calibrated for a ~2,000-row memory-extraction dataset on a 16 GB GPU (e.g., an RTX 3080 Ti, A10G, or similar Ampere+ GPU). T4 users: see the `fp16`/`bf16` note inside the recipe — one line must change. Adjust from here using the per-knob guidance above.

```python
# ch16_starting_recipe.py
# Recommended hyperparameter starting point for memory-extraction fine-tuning.
# Hardware target: 16 GB VRAM, Ampere+ GPU (RTX 3080 Ti, A10G, etc.).
#   T4 users: set fp16=True, bf16=False — see note inside.
# Dataset size: ~2,000 training rows.
# Model: Qwen3-8B or Gemma3-4B under QLoRA via Unsloth.

# SFTConfig is TRL's all-in-one training configuration class. It replaces
# the older pattern of passing a separate TrainingArguments object to
# SFTTrainer — fill in one config and hand it to the trainer directly.
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

# ── Step 1: Load model + tokenizer ────────────────────────────────────────────
# max_seq_length belongs HERE — it tells Unsloth how long the model's attention
# window should be during training. SFTConfig does NOT accept max_seq_length;
# putting it there is silently ignored or raises a TypeError.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B-bnb-4bit",
    max_seq_length=2048,   # Max tokens per example (input + output combined).
                           # A typical memory-extraction assistant turn —
                           # a JSON list of 3-5 objects with "text", "type",
                           # and "entities" fields — runs only 80-250 tokens,
                           # well within this limit.
                           # Lower to 1024 if OOM persists after other fixes.
    load_in_4bit=True,
    dtype=None,            # Unsloth auto-detects the best dtype for your GPU.
)

# ── Step 2: Attach LoRA adapters ───────────────────────────────────────────────
# LORA_CONFIG is a plain dict. It only takes effect when unpacked into
# get_peft_model() via **LORA_CONFIG. Without this call, no adapters are
# attached and nothing gets fine-tuned.
LORA_CONFIG = {
    "r": 16,               # Adapter rank. Raise to 32 if quality is poor.
    "lora_alpha": 16,      # Keep equal to r. Set to 32 for a stronger adapter.
    "lora_dropout": 0.05,  # Light regularization. Drop to 0 if dataset > 10k rows.
    "target_modules": [
        # All seven module types: attention (q/k/v/o) + feed-forward (gate/up/down).
        # The feed-forward modules matter for structured JSON output — include them.
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",  # Unsloth's memory saver — always on.
                                              # Trades a bit of compute for lower VRAM.
    "random_state": 42,
}

model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

# ── Step 3: Build training config ─────────────────────────────────────────────
training_args = SFTConfig(
    output_dir="./memory-extractor-checkpoints",

    # ── How long to train ───────────────────────────────────────────────────────
    num_train_epochs=3,            # 3 full passes. Drop to 2 if overfitting appears.
    # max_steps=500,               # Alternative: uncomment to cap by step count.

    # ── Batch size and accumulation ─────────────────────────────────────────────
    per_device_train_batch_size=2, # 2 examples per GPU step. Drop to 1 if OOM.
    gradient_accumulation_steps=8, # Effective batch = 2 x 8 = 16. Raise to 16 for
                                   # smoother gradients if you have headroom.

    # ── Learning rate and schedule ──────────────────────────────────────────────
    learning_rate=2e-4,            # LoRA sweet spot. Range: 5e-5 to 5e-4.
    lr_scheduler_type="cosine",    # Smooth decay. Do not change.
    warmup_ratio=0.05,             # Warm up over the first 5% of steps (~30 steps here).

    # ── Regularization ──────────────────────────────────────────────────────────
    weight_decay=0.01,             # Light penalty on large weights. Rarely needs tuning.

    # ── Optimizer ───────────────────────────────────────────────────────────────
    optim="adamw_8bit",            # 8-bit AdamW: free VRAM saving. Do not change.

    # ── Precision ───────────────────────────────────────────────────────────────
    # SAFE DEFAULT for all hardware including the T4 (Volta), which does NOT
    # support bf16. If you have an Ampere+ GPU (RTX 30/40xx, A10G, A100),
    # flip these two lines to fp16=False, bf16=True for better numeric stability.
    fp16=True,                     # Works on every GPU including T4.
    bf16=False,                    # Ampere+ only. Set True (and fp16=False) if eligible.

    # ── Logging and checkpointing ───────────────────────────────────────────────
    logging_steps=10,              # Record loss every 10 steps. Keep this small.
    save_strategy="steps",
    save_steps=100,                # Save checkpoint every 100 steps.
    save_total_limit=2,            # Keep only the 2 most recent checkpoints.

    # ── Validation ──────────────────────────────────────────────────────────────
    eval_strategy="steps",         # Evaluate on validation set periodically.
    eval_steps=50,                 # Evaluate every 50 steps.
    load_best_model_at_end=True,   # When done, load the checkpoint with lowest val loss.
    metric_for_best_model="eval_loss",

    # ── Reproducibility ─────────────────────────────────────────────────────────
    seed=42,
    data_seed=42,
)

# ── Step 4: Create the trainer and run ────────────────────────────────────────
# eval_dataset is REQUIRED because eval_strategy="steps" and
# load_best_model_at_end=True are both set above. Omitting it causes a
# runtime error. train_dataset and val_dataset come from Ch15's data-loading
# section — the 90/10 split created with dataset.train_test_split().
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,   # Your training split (see Ch15).
    eval_dataset=val_dataset,      # Your validation split — required for eval_strategy.
)
trainer.train()
```

This recipe gives you an effective batch size of 16, a cosine-decayed learning rate starting at `2e-4`, 5% warmup, and light regularization. It will fit on a 16 GB GPU with a 7B QLoRA model.

---

## Adjusting for different GPU sizes

Not everyone has a 16 GB GPU. Here's how to adapt the recipe:

```python
# ch16_vram_adaptation.py
# Four configurations for different hardware.
# All produce equivalent effective batch size = 16 (except 40 GB = 32).
# Only the per_device_train_batch_size / gradient_accumulation split changes.
# To use: copy the values from the matching config dict into your SFTConfig call,
# replacing the corresponding keys. For example:
#   training_args = SFTConfig(**config_8gb, output_dir="./checkpoints", ...)
# Or just copy individual values into the recipe above.
#
# Note: max_seq_length must also be updated in the FastLanguageModel.from_pretrained()
# call, not in SFTConfig. The values below show the recommended target per GPU tier.

# -- 8 GB GPU (RTX 3070, T4 with memory pressure) -----------------------------
config_8gb = {
    "per_device_train_batch_size": 1,   # Only 1 example fits at a time.
    "gradient_accumulation_steps": 16,  # Accumulate 16 mini-batches: effective = 16.
    # max_seq_length: use 1024 in FastLanguageModel.from_pretrained()
    "learning_rate": 2e-4,              # No change needed.
}

# -- 16 GB GPU (RTX 3080 Ti, A10G) -- the recipe default ---------------------
config_16gb = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,   # Effective = 16.
    # max_seq_length: use 2048 in FastLanguageModel.from_pretrained()
    "learning_rate": 2e-4,
}

# -- 24 GB GPU (RTX 3090, RTX 4090, A5000) ------------------------------------
config_24gb = {
    "per_device_train_batch_size": 4,   # Bigger physical batch: fewer accumulation steps.
    "gradient_accumulation_steps": 4,   # Effective = 16. Same total.
    # max_seq_length: use 2048 in FastLanguageModel.from_pretrained()
    "learning_rate": 2e-4,
    # Optionally: raise effective batch to 32 by doubling gradient_accumulation_steps.
    # If you do that, scale learning rate up slightly: 2e-4 -> ~2.8e-4.
}

# -- 40 GB GPU (A100, A40) -- plenty of headroom ------------------------------
config_40gb = {
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,   # Effective = 32.
    # max_seq_length: use 4096 in FastLanguageModel.from_pretrained()
    "learning_rate": 2.8e-4,           # Slightly higher to match larger effective batch.
}
```

---

## The three failure modes and their fixes

This is the most practical part of the chapter. When something goes wrong, you should be able to look at the symptom and know which knob to turn.

### Failure 1: Loss is not dropping

**Symptoms:** Loss starts somewhere above 2.0 and barely moves across hundreds of steps. Final loss is still above 1.5. Model output still looks like the base model, not a memory extractor.

**Probable causes and fixes, in order of likelihood:**

```python
# ch16_debug_loss_not_dropping.py
# Checklist: run through these in order when loss won't decrease.

# -- Check 1: Data format -------------------------------------------------------
# The most common culprit. Open your JSONL file and read 3 rows by eye.
# Every row must have {"messages": [{"role": "system", ...}, {"role": "user", ...},
# {"role": "assistant", ...}]}. If the format is wrong, loss won't drop.

import json
with open("data/train.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        row = json.loads(line)
        msgs = row["messages"]
        print(f"Row {i}: roles = {[m['role'] for m in msgs]}")
        # Expected: ['system', 'user', 'assistant']

        # Use a role lookup instead of a hard-coded index like msgs[2].
        # Hard-coding breaks silently if any row is missing a system prompt
        # or has extra turns (multi-turn conversations), causing an IndexError.
        asst = next(m for m in msgs if m["role"] == "assistant")
        parsed = json.loads(asst["content"])  # Must not raise — content must be valid JSON.
        print(f"         memories: {len(parsed)} items")
        # Each item should have "text", "type", and "entities" fields.

# -- Check 2: Learning rate too low --------------------------------------------
# If data looks fine, try raising learning rate.
# 2e-4 -> 4e-4 is a reasonable first move.
# "learning_rate": 4e-4

# -- Check 3: Warmup too long --------------------------------------------------
# If your run is short (under 200 steps), a warmup_ratio of 0.05 might mean
# 10+ steps of near-zero learning rate. During warmup, loss barely moves -- this
# is normal. Make sure you're reading loss AFTER warmup ends.

# -- Check 4: Wrong chat template ----------------------------------------------
# Qwen3 and Gemma 3 use different chat templates. Using the wrong one means the
# tokenizer inserts different special tokens than the model was trained with.
# The model is then trying to learn a token structure it doesn't recognize.
# Verify: tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
#         (or "gemma" for Gemma 3)
```

### Failure 2: Overfitting

**Symptoms:** Training loss drops smoothly to below 0.3. Validation loss starts rising after the midpoint of training. The model produces perfect JSON on training examples but makes errors — wrong types, missed entities, hallucinated memories — on new conversations.

```python
# ch16_debug_overfitting.py
# Overfitting fixes, in order of impact.

# Fix 1: Reduce epochs ---------------------------------------------------------
# The single most powerful fix. Go from 3 epochs to 2.
# "num_train_epochs": 2

# Fix 2: Raise dropout ---------------------------------------------------------
# More regularization on the adapter.
# "lora_dropout": 0.1

# Fix 3: Lower learning rate ---------------------------------------------------
# Slower updates give the model less opportunity to overfit fast.
# "learning_rate": 1e-4

# Fix 4: Add more data ---------------------------------------------------------
# The most durable fix. See Ch13 for synthetic data generation.
# More data diversity > all regularization tricks combined.

# Fix 5: Reduce rank -----------------------------------------------------------
# A smaller adapter has less capacity to memorize.
# "r": 8  (from 16)
# "lora_alpha": 8  (keep equal to r)
```

### Failure 3: Out of memory (OOM)

**Symptoms:** `torch.cuda.OutOfMemoryError` or `CUDA out of memory` during training. Training crashes, often in the first few steps.

```python
# ch16_debug_oom.py
# OOM fixes, applied in order. Stop as soon as the error goes away.

# Fix 1: Halve per_device_train_batch_size, double gradient_accumulation_steps --
# This keeps effective batch size constant but halves per-step memory.
# "per_device_train_batch_size": 1   (from 2)
# "gradient_accumulation_steps": 16  (from 8)

# Fix 2: Reduce max_seq_length -------------------------------------------------
# Memory cost scales with sequence length. Go from 2048 to 1024.
# Set this in FastLanguageModel.from_pretrained(), not in SFTConfig.
# max_seq_length=1024

# Fix 3: Confirm use_gradient_checkpointing is set -----------------------------
# This trades compute for memory -- recomputes activations during backprop
# instead of storing them. Unsloth's version is highly optimized.
# It is set via LORA_CONFIG and wired up in get_peft_model() (see recipe above):
#   "use_gradient_checkpointing": "unsloth"

# Fix 4: Reduce LoRA target modules --------------------------------------------
# Remove feed-forward modules and target only attention:
# "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
# This cuts trainable params roughly in half.

# Fix 5: Drop to a smaller model -----------------------------------------------
# A 4B model under QLoRA needs roughly 6-8 GB vs. 10-14 GB for a 7B model.
# For memory extraction, a well-trained 4B model often matches a 7B model's quality.
# See Ch10 for model selection guidance.
```

---

## Quick-reference table: all knobs at a glance

| Knob | Default | If loss won't drop | If overfitting | If OOM |
|---|---|---|---|---|
| `learning_rate` | `2e-4` | Raise to `4e-4` | Lower to `1e-4` | — |
| `num_train_epochs` | `3` | Raise to `4–5` | Lower to `2` | — |
| `per_device_train_batch_size` | `2` | — | — | Lower to `1` |
| `gradient_accumulation_steps` | `8` | — | — | Raise to `16` |
| `max_seq_length` | `2048` | — | — | Lower to `1024` |
| `r` (LoRA rank) | `16` | Raise to `32` | Lower to `8` | Lower to `8` |
| `lora_alpha` | `16` | Keep = `r` | Keep = `r` | Keep = `r` |
| `lora_dropout` | `0.05` | Lower to `0.0` | Raise to `0.1` | — |
| `warmup_ratio` | `0.05` | Lower to `0.02` | — | — |
| `weight_decay` | `0.01` | — | Raise to `0.05` | — |
| `lr_scheduler_type` | `"cosine"` | — | — | — |
| `optim` | `"adamw_8bit"` | — | — | — |

> Dash (—) means this knob does not directly affect this symptom. Focus elsewhere.

---

## Common mistakes

**1. Tuning multiple knobs at once.**

When something goes wrong, it's tempting to change three things at once. Don't. Change one knob, run training (even for 50 steps), look at the loss curve, then decide the next move. Changing multiple settings makes it impossible to know what helped.

**2. Choosing learning rate by feel, not by watching the curve.**

The loss curve in the first 50 steps tells you almost everything you need to know about learning rate. If it's dropping fast and smoothly: learning rate is fine. If it's oscillating: too high. If it's barely moving after warmup ends: too low. Don't guess — look at `logging_steps=10` data and read the early curve before committing to a full run.

**3. Forgetting that effective batch size changes gradient noise.**

If you drop `per_device_train_batch_size` from 2 to 1 to fix OOM without raising `gradient_accumulation_steps`, you've cut your effective batch size in half. The gradients get noisier, and you may need to lower the learning rate slightly to compensate. Always think in terms of effective batch size, not just the physical batch size.

**4. Setting `num_train_epochs` too high because "more is better."**

More epochs on a fixed dataset is not more training data — it's the same examples repeated. The model will start memorizing them. For datasets under 2,000 rows, 3 epochs is usually the ceiling. Watch validation loss — if it rises while training loss drops, you're past the optimal stopping point.

**5. Skipping the validation split.**

If you train on 100% of your data, you have no signal to detect overfitting. Always reserve 10–15% of your examples as a held-out validation set (covered in *Ch14 - Cleaning, Splitting, and Sanity-Checking Data*). Pass it to SFTTrainer via `eval_dataset` and watch the `eval_loss` column in your logs.

**6. Using `bf16=True` on a T4 or other non-Ampere GPU.**

`bfloat16` (bf16) is numerically more stable than `float16` (fp16) for fine-tuning, but bf16 requires Ampere architecture or newer (RTX 30-series, RTX 40-series, A100, A10G). The T4 is a Volta-architecture GPU and does NOT support bf16 — using `bf16=True` on a T4 will either crash or silently produce garbage. The recipe in this chapter defaults to `fp16=True` which works everywhere. If you have an Ampere+ GPU, flip to `fp16=False, bf16=True`.

---

## Recap

- A hyperparameter is any setting you choose before training — learning rate, epochs, batch size, LoRA rank, and so on.
- **Learning rate** is the most impactful knob. Start at `2e-4`. Drop by half if loss oscillates; raise if loss plateaus high.
- **Effective batch size** = `per_device_train_batch_size × gradient_accumulation_steps`. Target 16–32. Only split between the two based on VRAM constraints.
- **Epochs**: 3 is the right default for ~2,000-row datasets. Watch validation loss — if it rises while training loss drops, stop early.
- **`max_seq_length`**: set in `FastLanguageModel.from_pretrained()`, not in `SFTConfig`. 2048 covers most memory-extraction inputs (a typical `{text, type, entities}` assistant turn is only 80–250 tokens). Lower to 1024 if you're fighting OOM.
- **LoRA rank (`r`)**: 16 is the right default. Raise to 32 if quality is poor; drop to 8 if OOM persists.
- The three failure modes — loss not dropping, overfitting, OOM — each have a short checklist of knobs to try, in priority order.
- Change one knob at a time. The early loss curve (first 50 steps) tells you almost everything.

## Next

*Ch17 - Watching Training: Loss Curves and When to Stop* — your training script is running; now learn to read what the loss numbers actually mean, how to plot the curve, and exactly when to stop or restart.
