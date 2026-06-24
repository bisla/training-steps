# Ch17 - Watching Training: Loss Curves and When to Stop

Your training script is running. The GPU fan is spinning. Numbers are scrolling by in the terminal. Now what?

Most people ignore the logs until training finishes, then wonder why the model doesn't work. That's a mistake. The loss curve is a live diagnostic. It tells you whether training is going well, whether it's already done, or whether something is broken — while you can still do something about it.

This chapter teaches you to read that curve and act on what you see.

---

## What you'll learn

- What training loss and validation loss actually tell you — and why you need both
- How to wire up Weights & Biases (wandb) for live curve tracking, or stay in the terminal with stdout
- What a good curve, an overfitting curve, and a broken-data curve look like (with ASCII sketches)
- How to configure `logging_steps`, `eval_steps`, and checkpoint saving
- How to use early stopping so training stops itself when the model is done learning

---

## Concepts you need first

### Two losses, not one

In Ch7 you learned that loss measures how surprised the model was by the correct answer — lower is better. But one loss number is not enough. You need two:

**Training loss** — measured on the examples the model is actively learning from. This will almost always go down over time. It's the grade a student gives themselves on the material they just studied.

**Validation loss** — measured on a held-out set the model has never seen during training. This is the real grade. It tells you whether the model learned a *skill* or just memorized the training examples.

If you only watch training loss, you cannot tell the difference between a model that has genuinely learned memory extraction and one that has memorized your 800 training conversations word-for-word.

### The three states a training run can be in

At any moment, your training run is in one of three states:

1. **Still learning** — both losses are dropping. Keep going.
2. **Done** — both losses have plateaued (stopped meaningfully decreasing). Stop now or very soon.
3. **Overfitting** — training loss keeps dropping but validation loss starts rising. Stop immediately.

Everything in this chapter is about detecting which state you're in.

### What "plateau" means in practice

Loss never reaches exactly zero and stops moving. In practice, "plateau" means the loss has not improved by more than roughly 1-2% over the last 20-30% of your steps. A loss that moved from 0.42 to 0.41 over 200 steps has plateaued. A loss that moved from 0.80 to 0.42 has not.

---

## Option A: Watching training in the terminal (no extra tools)

The simplest monitoring setup is already built into Transformers and SFTTrainer. You just need to tell it how often to log.

The key parameter is `logging_steps`. Every `logging_steps` steps, the trainer prints a row like this to stdout:

```
{'loss': 1.4832, 'grad_norm': 2.341, 'learning_rate': 0.000124, 'epoch': 0.62, 'step': 80}
```

By default this is set to 500, which on a 600-step training run means you get *one* row. That's useless. Set it to something like 10 or 20 so you can actually see the curve develop.

Here is the relevant section of `TrainingArguments` to configure logging and evaluation:

```python
# ch17_training_args.py
# These are the monitoring-focused parameters you add to your TrainingArguments.
# Plug these into the full training script from Ch15.

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./memory-extractor-checkpoints",

    # --- How often to print loss to the terminal ---
    logging_steps=10,           # Print a loss row every 10 steps.
                                # With ~500 total steps, this gives 50 data points —
                                # enough to see the curve shape clearly.
    logging_strategy="steps",   # "steps" means log every N steps (not every epoch).

    # --- How often to evaluate on the validation set ---
    eval_strategy="steps",      # Run validation periodically during training.
    eval_steps=50,              # Evaluate every 50 steps.
                                # Rule of thumb: eval roughly 8-10 times per run.
                                # Too frequent = slow; too rare = blind to overfitting.

    # --- Save checkpoints so you can recover a good model ---
    save_strategy="steps",
    save_steps=50,              # Match eval_steps so you can load any evaluated checkpoint.
    save_total_limit=3,         # Keep only the 3 most recent checkpoints to save disk space.
                                # Each checkpoint for a LoRA model is ~100-400 MB.

    # --- Load the best checkpoint at the end (not just the last one) ---
    load_best_model_at_end=True,    # After training finishes, the trainer automatically
                                    # reloads the checkpoint with the lowest validation loss.
                                    # This is the version you should export and use.
    metric_for_best_model="eval_loss",  # The metric to compare checkpoints by.
    greater_is_better=False,        # Lower loss = better model.

    # --- Other required args (set to your values from Ch16) ---
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    optim="adamw_8bit",
    fp16=True,                  # or bf16=True on newer GPUs (A100, H100, 4090)
)
```

With these settings, your terminal output will look like this during a healthy run:

```
{'loss': 2.4812, 'grad_norm': 4.21, 'learning_rate': 0.000025, 'epoch': 0.06, 'step': 10}
{'loss': 2.1034, 'grad_norm': 3.87, 'learning_rate': 0.000050, 'epoch': 0.12, 'step': 20}
{'loss': 1.7821, 'grad_norm': 3.12, 'learning_rate': 0.000100, 'epoch': 0.19, 'step': 30}
{'loss': 1.4203, 'grad_norm': 2.54, 'learning_rate': 0.000150, 'epoch': 0.25, 'step': 40}
{'loss': 1.1944, 'grad_norm': 2.01, 'learning_rate': 0.000180, 'epoch': 0.31, 'step': 50}
{'eval_loss': 1.2301, 'eval_runtime': 18.4, 'step': 50}
{'loss': 1.0021, 'grad_norm': 1.82, 'learning_rate': 0.000195, 'epoch': 0.37, 'step': 60}
...
{'loss': 0.3812, 'grad_norm': 0.71, 'learning_rate': 0.000041, 'epoch': 2.88, 'step': 490}
{'loss': 0.3744, 'grad_norm': 0.68, 'learning_rate': 0.000020, 'epoch': 2.94, 'step': 500}
{'eval_loss': 0.3901, 'eval_runtime': 18.1, 'step': 500}
```

Note that `eval_loss` rows appear less frequently than `loss` rows — that's correct; you configured `eval_steps=50` and `logging_steps=10`.

---

## Reading the curve: four shapes and what they mean

Here are the four curve patterns you'll actually encounter. For each one, the ASCII sketch shows training loss as `T` and validation loss as `V`. Steps increase left to right; loss decreases top to bottom.

### Shape 1: Healthy learning

```
Loss
 │
2.5│ T
   │  T
2.0│   T  V
   │    T  V
1.5│     T   V
   │      T   V
1.0│       TT  VV
   │         TT  VV
0.5│           TTT  VVV
   │               TTTTTT  VVVVV
0.0└────────────────────────────── Steps
```

**What you see:** Both T and V are dropping together. V follows T with a small lag (validation loss is usually slightly higher than training loss — that's normal). Both flatten toward the end.

**Diagnosis:** The model is learning. Training is working.

**Action:** Let it finish. When both curves plateau, stop.

---

### Shape 2: Overfitting

```
Loss
 │
2.5│ T V
   │  T V
2.0│   T  V
   │    T   V
1.5│     T    V
   │      TT    V
1.0│        TTT  V←─── V starts rising here
   │           TTTTT  VVV
0.5│               TTTTTTT    VVVVVV↑
   │                                VVV↑
0.0└────────────────────────────────────── Steps
```

**What you see:** Training loss keeps falling, but at some point validation loss stops improving and starts climbing. The gap between T and V widens.

**Diagnosis:** Overfitting. The model has memorized your training conversations instead of learning the memory-extraction skill. On new conversations it hasn't seen, it will perform worse as training continues.

**Action:** Stop immediately. The best model is at the checkpoint *just before* V started rising — which is why you saved checkpoints and set `load_best_model_at_end=True`. If this happens early, you need more training data or a lower learning rate (see Ch19 for the full debugging playbook).

---

### Shape 3: Underfitting / not learning

```
Loss
 │
2.5│ T V
   │ T  V
2.0│  T  V
   │  T  V
1.5│   T  V
   │   T  V
1.0│   TT  VV     ← both flatten high and early
   │    TT  VV
0.8│      TTT  VVV
   │          TTTT VVVV ← barely moving
0.6└──────────────────────────── Steps
```

**What you see:** Both losses drop a little, then plateau much higher than expected (above ~0.8 for this task). The curve flattens too early.

**Diagnosis:** The model is not learning enough. Either the learning rate is too low, you ran too few steps, your data is too small, or there's a formatting problem in your training examples.

**Action:** First check your data format — make sure your training examples actually follow the expected prompt/response structure (Ch12). Then try increasing the learning rate slightly (e.g., from `2e-4` to `3e-4`) or adding more training steps. Ch19 has a systematic debugging checklist.

---

### Shape 4: Broken data / exploding loss

```
Loss
 │
4.0│    T      T
   │ T    T  T   T
3.0│   T    T      T     T
   │                  T
2.0│                     T    T
   │    V  V  V  V  V  V  V  V   ← V doesn't match T at all
1.0└──────────────────────────────── Steps
```

**What you see:** Training loss is wildly oscillating instead of smoothly declining. Or training loss appears to be dropping but validation loss never moves at all. Or loss jumps suddenly to NaN or very high values.

**Diagnosis:** Something is structurally wrong. Common causes: mixed-up data format (the label is appearing in the wrong field), learning rate too high, very small dataset with uneven batches, or a tokenization issue.

**Action:** Stop immediately. Don't wait for training to finish. Go back to Ch14's data sanity checks and verify a sample of your training rows look exactly right. If the data looks fine, cut the learning rate by 5-10x and retry.

---

## Option B: Weights & Biases (wandb) for live visual curves

The terminal works fine, but staring at scrolling numbers to spot an overfitting inflection point is hard. Weights & Biases (wandb) gives you a live web dashboard with actual charts. It's free for personal use and takes about five minutes to set up.

**Why wandb over TensorBoard?** TensorBoard is also free and ships with TensorFlow, but requires running a local server and navigating to a localhost URL. wandb streams metrics to a hosted dashboard you can share, view from another machine, or keep as a record of your training runs. For a one-person project, either works; wandb is generally easier to get started.

### Setup

```bash
# Install the wandb library (one time)
pip install wandb

# Authenticate with your free wandb account (one time)
# Creates a ~/.netrc entry with your API key
wandb login
```

You'll be prompted for an API key. Get it from https://wandb.ai/settings after creating a free account.

### Wiring wandb into your training script

```python
# ch17_wandb_setup.py
# Add this block near the top of your training script (before TrainingArguments).
# Everything else stays the same — wandb is just a logger; it doesn't change
# how training runs.

import wandb

# Initialize a wandb run. This creates a new entry in your wandb dashboard.
# Think of it like opening a logbook for this specific training run.
wandb.init(
    project="memory-extractor",   # Groups runs under one project in the dashboard.
    name="qwen3-run-1",           # A human-readable name for this specific run.
                                  # Use something you'll recognize later, like the
                                  # model name + date + any key change you made.
    config={                      # Log your hyperparameters so you can compare runs.
        "model": "Qwen/Qwen3-1.7B",
        "learning_rate": 2e-4,
        "epochs": 3,
        "batch_size": 2,
        "grad_accum": 4,
        "train_examples": 800,
    }
)
```

Then in your `TrainingArguments`, add one line:

```python
training_args = TrainingArguments(
    output_dir="./memory-extractor-checkpoints",

    # --- Add this line to enable wandb logging ---
    report_to="wandb",          # Tells the trainer to send metrics to wandb.
                                # The trainer detects wandb is installed and
                                # streams loss, learning_rate, grad_norm automatically.

    # ... all other args as before (logging_steps, eval_steps, etc.) ...
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    optim="adamw_8bit",
    fp16=True,
)
```

After you run `trainer.train()`, a URL will print to your terminal like:

```
wandb: 🚀 View run qwen3-run-1 at: https://wandb.ai/yourname/memory-extractor/runs/abc123
```

Open that in a browser and you'll see live charts for training loss, validation loss, learning rate, and gradient norm — updating every `logging_steps` steps as training progresses.

### Finishing cleanly

```python
# At the very end of your training script, after trainer.train():
trainer.train()

# Tell wandb the run is finished. This marks it as complete in the dashboard
# and flushes any remaining metrics. Without this, the run stays "running"
# in the UI even after your script ends.
wandb.finish()
```

### What to look for on the wandb dashboard

Once your run is live, open the **Charts** tab. You'll see:

- `train/loss` — your training loss over steps
- `eval/loss` — validation loss (appears at each `eval_steps` interval)
- `train/learning_rate` — the warmup and decay curve (useful to confirm warmup is happening)
- `train/grad_norm` — gradient magnitude; very high values (above ~10) can signal instability

Pin `train/loss` and `eval/loss` to the same chart (wandb lets you overlay runs) so you can directly see the gap between them. That gap is your overfitting signal.

---

## Early stopping: making training stop itself

Manually watching a curve and killing a training run at the right moment is tedious. Early stopping automates it: the trainer watches validation loss for you, and if it hasn't improved for a set number of evaluations, it stops training automatically.

```python
# ch17_early_stopping.py
# Add EarlyStoppingCallback to your trainer to stop automatically
# when validation loss stops improving.

from transformers import EarlyStoppingCallback
from trl import SFTTrainer

# EarlyStoppingCallback takes one key argument:
#   early_stopping_patience — how many consecutive eval checkpoints
#   without improvement before stopping.
#
# With eval_steps=50 and patience=3, training stops if validation loss
# hasn't improved over any of the last 3 evaluations (150 steps of patience).
# That's usually enough to confirm a plateau versus a temporary dip.
early_stop = EarlyStoppingCallback(
    early_stopping_patience=3,       # Stop after 3 consecutive non-improvements.
    early_stopping_threshold=0.001,  # Improvement must be at least 0.001 to count.
                                     # Without a threshold, noise can reset the counter.
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,      # Required for early stopping — you must have a
                                    # validation set. See Ch14 for how to create one.
    args=training_args,             # Must include load_best_model_at_end=True and
                                    # eval_strategy != "no"
    callbacks=[early_stop],         # Pass the callback here.
)

trainer.train()

# After training, check how many steps actually ran.
# If early stopping triggered, this will be less than your max.
print(f"Training stopped at step: {trainer.state.global_step}")
print(f"Best eval loss achieved:  {trainer.state.best_metric:.4f}")
```

**A note on `early_stopping_patience`:** Don't set this too low. A patience of 1 means training stops the first time validation loss fails to improve — that's too trigger-happy. Validation loss can temporarily tick up for one evaluation due to batch randomness, then resume improving. Patience of 3 is a safe default. For larger datasets (5,000+ examples), consider 5.

**Requirement:** Early stopping requires `load_best_model_at_end=True` in your `TrainingArguments`. Without it, the callback raises an error because there's no mechanism to recover the best checkpoint.

---

## Diagnosing the four states, step by step

Here is a concrete decision procedure. Run through it at the midpoint of training (e.g., step 250 of 500) and again near the end:

```python
# ch17_diagnosis.py
# A simple diagnostic script you can run mid-training or after to
# classify your training run into one of four states.
# Assumes you have a loss_history.json from the LossHistoryCallback in Ch7,
# or you can paste in values you read from the terminal or wandb.

import json
from pathlib import Path

def diagnose_training(train_losses: list[float], val_losses: list[float]) -> str:
    """
    Given lists of training and validation loss values (in order),
    print a diagnosis of the training run's current state.

    train_losses: list of training loss values over time
    val_losses:   list of validation loss values (can be shorter — evaluated less often)
    """
    if len(train_losses) < 4:
        return "Too few data points to diagnose. Check your logging_steps setting."

    # --- Check for NaN or explosions first ---
    if any(l != l for l in train_losses):  # NaN check (NaN != NaN is True)
        return "BROKEN: NaN detected in training loss. Check learning rate and data format."
    if train_losses[-1] > train_losses[0] * 1.5:
        return "BROKEN: Loss is higher now than at the start. Likely a data or LR problem."

    # --- Compute recent trend in training loss ---
    # Compare the last 20% of steps to the 20% before that
    n = len(train_losses)
    slice_size = max(1, n // 5)
    recent_train = sum(train_losses[-slice_size:]) / slice_size
    prev_train   = sum(train_losses[-2*slice_size:-slice_size]) / slice_size
    train_improving = (prev_train - recent_train) / prev_train > 0.01  # >1% improvement

    # --- Check validation loss trend if available ---
    if len(val_losses) >= 4:
        val_n = len(val_losses)
        val_slice = max(1, val_n // 4)
        recent_val = sum(val_losses[-val_slice:]) / val_slice
        prev_val   = sum(val_losses[-2*val_slice:-val_slice]) / val_slice

        val_improving = (prev_val - recent_val) / prev_val > 0.01
        val_rising    = (recent_val - prev_val) / prev_val > 0.02  # >2% increase

        gap = val_losses[-1] - train_losses[-1]

        if val_rising and not train_improving:
            return (f"OVERFITTING: Val loss is rising ({val_losses[-1]:.4f}) while "
                    f"train loss is flat ({train_losses[-1]:.4f}). Stop and use best checkpoint.")
        if val_rising and train_improving:
            return (f"EARLY OVERFITTING: Train still improving but val loss has turned up. "
                    f"Watch closely — stop within the next eval cycle if val keeps rising.")
        if not train_improving and not val_improving:
            return (f"PLATEAU: Both losses have stopped improving. "
                    f"train={train_losses[-1]:.4f}, val={val_losses[-1]:.4f}. Safe to stop.")
        if train_improving and val_improving:
            return (f"LEARNING: Both losses still dropping. "
                    f"train={train_losses[-1]:.4f}, val={val_losses[-1]:.4f}. Keep going.")
        return (f"UNCERTAIN: Mixed signals. train={train_losses[-1]:.4f}, "
                f"val={val_losses[-1]:.4f}. Check more data points.")

    # --- Only training loss available ---
    if not train_improving:
        return (f"PLATEAU (train only): Training loss has flattened at {train_losses[-1]:.4f}. "
                f"Add a validation set to confirm it's safe to stop.")
    return (f"LEARNING (train only): Training loss still dropping ({train_losses[-1]:.4f}). "
            f"Add a validation set to check for overfitting.")


# --- Example usage ---
# Paste in values from your terminal output or load from a saved log.
example_train = [2.48, 2.10, 1.78, 1.42, 1.19, 0.95, 0.78, 0.62, 0.51, 0.43,
                 0.39, 0.38, 0.37, 0.37, 0.37, 0.37]
example_val   = [2.51, 1.88, 1.31, 0.97, 0.71, 0.52, 0.45, 0.42, 0.41, 0.41]

print(diagnose_training(example_train, example_val))
# Expected output: "PLATEAU: Both losses have stopped improving. ..."
```

---

## How to tell "the data is broken" specifically

The curve shapes above cover most cases, but broken data has its own tells that are worth naming explicitly because they look similar to other problems.

**Symptom: validation loss never moves at all, no matter how long you train.**
Almost certainly a data split problem. Your validation set may have leaked into your training set (the model has already seen those examples), or your validation examples have a different format than training examples (so the model is being evaluated on something it was never trained to do). Go back to Ch14 and re-run your split with a fixed random seed.

**Symptom: training loss drops to near zero very fast (in under 50 steps on 800 examples).**
The dataset is too small or too repetitive. If you have duplicate conversations in your training data, the model learns to reproduce them almost instantly, which looks like great training loss but is actually memorization. Deduplicate your dataset (Ch14) and check how many unique conversations you actually have.

**Symptom: training loss oscillates but the oscillation amplitude never decreases.**
The learning rate is too high for your dataset size. Each step is overshooting the minimum. Cut `learning_rate` by 5x and restart.

**Symptom: `grad_norm` consistently above 5-10 throughout training.**
Gradient clipping may not be working, or your data has extreme outliers (very long inputs or outputs mixed with very short ones). SFTTrainer clips gradients by default at 1.0; check that `max_grad_norm` hasn't been overridden. Also check that your training examples have a consistent length distribution — extreme length variance makes batches noisy.

---

## Checkpoint strategy: which file is your actual model

By the end of training, your `output_dir` will contain several checkpoint subdirectories:

```
memory-extractor-checkpoints/
  checkpoint-50/
  checkpoint-100/
  checkpoint-150/
  checkpoint-200/   ← might be the best eval checkpoint
  checkpoint-250/   ← last checkpoint (not necessarily the best)
```

The checkpoint number is the step count. With `load_best_model_at_end=True`, the trainer automatically loads the best checkpoint into memory after training completes — so `trainer.model` at the end of the script is already the best version. But the checkpoint subdirectory files are still there if you need to reload manually.

To find which checkpoint was best:

```python
# ch17_find_best_checkpoint.py
# After training, find which checkpoint had the best validation loss.

# trainer.state.best_model_checkpoint gives the path to the best checkpoint directory.
print("Best checkpoint:", trainer.state.best_model_checkpoint)
print("Best eval loss:", trainer.state.best_metric)

# If you need to reload this checkpoint later (e.g., in a new session):
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=trainer.state.best_model_checkpoint,  # path to the best checkpoint dir
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```

Ch21 covers saving and merging the final model in depth.

---

## Quick reference: parameter cheat-sheet

| Parameter | What it does | Recommended starting value |
|---|---|---|
| `logging_steps` | How often to print training loss | 10 |
| `eval_strategy` | When to run validation | `"steps"` |
| `eval_steps` | How often to evaluate | ~10% of total steps |
| `save_strategy` | When to save checkpoints | `"steps"` (match eval_steps) |
| `save_steps` | How often to save | Same as eval_steps |
| `save_total_limit` | Max checkpoints to keep | 3 |
| `load_best_model_at_end` | Reload best checkpoint after training | `True` |
| `metric_for_best_model` | Metric to judge "best" | `"eval_loss"` |
| `greater_is_better` | Is higher metric better? | `False` (lower loss = better) |
| `report_to` | Where to send metrics | `"wandb"` or `"tensorboard"` |
| `early_stopping_patience` | Evaluations without improvement before stopping | 3 |

---

## Common mistakes

**Mistake 1: Setting `logging_steps` too high.**

With the default of 500, a 600-step run gives you exactly one loss reading. You can't see any curve at all. Set `logging_steps=10` or `logging_steps=20` for every run so you always have enough data points to read the shape.

**Mistake 2: Not passing `eval_dataset` to SFTTrainer.**

If you configure `eval_strategy="steps"` but forget to pass `eval_dataset` to the trainer, it silently skips evaluation. You'll never see an `eval_loss` line. Always pass both `train_dataset` and `eval_dataset`. See Ch14 for how to create a train/validation split.

**Mistake 3: Using `load_best_model_at_end=True` without `save_strategy`.**

This raises a cryptic error at the end of training: `TrainerCallback requires save_strategy to match eval_strategy`. The fix: make sure `save_strategy="steps"` and `save_steps` matches or divides `eval_steps`. If you evaluate every 50 steps, save every 50 steps.

**Mistake 4: Trusting only training loss when it looks great.**

A training loss that drops to 0.05 sounds incredible. But if your training dataset has 200 examples and you ran 10 epochs, the model has seen each example 10 times — it's basically memorized them. The only way to know if it generalizes is to check validation loss. If you trained without a validation split and aren't sure whether you've overfit, run the evaluation script from Ch18 on a small set of conversations you held back.

**Mistake 5: Stopping too early because of one bad eval step.**

Validation loss occasionally ticks up for one eval interval due to random batch effects, then resumes dropping. Don't stop training because of a single uptick. Use `early_stopping_patience=3` to require three consecutive non-improvements before stopping. Looking at the trend over several evaluations, not a single point, is the reliable signal.

**Mistake 6: Running wandb in a cloud notebook without calling `wandb.finish()`.**

Cloud notebook kernels sometimes get killed before the wandb process flushes its final metrics. Always call `wandb.finish()` explicitly at the end of your script. Otherwise your wandb dashboard shows the run as still "running" indefinitely, and the last batch of metrics may not appear.

**Mistake 7: Comparing loss numbers across different models.**

A validation loss of 0.45 on Qwen3-1.7B is not the same as 0.45 on Gemma 3-4B. Loss is relative to the model's tokenizer and architecture. What matters is: (a) is training loss going down on this run, and (b) does validation loss track training loss without diverging. Cross-model comparisons require task-specific evaluation metrics, which Ch18 covers.

---

## Recap

- Training loss tells you the model is learning. Validation loss tells you whether it's generalizing. You need both.
- A healthy run: both losses drop together and plateau near the end.
- Overfitting: training loss keeps dropping but validation loss starts rising. Stop and use the best checkpoint.
- Underfitting: both losses plateau high and early. Check data format, increase steps or learning rate.
- Broken data: loss oscillates wildly, or validation loss never moves. Diagnose with Ch14's data checks.
- Set `logging_steps=10` so you can actually see the curve shape in the terminal.
- Set `eval_steps` to roughly 10% of your total steps to evaluate often enough to catch problems early.
- Set `save_steps` to match `eval_steps` and `load_best_model_at_end=True` so the trainer automatically gives you the best checkpoint, not just the last one.
- wandb gives you live visual charts with one extra line (`report_to="wandb"`) — worth the five-minute setup.
- `EarlyStoppingCallback` with `patience=3` stops training automatically when validation loss stops improving.

## Next

**Ch18 - Did It Actually Work? Evaluating Memory Extraction** — now that training is done and you have a model, how to rigorously measure whether it actually extracts memories correctly from conversations it has never seen.
