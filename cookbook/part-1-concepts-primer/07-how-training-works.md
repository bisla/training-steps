# Ch7 - How Training Actually Works (Loss, Gradients, Epochs)

You've picked a base model (Ch10 will cover the specifics) and you understand why LoRA lets you
fine-tune it without touching most of its weights (Ch6). Now the obvious question: what actually
*happens* when you run a training script? What is the machine doing for those 20-40 minutes
while the GPU fan spins?

This chapter answers that. No calculus. No Greek letters. Just the loop that every fine-tune
runs, explained in terms a Python developer can reason about.

---

## What you'll learn

- The four-step learning loop that repeats thousands of times during training
- What "loss" is and why minimizing it is the whole game
- What epochs, steps, batch size, learning rate, and warmup mean — and which knob each maps to
- How to tell overfitting from underfitting, and what to do about each
- What a healthy loss curve looks like so you know when to stop

---

## Concepts you need first

### Loss: measuring how wrong the model is

Think of loss as a score for *surprise*. If you show a language model the text
`"The cat sat on the"` and it confidently predicts `"mat"` — and that was the right answer —
it's not very surprised. Low surprise → low loss. If it confidently predicts `"volcano"`, that's
very surprising given the context. High surprise → high loss.

More precisely: loss is a single number that measures how badly the model's predictions differ
from the correct answers on a batch of examples. Lower is better. Zero would mean perfect
predictions every time (which you don't want — more on that in a moment).

For our memory-extraction task, the model is predicting the tokens in the correct JSON output
one at a time. If the right next token is `"text"` (a field name in our schema) and the model
assigns it a high probability, loss stays low. If the model wanted `"content"` instead, loss
goes up.

### Gradient: the direction to nudge

Here's the only piece of calculus you need, translated into plain English: a gradient is a
vector that answers the question *"if I increase this weight slightly, does the loss go up or
down, and by how much?"*

You can think of it like hiking in fog. You can't see the valley (the minimum loss), but you
can feel the slope under your feet. The gradient tells you which direction is downhill. The
optimizer uses that direction to take a small step. Repeat thousands of times, and you walk
downhill toward lower loss.

The optimizer is the algorithm that decides exactly how big a step to take and in what
direction. The most common one you'll encounter is AdamW. You don't need to understand its
internals — just know it's battle-tested and the default in almost every fine-tuning setup.

### Learning rate: the step size

The learning rate controls how large each step downhill is. Too large: you overshoot the valley
and bounce around. Too small: training takes forever and may get stuck. A typical starting value
for LoRA fine-tuning is somewhere in the range of `2e-4` to `5e-4`. Unsloth's defaults are
sensible; you'll rarely need to stray far from them.

### Warmup: easing into training

At the very start of training, the model's predictions are chaotic and the gradients are huge.
If you take full-size steps immediately, you can damage the model's existing knowledge in the
first few batches. Warmup solves this by starting with a near-zero learning rate and ramping it
up over the first few hundred steps. It's like letting an engine warm up before flooring the
accelerator.

### Batch size: how many examples per step

Instead of updating weights after every single training example (slow and noisy) or after the
entire dataset (uses too much memory), training processes a *batch* of examples at once, then
does one weight update. A batch size of 2-4 is common for LoRA fine-tuning on a 24 GB GPU.
Larger batches give smoother gradient estimates but use more VRAM.

### Gradient accumulation: faking a bigger batch

If you want the effect of a batch size of 16 but only have VRAM for 2 examples at a time, you
can process 8 mini-batches of 2, accumulate (add together) the gradients across all of them,
and then do a single weight update. The model sees it as one batch of 16. The
`gradient_accumulation_steps` parameter controls how many mini-batches to accumulate before
updating.

### Epoch: one full pass through your data

An epoch is one complete pass through your entire training dataset. If you have 1,000 training
examples and a batch size of 4, one epoch is 250 steps. Fine-tuning small task-specific
datasets typically uses 2-5 epochs. Running many more risks overfitting (explained below).

### Step: one weight update

A step is one gradient computation + one weight update. Step count = (dataset size / batch size)
× epochs × (1 / gradient_accumulation_steps). This is the number you'll watch ticking up in
your training logs.

---

## The training loop, in full

Here's the loop that runs thousands of times when you execute a fine-tune:

```
for each step:
    1. PREDICT  — feed a batch of input/output pairs through the model
    2. MEASURE  — compute the loss (how wrong were the predictions?)
    3. BACKPROP — compute gradients (which direction improves each weight?)
    4. UPDATE   — take a step downhill (optimizer adjusts the weights)
```

That's it. Four operations, repeated. All the complexity in training libraries is scaffolding
around these four lines.

Let's make this concrete with a real example from our memory-extraction task.

---

## Walking through one training step

Suppose your training dataset has this row:

**Input (the prompt):**
```
Extract memories from this conversation:

User: I prefer dark mode in all my apps.
Assistant: Got it! I'll remember that.
```

**Output (the label, i.e., what the model should produce):**
```json
[
  {
    "text": "User prefers dark mode in all apps.",
    "type": "preference",
    "entities": ["dark mode"]
  }
]
```

During training, the model sees the input and tries to predict the output token by token. At
each position, it assigns a probability to every token in its vocabulary. If the correct next
token is `[` (opening bracket of the JSON array) and the model gives it a 90% probability,
that step contributes very little to the loss. If it gives `[` only a 5% probability and wants
to emit something else, that step contributes a lot to the loss.

After predicting all the output tokens, the loss for this example is computed. Then gradients
are calculated (which LoRA weights nudged the prediction in the wrong direction?) and those
weights are adjusted by a tiny amount in the direction that would have made the prediction
better.

Run this on enough examples and the model starts to internalize: when the prompt is in this
format, the output should be a JSON array with `text`, `type`, and `entities` fields. The
schema becomes part of the model's behavior.

---

## Runnable code: instrumenting the training loop

You won't write the training loop yourself — `trl`'s `SFTTrainer` handles it. But you can
observe every step by passing a custom callback. Here's a complete, runnable example that
attaches a loss logger to a training run and saves the loss history to a file you can inspect
later.

```python
# ch07_loss_logger.py
# Run this AFTER you have a dataset and a model set up (see Ch15 for the full script).
# This snippet shows how to hook into the training loop and record loss at every step.

import json
from pathlib import Path
from transformers import TrainerCallback

# -----------------------------------------------------------------------------
# 1. Define a callback that records loss at each logging step.
#    A callback is just a Python class with methods that fire at specific
#    points in training (on_log, on_epoch_end, etc.).
# -----------------------------------------------------------------------------
class LossHistoryCallback(TrainerCallback):
    def __init__(self):
        self.history = []  # will hold {"step": ..., "loss": ...} dicts

    def on_log(self, args, state, control, logs=None, **kwargs):
        # `logs` is a dict the trainer populates with training metrics.
        # It always contains "loss" during training steps.
        if logs and "loss" in logs:
            self.history.append({
                "step": state.global_step,
                "loss": round(logs["loss"], 4),
                # learning_rate is also logged if you want to watch warmup
                "lr": logs.get("learning_rate", None),
            })

    def save(self, path: str = "loss_history.json"):
        # Dump the history to a JSON file so you can inspect or plot it later.
        Path(path).write_text(json.dumps(self.history, indent=2))
        print(f"Loss history saved to {path} ({len(self.history)} steps recorded)")


# -----------------------------------------------------------------------------
# 2. Wire the callback into SFTTrainer.
#    Assume `model`, `tokenizer`, `train_dataset`, and `training_args` are
#    already set up (see Ch15 for the full setup). This snippet focuses only
#    on the callback attachment.
# -----------------------------------------------------------------------------
from trl import SFTTrainer

loss_cb = LossHistoryCallback()

# Pass the callback in the `callbacks` list — SFTTrainer accepts a list of
# any TrainerCallback subclasses.
trainer = SFTTrainer(
    model=model,                  # your LoRA-wrapped model from Unsloth
    tokenizer=tokenizer,
    train_dataset=train_dataset,  # your memory-extraction dataset
    args=training_args,           # TrainingArguments you configured
    callbacks=[loss_cb],          # <-- our custom loss recorder
)

# -----------------------------------------------------------------------------
# 3. Train, then save the loss history.
# -----------------------------------------------------------------------------
trainer.train()
loss_cb.save("loss_history.json")  # inspect this file after training

# -----------------------------------------------------------------------------
# 4. Quick sanity-check: print first and last few steps.
#    If loss is decreasing over time, training is working.
# -----------------------------------------------------------------------------
history = loss_cb.history
if history:
    print("First 3 steps:", history[:3])
    print("Last  3 steps:", history[-3:])
    first_loss = history[0]["loss"]
    final_loss = history[-1]["loss"]
    drop_pct = (first_loss - final_loss) / first_loss * 100
    print(f"Loss dropped from {first_loss} → {final_loss} ({drop_pct:.1f}% reduction)")
```

When you run a full training job, `loss_history.json` will contain something like:

```json
[
  {"step": 10,  "loss": 2.431, "lr": 0.000012},
  {"step": 20,  "loss": 1.987, "lr": 0.000025},
  {"step": 30,  "loss": 1.654, "lr": 0.000050},
  ...
  {"step": 500, "loss": 0.312, "lr": 0.000180}
]
```

Loss starting high (above 2.0 is normal), dropping steadily, and flattening out below 0.5 is
the classic healthy shape. We'll look at how to plot and interpret this in Ch17.

---

## Overfitting vs. underfitting: the core tension

This is the most important conceptual divide in all of machine learning. And you don't need any
math to understand it — just this analogy.

Imagine you're studying for a driving test using a practice booklet of 50 questions. Two
students approach it differently:

**Student A (underfitting):** Barely glances at the booklet. On exam day, knows almost nothing.
The model hasn't learned the underlying skill.

**Student B (overfitting):** Memorizes every question and answer word-for-word. Can recite the
practice booklet perfectly. On exam day — with different question wording — freezes. The model
has memorized the training data instead of learning the skill.

**The goal:** A student who understood *why* each answer is correct, so they can handle new
questions they haven't seen before.

In fine-tuning terms:

| Situation | Training loss | Validation loss | What it means |
|---|---|---|---|
| Underfitting | High | High | Model hasn't learned the task yet |
| Healthy | Low | Low (similar to train) | Model has learned the skill |
| Overfitting | Very low | Much higher than train | Model memorized training examples |

You detect overfitting by watching the **validation loss** (loss on examples the model never
trained on). If training loss keeps dropping but validation loss starts climbing — the model is
memorizing, not generalizing.

For memory extraction specifically, overfitting looks like this: the model produces perfect
JSON for conversations it saw during training, but starts hallucinating fields, dropping
memories, or mis-classifying types on new conversations it hasn't seen.

---

## What a healthy loss curve looks like

A healthy training run has three phases:

**Phase 1 — Warmup (first ~5% of steps):** Loss may be erratic or even briefly rise. The
learning rate is ramping up. This is normal; don't panic.

**Phase 2 — Descent (the bulk of training):** Loss drops steadily, often rapidly at first then
more gradually. This is the model learning. You want to see this curve be smooth and
consistently decreasing.

**Phase 3 — Plateau:** Loss flattens. Further training yields diminishing returns. This is your
signal to stop (or to try a different learning rate or more data).

A loss curve that looks like a jagged EKG the entire time usually means your learning rate is
too high. A curve that barely moves means your learning rate is too low or your data has
problems (Ch14 covers data sanity-checking).

Here's a simple script that reads your saved loss history and prints an ASCII sparkline so you
can see the shape without leaving the terminal:

```python
# ch07_sparkline.py
# Reads loss_history.json and prints a rough ASCII loss curve.
# Useful for quick inspection without needing matplotlib.

import json
import math
from pathlib import Path

def sparkline(values: list[float], width: int = 60) -> str:
    """
    Map a list of floats to a string of ASCII block characters.
    Higher values appear taller; lower values appear shorter.
    This gives a rough visual of the curve shape.
    """
    blocks = " ▁▂▃▄▅▆▇█"  # 9 levels, from empty to full block
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return blocks[0] * len(values)
    # Downsample to `width` characters if history is longer
    step = max(1, len(values) // width)
    sampled = values[::step][:width]
    chars = []
    for v in sampled:
        # Normalize to 0-1, then map to 0-8 (9 levels)
        normalized = (v - min_v) / (max_v - min_v)
        # Invert: high loss → high block, low loss → short block
        idx = int(normalized * 8)
        chars.append(blocks[idx])
    return "".join(chars)

# Load the history you saved during training
history = json.loads(Path("loss_history.json").read_text())
losses = [row["loss"] for row in history]

print(f"\nLoss curve ({len(losses)} steps recorded):")
print(f"  Start: {losses[0]:.4f}  →  End: {losses[-1]:.4f}")
print()
print("  HIGH │" + sparkline(losses))
print("   LOW │" + " " * 60)
print()

# Rough verdict
ratio = losses[-1] / losses[0]
if ratio < 0.3:
    print("Looks good: loss dropped by more than 70%. Model is learning.")
elif ratio < 0.6:
    print("Moderate drop. Consider more data or more epochs.")
else:
    print("Small drop. Check your data format and learning rate.")
```

---

## Tying it to the TrainingArguments you'll use in Ch15

When you actually run your fine-tune (Ch15), you'll configure a `TrainingArguments` object.
Here's a preview of how the concepts in this chapter map to specific parameters:

```python
# ch07_training_args_preview.py
# This is not a runnable training script — it's a reference showing which
# parameter controls which concept from this chapter.

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./memory-extractor-checkpoints",

    # --- Epochs and steps ---
    num_train_epochs=3,          # How many full passes through your data
    # Alternatively: max_steps=500  (use whichever you find easier to reason about)

    # --- Batch size and gradient accumulation ---
    per_device_train_batch_size=2,   # Examples per GPU per step (keep low for VRAM)
    gradient_accumulation_steps=4,   # Effective batch = 2 × 4 = 8

    # --- Learning rate and warmup ---
    learning_rate=2e-4,          # Step size; 2e-4 is a reasonable LoRA default
    warmup_ratio=0.05,           # Warm up over the first 5% of total steps
    # warmup_steps=100           # Alternative: set warmup as a fixed step count

    # --- Logging (how often to record loss) ---
    logging_steps=10,            # Record loss every 10 steps
    save_steps=100,              # Save a checkpoint every 100 steps

    # --- Evaluation (requires a validation split) ---
    eval_strategy="steps",       # Run evaluation periodically
    eval_steps=50,               # Evaluate every 50 steps

    # --- Optimizer ---
    optim="adamw_8bit",          # AdamW in 8-bit precision; saves VRAM, same behavior
)
```

You don't need to memorize all of these now. Ch15 walks through the full script with every
parameter explained in context. This preview is here so that when you see `warmup_ratio` in
the wild, you know exactly what concept it connects to.

---

## Common mistakes

**Mistake 1: Training for too many epochs on a small dataset.**

If you have 500 training examples and you train for 20 epochs, the model sees each example 20
times. It will start memorizing specific conversations from your dataset. The fix: stick to 2-5
epochs for datasets under a few thousand examples, and always watch validation loss.

**Mistake 2: Setting the learning rate too high.**

Symptoms: loss oscillates wildly and doesn't settle, or jumps upward after initially dropping.
Fix: cut the learning rate by 5-10x and restart. `2e-4` is usually a safe starting point for
LoRA; going above `5e-4` is risky with small datasets.

**Mistake 3: Ignoring validation loss entirely.**

It's tempting to only watch training loss because it's always right there in the logs. But
training loss alone tells you nothing about generalization. Always hold out 10-15% of your data
as a validation split (Ch14 covers this) and watch both curves.

**Mistake 4: Stopping too early because the curve looks flat.**

Sometimes loss plateaus briefly and then drops again after more steps. The safe approach:
let training finish unless validation loss is clearly rising (which means overfitting is
underway). If you're using `max_steps`, add a conservative buffer.

**Mistake 5: Forgetting that loss units depend on the model and task.**

A loss of 1.2 might be excellent for one model and terrible for another. What matters is the
*trend* (is it going down?) and the *ratio* (where did it end relative to where it started?),
not the absolute number. Comparing loss values across completely different model architectures
is not meaningful.

**Mistake 6: Not logging frequently enough.**

The default logging interval in some setups is every 500 steps. On a 600-step training run,
that means you see only one loss reading. Set `logging_steps=10` so you can actually see the
curve shape.

---

## Recap

- Training is a loop: predict → measure loss → compute gradients → update weights. Repeat.
- Loss is a single number measuring how surprised the model was by the correct answer. Lower is
  better.
- A gradient points in the direction that would reduce loss. The optimizer uses it to take a
  small step in that direction each step.
- Learning rate controls step size. Too large → chaotic training. Too small → slow progress.
- Warmup prevents early large steps from damaging the model's existing knowledge.
- Batch size is examples per step. Gradient accumulation lets you simulate larger batches with
  less VRAM.
- An epoch is one full pass through your dataset. Fine-tuning typically uses 2-5 epochs.
- Overfitting = memorizing training examples. Underfitting = not learning enough. Both are
  visible in the gap between training loss and validation loss.
- A healthy loss curve drops quickly early, then gradually flattens. Wild oscillations signal
  a learning rate problem.

## Next

**Ch8 - Hardware, GPUs, and Setting Up Your Environment** — where to actually run this training
loop, what GPU you need, and how to get your machine ready to run Unsloth.
