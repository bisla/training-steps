# Ch6 - LoRA and QLoRA Without the Math Headache

You've picked your base model (Ch10 will help you choose between Qwen3 and Gemma 3). Now the obvious question: how do you actually change it? A 7-billion-parameter model is a file that weighs ~14 GB in its raw form. Rewriting every number in it from scratch — what researchers call "full fine-tuning" — would need 80 GB of GPU memory or more, cost hundreds of dollars per run, and take days. That's not a realistic option for most people.

LoRA and QLoRA are the engineering workarounds that make fine-tuning feasible on a single consumer GPU or a free Colab session. This chapter explains both — no calculus required.

---

## What you'll learn

- What LoRA is and why the "sticky notes on a textbook" mental model is accurate
- What `r` (rank) and `alpha` mean, and which values to start with
- What quantization is and how QLoRA stacks it on top of LoRA to slash memory further
- The rough VRAM you need for 1B, 4B, and 7–8B models under QLoRA
- Which LoRA parameters to set in your training script and what typical starting values look like

---

## Concepts you need first

### What are model weights, anyway?

A language model is, at its core, a very large table of numbers. Those numbers — called **weights** or **parameters** — encode everything the model learned during its original training: grammar, facts, reasoning patterns, how to format JSON. When the model reads your prompt and produces an output, it's doing arithmetic with those numbers.

Fine-tuning means adjusting those numbers so the model gets better at your specific task — in our case, extracting memories from conversation chunks and formatting them as a JSON list.

The problem: a 7B model has 7,000,000,000 of these numbers. Changing all of them at once is expensive in memory and time.

### The sticky-note idea (LoRA's core trick)

Imagine you buy a medical textbook and it's perfect — except it knows nothing about your hospital's specific procedures. You have two options:

1. Reprint the entire textbook with your procedures woven in. Expensive. Slow.
2. Write your procedures on sticky notes and attach them to the relevant pages. Fast, cheap, and you can peel them off if you change your mind.

LoRA (**Low-Rank Adaptation**) is option two. Instead of rewriting the original weights, it freezes them completely and learns a pair of small "adapter" matrices that sit alongside the original. During inference (when the model generates output), the model adds the adapter's contribution to each layer's output. The original weights are never touched.

**One-line definition:** LoRA freezes the base model's weights and learns small side matrices that nudge each layer's behavior toward your task.

**Why it matters for memory extraction:** our training dataset (built in Ch13) might have a few thousand examples. LoRA lets the model learn the JSON output format and memory-typing logic from those examples without forgetting its general language ability — and without needing a server-grade GPU to do it.

### What is "rank" (the `r` parameter)?

The adapter matrices LoRA learns are intentionally tiny. Their size is controlled by `r`, the **rank**.

Think of rank as the number of "dimensions of change" you're allowing. A rank of 1 means the adapter can only learn one kind of adjustment. A rank of 64 means it can learn 64 independent adjustments. Higher rank = more expressive adapter = more VRAM and slower training.

Typical starting value: **`r=16`**. For a narrow task like memory extraction, this is usually enough. If your results are poor after a full training run, you can try `r=32` or `r=64` — but start low.

### What is `alpha`?

Alpha controls the **scaling** of the adapter's contribution. Specifically, LoRA multiplies the adapter output by `alpha / r` before adding it to the original layer output.

In practice, the common convention is to set `alpha` equal to `r` (so the scale factor is 1.0) or to `2 * r` (scale factor 2.0, which gives the adapter a slightly stronger voice).

Typical starting value: **`alpha=16`** (matching `r=16`) or **`alpha=32`** (double `r`). Don't overthink this — it's the least sensitive knob on the board.

### What is `lora_dropout`?

Dropout is a regularization trick: during training, it randomly zeroes out some fraction of the adapter's activations. This prevents the adapter from memorizing your training data too closely (a problem called overfitting).

For small datasets (under ~10,000 examples), a small dropout helps. For large datasets, you can set it to 0.

Typical starting value: **`dropout=0.05`** (5%).

### Quantization: rounding numbers to save memory

A model weight is normally stored as a 32-bit or 16-bit floating-point number. "Floating-point" means it can represent a huge range of values with high precision — but it costs memory.

**Quantization** rounds those numbers down to a coarser representation: instead of a full 32-bit float, store a 4-bit integer. That's 8× fewer bits per weight.

The analogy: imagine you're storing a color. You could store it as `(R=173, G=42, B=91)` — precise, 24 bits. Or you could round it to the nearest 16 named colors — imprecise, 4 bits. You lose some fidelity, but for most purposes the color still looks right.

**One-line definition:** quantization rounds model weights to fewer bits to shrink how much memory they occupy.

The quality tradeoff is real but usually small. A 4-bit quantized model performs close to — often within 1–2% of — its full-precision version on most tasks. For a domain-specific fine-tune like ours, the gap is rarely noticeable.

### QLoRA: stacking both tricks

**QLoRA** combines quantization and LoRA:

1. Load the frozen base model in **4-bit precision** (using a library called `bitsandbytes`). This cuts the base model's VRAM footprint by ~4×.
2. Attach LoRA adapters in **16-bit precision** (adapters stay full-precision so they train cleanly).
3. Train only the adapter weights; the frozen 4-bit base never changes.

The result: you get the quality of a large model with the memory footprint of something much smaller.

---

## Rough VRAM math

Here are ballpark numbers for loading a model under QLoRA for training. These are estimates — actual usage depends on batch size, sequence length, and optimizer state.

| Model size | 4-bit base | + adapters + optimizer | Minimum GPU VRAM |
|---|---|---|---|
| 1B params | ~0.7 GB | ~3–4 GB total | 6 GB (RTX 3060) |
| 4B params | ~2.5 GB | ~6–8 GB total | 8 GB (RTX 3070) |
| 7–8B params | ~4.5 GB | ~10–14 GB total | 16 GB (RTX 3080 Ti / A10G) |

Colab's free T4 has 15 GB of VRAM — enough for a 7B model with careful batch-size settings. Colab Pro's A100 (40 GB) gives you comfortable headroom.

> **Note on Unsloth:** The Unsloth library (introduced in Ch9 and used throughout Ch15 onward) applies additional memory optimizations on top of QLoRA. In practice, Unsloth users often see 30–40% less VRAM usage than the table above suggests. The numbers above are conservative baselines without Unsloth's tricks.

---

## Applying LoRA in code

Let's make this concrete. Here is how you attach a LoRA adapter to a model using Unsloth — the same pattern you'll use in your full training script in Ch15.

```python
# Install if you haven't already:
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# pip install bitsandbytes transformers trl peft

from unsloth import FastLanguageModel

# ---------------------------------------------------------------------------
# Step 1: Load the base model in 4-bit (this is the QLoRA part).
# Unsloth handles the quantization automatically — you just set load_in_4bit=True.
# max_seq_length is the longest token sequence the model will see during training.
# For our memory-extraction task, 2048 is plenty: conversations aren't that long.
# ---------------------------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B",  # base model from Hugging Face Hub
    # ^ This is the Qwen3 8B model. Ch15 will confirm this exact name (or the
    #   Gemma 3 alternative: "unsloth/gemma-3-4b-it"). Use whichever your
    #   training chapter specifies — the LoRA setup code below is identical
    #   for both models.
    max_seq_length=2048,
    dtype=None,          # None = auto-detect; usually bfloat16 on modern GPUs
    load_in_4bit=True,   # enables QLoRA 4-bit quantization of the frozen base
)

# ---------------------------------------------------------------------------
# Step 2: Attach LoRA adapters to the model.
# This is what "adds the sticky notes." The base weights are frozen;
# only the adapter parameters (a tiny fraction of total params) will train.
# ---------------------------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,            # rank — how many "dimensions of change" the adapter can learn
    lora_alpha=16,   # scaling factor; equal to r means scale=1.0, a safe default
    lora_dropout=0.05,  # 5% dropout to reduce overfitting on small datasets
    target_modules=[
        # These are the layer types inside the transformer that LoRA will adapt.
        # "q_proj" and "v_proj" are the query and value attention matrices —
        # the parts most responsible for "what to pay attention to."
        # Adding k_proj, o_proj, and the feed-forward layers (gate/up/down)
        # gives the adapter more expressiveness without a huge VRAM cost.
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",     # don't add a trainable bias — keeps adapters small
    use_gradient_checkpointing="unsloth",
    # ^ Gradient checkpointing is a memory-saving technique: instead of storing
    #   every intermediate activation in GPU memory during the forward pass, it
    #   recomputes them on the fly during the backward pass. You trade a bit of
    #   extra compute for a significant reduction in VRAM usage — essential for
    #   long sequences. Passing the string "unsloth" (rather than True) uses
    #   Unsloth's own optimized implementation, which is more memory-efficient
    #   than the standard PyTorch version. If you're on an older Unsloth release
    #   that doesn't accept a string here, pass True instead.
    random_state=42,  # for reproducibility
)

# ---------------------------------------------------------------------------
# Step 3: Verify what's actually trainable.
# The output will show that only ~1-2% of total parameters are trainable —
# that's the LoRA adapter. Everything else is frozen.
# ---------------------------------------------------------------------------
model.print_trainable_parameters()
# Expected output (approximate):
# trainable params: 41,943,040 || all params: 7,656,849,408 || trainable%: 0.5476
```

That last line is the key insight: you are training roughly **0.5% of the model's total parameters**. The other 99.5% are frozen and loaded in 4-bit. This is why QLoRA fits on a single GPU.

---

### Which layers should you target?

The `target_modules` list tells LoRA which layer types to attach adapters to. Here's a quick guide:

| What to include | When |
|---|---|
| `q_proj`, `v_proj` only | Minimal VRAM, good for very small GPUs (6–8 GB). Try this first if you're tight on memory. |
| + `k_proj`, `o_proj` | Default for most tasks. What the code above uses. |
| + `gate_proj`, `up_proj`, `down_proj` | Adds adapters to the feed-forward layers too. More expressive; better for tasks that need new output formats (like our JSON schema). Costs ~10–15% more VRAM. |

For memory extraction — where we're teaching the model a specific structured output format — targeting all seven module types is worth it.

---

### Verifying the adapter is wired up

Before committing to a full training run, it's useful to do a quick sanity check: can the model (with adapters attached but before training) still produce output? If it crashes here, something is misconfigured.

```python
# Quick sanity check: run one forward pass with a dummy input.
# This verifies the model loaded correctly and the adapter is attached.
# This is NOT training — we're just checking the plumbing.

from unsloth.chat_templates import get_chat_template

# Apply the correct chat template for this model family.
# Each model family has its own prompt format, and using the wrong one causes
# silent garbling of your inputs (the model sees malformed text and produces
# nonsense). For Qwen3 use "qwen-2.5" (Qwen3 shares Qwen2.5's chat format);
# for Gemma 3 use "gemma". To find the right string for any model, check the
# model card on Hugging Face or the Unsloth docs — search for "chat_template".
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
# If you switched to Gemma 3 ("unsloth/gemma-3-4b-it"), change this to:
# tokenizer = get_chat_template(tokenizer, chat_template="gemma")

# A minimal memory-extraction prompt using the book's shared JSON schema.
# Every memory object in this book has exactly three fields:
#   - "text"     : the memory string (e.g. "User lives in Austin")
#   - "type"     : a category label (e.g. "location", "preference")
#   - "entities" : a list of named entities mentioned (e.g. ["Austin"])
# Ch12 builds the full dataset around this schema; Ch15 trains on it.
# The test below verifies the model can at least attempt this format before training.
test_messages = [
    {
        "role": "user",
        "content": (
            "Extract memories from this conversation as a JSON list.\n\n"
            "User: I just moved to Austin last month. Loving the weather.\n"
            "Assistant: Nice! Do you miss the East Coast?"
        )
    }
]

# Tokenize the prompt.
inputs = tokenizer.apply_chat_template(
    test_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

# Enable inference mode (faster; turns off gradient tracking).
FastLanguageModel.for_inference(model)

# Generate a short response. max_new_tokens=128 is enough for a few memories.
outputs = model.generate(input_ids=inputs, max_new_tokens=128, use_cache=True)

# Decode and print.
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Before any training, the base model will probably produce plausible-looking JSON, or it might produce something malformed — both are expected. The point is that it runs. After training (Ch15), the output will consistently match our schema.

---

### The full parameter picture

Here's a summary table of every LoRA parameter you'll set, what it does, and what to start with:

| Parameter | What it controls | Starting value | When to change |
|---|---|---|---|
| `r` | Adapter expressiveness | `16` | Raise to `32` or `64` if quality is poor after full training |
| `lora_alpha` | Adapter output scaling | `16` (= r) or `32` (= 2×r) | Leave at `r` unless results are flat |
| `lora_dropout` | Overfitting prevention | `0.05` | Set to `0` for large datasets (>20k rows); raise to `0.1` for very small ones |
| `target_modules` | Which layers get adapters | All 7 (q/k/v/o + gate/up/down) | Reduce to q+v if VRAM is tight |
| `bias` | Whether biases are trainable | `"none"` | Rarely changed |

---

## Common mistakes

**Setting `r` way too high.** A rank of 128 or 256 sounds like "more learning" but it mainly means more VRAM, slower training, and a higher risk of the adapter memorizing noise in your training data. Start at 16.

**Setting `alpha` much lower than `r`.** If `alpha` is, say, 1 and `r` is 16, the adapter's scale factor is 0.0625 — so tiny that its adjustments barely register. Keep `alpha >= r`.

**Forgetting to call `FastLanguageModel.for_inference(model)`** before generating output during evaluation. Without this call, Unsloth doesn't apply its inference-time optimizations, and generation will be noticeably slower.

**Targeting too few layers on a structured-output task.** For tasks that require a specific format — like our JSON memory schema — the feed-forward layers (`gate_proj`, `up_proj`, `down_proj`) carry a lot of the formatting behavior. If you only target `q_proj` and `v_proj`, the adapter may struggle to reliably produce valid JSON.

**Conflating quantization with training precision.** The base model is quantized to 4-bit for storage efficiency. The adapter trains in 16-bit (bfloat16). These are separate things. Don't try to train the adapters in 4-bit — gradient math at that precision is unreliable.

**Running out of VRAM mid-training.** The VRAM numbers in the table above are for loading. During training, the optimizer also needs memory. If you see an out-of-memory error, the first fix is to halve your `per_device_train_batch_size`. More on this in Ch16 (Hyperparameters).

---

## Recap

- **LoRA** freezes the base model's weights and learns tiny "adapter" side matrices — like sticky notes on a textbook. Only the adapters train.
- **`r` (rank)** controls how expressive the adapter is. Start at `r=16`; raise if quality is poor.
- **`alpha`** scales the adapter's output. Start equal to `r` or double it.
- **`lora_dropout`** reduces overfitting. Start at `0.05`.
- **QLoRA** stacks quantization on top of LoRA: the frozen base loads in 4-bit (~4× smaller), while the adapters stay in 16-bit for clean training.
- A 7–8B model under QLoRA needs roughly 10–14 GB of VRAM for training; Unsloth's extra optimizations often shave this down further.
- Only ~0.5% of parameters are actually trainable — that's what makes this feasible on a single GPU.
- For memory extraction, target all seven module types (q/k/v/o + gate/up/down) to give the adapter enough expressiveness to learn the JSON output format reliably.

## Next

**Ch7 - How Training Actually Works (Loss, Gradients, Epochs)** — now that you know what gets trained (the adapters) and how they're sized, we'll walk through what actually happens during the training loop: what a loss number means, how gradients update the adapter weights, and how to tell when the model has trained long enough.
