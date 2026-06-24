# Ch10 - Choosing Your Base Model: Qwen vs Gemma

You've spent the last few chapters understanding *how* training works. Now comes a decision that shapes everything downstream: which model do you start from?

This chapter is your buyer's guide. By the end you'll have made a concrete choice — one specific model, one Unsloth model ID string, and a working load script — and you'll understand exactly why you made that choice instead of any other.

> **What you're building toward.** Every chapter in this book targets the same output format. Your fine-tuned model will take a conversation as input and produce structured JSON memories that look like this:
>
> ```json
> {"text": "User prefers dark mode", "type": "preference", "entities": ["User"]}
> ```
>
> Three fields: `text` (the memory as a plain sentence), `type` (the category — `preference`, `fact`, `relationship`, etc.), and `entities` (the people or things the memory is about). This schema is what your model needs to learn to produce reliably. The model choice in this chapter determines how easily it will learn that pattern.

---

## What you'll learn

- What "base" vs "instruct" means, and which one you actually want for fine-tuning
- How model size (measured in billions of parameters) trades off against VRAM, speed, and quality
- The practical differences between the Qwen3 and Gemma 3 families for a structured extraction task
- How to read an Unsloth model ID and load the model in two dozen lines of code
- How to run a one-call smoke test that proves your model can generate text before you spend hours on training
- The one default recommendation to start with, and when to switch to something smaller

---

## Concepts you need first

### Parameters: the model's "memory"

Think of a neural network as a massive lookup table with billions of dials. Each dial is called a **parameter** — a number the model learned during pretraining. When you say a model is "7B", you mean it has roughly seven billion of those dials.

More dials = more capacity to store patterns = better quality, but also more RAM to hold them all. A 7B model needs roughly 14 GB of GPU memory just to sit there at full precision (more on precision in a moment). A 1B model needs roughly 2 GB.

For our memory-extraction task — turning a conversation into a JSON list — you don't need the model to write poetry or reason through a chess game. You need it to identify facts and format them reliably. That's a task where a smaller, well-fine-tuned model routinely beats a larger, general-purpose one.

### Base vs instruct: which starting point

Every model family ships in two flavors:

- **Base**: the raw pretrained model. It has absorbed huge amounts of text and learned patterns, but it has no concept of "answer a question" or "follow an instruction." Ask it something and it will continue your text, not reply to it.
- **Instruct** (sometimes called "chat"): the same weights, but further trained to follow human instructions in a conversational format. It expects a prompt in a specific chat template and produces a reply, not a continuation.

For fine-tuning, instruct is almost always the right starting point. Here's why: instruction-following is a hard skill to learn from scratch. The base model already has language knowledge; the instruct model has *also* learned to follow directions. Fine-tuning on top of instruct means you're teaching it a new *behavior* (extract memories, format as JSON), not teaching it the concept of following instructions at the same time. You get better results faster.

The one exception is if you have tens of thousands of examples and a very unusual task that fighting against the instruct model's trained habits. For memory extraction with a few hundred to a few thousand examples, instruct is the right call.

### Quantization: fitting a big model into less VRAM

VRAM is GPU memory — the thing that runs out first when you try to load a large model. Full-precision weights (32-bit floats) take about 4 bytes per parameter. A 7B model at full precision = ~28 GB. Most people don't have that.

**Quantization** rounds the weights to lower precision (4-bit or 8-bit), like compressing a WAV audio file to MP3. You lose a little quality, but the model goes from 28 GB to roughly 4-5 GB. This is the entire trick that makes fine-tuning accessible on a single consumer GPU.

QLoRA (covered in depth in *Ch6 - LoRA and QLoRA Without the Math Headache*) combines quantization with LoRA so you can both load *and train* a large model in a fraction of the VRAM. Unsloth handles all of this for you — you just pass `load_in_4bit=True` and it works.

---

## The two families: Qwen3 and Gemma 3

Both are strong open-weight model families from 2024-2025. Both are genuinely good for structured extraction. Here's how they differ in practice:

| | **Qwen3** | **Gemma 3** |
|---|---|---|
| Made by | Alibaba (DAMO Academy) | Google DeepMind |
| License | Apache 2.0 (commercial OK) | Gemma Terms of Use (commercial OK with conditions — check the [license page](https://ai.google.dev/gemma/terms) for your use case) |
| Sizes available | 0.5B, 1.5B, 4B, 8B, 14B, 32B, 72B | 1B, 4B, 12B, 27B |
| Context window | 32K–128K depending on size | 8K–128K depending on size |
| Instruct variant? | Yes (`-Instruct` suffix) | Yes (`-it` suffix, short for "instruction-tuned") |
| Unsloth support | Full | Full |
| Multilingual | Very strong | Good |
| Structured output tendency | Strong — follows JSON schemas reliably | Strong — comparable at 4B+ |

For memory extraction, both families work. The differences are small at the 4B range. The reason to prefer **Qwen3-4B** as the default:

1. Apache 2.0 license — no conditions to worry about for any commercial use
2. Slightly better JSON-following behavior out of the box at the 1B-4B range, based on community benchmarks for structured extraction tasks
3. 32K context window at 4B means you can feed it longer conversation chunks without truncating

Neither choice is wrong. If you're already in the Google ecosystem or prefer Gemma's safety tuning approach, Gemma 3 4B-it is an equally valid starting point.

---

## Picking a size: the VRAM trade-off table

Here's the practical guide. "Fine-tune VRAM" means approximate GPU memory needed to run QLoRA training with Unsloth. "Load VRAM" means memory to load the model for inference only.

| Model | Parameters | Load VRAM (4-bit) | Fine-tune VRAM (QLoRA) | Relative quality for extraction |
|---|---|---|---|---|
| Qwen3-0.6B | 0.5B | ~1 GB | ~3 GB | Passable — tends to miss subtle facts |
| Qwen3-1.7B | 1.5B | ~2 GB | ~5 GB | Good for simple extractions |
| **Qwen3-4B** | **4B** | **~3 GB** | **~8 GB** | **Best quality-per-VRAM ratio** |
| Qwen3-8B | 8B | ~6 GB | ~14 GB | Better, but diminishing returns |
| Gemma-3-1B-it | 1B | ~1.5 GB | ~4 GB | Similar to Qwen3-1.7B |
| Gemma-3-4B-it | 4B | ~3 GB | ~8 GB | Comparable to Qwen3-4B |

All VRAM numbers are approximate ballparks for a batch size of 1-2 with sequence length around 1024 tokens.

**The recommendation:** Start with `unsloth/Qwen3-4B-bnb-4bit`. It fits comfortably in 8 GB of VRAM — the amount on an NVIDIA RTX 3060 or many entry-level cloud GPU instances. Colab's T4 has 15 GB, which is more than enough. It produces high-quality JSON extraction after fine-tuning on even a few hundred examples.

**The fallback:** If you only have 4-6 GB of VRAM (older GPU, constrained cloud instance), drop to `unsloth/Qwen3-1.7B-bnb-4bit`. You'll need more training examples to compensate for the smaller capacity, but it will run.

---

## Reading an Unsloth model ID

Let's decode the string `unsloth/Qwen3-4B-bnb-4bit`:

```
unsloth/         → the Unsloth organization on Hugging Face (they host pre-quantized copies)
Qwen3-          → the model family
4B-             → parameter count
Instruct-       → the instruct-tuned variant (not base)
bnb-4bit        → pre-quantized to 4-bit using bitsandbytes (so Unsloth already did the compression)
```

The `bnb-4bit` suffix means you're downloading a version that's already been compressed. This is Unsloth's key convenience: they pre-quantize popular models and host them so you skip the quantization step at load time. Faster startup, same result.

---

## Loading your model: the full script

Here's the complete, runnable code to load Qwen3-4B ready for fine-tuning. You'll run this at the top of every training script going forward (Ch15 - Your First Fine-Tune with Unsloth builds directly on this).

```python
# load_model.py
# Loads Qwen3-4B in 4-bit quantization via Unsloth.
# Run this to verify your environment works before you start training.

from unsloth import FastLanguageModel
import torch  # torch (PyTorch) is installed automatically as a dependency of Unsloth — no separate install needed

# --- Configuration ---
# Change MODEL_NAME to switch to a different model (see the table above).
# Everything else stays the same.
MODEL_NAME = "unsloth/Qwen3-4B-bnb-4bit"

# The maximum sequence length your model will see during training.
# Longer = more context per example, but uses more VRAM.
# 2048 is a safe default for memory-extraction conversations.
# Most conversations + extracted memories fit comfortably in 1024-2048 tokens.
MAX_SEQ_LENGTH = 2048

# LoRA rank: how many "adapter" dimensions to add during fine-tuning.
# Higher = more capacity to learn, but more VRAM and longer training.
# 16 is a solid default; use 8 if you're VRAM-constrained.
LORA_RANK = 16

# --- Load the model and tokenizer together ---
# FastLanguageModel is Unsloth's wrapper around Hugging Face's model loader.
# It handles quantization, LoRA setup, and speed optimizations automatically.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,          # Use 4-bit quantization to save VRAM
    dtype=None,                 # Let Unsloth pick the best dtype for your GPU
)

# --- Add LoRA adapters ---
# This is what makes fine-tuning feasible. Instead of updating all 4 billion
# parameters (which would require enormous VRAM and compute), LoRA inserts
# small trainable matrices at key points. During training, only those matrices
# get updated — everything else stays frozen.
# See Ch6 for the full explanation of how LoRA works.
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,                # LoRA rank (adapter size)
    target_modules=[            # Which parts of the model get LoRA adapters.
        "q_proj", "k_proj",     # These are the attention projection layers —
        "v_proj", "o_proj",     # the parts most responsible for "which information
        "gate_proj",            # matters here." Targeting them gives the best
        "up_proj", "down_proj", # fine-tuning results per parameter updated.
                                # You don't need to understand what these are —
                                # these defaults work correctly for all Qwen and Gemma models.
    ],
    lora_alpha=LORA_RANK * 2,   # A scaling factor that controls how strongly the LoRA updates
                                # influence the model. alpha == rank is a common default;
                                # we use 2x rank here because it gives the adapter more influence,
                                # which helps when fine-tuning on small datasets (a few hundred
                                # examples). Ch16 explains the full trade-off and when to tune this.
    lora_dropout=0,             # Dropout for LoRA layers. 0 works well with Unsloth.
    bias="none",                # Don't add bias terms to LoRA layers.
    use_gradient_checkpointing="unsloth",  # Unsloth's memory-saving trick.
    random_state=42,            # Seed for reproducibility.
)

# --- Quick sanity check ---
# Count how many parameters are actually trainable (LoRA only)
# vs. the total parameters (the frozen base model + LoRA).
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,}")
print(f"Total parameters:     {total:,}")
print(f"Trainable %:          {100 * trainable / total:.2f}%")

# Expected output (approximate):
# Trainable parameters: 41,943,040
# Total parameters:     3,985,514,496
# Trainable %:          1.05%
#
# That's the LoRA magic: you're only training ~1% of the model's parameters,
# but that 1% is targeted at the right places to change the model's behavior.

print("\nModel loaded successfully. Ready for fine-tuning.")
print(f"Model: {MODEL_NAME}")
print(f"Max sequence length: {MAX_SEQ_LENGTH} tokens")

# --- Smoke test: one inference call ---
# This proves the model can actually produce text before you spend hours on data prep.
# A broken model (wrong chat template, misconfigured weights) will fail here rather
# than silently producing garbage in Chapter 15.
#
# We put the model into inference mode first (FastLanguageModel.for_inference disables
# the LoRA training hooks and enables Unsloth's fast generation path).
FastLanguageModel.for_inference(model)

# Build a minimal chat message in the format the instruct model expects.
# apply_chat_template converts a list of {"role": ..., "content": ...} dicts into
# the token sequence the model was trained to respond to.
messages = [
    {"role": "user", "content": "Extract memories from: 'I love hiking on weekends.'"}
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,           # Return token IDs (not the raw text string)
    add_generation_prompt=True,  # Append the assistant-turn header so the model knows to reply
    return_tensors="pt",     # Return a PyTorch tensor (not a plain Python list)
).to(model.device)           # Move input to the same device (GPU) as the model

# Generate a short reply — max_new_tokens=80 is enough to see a JSON object.
outputs = model.generate(input_ids=inputs, max_new_tokens=80, use_cache=True)

# Decode back to text and strip the input tokens so we only see the model's reply.
reply = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
print("\n--- Smoke test output ---")
print(reply)
print("--- End smoke test ---")

# Expected: some JSON-like text, e.g.:
# [{"text": "User loves hiking on weekends", "type": "activity", "entities": ["User"]}]
#
# The exact output will vary — the model hasn't been fine-tuned yet, so it may not
# follow the {text, type, entities} schema perfectly. What matters is that it produces
# coherent text. If you see an error here, your environment has a problem worth fixing
# now rather than after hours of training.
```

Run this with:

```bash
python load_model.py
```

The first run downloads the model weights from Hugging Face (roughly 2.5–3 GB for the 4-bit version — on a typical cloud instance this takes 2-5 minutes; on a home connection it may take longer). The script will appear to hang silently during the download — this is normal, it is not crashed. Subsequent runs load from the local cache in seconds.

If you see the "Model loaded successfully" line with a trainable % around 1%, and the smoke test prints some text under "--- Smoke test output ---", your environment is working and you're ready for the training chapters. If the smoke test crashes or prints nothing, fix the error before continuing — it will only get harder to diagnose once training is involved.

### What if you want to try Gemma instead?

Swap the model name. Everything else — LoRA setup, the tokenizer call, the sanity check — stays identical:

```python
# Gemma 3 4B instruct variant
MODEL_NAME = "unsloth/gemma-3-4b-it-bnb-4bit"

# Gemma 3 1B fallback (if VRAM is tight)
MODEL_NAME = "unsloth/gemma-3-1b-it-bnb-4bit"
```

Unsloth normalizes the interface across model families. The code above is genuinely model-agnostic — the only thing you ever change is the `MODEL_NAME` string.

**Important — Gemma models are gated.** Before the download will work, you need to:

1. Go to `huggingface.co/google/gemma-3-4b-it` and accept Google's license agreement (one-time, takes 30 seconds).
2. Create a Hugging Face access token at `huggingface.co/settings/tokens`.
3. Run `huggingface-cli login` in your terminal and paste the token when prompted. (`huggingface-cli` comes from the `huggingface_hub` package — if the command is not found, install it first with `pip install huggingface_hub`.)

Qwen3 models are ungated — no login or license acceptance needed. This is one more practical reason Qwen3 is the recommended default for getting started quickly.

### Context window note

Qwen3-4B has a 32K token context window, but we're capping `MAX_SEQ_LENGTH` at 2048 for training. Why? Because VRAM usage scales with sequence length. At 2048 tokens, a training batch fits comfortably in 8 GB. At 32K tokens, you'd need far more memory. For memory-extraction conversations — which are typically a few dozen messages — 2048 tokens is plenty. If you're working with very long documents, bump this to 4096 and watch your VRAM usage.

---

## Common mistakes

**Loading the base model instead of instruct.**
If you accidentally use `unsloth/Qwen3-4B-bnb-4bit` (no `-Instruct`), fine-tuning will still run, but you'll need significantly more training data to teach it the concept of instruction-following on top of the extraction task. Always verify your model ID has `-Instruct` (Qwen) or `-it` (Gemma) in it.

**Going straight for the 8B because "bigger is better."**
The 8B model needs roughly 14 GB of VRAM for QLoRA training. If your GPU has 8-10 GB, you'll hit out-of-memory errors mid-training, not at load time. Start with 4B, get a working pipeline, then scale up if the quality isn't good enough. A trained 4B almost always beats an undertrained 8B.

**Forgetting that the first download is slow.**
The Hugging Face download happens silently inside `from_pretrained`. If your script appears to hang for 10 minutes after that line, it's downloading, not crashed. Add `print("Downloading model...")` before the call if you want reassurance.

**Mismatching `MAX_SEQ_LENGTH` between training and inference.**
Whatever you set `MAX_SEQ_LENGTH` to during fine-tuning, use the same value when you load the model for inference later. The LoRA adapters encode assumptions about positional embeddings based on this value. A mismatch won't always crash — it can just silently produce worse results. The simplest fix: define `MAX_SEQ_LENGTH` as a constant at the top of your training script, then copy that exact value to the top of your inference script. Keep them in sync manually — one constant, two files, same number.

**Choosing a model not in Unsloth's pre-quantized library.**
Unsloth's `bnb-4bit` models are the ones they've pre-optimized and tested. If you load a model ID from Hugging Face that isn't in Unsloth's library, you can still use it, but you lose some of Unsloth's speed optimizations and the download will be larger (full precision weights that get quantized at load time). Stick to the `unsloth/` namespace for training.

---

## Recap

- **Instruct over base**: instruct models already know how to follow directions; fine-tuning teaches them *what* to do, not *how* to listen.
- **4B is the sweet spot**: enough capacity for reliable JSON extraction, fits in 8 GB VRAM, fast enough to iterate on in an afternoon.
- **Qwen3-4B is the recommended default**: Apache 2.0 license, strong JSON behavior, 32K context.
- **Gemma 3 4B is a legitimate alternative**: comparable quality, different license terms.
- **Unsloth model IDs follow a pattern**: `unsloth/{Family}-{Size}-{Variant}-bnb-4bit` — memorize this and you can navigate their library.
- **The load code is model-agnostic**: swap `MODEL_NAME` and everything else stays the same.
- **You're training ~1% of the parameters via LoRA**: the other 99% stay frozen, which is why this fits in consumer hardware.
- **First download is ~2.5 GB and may take 5-15 minutes**: it's cached after that, so subsequent runs load in seconds.

## Next

*Ch11 - Defining the Task: What "Memory Extraction" Means* — where we get precise about the input/output contract your model needs to learn, and nail down the JSON schema your fine-tuned model will always produce.
