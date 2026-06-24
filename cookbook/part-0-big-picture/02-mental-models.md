# Ch2 - Mental Models: What a Model Actually Is

You finished Chapter 1 knowing *why* you'd want to fine-tune a model. Before you touch a single line of training code, you need a picture in your head of what the thing you're fine-tuning actually *is*. Without that picture, every decision in the rest of the book — choosing a base model, picking a LoRA rank, reading a loss curve — is guesswork.

This chapter builds that picture. No math. No academic paper citations. Just a working mental model you can reason with.

---

## What you'll learn

- What a language model actually is: a giant box of numbers that maps text to probabilities
- What "7 billion parameters" means in plain English and in RAM
- The difference between a base model and an instruct/chat model, and why it matters
- What pretraining did to those numbers — and what fine-tuning changes (and doesn't)
- How to load a model and peek inside it with a few lines of Python

---

## Concepts you need first

### Parameters: the box of numbers

Think of a language model as a massive calculator with billions of dials. Each dial is a number — a *parameter* (also called a *weight*). When you feed the model some text, those dials interact mathematically to produce an output. You never turn the dials by hand; training turns them automatically. Once training is done, the dials are frozen: that frozen set of numbers *is* the model.

**One-line definition:** a parameter is a single floating-point number stored inside the model that was tuned during training.

**Why it matters for memory extraction:** our goal is to take a model whose dials were set on the whole internet and nudge a small fraction of those dials so the model reliably outputs memory JSON instead of generic prose.

### What "7B parameters" means on disk

A floating-point number can be stored at different precisions:

| Precision | Bits per number | Common name |
|-----------|----------------|-------------|
| 32-bit float | 4 bytes | `float32` / `fp32` |
| 16-bit float | 2 bytes | `float16` / `bfloat16` |
| 8-bit int | 1 byte | `int8` |
| 4-bit int | 0.5 bytes | `int4` / `nf4` |

A 7-billion-parameter model in full 16-bit precision:

```
7,000,000,000 × 2 bytes = 14 GB
```

That's why a "7B model" is roughly 14 GB on disk and needs roughly 16–20 GB of GPU RAM to run in 16-bit mode (the extra headroom is for activations — the intermediate values the model computes while processing a batch; think of them as scratchpad memory that exists only during a forward pass — and the optimizer during training).

Load it in 4-bit quantization — which Chapter 6 explains — and that same model fits in about **4–5 GB of VRAM**. That's how you run a 7B model on a consumer GPU.

### Base model vs instruct model

Every model you'll use comes in (at least) two flavors.

**Base model** — fresh off pretraining. It has read enormous amounts of text and learned to continue it fluently. Ask it "What is the capital of France?" and it might respond "What is the capital of Germany? What is the capital of Spain?" — because it learned that lists of questions often follow each other. It doesn't know it's supposed to *answer* you.

**Instruct model** (also called a chat model) — the base model after a second training pass that taught it to follow instructions and hold a conversation. Ask the same question and it answers "Paris." Internally it's the same architecture; the instruct version just has differently tuned dials plus a special *chat template* that structures messages.

**Rule of thumb for fine-tuning:** almost always start from the instruct version, not the base. It already knows how to follow instructions; you're just teaching it your specific task on top of that foundation.

---

## How a model produces output

Here's the core loop, in plain terms:

1. Your text is chopped into *tokens* (roughly word-pieces — covered in Chapter 5).
2. Each token becomes a list of numbers (an *embedding*).
3. That list flows through dozens of *layers* — each layer is a stack of matrix multiplications involving the model's parameters.
4. At the end, the model outputs a probability over every token in its vocabulary: "given everything so far, the next token is `{` with 42% probability, `[` with 31%, `the` with 3%…"
5. You sample from that distribution (or take the top pick) and append the chosen token to the sequence.
6. Repeat from step 1 until you hit a stop token or a length limit.

That's it. The model never "knows" anything in a human sense; it just computes "what token comes next?" over and over, guided by billions of tuned parameters.

**For memory extraction:** we want those probabilities to strongly favor valid JSON structures — specifically our schema with `text`, `type`, and `entities` fields — whenever the input looks like a conversation chunk.

---

## What pretraining did

Training a base model from scratch — *pretraining* — costs millions of dollars and months of compute on thousands of GPUs. During pretraining, the model reads trillions of tokens of internet text and, at each step, tries to predict the next token. Every time it's wrong, the error signal flows backward through the network and nudges each parameter very slightly in the direction that would have made a better prediction. Do this trillions of times and the parameters settle into values that encode an enormous amount about language, facts, reasoning patterns, and structure.

After pretraining the model has, baked into its weights:

- Grammar and syntax of dozens of languages
- Factual associations ("Paris is the capital of…")
- Code patterns, JSON formatting, markdown structure
- Vague intuitions about what "makes sense" in a sentence

All of that is frozen in 7 billion numbers. You didn't pay for any of it.

---

## What fine-tuning changes

Fine-tuning is a second, much cheaper training pass on your own data. The process is identical to pretraining — feed in examples, measure prediction error, nudge parameters — but:

- You run it for thousands of steps, not trillions
- You update only a small fraction of the parameters (with LoRA — see Chapter 6)
- You use your domain-specific examples, not the whole internet

The result: the model's general language ability stays intact (it still knows what Paris is, still knows JSON syntax), but its *behavior* shifts toward your task. Give it a conversation chunk and it now reaches for the memory-extraction output format instead of generic prose.

A useful analogy: pretraining taught the model the entire culinary canon. Fine-tuning is a two-week apprenticeship in your specific restaurant's kitchen. It won't forget how to cook; it'll just start plating things the way *you* plate them.

---

## Hands-on: loading a model and seeing its parameters

The code below loads a small quantized model, prints its parameter count, and runs a quick memory-extraction prompt. You don't need a powerful GPU for this — a machine with 6 GB of VRAM (or even a free Colab T4) is enough.

```python
# Install dependencies first (if you haven't already):
# pip install unsloth transformers accelerate bitsandbytes

from unsloth import FastLanguageModel   # Unsloth wraps HuggingFace + optimizes for fine-tuning
import torch

# --- 1. Load the model in 4-bit mode ---
# max_seq_length: the longest input+output in tokens we'll ever use.
# We'll use 2048 for now — enough for a conversation chunk + JSON output.
# load_in_4bit: compresses parameters from 16-bit to 4-bit to save VRAM.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-8B-bnb-4bit",  # Qwen3 instruct, pre-quantized to 4-bit — the same checkpoint used throughout this book
    max_seq_length = 2048,
    dtype = None,          # Let Unsloth pick the right dtype for your GPU
    load_in_4bit = True,   # ~4-5 GB VRAM instead of ~14 GB
)

# --- 2. Count the parameters ---
# Each parameter is one number in the model's giant dial-board.
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters:     {total_params / 1e9:.2f}B")       # Should be ~7B
print(f"Trainable parameters: {trainable_params / 1e9:.2f}B")   # Also ~7B before we add LoRA
# After Chapter 6's LoRA setup, trainable will drop to ~1-2% of total.

# --- 3. Run a quick memory-extraction prompt ---
# This shows the model's CURRENT behavior before any fine-tuning.
# Expect messy output — we haven't trained it for our specific schema yet.

messages = [
    {
        "role": "system",
        "content": (
            "You extract memories from conversations. "
            "Output a JSON list of memories, each with fields: "
            "text (string), type (one of: fact | preference | decision | relationship), "
            "entities (list of strings)."
        ),
    },
    {
        "role": "user",
        "content": (
            "Here is a conversation chunk:\n\n"
            "Alice: I just moved to Berlin last month. I hate coffee but I love tea.\n"
            "Bob: Oh nice! I've been here five years. We should grab tea sometime.\n\n"
            "Extract all memories."
        ),
    },
]

# Apply the model's built-in chat template to format messages correctly.
# The template adds special tokens the model was trained to recognize.
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize = False,       # Return a string, not token IDs — easier to inspect
    add_generation_prompt = True,  # Append the marker that tells the model to start replying
)

# Tokenize and move to GPU
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate a response (greedy decoding: always pick the most likely next token)
with torch.no_grad():   # No gradient tracking needed — we're just running inference
    output_ids = model.generate(
        **inputs,
        max_new_tokens = 300,   # Cap output length
        do_sample = False,      # Greedy — always pick the most likely token; deterministic and good for inspection
        # temperature is ignored when do_sample=False, so we omit it here to avoid confusion
    )

# Decode only the newly generated tokens (not the prompt)
generated = tokenizer.decode(
    output_ids[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens = True,
)

print("\n--- Model output (before fine-tuning) ---")
print(generated)
# You'll likely see *some* JSON, but it may miss fields, hallucinate types,
# or add prose around the JSON. Fine-tuning fixes all of this.
```

### What to look at in the output

Run this and you'll see something like:

```
Total parameters:     7.62B
Trainable parameters: 7.62B
```

The model has 7.62 billion numbers. At 4-bit each, that's about 3.8 GB of weights — which is why it loads on modest hardware.

The generated text will be *close* to what we want — JSON with memories — but probably imperfect. Fields might be in wrong formats, the `type` enum might drift, entities might be missing. That gap between "close" and "reliable and structured" is exactly what fine-tuning closes.

---

## Base model vs instruct model: a concrete test

Here's a one-liner to see the behavioral difference yourself. The code above uses the instruct model. Swap the `model_name` to the base model and run the same prompt:

```python
# Swap this line to compare:
model_name = "unsloth/Qwen3-8B-bnb-4bit"             # Base model — no instruction tuning
# vs
model_name = "unsloth/Qwen3-8B-bnb-4bit"    # Instruct model (used throughout this book)
```

The base model will likely continue the conversation as if it were text to complete, or produce something incoherent in response to the chat template. The instruct model will at least attempt to answer. This is why Chapter 10 (Choosing Your Base Model) defaults to instruct variants for fine-tuning.

---

## Common mistakes

**Confusing "parameters" with "intelligence."** More parameters doesn't always mean smarter for your task. A 7B model fine-tuned on your data will almost always outperform a 70B model that's just prompted. The parameters encode general language; your fine-tune encodes your task.

**Starting from the base model instead of instruct.** Base models need more training data and careful formatting to learn instruction-following from scratch. Unless you have a specific reason (e.g., building a custom chat template from scratch), always start from the instruct variant.

**Assuming fine-tuning "overwrites" the base knowledge.** It doesn't — not if you do it correctly with a small learning rate. The model retains everything pretraining gave it; fine-tuning nudges behavior without erasing the foundation. Aggressive training with a large learning rate *can* cause "catastrophic forgetting" — where the model overwrites so much of its pretrained knowledge that it loses general language ability — Chapter 16 covers how to avoid it.

**Forgetting that 4-bit models are slightly less accurate.** Quantization (compressing from 16-bit to 4-bit) introduces minor rounding errors. For most tasks, the quality difference is negligible. For production systems where every percentage point matters, you can fine-tune in 4-bit and then export the merged model in 16-bit — Chapter 21 walks through this.

**Running `.generate()` without `torch.no_grad()`.** During inference you don't need gradients. Without `no_grad()`, PyTorch stores intermediate values in case you want to backpropagate (backpropagate: compute how much each parameter contributed to the error so training can update them — irrelevant during inference, but PyTorch does it by default unless told otherwise), which wastes VRAM silently and can cause out-of-memory errors on larger inputs.

---

## Recap

- A language model is a box of billions of numbers (parameters/weights) that maps token sequences to probability distributions over the next token.
- A 7B model in 16-bit is ~14 GB; in 4-bit quantization it shrinks to ~4–5 GB of VRAM.
- Base models predict the next token; instruct models were additionally trained to follow instructions. Always fine-tune from the instruct variant.
- Pretraining is expensive and bakes in general language knowledge. Fine-tuning is cheap and shifts behavior toward your specific task without erasing what pretraining built.
- Our goal: nudge the model's probability distribution so it reliably outputs well-formed memory JSON when given a conversation chunk.
- You can count parameters with a one-liner: `sum(p.numel() for p in model.parameters())`.
- After a LoRA fine-tune (Chapter 6), only 1–2% of parameters are trainable — the rest stay frozen.

## Next

**Ch3 - Prompting vs RAG vs Fine-Tuning vs Full Training** — a map of the whole landscape so you can explain to anyone (and to yourself) exactly why fine-tuning is the right tool for memory extraction, and when it isn't.
