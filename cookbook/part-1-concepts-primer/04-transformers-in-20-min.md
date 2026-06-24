# Ch4 - Transformers and LLMs in 20 Minutes

> **Before you read this:** You do not need to understand matrices, calculus, or neural network theory to use this chapter. This chapter gives you just enough of the picture to reason about what a model does — so you can frame memory extraction as a task the model can learn.

---

## What you'll learn

- What actually happens inside a language model, from raw text in to predicted text out
- What "attention" means in plain English — and why it's the key ingredient
- What autoregressive generation is, and why it matters for how we write training examples
- What temperature and sampling are, and when you'd care about them
- Exactly what you can safely skip for now (and come back to later)

---

## Concepts you need first

### Concept 1 — Tokens (a quick refresher)

You met tokens in **Ch3 - Prompting vs RAG vs Fine-Tuning vs Full Training**, but here's the one-line recap: a token is a small chunk of text — roughly a word or part of a word. The model never sees characters or bytes; it sees tokens. The whole pipeline starts with turning your text into a list of integer IDs, one per token. (Ch5 goes much deeper on tokens — don't worry about the details yet.)

### Concept 2 — Probability distributions

**Everyday analogy:** Imagine a very well-read friend. You say "The sky is…" and ask them what word comes next. They don't say just one word with certainty — they assign rough odds: "blue" (50%), "clear" (20%), "dark" (10%), etc. That ranked list of possibilities is a probability distribution.

**One-line definition:** A probability distribution over the vocabulary is a list of scores, one for every possible next token, that sum to 1.0.

**Why it matters here:** Every forward pass through a language model outputs exactly this — a probability distribution over all ~100k tokens in its vocabulary. That's all a model is doing: ranking what token is most likely to come next.

### Concept 3 — Layers

**Everyday analogy:** Think of an editor who reads a draft five times, each pass adding a different kind of polish: first for spelling, then for grammar, then for logic, then for tone, then for flow. Each pass refines the same text, and by the final pass, problems that weren't visible in pass one have been caught.

**One-line definition:** A transformer is a stack of identical processing blocks called layers; each layer refines the model's internal representation of the input.

**Why it matters here:** The more layers, the more the model can capture subtle patterns. A 7-billion-parameter model has around 32 layers; a 70-billion one has around 80. You don't choose the number of layers — that's fixed in the base model you pick.

---

## The full picture in one paragraph

Here is everything the model does, no math required:

1. Your text gets split into tokens, which become a list of integers.
2. Each integer gets looked up in a big table to become a vector — a list of numbers that represents "what this token means." (A vector is just a list of numbers. That's it.)
3. That list of vectors flows through ~32 layers. Each layer does two things: (a) every token looks at all the other tokens and updates its own representation based on what it sees — this is **attention**; (b) each token's representation gets individually refined through a small neural network — this is the **feed-forward** step.
4. After the last layer, the final representation of the very last token gets projected onto the vocabulary: a score for every possible next token. Apply a softmax (just a normalizer that forces the scores to sum to 1) and you have your probability distribution.
5. Sample one token from that distribution. Append it to the input. Repeat from step 1.

That's the whole loop. Every LLM you will ever use does exactly this.

---

## Attention: the one idea that changed everything

**Everyday analogy:** Imagine reading the sentence: *"The trophy didn't fit in the suitcase because it was too big."* What does "it" refer to? Your brain immediately looks back at "trophy" and "suitcase" and decides: it's the trophy, because the trophy is the thing that might not fit. You didn't read every word equally — you pulled attention toward the relevant ones.

Attention in a transformer is the same idea, formalized. When the model processes the token "it" in that sentence, the attention mechanism lets "it" look at every other token in the context and decide how much weight to give each one. "Trophy" gets a high weight. "The" gets almost none.

This happens for every token, simultaneously, in every layer.

**Why this was revolutionary:** Before transformers (2017), models read text left to right, one token at a time, and had to compress everything they'd seen into a single fixed-size memory. Long-range relationships — like a pronoun referring to a noun fifty words earlier — were hard to preserve. Attention lets every token directly query every other token, regardless of distance. Long context became tractable.

**What you can safely ignore for now:** You don't need to know about "query, key, value" matrices, multi-head attention, or scaled dot-product attention. Those are implementation details. The concept you need is: *each token updates its own representation by looking at and weighting all the other tokens.* That's it.

---

## Why this matters for memory extraction

Your memory extraction task looks like this:

**Input:**
```
User: I moved to Austin last year for work.
Assistant: Oh nice, what do you do?
User: I'm a backend engineer, mostly Python.
```

**Output:**
```json
[
  {"text": "User lives in Austin", "type": "fact", "entities": ["User", "Austin"]},
  {"text": "User moved to Austin for work", "type": "fact", "entities": ["User", "Austin"]},
  {"text": "User is a backend engineer", "type": "fact", "entities": ["User"]},
  {"text": "User primarily codes in Python", "type": "preference", "entities": ["User", "Python"]}
]
```

> This is the exact schema we will use throughout the book — `text`, `type`, and `entities` — introduced in Ch2 and used in every training example going forward. When you see this structure anywhere in the book, it is always the same three fields with the same meaning.

To a language model, both the input and the output are just text. The model will learn to predict the output tokens one by one, conditioned on the input tokens. It doesn't "understand" what a memory is in any deep sense — it learns a statistical pattern: *given a conversation that looks like this, the next tokens are probably a JSON list that looks like that.*

This is why framing counts. We must present the task as a text-in / text-out prediction problem, with a consistent format the model can learn to reproduce. **Ch11 - Defining the Task: What "Memory Extraction" Means** and **Ch12 - Data Format: Turning the Task into Training Rows** go into exactly how to do this.

---

## Autoregressive generation: one token at a time

The word "autoregressive" sounds fancy. It just means: the model generates token N by conditioning on tokens 1 through N-1, including the tokens it just generated.

In practice:

```
Input:  [SYSTEM] You extract memories. [USER] I moved to Austin...
Step 1: Model predicts "[" (the first JSON bracket)
Step 2: Model predicts "\n" 
Step 3: Model predicts " " 
Step 4: Model predicts "{" 
... and so on, one token at a time, until it generates the stop token.
```

**Why this matters for training:** When we build training examples, the model learns to predict every output token. A single training row with 50 output tokens gives the model 50 separate prediction signals to learn from. This is called **teacher forcing** — during training, we always feed the model the correct previous tokens (from the ground-truth output), rather than its own guesses. During inference (actual use), we feed it its own output. You don't need to implement teacher forcing; it happens automatically inside the training framework.

---

## Temperature and sampling: controlling how the model picks

When the model produces a probability distribution over the vocabulary, you have to pick one token. There are two main strategies:

**Greedy decoding:** Always pick the highest-probability token. Deterministic, but tends to produce repetitive, "safest" output.

> **Logit** — the raw, unnormalized score a model assigns to each possible next token, before softmax converts it into a probability. Think of it as a vote tally before the percentages are calculated. You will see this term throughout the book (and throughout ML writing in general); it always means the same thing: a raw score, not yet a probability.

**Temperature sampling:** Scale the logits (the raw scores before the softmax) by a temperature value, then sample randomly from the resulting distribution.

- **Temperature = 1.0**: Sample from the distribution as-is.
- **Temperature < 1.0** (e.g., 0.3): Make the distribution "sharper" — the highest-probability tokens get even more dominant. Output becomes more deterministic and focused.
- **Temperature > 1.0** (e.g., 1.5): Flatten the distribution — lower-probability tokens get a better chance. Output becomes more random and creative.

**For memory extraction**, you almost always want low temperature (0.1–0.3) or greedy decoding. You want consistent, structured JSON output — not creative variation. A high temperature might cause the model to output `"type": "fact"` sometimes and `"type": "Fact"` other times, breaking your downstream parser.

We'll come back to this in **Ch22 - Serving Your Model and Using It in an App** when we set inference parameters.

---

## A runnable demo: watching a model predict

Let's make this concrete. The code below loads a small, fast model and shows you exactly what the model sees — the probability distribution over next tokens — at each step. We're not fine-tuning here; this is purely to build intuition.

> **Hardware note:** This demo runs fine on CPU with the tiny `Qwen/Qwen3-0.6B` model (approx. 1 GB RAM). It takes about 30–60 seconds on CPU. On a GPU it's near-instant.

```python
# ch04_demo.py
# Goal: watch the model assign probabilities to next tokens,
# one step at a time, to build intuition for what's happening inside.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# We use a tiny 0.5B-parameter Qwen model here — just for demonstration.
# It's small enough to run on CPU without a GPU or any special setup.
# (When you fine-tune in Ch15, you'll use the 7B version with Unsloth.)
MODEL_ID = "Qwen/Qwen3-0.6B"

print("Loading tokenizer and model... (first run downloads ~1 GB)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,  # float32 so this runs on CPU without issues
)
model.eval()  # put model in inference mode — tells PyTorch we are not training,
              # which makes outputs stable and deterministic (no training-time randomness)
              # and avoids computing things only needed during training.

# A simple prompt that mimics the start of our memory-extraction task.
# We're asking the model to complete a JSON list of memories.
prompt = """Extract memories from this conversation as JSON.

Conversation:
User: I moved to Austin last year for work.

Memories:
["""

# Tokenize the prompt — convert string to a list of integer token IDs.
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print(f"\nPrompt tokenized into {input_ids.shape[1]} tokens.\n")

# --- Single forward pass: no generation, just predictions ---
# torch.no_grad() tells PyTorch not to track gradients — we're not training,
# so there's no need, and it saves memory.
with torch.no_grad():
    outputs = model(input_ids)

# outputs.logits has shape (batch_size, sequence_length, vocab_size)
# We want the predictions for what comes AFTER the last token in our prompt.
# So we take index [-1] on the sequence dimension.
last_token_logits = outputs.logits[0, -1, :]  # shape: (vocab_size,)

# softmax converts raw logit scores into probabilities that sum to 1.
# Each score goes from "how much the model favors this token" to "what fraction of the total."
probs = torch.softmax(last_token_logits, dim=-1)

# Find the top 10 most likely next tokens.
top_k = 10
top_probs, top_indices = torch.topk(probs, top_k)

print("Top 10 predicted next tokens after the prompt:")
print(f"{'Token':<20} {'Probability':>12}")
print("-" * 34)
for prob, idx in zip(top_probs, top_indices):
    # Decode the token ID back to a string.
    # We add a space prefix for cleaner display of leading-space tokens.
    token_str = tokenizer.decode([idx])
    # repr() shows whitespace/newlines explicitly so we can see them
    print(f"{repr(token_str):<20} {prob.item():>12.4f}")

# --- Now watch autoregressive generation step by step ---
print("\n\nAutoregressive generation (5 steps):")
print("=" * 50)

current_ids = input_ids.clone()

for step in range(5):
    with torch.no_grad():
        step_outputs = model(current_ids)

    # Same as above: grab predictions for the next token position
    step_logits = step_outputs.logits[0, -1, :]
    step_probs = torch.softmax(step_logits, dim=-1)

    # Greedy: pick the single most likely token
    next_token_id = torch.argmax(step_probs).unsqueeze(0).unsqueeze(0)
    next_token_str = tokenizer.decode(next_token_id[0])

    # What's the probability the model assigned to its own top choice?
    top_prob = step_probs[next_token_id[0, 0]].item()

    print(f"Step {step + 1}: chose {repr(next_token_str):<15} (p={top_prob:.3f})")

    # Append the chosen token to the running sequence.
    # On the next loop iteration, the model conditions on this longer input.
    current_ids = torch.cat([current_ids, next_token_id], dim=1)

# Decode the full generated sequence so far
generated_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)
print("\nFull text so far (prompt + 5 generated tokens):")
print("-" * 50)
print(generated_text)
```

Run it:

```bash
pip install "transformers>=4.40" "torch>=2.2" "accelerate>=0.30"
python ch04_demo.py
```

> **Why `accelerate`?** The `transformers` library uses `accelerate` under the hood when loading models. Without it you may get a confusing `ImportError` that doesn't mention `accelerate` by name. Installing it now prevents that. You don't need to call it directly — `transformers` calls it for you.

You should see output like:

```
Top 10 predicted next tokens after the prompt:
Token                  Probability
----------------------------------
'\n'                        0.1823
'{'                         0.1541
' {'                        0.1302
'  '                        0.0891
...

Autoregressive generation (5 steps):
==================================================
Step 1: chose '\n'          (p=0.182)
Step 2: chose '  '          (p=0.341)
Step 3: chose '{'           (p=0.612)
Step 4: chose '"'           (p=0.489)
Step 5: chose 'text'        (p=0.291)
```

What you're watching: the model, trained only on general web text, already has some idea that after `[` in a JSON context, a `{` probably follows. After fine-tuning on memory-extraction examples, the probabilities for the correct JSON structure will be dramatically higher.

---

## What you can safely ignore for now

You do not need to understand any of the following to complete this book:

- **The math of attention** (query/key/value matrices, scaled dot-product)
- **Positional encodings** (how the model knows token order)
- **The feed-forward sublayer** (the MLP inside each transformer block)
- **Layer normalization** (a numerical stability trick)
- **How the model was pretrained** (next-token prediction on trillions of tokens)
- **Backpropagation** (how gradients flow during training — a later chapter covers the intuition when you need it)
- **CUDA kernels and GPU internals** (the GPU setup chapter handles this when we get to training)

These concepts exist, and they matter for researchers. For you, right now, they are details you can look up if you ever need them. Everything in this book works without understanding them.

---

## Common mistakes

**Mistake 1: Treating the model as a "lookup" rather than a predictor**

The model doesn't retrieve facts from a database — it predicts the next token based on learned patterns. This means it can "hallucinate" content that was never in the training data. For memory extraction, this is why we need carefully structured prompts and training data that reinforces exact output formats. A model that has seen thousands of clean JSON examples during fine-tuning has a much higher probability of outputting valid JSON than a base model that's just guessing.

*Fix:* Frame every task as "what token sequence do we want the model to learn to predict?" not "what knowledge do we want the model to store?"

**Mistake 2: Confusing generation with determinism**

New practitioners often expect the model to always give the same output for the same input. It won't, by default — sampling introduces randomness. If you set `temperature=0` (or use greedy decoding), the output becomes deterministic. For production memory extraction, this is usually what you want.

*Fix:* When writing your inference code in **Ch22**, explicitly set `temperature=0` or `do_sample=False` until you have a reason to do otherwise.

**Mistake 3: Thinking more tokens in = smarter output**

The model's context window has a fixed maximum length (e.g., 32k or 128k tokens). Stuffing it with unrelated conversation history won't make the model "smarter" — it may actually dilute the signal and hurt precision. For memory extraction, a focused chunk (one session or one topic) often outperforms a long, sprawling context.

*Fix:* In **Ch11**, we'll discuss how to chunk conversations into manageable inputs before passing them to the model.

**Mistake 4: Mistaking a large model for a fine-tuned one**

GPT-4 or Claude can extract memories from a conversation using prompting. A fine-tuned 7B model is not necessarily better than those — it's *faster*, *cheaper*, *private*, and *controllable* for your specific schema. If raw accuracy on a general task is what you need, a big general model may win. Fine-tuning wins when you need a specific schema, low latency, zero API cost, and data privacy.

*Fix:* Keep the trade-offs from **Ch3 - Prompting vs RAG vs Fine-Tuning vs Full Training** in mind throughout the project.

---

## Recap

- A language model is a next-token predictor: text tokens in, a probability distribution over the vocabulary out.
- Attention is the mechanism by which each token looks at every other token in the context and updates its own representation accordingly. It's what makes transformers handle long-range relationships well.
- Autoregressive generation is the loop: predict one token, append it, repeat. Every LLM generates this way.
- Temperature controls how random the sampling is. For structured output like JSON, use low temperature or greedy decoding.
- For memory extraction, the entire task is a text-prediction problem: given a conversation as input text, predict a JSON list of memories as output text. Fine-tuning teaches the model this specific mapping.
- You do not need to understand attention math, positional encodings, backpropagation, or GPU internals to build a working fine-tune.

## Next

**Ch5 - Tokens, Context Windows, and Chat Templates** — how text becomes numbers, how long a conversation can be, and the exact format you must use when feeding data to a chat-style model.
