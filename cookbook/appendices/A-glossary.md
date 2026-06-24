# Appendix A - Glossary of Every Term Used

This appendix is your safety net. Every time you hit an unfamiliar word in the main chapters,
look it up here. Definitions are in plain English first, with a precise one-liner after each,
and a note on where the term shows up in the memory-extraction project when that's useful.

Entries are alphabetical. Terms that have their own entry are written in **bold** when they appear
inside another definition.

---

## What you'll learn

- A plain-English definition for every ML term used in this book
- How each term connects to the memory-extraction task you built throughout
- Quick cross-references to the chapters where each concept is explained in depth
- A short Python code section at the end that prints a glossary "cheat sheet" for the key
  numeric concepts — useful to have open when you're reading training logs
- Common places where term confusion causes real bugs

---

## Concepts you need first

No prerequisites. This appendix assumes nothing. If you need context on why any of these terms
exist in the first place, Chapter 1 (Why Teach a Model Your Own World) and Chapter 2 (Mental
Models: What a Model Actually Is) are the best starting points.

---

## The Glossary

---

### Accelerate

A Hugging Face library that handles the low-level details of running PyTorch on GPUs: moving
tensors to the right device, handling mixed-precision arithmetic, and distributing work across
multiple GPUs if you have them. You rarely call it directly — **Unsloth** and **TRL** call it
for you. Think of it as the plumbing under the floorboards.

*One-line definition:* a library that abstracts GPU memory management and multi-device training
so you write device-agnostic code.

---

### Activation / Activation function

After a layer in a neural network does its matrix math, it passes the result through an
activation function — a simple mathematical operation that introduces non-linearity. Without it,
stacking layers would be equivalent to one layer. The most common one in modern transformers is
called **SiLU** or **GELU**. You will rarely need to configure this; it is baked into the model
architecture.

*One-line definition:* a function applied to each layer's output to allow the network to learn
non-linear patterns.

---

### AdamW

The optimizer almost every fine-tuning run uses. An optimizer is the algorithm that decides how
much to adjust each **weight** after each training step. AdamW is a variant of the classic Adam
optimizer with a small fix ("weight decay") that helps prevent **overfitting**. Unsloth defaults
to `adamw_8bit`, which is the same algorithm stored in 8-bit integers to save **VRAM**.

*One-line definition:* the default optimizer for fine-tuning; adjusts weights using gradient
history and weight decay to converge efficiently.

See: Chapter 7 (How Training Actually Works).

---

### Adapter/dataset versioning & model registry

Once you retrain on a schedule (Part 8), you stop having "the model" and start having a
*lineage* of models, each trained from a specific **dataset** snapshot. Versioning means giving
every trained **LoRA** adapter and every dataset a unique, immutable name (a version tag or a
content hash) so you can always answer "which data produced the adapter currently serving
traffic?" A **model registry** is just the catalog that stores those versioned artifacts
together with their metadata — training config, eval scores, parent version — so a deploy is
"promote adapter v7" rather than "copy some files I hope are the right ones." This is what makes
**rollback** a one-line operation instead of a panic.

*One-line definition:* the discipline of giving every trained adapter and training dataset an
immutable version, tracked in a registry, so any deployed model is traceable and reversible.

See: Chapter 34 (Production Ops: Monitoring, Versioning, Gating, and Rollback).

---

### Advantage (and GAE)

A number used in reinforcement-learning fine-tuning (**PPO**, **GRPO**, **RLOO**) that answers a
sharper question than raw reward: "was this output *better than expected*?" If every answer to a
prompt scores about 0.6 and this one scored 0.6, its advantage is roughly zero — nothing to learn.
If it scored 0.9, the advantage is positive and the model is nudged to do more of that. Centering
on "better than expected" instead of the absolute score is what keeps RL training stable.
**PPO** estimates the baseline with a **value head** and smooths it across timesteps with a method
called **GAE** (Generalized Advantage Estimation); **GRPO** skips all that and uses the average
score of a *group* of samples for the same prompt as the baseline.

*One-line definition:* how much better (or worse) an output was than the expected baseline; the
signal RL methods actually train on, rather than the raw reward.

See: Chapter 27 (PPO and the Full RL Loop: Why We Don't Use It Here).

---

### Alpha (LoRA alpha, `lora_alpha`)

One of the two main numbers you set when configuring a **LoRA** adapter. Alpha controls how
loudly the adapter speaks relative to the frozen base model. Specifically, the adapter's output
is multiplied by `alpha / r` before being added to the layer. Setting `alpha = r` gives a scale
factor of 1.0 (neutral). Setting `alpha = 2 * r` gives 2.0 (the adapter has double the
influence).

*One-line definition:* a scaling factor that controls how strongly the LoRA adapter nudges each
layer's output.

Typical starting value: `lora_alpha=16` (matching `r=16`). See: Chapter 6 (LoRA and QLoRA Without
the Math Headache).

---

### Attention / Self-attention

The core mechanism inside every transformer layer. Attention lets each token in the sequence
"look at" every other token and decide how much to borrow from them when computing its own
representation. When your memory-extraction model reads "Alice prefers dark mode," attention is
what connects the word "prefers" back to "Alice" so the model knows who the preference belongs to.

The attention calculation involves three learned matrices per layer: **Query (Q)**, **Key (K)**,
and **Value (V)**. These are also the matrices most commonly targeted by **LoRA** adapters.

*One-line definition:* a mechanism that lets each token weight-sum information from all other
tokens in the sequence, allowing the model to track relationships across long contexts.

See: Chapter 4 (Transformers and LLMs in 20 Minutes).

---

### Base model

A language model straight off **pretraining** — it has read enormous amounts of text and learned
to predict the next token fluently, but it has not been taught to follow instructions. If you ask
a base model "What is the capital of France?" it may respond with another question, continue in
the style of a quiz, or produce something incoherent. It doesn't know it's supposed to answer you.

Contrast with **instruct model**. For fine-tuning purposes, always start from the instruct
variant unless you have a specific reason not to.

*One-line definition:* a pretrained model that has not received instruction-tuning; knows language
but does not know how to be an assistant.

See: Chapter 2 (Mental Models: What a Model Actually Is), Chapter 10 (Choosing Your Base Model).

---

### Batch / Batch size (`per_device_train_batch_size`)

Instead of updating the model after every single training example (too slow and noisy) or after
the entire dataset (uses too much memory), training groups examples into **batches** and does one
weight update per batch. Batch size is the number of examples in each group.

For LoRA fine-tuning on a typical GPU, a batch size of 2–4 is common. Larger batches give
smoother **gradient** estimates but use more **VRAM**. If you're tight on VRAM, lower the batch
size and raise **gradient accumulation** to compensate.

*One-line definition:* the number of training examples processed together before one weight update
is made.

See: Chapter 7 (How Training Actually Works), Chapter 16 (Hyperparameters).

---

### BF16 / bfloat16

A 16-bit floating-point number format designed by Google Brain. It uses fewer bits than a
standard `float32` (32-bit) but keeps the same exponent range, which makes it well-suited for
deep learning — large or very small gradient values don't lose meaning. Modern GPUs (A100, H100,
RTX 30/40 series) have hardware acceleration for bf16. Unsloth uses bf16 for the LoRA adapter
weights during training.

*One-line definition:* a 16-bit float format with wide dynamic range, the standard training
precision for modern GPU fine-tuning.

---

### bitsandbytes

A Python library that implements **quantization** — specifically 4-bit and 8-bit quantization of
model weights. It is what makes **QLoRA** possible: it loads the frozen base model in 4-bit,
drastically reducing **VRAM** usage. Unsloth calls bitsandbytes internally; you set
`load_in_4bit=True` and it handles the rest.

*One-line definition:* a library that quantizes model weights to 4-bit or 8-bit integers to
reduce GPU memory usage.

See: Chapter 6 (LoRA and QLoRA Without the Math Headache), Chapter 9 (The Toolbox).

---

### Canary deploy / shadow deploy

Two ways to ship a new model without betting all your traffic on it at once. A **canary deploy**
sends a small slice of real traffic (say 5%) to the new adapter while the rest keeps hitting the
old one; if the canary's metrics look healthy, you ramp it up. A **shadow deploy** is even safer:
the new model sees a copy of real requests but its outputs are *not returned to users* — you only
log them and compare. Shadow mode is ideal for memory extraction because you can diff the new
model's JSON against the current model's on live conversations before trusting it. Both pair
naturally with a **canary / regression eval set** and **eval gating**.

*One-line definition:* progressive rollout strategies — canary routes a small fraction of live
traffic to the new model; shadow runs it on real traffic without serving its output — so problems
surface before full release.

See: Chapter 34 (Production Ops: Monitoring, Versioning, Gating, and Rollback).

---

### Canary / regression eval set

A small, *frozen* set of examples you never train on, kept aside specifically to catch
regressions across retraining rounds. The name comes from "canary in a coal mine": if a fresh
round of fine-tuning quietly breaks something the model used to handle, scores on this fixed set
drop and warn you before the model reaches users. Unlike your main **validation split** (which
can grow and change), the canary set stays identical round after round so the numbers are
comparable over time. For memory extraction, stock it with tricky conversations — empty inputs
that should yield `[]`, multi-entity exchanges, edge-case types — plus a few general-language
prompts to detect **catastrophic forgetting**.

*One-line definition:* a fixed, never-trained-on evaluation set held constant across retraining
rounds, used to detect regressions and forgetting before they ship.

See: Chapter 33 (Catastrophic Forgetting Over Many Rounds), Chapter 34 (Production Ops:
Monitoring, Versioning, Gating, and Rollback).

---

### Catastrophic forgetting

When fine-tuning goes wrong in a specific way: the model is trained so aggressively on your new
task that it overwrites the general language knowledge it got from **pretraining**. After
catastrophic forgetting, the model might be good at memory extraction but terrible at everything
else — including following instructions correctly, writing valid JSON, or understanding natural
language.

The main defenses: use **LoRA** (only updates a tiny fraction of weights), keep the **learning
rate** small, and don't train for too many **epochs**. Chapter 16 (Hyperparameters) covers the
knobs to watch.

In continual learning the risk compounds: each retraining round can shave off a little more
general ability, so forgetting that looks negligible in one round becomes severe after ten. The
round-to-round defenses are **replay** (mix ~10–30% prior/general data back into each round) and a
frozen **canary / regression eval set** that includes general-language prompts, so a drop shows up
immediately. Chapter 33 covers this over-many-rounds version specifically.

*One-line definition:* the failure mode where aggressive fine-tuning overwrites a model's
previously learned general abilities; in continual learning it accumulates across rounds unless
countered with replay and a frozen canary set.

See: Chapter 33 (Catastrophic Forgetting Over Many Rounds), Chapter 16 (Hyperparameters: Which
Knobs to Turn and When).

---

### Chat template

A formatting convention that wraps your messages (system prompt, user message, assistant reply)
in special tokens the model was trained to recognize. Different model families use different
templates. Qwen3 uses `<|im_start|>` / `<|im_end|>` markers; Gemma 3 uses `<start_of_turn>` /
`<end_of_turn>`. Using the wrong template — or no template — causes the model to behave as if
it's reading a garbled prompt.

Unsloth's `get_chat_template()` function applies the correct template for each model family
automatically.

*One-line definition:* the model-specific special-token wrapper that tells a chat model where
messages start and end and who is speaking.

See: Chapter 5 (Tokens, Context Windows, and Chat Templates).

---

### Checkpoint

A snapshot of the model's weights saved to disk at a specific point during training. If training
crashes or you want to experiment with an earlier version, you load from a checkpoint. Configured
by `save_steps` in `TrainingArguments`. A typical run saves a checkpoint every 100–200 steps.

*One-line definition:* a saved copy of the model's weights at a given training step, allowing
recovery or rollback.

---

### Context window (context length)

The maximum number of **tokens** a model can "see" at once — both input and output together. If
the context window is 2048 tokens and your prompt is 1800 tokens, you only have 248 tokens left
for the model's response. Exceeding the context window causes truncation (text gets silently
cut) or an error.

For memory extraction, a context window of 2048 is enough for most conversations. Set
`max_seq_length=2048` in Unsloth to match.

*One-line definition:* the hard upper limit on how many tokens the model can process in a single
forward pass (prompt + response combined).

See: Chapter 5 (Tokens, Context Windows, and Chat Templates).

---

### Dataset (Hugging Face `datasets` library)

The `datasets` library from Hugging Face is the standard way to load, process, and stream
training data in Python. It handles JSONL files, CSV files, Parquet, and data hosted on the
Hugging Face Hub. It also supports efficient data processing with `map()` and automatic caching
so you don't re-process data on every run.

In our memory-extraction project, `datasets` loads our JSONL file of conversation/memory pairs
and feeds rows to the trainer one batch at a time.

*One-line definition:* a Hugging Face library for loading, processing, and streaming training
datasets efficiently.

See: Chapter 9 (The Toolbox), Chapter 14 (Cleaning, Splitting, and Sanity-Checking Data).

---

### Decoding / Decode

Two meanings depending on context:

1. **Token decoding:** converting token IDs back into human-readable text. After the model
   generates a sequence of integer token IDs, `tokenizer.decode()` turns them back into the
   string you can read.

2. **Generation strategy (decoding strategy):** how the model chooses the next token at
   inference time. Options include **greedy decoding** (always pick the highest-probability
   token), **sampling** (pick probabilistically), and **beam search** (simultaneously explore
   the top-N most likely token sequences in parallel, then keep whichever full sequence scores
   highest — more thorough than greedy but slower and not usually needed for structured output).
   For memory extraction, greedy or low-temperature sampling works well — you want consistent,
   not creative, output.

*One-line definition:* (1) converting token IDs back to text; (2) the strategy for selecting
tokens during generation.

---

### Distillation (knowledge distillation)

A training technique where a small "student" model learns to mimic the outputs of a large
"teacher" model, rather than learning directly from labeled data. The teacher's probability
distributions (not just its final answers) are used as training targets, which transfers more
nuance. In the memory-extraction project, generating synthetic training data with a powerful
model like GPT-4o or Claude acts as a form of informal distillation: the small model learns the
behavior of a much larger one.

*One-line definition:* training a smaller model to mimic the output distributions of a larger
teacher model, transferring capability more efficiently than training from labels alone.

See: Chapter 13 (Creating Your Training Data with Synthetic Generation).

---

### DPO (Direct Preference Optimization)

The simplest way to teach a model "prefer this kind of answer over that kind." You show it
**preference pairs** — for the same prompt, a *chosen* (better) answer and a *rejected* (worse)
one — and DPO directly adjusts the model to raise the probability of chosen answers and lower the
probability of rejected ones. Its big selling point: unlike **RLHF**/**PPO**, DPO needs *no
separate reward model and no reinforcement-learning loop* — the preference data *is* the training
signal, so it runs almost like a slightly fancier **SFT** job. It keeps the model from drifting
too far using a **reference model** and a `beta` knob (the **KL** strength; TRL's default is 0.1).
For memory extraction, DPO is the recommended first step beyond plain SFT — e.g. teach it to
prefer clean `[]` over a hallucinated memory.

*One-line definition:* a preference-tuning method that trains directly on chosen/rejected pairs to
prefer better outputs, with no separate reward model and no RL loop.

See: Chapter 26 (DPO: Learning Directly From Preference Pairs), Chapter 29 (Choosing Your Method).

---

### Dropout (`lora_dropout`)

A regularization technique: during training, a random fraction of the neurons (or, in LoRA's
case, adapter values) are set to zero at each step. This forces the model to not rely too heavily
on any single path and helps prevent **overfitting** on small datasets. At inference time,
dropout is always turned off.

For LoRA fine-tuning with small datasets (under ~10k examples), `lora_dropout=0.05` (5%) is a
reasonable default.

*One-line definition:* randomly zeroing a fraction of values during training to prevent the model
from memorizing training data.

---

### Embedding

A list of numbers (a vector) that represents a token or a concept in a high-dimensional space.
When the model reads the token `"preference"`, it looks up that token's embedding — a list of,
say, 4096 numbers — and that list encodes everything the model has learned about what
"preference" means in context. Similar concepts have embeddings that are mathematically close
to each other.

*One-line definition:* a dense vector of numbers that encodes the meaning of a token as
understood by the model.

See: Chapter 4 (Transformers and LLMs in 20 Minutes).

---

### Epoch

One complete pass through your entire training dataset. If you have 1,000 training examples and
a batch size of 4, one epoch is 250 **steps**. Fine-tuning typically uses 2–5 epochs. Training
for too many epochs risks **overfitting**; too few risks **underfitting**.

*One-line definition:* one full pass through the training dataset from start to finish.

See: Chapter 7 (How Training Actually Works).

---

### Eval gating

A safety rule that says: a newly trained model is *not allowed to deploy* unless it clears your
evaluation bar automatically. Instead of a human eyeballing scores and deciding, you encode the
bar as a check — "**F1** on the **canary / regression eval set** must be at least as good as the
currently deployed model, and structural-validity must stay above 99%" — and the deploy pipeline
refuses to promote anything that fails. This is what makes a continual-learning loop safe to run
often: a bad retraining round gets blocked by the gate rather than reaching users. Pairs naturally
with **canary deploy** and **rollback**.

*One-line definition:* an automated pass/fail check on eval metrics that a new model must clear
before it is allowed to deploy.

See: Chapter 34 (Production Ops: Monitoring, Versioning, Gating, and Rollback).

---

### Evaluation / Eval loss

Evaluation is running the model on examples it was never trained on (the **validation split**) to
measure how well it generalizes. Eval loss is the **loss** computed on those held-out examples.
If training loss keeps dropping but eval loss starts rising, the model is **overfitting**.

For memory extraction specifically, loss alone isn't enough — you also want to measure whether
the JSON output is structurally valid and whether the extracted memories are correct. Chapter 18
(Did It Actually Work? Evaluating Memory Extraction) covers a full evaluation harness.

*One-line definition:* measuring the model's **loss** (and other metrics) on held-out data it
never saw during training.

See: Chapter 18 (Did It Actually Work? Evaluating Memory Extraction).

---

### F1 score

A combined accuracy metric that balances **precision** and **recall** into a single number. F1 is
the harmonic mean of the two: `F1 = 2 × (precision × recall) / (precision + recall)`. A score of
1.0 is perfect; 0.0 is worst possible.

For memory extraction, F1 measures how well the model's extracted memories overlap with the gold
standard: are you catching the right memories (recall) without adding noise (precision)?

*One-line definition:* the harmonic mean of precision and recall; a single number summarizing
both the "did you catch everything?" and "did you add noise?" dimensions of accuracy.

See: Chapter 18 (Did It Actually Work? Evaluating Memory Extraction).

---

### Fine-tuning

A second, cheaper training pass on your own data, starting from a **pretrained** model. Fine-tuning
does not train the model from scratch — it nudges the existing weights (or, with **LoRA**, adds
small adapter weights) to shift the model's behavior toward your specific task.

The result: a model that retains the general language knowledge from pretraining while reliably
performing your task — like outputting memory JSON from conversation input.

*One-line definition:* retraining a pretrained model on task-specific examples to specialize its
behavior without starting from scratch.

See: Chapter 1 (Why Teach a Model Your Own World), Chapter 15 (Your First Fine-Tune with Unsloth).

---

### FP16 / float16

A 16-bit floating-point number format. Older GPUs use fp16; newer ones prefer **bf16** (which
has better dynamic range). The practical difference for fine-tuning: bf16 is slightly safer
because it handles very small gradient values without rounding them to zero. Unsloth auto-detects
which format your GPU supports.

*One-line definition:* a 16-bit float format; functionally similar to bf16 but with a narrower
range that can cause numeric instability in some training setups.

---

### Frozen weights

Weights that are held fixed during training — they are not updated by the **optimizer**. In
**LoRA** fine-tuning, the entire base model is frozen. Only the small adapter matrices are
trainable. Freezing is what makes LoRA memory-efficient: the optimizer only needs to track
gradients for the ~0.5% of weights that actually train.

*One-line definition:* model weights that are not updated during training; held constant while
only the adapter weights learn.

---

### GGUF

A file format for storing quantized language models, designed for use with **llama.cpp** and
**Ollama**. A GGUF file bundles the model weights and metadata into one portable binary. It is
the format you export to when you want to run your fine-tuned model on a Mac, a CPU, or any
machine without a dedicated NVIDIA GPU. A 1.7B model exported to GGUF at Q4_K_M quantization is
roughly 1.1 GB.

*One-line definition:* a self-contained binary format for quantized models; the standard for
CPU/Mac inference via llama.cpp and Ollama.

See: Chapter 21 (Saving, Merging, and Exporting Your Model).

---

### Gradient

After the model makes a prediction and the **loss** is computed, PyTorch works backward through
every layer and calculates: "if I nudge this weight slightly higher, does the loss go up or
down, and by how much?" That calculation for every weight is the gradient. The gradient is then
used by the **optimizer** to take one step in the direction that reduces loss.

Think of it like a slope. You're trying to walk downhill to minimum loss, and the gradient is
the slope reading under your feet at each step.

*One-line definition:* a vector describing how much each weight should change to reduce loss by a
small amount; the output of the backward pass.

See: Chapter 7 (How Training Actually Works).

---

### Gradient accumulation (`gradient_accumulation_steps`)

A trick for simulating a larger **batch size** when you don't have enough **VRAM** to fit a
large batch. Instead of updating weights after each mini-batch, you process several mini-batches,
sum their **gradients**, and only update weights once at the end.

Example: `per_device_train_batch_size=2` with `gradient_accumulation_steps=4` gives an effective
batch size of 8, but only 2 examples are in VRAM at any one time.

*One-line definition:* accumulating gradients over multiple mini-batches before doing a weight
update, to simulate a larger batch size without the VRAM cost.

See: Chapter 7 (How Training Actually Works), Chapter 16 (Hyperparameters).

---

### Gradient checkpointing

A memory optimization technique: instead of storing all intermediate values computed during the
forward pass (needed for the backward pass), only a subset are stored. The rest are recomputed
on-the-fly during the backward pass. This trades a small amount of compute time for a
significant reduction in VRAM.

Unsloth enables gradient checkpointing by default via `use_gradient_checkpointing="unsloth"`.

*One-line definition:* a VRAM optimization that recomputes some intermediate values during
backprop instead of storing them all, reducing peak memory at the cost of ~10% extra compute.

---

### GRPO (Group Relative Policy Optimization)

The reinforcement-learning method this book actually recommends when you want RL. Its key trick:
for each prompt, generate a *group* of several answers (TRL defaults to `num_generations=8`),
score them all with a **reward function**, and use the group's own average score as the baseline
— so each answer's **advantage** is simply "how much better than its siblings was it?" Because the
group provides the baseline, **GRPO drops the value model (critic) entirely**, which is what makes
it dramatically lighter than **PPO** (no **value head** to train). You still optionally keep a
**reference model** via a **KL** penalty (`beta`, which defaults to 0.0 in TRL — i.e. *off* unless
you raise it). For memory extraction, GRPO shines when correctness is easy to *check* but hard to
hand-label: a programmatic **reward function** can score valid JSON and entity overlap directly.

*One-line definition:* an RL fine-tuning method that scores a group of samples per prompt and uses
their average as the baseline, dropping the separate value model that PPO requires.

See: Chapter 28 (GRPO: Practical RL With Reward Functions), Chapter 29 (Choosing Your Method).

---

### Hallucination

When a model generates text that sounds confident and plausible but is factually wrong or
invented. In the context of memory extraction, hallucination means the model outputs memories
that were not present in the input conversation, invents entities, or uses field names not in
your schema.

Fine-tuning on high-quality labeled examples is one of the most effective tools for reducing
hallucination in a specific task.

*One-line definition:* model-generated output that is plausible-sounding but factually incorrect
or entirely fabricated.

---

### Hard-example mining

A data-selection strategy for continual learning: instead of retraining on a random pile of new
data, you deliberately go find the examples the current model *gets wrong* and prioritize those.
The intuition is the same as studying: re-reading what you already know is wasted effort; you
learn fastest from the problems you keep missing. In practice you run the deployed model over
recent inputs, flag the failures (invalid JSON, missed memories, wrong entities), and feed
corrected versions of those into the next round. It's the opposite of **importance sampling**'s
broad reweighting — here you're surgically targeting known weak spots.

*One-line definition:* deliberately selecting the examples the current model fails on and
prioritizing them in the next training round, since failures carry the most learning signal.

See: Chapter 31 (Selecting and Curating Data That Actually Helps).

---

### Hugging Face Hub

An online platform for sharing and downloading pre-trained models, datasets, and training spaces.
When you call `FastLanguageModel.from_pretrained("unsloth/Qwen3-8B")`, Unsloth
downloads the model from the Hugging Face Hub. You can also push your fine-tuned model to the
Hub with `model.push_to_hub("your-username/your-model-name")`.

*One-line definition:* the central online repository for sharing, downloading, and versioning
models and datasets in the Hugging Face ecosystem.

---

### Hyperparameter

A number you set before training starts that controls how training behaves — as opposed to the
model's **weights**, which are learned during training. Common hyperparameters: **learning rate**,
**batch size**, **epochs**, **LoRA rank**, **LoRA alpha**, **warmup steps**. Getting hyperparameters
roughly right matters more than getting them perfect; the defaults in this book are solid starting
points.

*One-line definition:* a training setting you choose in advance (like learning rate or batch size)
that controls the training process but is not itself learned.

See: Chapter 16 (Hyperparameters: Which Knobs to Turn and When).

---

### Importance sampling (data)

A way of reweighting your training data so the mix the model trains on matches the mix it will
actually face in production — without throwing away or duplicating rows by hand. If 60% of real
conversations are short customer-support chats but your **dataset** is mostly long meeting
transcripts, you can up-weight the support examples (and down-weight the over-represented ones) so
each round's gradient reflects the real-world distribution. The phrase is borrowed from
statistics, but in this book it just means "tilt the training mixture toward what matters" rather
than treating every example as equally important. Contrast with **hard-example mining**, which
targets specific failures rather than reshaping the overall mix.

*One-line definition:* reweighting training examples so the distribution the model learns from
matches the distribution it will see in production.

See: Chapter 31 (Selecting and Curating Data That Actually Helps).

---

### Inference

Running the model to generate output — the opposite of training. During inference, weights are
frozen and gradients are not computed. Inference is what happens when you send a conversation
to your fine-tuned model and get back a JSON list of memories. Always call
`FastLanguageModel.for_inference(model)` before generating output with Unsloth — it applies
additional speed optimizations.

*One-line definition:* using a trained model to generate output; weights are frozen, no learning
happens.

---

### Instruct model (instruction-tuned model)

A model that started as a **base model** and then received a second training pass specifically to
follow instructions and hold conversations. For example, `Qwen3-1.7B` and
`gemma-3-1b-it` are both instruct variants. The "it" or "Instruct" suffix is the giveaway.
These models also come with a **chat template** that structures the conversation format.

*One-line definition:* a base model that has been further trained to follow instructions and
respond helpfully to user messages.

See: Chapter 2 (Mental Models: What a Model Actually Is), Chapter 10 (Choosing Your Base Model).

---

### JSON / JSONL

**JSON** (JavaScript Object Notation) is the structured text format used throughout this book
for memory extraction output. Our schema outputs a JSON array of objects with fields `text`,
`type`, and `entities`.

**JSONL** (JSON Lines) is a variant where each line of a file is a separate, complete JSON
object. It is the standard format for training datasets because it is easy to stream line by
line without loading the entire file into memory.

```json
{"input": "Alice prefers dark mode.", "output": "[{\"text\": \"Alice prefers dark mode.\", \"type\": \"preference\", \"entities\": [\"Alice\"]}]"}
```

*One-line definition:* JSON is the structured output format for memories; JSONL is the line-by-line
variant used for training data files.

See: Chapter 12 (Data Format: Turning the Task into Training Rows).

---

### Key (K) / Query (Q) / Value (V)

The three learned projection matrices inside each **attention** layer. The query represents
"what am I looking for?", the key represents "what do I contain?", and the value represents
"what information do I carry?". During attention, each token's query is compared to every other
token's key; the similarity scores determine how much of each token's value gets mixed in.

In **LoRA**, adding adapters to `q_proj`, `k_proj`, and `v_proj` is the most impactful place to
attach adapters because these matrices determine how the model allocates attention — and attention
is where structured output behaviors are encoded.

*One-line definition:* the three projection matrices that implement the attention mechanism; Q asks
questions, K answers them, V supplies the content that gets mixed in.

---

### KL divergence / KL penalty

KL divergence is a number measuring how far one probability distribution has drifted from another.
In preference and RL fine-tuning it has a very concrete job: it measures how far the model you're
training (the **policy model**) has wandered from its starting point (the **reference model**).
Think of it as a *leash*. RL rewards push the model to chase the **reward function**, and without
a leash it will happily walk off a cliff — producing degenerate, reward-hacking text that scores
high but reads like nonsense. The **KL penalty** is a term added to the objective that pulls the
model back toward the reference, trading a little reward for staying sane. The leash length is the
`beta` knob: higher `beta` = shorter leash (stays close to the reference). It appears in **PPO**,
**GRPO** (default `beta=0.0`, i.e. *no* leash unless you raise it), and inside **DPO**'s loss.

*One-line definition:* a measure of how far the trained model has drifted from its reference; used
as a penalty (a "leash") to keep RL/preference training from degenerating, tuned by `beta`.

See: Chapter 27 (PPO and the Full RL Loop: Why We Don't Use It Here), Chapter 28 (GRPO: Practical
RL With Reward Functions).

---

### KTO (Kahneman-Tversky Optimization)

A preference-tuning method that's even cheaper to collect data for than **DPO**. DPO needs *pairs*
(for one prompt, a better and a worse answer side by side). KTO needs only a single **thumbs-up or
thumbs-down label** per example — "was this output good or bad?" — which is exactly the kind of
signal you already get from production (a 👍/👎 button, an accepted-vs-edited memory). The name
nods to the prospect-theory work of psychologists Kahneman and Tversky on how people weigh gains
and losses. Use KTO when you can't easily produce matched pairs but you *can* label individual
outputs as acceptable or not. TRL exposes it top-level as `KTOTrainer`.

*One-line definition:* a preference method that learns from single per-example good/bad labels
instead of chosen/rejected pairs, making feedback far easier to collect than DPO's.

See: Chapter 26 (DPO: Learning Directly From Preference Pairs), Chapter 29 (Choosing Your Method).

---

### Layer

A single processing block inside a neural network. A transformer model like Qwen3-8B has 32 or
more layers stacked in sequence. Each layer has an **attention** sub-block and a feed-forward
sub-block. As text flows through the layers, each layer refines the representation. Deep in the
stack (later layers) the model is reasoning about meaning; early layers are handling surface
syntax.

*One-line definition:* one processing block in the neural network stack; a transformer is dozens
of these layers applied in sequence.

---

### Learning rate (`learning_rate`)

The size of each step the **optimizer** takes when updating weights. Too large and training
overshoots and oscillates. Too small and training stalls or takes forever. For LoRA fine-tuning,
a learning rate in the range `2e-4` to `5e-4` is a solid starting point. Unsloth's defaults are
sensible; this is rarely the first knob to touch when debugging.

*One-line definition:* controls how large each gradient-based weight update is; the most
important hyperparameter to get roughly right.

See: Chapter 7 (How Training Actually Works), Chapter 16 (Hyperparameters).

---

### llama.cpp

An open-source C++ inference engine for running quantized language models on CPUs, Macs, and
low-VRAM machines. It is the engine behind **Ollama**. To use your fine-tuned model with
llama.cpp, you export it as a **GGUF** file.

*One-line definition:* a C++ library for running quantized LLMs locally on CPU or Apple Silicon
without a dedicated NVIDIA GPU.

See: Chapter 21 (Saving, Merging, and Exporting Your Model), Chapter 22 (Serving Your Model).

---

### LLM (Large Language Model)

A neural network trained on large amounts of text to model the probability of token sequences.
"Large" refers to the number of **parameters** (billions to hundreds of billions). Modern LLMs
like Qwen3 and Gemma 3 are **transformer**-based and can generate coherent, contextually aware
text. In this book "LLM" and "model" are used interchangeably.

*One-line definition:* a transformer-based neural network with billions of parameters trained to
predict and generate human language.

---

### LLM-as-judge

An evaluation technique where you ask a powerful model (e.g. GPT-4o or Claude) to score or
compare the outputs of your fine-tuned model. Useful when "correctness" is hard to measure
mechanically — for example, judging whether an extracted memory is semantically equivalent to
the gold standard even if worded differently.

*One-line definition:* using a capable LLM to evaluate the quality of another model's outputs,
acting as an automated scorer or ranker.

See: Chapter 18 (Did It Actually Work? Evaluating Memory Extraction).

---

### LoRA (Low-Rank Adaptation)

A fine-tuning technique that avoids changing the original model weights. Instead, it freezes the
base model and attaches small "adapter" matrices alongside the existing weight matrices in each
layer. Only the adapter matrices are trained. Because adapters are much smaller than the original
matrices, the number of trainable **parameters** drops from billions to tens of millions — roughly
0.5% of the total.

The "low-rank" part means each adapter is decomposed into two thin matrices (A and B) whose
product approximates a full-rank weight update. Rank `r` controls how thin (and thus how
expressive) the adapter is.

*One-line definition:* a fine-tuning method that trains only small adapter matrices alongside
frozen base weights, reducing trainable parameters by ~99.5%.

See: Chapter 6 (LoRA and QLoRA Without the Math Headache).

---

### Loss (training loss, eval loss)

A single number that measures how wrong the model's predictions were on the current batch.
Mathematically it is cross-entropy loss: it penalizes the model for assigning low probability
to the correct next token. Lower is better. A loss of 2.5 at the start of fine-tuning, dropping
to 0.3 by the end, means the model went from very surprised by the correct tokens to barely
surprised — it has learned the pattern.

Training loss is computed on the training data. Eval loss is computed on the held-out
**validation** split.

*One-line definition:* a number measuring how surprised the model was by the correct answer;
the quantity that the optimizer minimizes throughout training.

See: Chapter 7 (How Training Actually Works), Chapter 17 (Watching Training: Loss Curves).

---

### Loss curve

A chart (or ASCII sparkline) of **loss** over training **steps**. A healthy loss curve starts
high, drops steeply in the first few hundred steps, then gradually flattens. If the curve never
drops, your **learning rate** or data format is likely wrong. If it drops and then rises on the
eval split, you are **overfitting**.

*One-line definition:* a visualization of loss over training steps; the primary diagnostic tool
for assessing whether training is progressing normally.

See: Chapter 17 (Watching Training: Loss Curves and When to Stop).

---

### Max new tokens (`max_new_tokens`)

A generation parameter that caps how many tokens the model can output in a single call. For
memory extraction, 512 tokens is usually enough for most conversations. Setting it too low
truncates the JSON mid-output; setting it too high wastes inference time on pathological inputs.

*One-line definition:* the maximum number of tokens the model is allowed to generate in one
inference call.

---

### Memory extraction

The running example task for this entire book: given a chunk of conversation (chat messages,
meeting notes, journal entries), output a structured JSON list of atomic, standalone facts,
preferences, decisions, and relationships embedded in that text. Each memory follows this schema:

```json
{
  "text": "Alice prefers dark mode in all her apps.",
  "type": "preference",
  "entities": ["Alice"]
}
```

This mirrors what products like mem0 do: they listen to conversations and extract durable facts
so an AI can remember things about you over time.

*One-line definition:* the task of extracting structured, atomic facts from raw conversation text
and formatting them as a JSON array.

See: Chapter 11 (Defining the Task: What "Memory Extraction" Means).

---

### Merge (adapter merge)

The process of baking a **LoRA** adapter's weights permanently into the base model weights,
producing a single self-contained model file. A merged model is larger than the adapter alone
but works with any inference tool — it looks like a standard model with no LoRA dependency.

Unsloth does this with `model.save_pretrained_merged(...)`.

*One-line definition:* permanently combining a LoRA adapter's learned adjustments into the base
model weights to produce a standalone model.

See: Chapter 21 (Saving, Merging, and Exporting Your Model).

---

### Mixed precision

A training technique that uses **bf16** (or **fp16**) for most operations (faster, less VRAM)
but **fp32** (32-bit) for numerically sensitive parts like the optimizer's internal state. This
gives near full-precision accuracy with significantly better memory efficiency. Unsloth and
**accelerate** handle this automatically.

*One-line definition:* training that uses lower-precision floats (bf16/fp16) for most math and
full-precision (fp32) only where numerical stability requires it.

---

### Model weights (weights, parameters)

The billions of floating-point numbers inside a language model that encode everything it learned
during training. "Weights" and "parameters" are used interchangeably. In a 7B model there are
roughly 7,000,000,000 of these numbers. Fine-tuning adjusts a small fraction of them.

*One-line definition:* the learned numeric values stored inside a model; collectively, they
define the model's behavior.

See: Chapter 2 (Mental Models: What a Model Actually Is).

---

### NF4 (Normal Float 4)

A specific 4-bit quantization format designed for language model weights. Unlike a naive 4-bit
integer, NF4 uses a non-uniform scale derived from the normal distribution, which better matches
the statistical distribution of actual model weights. **bitsandbytes** uses NF4 by default when
you set `load_in_4bit=True`.

*One-line definition:* a 4-bit number format optimized for the distribution of neural network
weights, offering better quality than standard int4 quantization.

---

### Ollama

A tool that makes it easy to run quantized language models locally via a simple CLI and REST API.
It uses **llama.cpp** under the hood and reads **GGUF** files. After exporting your fine-tuned
model to GGUF, you can create an Ollama modelfile and `ollama run your-model` from the terminal.

*One-line definition:* a user-friendly wrapper around llama.cpp for running quantized models
locally with a CLI and API.

See: Chapter 22 (Serving Your Model and Using It in an App).

---

### OOM (Out of Memory)

The error you see when you try to allocate more **VRAM** than your GPU has. It crashes training
with a `RuntimeError: CUDA out of memory` message. The most common fixes: reduce **batch size**,
reduce `max_seq_length`, enable **gradient checkpointing**, or use a smaller model.

*One-line definition:* a GPU crash caused by trying to allocate more memory than the GPU has;
the most common failure mode in fine-tuning.

See: Chapter 8 (Hardware, GPUs, and Setting Up Your Environment), Chapter 19 (When It Goes Wrong).

---

### Optimizer

The algorithm that uses **gradients** to update the model's **weights** after each training step.
Think of it as the thing that actually moves the dials. The standard optimizer for fine-tuning
is **AdamW**. Unsloth uses an 8-bit version (`adamw_8bit`) by default to save VRAM without
losing meaningful quality.

*One-line definition:* the algorithm that applies gradients to update model weights at each
training step.

See: Chapter 7 (How Training Actually Works).

---

### ORPO (Odds Ratio Preference Optimization)

A preference method that folds **SFT** and preference tuning into a *single* step. Normally you'd
fine-tune on good outputs (SFT) and then run **DPO** to push away from bad ones — two passes.
ORPO does both at once by adding a small "odds-ratio" penalty to an ordinary SFT loss: the model
learns the chosen answers while being gently pushed away from the rejected ones in the same go.
Its defining property is that it is **reference-free** — unlike **DPO**, **PPO**, and **GRPO**, it
needs *no separate reference model* held in memory, which saves VRAM. In TRL 1.6.0 it lives in
`trl.experimental.orpo`. Worth knowing as a one-pass alternative when you want SFT and preference
shaping together.

*One-line definition:* a reference-free preference method that combines SFT and preference tuning
into one step via an odds-ratio penalty, needing no separate reference model.

See: Chapter 26 (DPO: Learning Directly From Preference Pairs), Chapter 29 (Choosing Your Method).

---

### Overfitting

When the model learns the training examples too well — memorizing specific conversations instead
of learning the underlying pattern. Symptoms: training **loss** is very low, but **eval loss**
is much higher, and the model produces good output on training examples but struggles with new
inputs.

For memory extraction, overfitting looks like perfect JSON on conversations the model saw during
training but garbled or hallucinated output on new conversations.

Fixes: reduce **epochs**, increase **dropout**, add more training data, or raise **weight decay**.

*One-line definition:* the failure mode where the model memorizes training examples instead of
generalizing; detected by a gap between training and eval loss.

See: Chapter 7 (How Training Actually Works), Chapter 19 (When It Goes Wrong).

---

### PEFT (Parameter-Efficient Fine-Tuning)

A Hugging Face library that implements multiple parameter-efficient fine-tuning methods,
including **LoRA**. When Unsloth calls `FastLanguageModel.get_peft_model(...)`, it is using
PEFT's LoRA implementation under the hood. You rarely interact with PEFT directly.

*One-line definition:* a Hugging Face library implementing efficient fine-tuning methods like LoRA;
called by Unsloth behind the scenes.

See: Chapter 9 (The Toolbox).

---

### Perplexity

A measure of how "confused" the model is by a piece of text. It is calculated from **loss**:
`perplexity = exp(loss)`. A perplexity of 1.0 means the model predicted every token perfectly.
A perplexity of 100 means the model was very surprised by the text. Lower is better.

You will see perplexity mentioned in model evaluation papers. For our purposes, watching loss
directly is more practical — perplexity is just loss on a different scale.

*One-line definition:* exp(loss); a measure of how surprised the model is by a text sample —
lower means the model finds the text more predictable.

---

### Policy model and Reference model

Two copies of the model that show up everywhere in RL and preference fine-tuning (**PPO**,
**GRPO**, **DPO**). The **policy model** is the one you're actually training — "policy" is the RL
word for "the thing that decides what to do," and here it decides which token to emit next. The
**reference model** is a *frozen snapshot* of where you started (usually your **SFT** model), kept
untouched as a fixed point of comparison. Training measures how far the policy has drifted from
the reference using **KL divergence**, and the **KL penalty** uses that to keep the policy on a
leash. So: the policy moves, the reference stands still, and the gap between them is what you
control. (**ORPO** is notable for being *reference-free* — it skips the second copy.)

*One-line definition:* the policy is the model being trained; the reference is a frozen copy of
the starting model used as the anchor that the KL penalty keeps the policy close to.

See: Chapter 27 (PPO and the Full RL Loop: Why We Don't Use It Here).

---

### PPO (Proximal Policy Optimization)

The classic, full-strength reinforcement-learning algorithm behind the original **RLHF** — and a
cautionary tale for this book's reader. PPO is powerful but *heavy*: a single training step juggles
**four** models at once — the **policy model** being trained, a frozen **reference model**, a
**reward model** that scores outputs, and a **value head / critic** that predicts the baseline for
computing **advantage** — all kept in memory together, with a **KL penalty** leashing the policy to
the reference. That's a lot of moving parts and VRAM for a single consumer GPU. In TRL 1.6.0 PPO
has been relocated to `trl.experimental.ppo` and the old hand-rolled `.step()` loop is gone, so
the book treats PPO as *conceptual*: understand the full loop, then reach for **GRPO**, which gets
most of the benefit by dropping the value model.

*One-line definition:* the full RLHF algorithm that trains a policy against a reward model using a
value-head baseline and a KL leash; powerful but heavy, so this book teaches it conceptually and
uses GRPO instead.

See: Chapter 27 (PPO and the Full RL Loop: Why We Don't Use It Here), Chapter 29 (Choosing Your
Method).

---

### Precision (metric)

In evaluation, precision answers: "of all the memories the model extracted, what fraction were
actually correct?" A precision of 1.0 means every extracted memory was valid. Low precision means
the model is adding noise — extracting things that aren't real memories.

Precision is paired with **recall** and combined into **F1 score**.

*One-line definition:* the fraction of model-extracted memories that are correct; measures how
much noise the model adds.

See: Chapter 18 (Did It Actually Work? Evaluating Memory Extraction).

---

### Preference optimization / preference tuning

The umbrella name for the whole family of techniques that teach a model to *prefer better answers
over worse ones*, rather than just imitating a single "correct" answer the way **SFT** does. The
shift in mindset: **SFT** shows the model one gold output per input and says "produce this"; 
preference tuning shows it *comparisons* ("this answer is better than that one") and says "lean
toward the better kind." This matters when there's no single right answer but there *is* a clear
better/worse — e.g. two valid memory extractions where one is cleaner or avoids a borderline
hallucination. The family includes **DPO**, **KTO**, **ORPO**, and the RL methods **PPO**,
**GRPO**, and **RLOO**.

*One-line definition:* the family of methods that train a model from comparisons of better-vs-worse
outputs instead of single gold labels; the step beyond imitation-style SFT.

See: Chapter 24 (Beyond Imitation: Why Preference and RL).

---

### Preference pair (chosen / rejected)

The basic unit of training data for **DPO** (and for training a **reward model**). A preference
pair is, for *one prompt*, two responses: a **chosen** one (the better answer) and a **rejected**
one (the worse answer). The model never sees an absolute score — only the relative judgment "this
beats that," which is far easier for humans (or a stronger model) to produce reliably than a
precise quality number. In TRL these arrive as dataset columns named `prompt`, `chosen`, and
`rejected`. For memory extraction, a natural pair is the same conversation with a clean correct
extraction as *chosen* and a near-miss (a hallucinated entity, a bundled fact) as *rejected*.
(**KTO** relaxes this to single good/bad labels instead of pairs.)

*One-line definition:* a single prompt paired with a better (chosen) and a worse (rejected)
response; the training unit for DPO and reward models.

See: Chapter 25 (Rewards: Functions and Reward Models), Chapter 26 (DPO: Learning Directly From
Preference Pairs).

---

### Pretraining

The original, expensive training run that creates a **base model** from scratch. It feeds the
model trillions of tokens of public text (web pages, books, code) and trains it to predict the
next token. This process costs millions of dollars and months of GPU time. You never do this —
you start from the result of someone else's pretraining run and apply **fine-tuning** on top.

*One-line definition:* the original large-scale training run on public text that creates the base
model; extremely expensive, done once by the model developer.

See: Chapter 2 (Mental Models: What a Model Actually Is).

---

### QLoRA (Quantized LoRA)

**LoRA** plus **quantization**, stacked together. The base model is loaded in 4-bit precision
(via **bitsandbytes** / **NF4**), slashing **VRAM** usage by ~4×. The **LoRA** adapters are still
trained in 16-bit (**bf16**) precision so gradient math stays stable. Only the adapter weights are
updated; the frozen 4-bit base never changes.

Result: you can fine-tune a 7B model on a 16 GB consumer GPU instead of needing 80 GB.
**Unsloth** makes QLoRA the default; you just set `load_in_4bit=True`.

*One-line definition:* LoRA fine-tuning with the base model loaded in 4-bit quantization;
reduces VRAM by ~4× while preserving training quality.

See: Chapter 6 (LoRA and QLoRA Without the Math Headache).

---

### Quantization

Rounding model **weights** from high-precision floating-point numbers (32-bit or 16-bit) down to
lower-precision integers (8-bit or 4-bit). This shrinks the model's memory footprint dramatically.
A 7B model that is 14 GB in 16-bit becomes roughly 4–5 GB in 4-bit. Quality loss is typically
1–3% on most benchmarks — rarely noticeable for a task-specific fine-tune.

*One-line definition:* compressing model weights from 16-bit floats to 4-bit or 8-bit integers
to dramatically reduce GPU memory usage.

See: Chapter 6 (LoRA and QLoRA Without the Math Headache).

---

### RAG (Retrieval-Augmented Generation)

A technique that pairs an LLM with a search system: before answering, the system retrieves
relevant documents from a database and injects them into the prompt as context. Good for large,
changing knowledge bases. Not a substitute for fine-tuning when the goal is to teach reliable
output *behavior* (like structured JSON) rather than inject new facts.

*One-line definition:* augmenting an LLM's response by retrieving and injecting relevant
documents at query time; good for knowledge, not for behavior.

See: Chapter 1 (Why Teach a Model Your Own World), Chapter 3 (Prompting vs RAG vs Fine-Tuning).

---

### Rank (`r`, LoRA rank)

The key size parameter in **LoRA**. It controls how expressive the adapter is — how many
"dimensions of change" it can learn. A rank of 16 means the adapter can represent 16 independent
directions of adjustment per layer. Higher rank = more expressive = more trainable parameters =
more VRAM. For narrow tasks like memory extraction, `r=16` is a solid default.

*One-line definition:* the number of latent dimensions in the LoRA adapter matrices; controls
expressiveness and trainable parameter count.

See: Chapter 6 (LoRA and QLoRA Without the Math Headache).

---

### Recall (metric)

In evaluation, recall answers: "of all the memories that should have been extracted, what
fraction did the model actually find?" A recall of 1.0 means the model missed nothing. Low recall
means the model is under-extracting — leaving real memories behind.

Recall is paired with **precision** and combined into **F1 score**.

*One-line definition:* the fraction of true memories that the model successfully extracted;
measures how much the model misses.

See: Chapter 18 (Did It Actually Work? Evaluating Memory Extraction).

---

### Replay / replay ratio / rehearsal

The single most important defense against **catastrophic forgetting** in a continual-learning
loop. The idea is borrowed from how you'd keep a skill sharp: when you study something new, you
also re-review the old material so you don't lose it. In training terms, each retraining round
mixes a fraction of *prior and general* data back in alongside the fresh new data — this is
**replay** (also called **rehearsal**). The **replay ratio** is that fraction; this book's rule of
thumb is ~10–30% (default around 20%). Too little and the model forgets its older abilities; too
much and it barely learns the new task. For memory extraction, replay means each round still
includes a slice of earlier conversations and some general instruction-following examples.

*One-line definition:* mixing a fraction of prior/general data into each new training round
(the replay ratio, ~10–30%) so the model retains old abilities; also called rehearsal.

See: Chapter 32 (How Much Data, and How Often to Retrain), Chapter 33 (Catastrophic Forgetting
Over Many Rounds).

---

### Reward function (programmatic)

A plain Python function that scores a model's output — no neural network, no training required.
Where a **reward model** *learns* what "good" means from human preferences, a programmatic reward
function just *encodes* it in code: you write the rules. For memory extraction this is a
beautiful fit, because much of "good" is mechanically checkable — does it parse as valid JSON?
does every object have the right fields? do the entities actually appear in the conversation? Each
of those becomes a few lines that return a number. In TRL's **GRPO**, a reward function has the
signature `fn(completions, **kwargs) -> list[float]`, and you can pass several (weighted) at once.
The catch is **reward hacking**: the model will exploit any loophole your rules leave open.

*One-line definition:* a hand-written Python function that scores outputs by encoded rules (e.g.
valid JSON, correct entities), used as the reward signal in GRPO — no trained model needed.

See: Chapter 25 (Rewards: Functions and Reward Models), Chapter 28 (GRPO: Practical RL With Reward
Functions).

---

### Reward hacking

When a model learns to maximize the **reward** in a way that technically scores high but defeats
the spirit of what you wanted — gaming the metric instead of doing the task. The classic shape:
your **reward function** gives points for valid JSON, so the model discovers that emitting `[]`
(an empty but perfectly valid array) every single time scores well while extracting *nothing*.
The reward went up; the model got worse. The fixes are to make the reward harder to game (reward
*correct* extraction, not just *valid* JSON), combine several reward signals so no single loophole
dominates, and watch a **canary / regression eval set** for scores that rise while real quality
falls. It's the central hazard of any RL method (**PPO**, **GRPO**).

*One-line definition:* the failure mode where a model maximizes the reward signal through a
loophole that satisfies the metric but not the actual goal.

See: Chapter 25 (Rewards: Functions and Reward Models), Chapter 28 (GRPO: Practical RL With Reward
Functions).

---

### Reward model

A *separate, trained* model whose only job is to look at an output and emit a single number: how
good is this? You train it on **preference pairs** — shown a *chosen* and a *rejected* response,
it learns to score the chosen one higher — and then **PPO**-style RL uses its scores as the reward
signal. In TRL you build one with `RewardTrainer` on top of an
`AutoModelForSequenceClassification` (`num_labels=1`). It is the right tool when "good" is too
fuzzy to write as code (tone, helpfulness, subtle correctness). Contrast with a **reward function
(programmatic)**, which is just Python rules — for memory extraction the programmatic route is
usually enough, so a full reward model is more often discussed than built here.

*One-line definition:* a separately trained model that scores output quality as a number, learned
from preference pairs; supplies the reward signal in RLHF/PPO-style training.

See: Chapter 25 (Rewards: Functions and Reward Models).

---

### RLHF (Reinforcement Learning from Human Feedback)

The training recipe that famously turned raw language models into helpful assistants. It's a
three-stage pipeline: (1) **SFT** to get a baseline that follows instructions, (2) collect human
**preference pairs** and train a **reward model** on them, then (3) use reinforcement learning —
classically **PPO** — to push the model toward outputs the reward model scores highly, with a
**KL penalty** keeping it from drifting too far from the SFT starting point. RLHF is the *idea*
("optimize against human preferences with RL"); **PPO** is one *implementation* of stage 3.
Modern methods like **DPO** reach a similar goal while collapsing stages 2–3 into one and skipping
the reward model and RL loop entirely.

*One-line definition:* the SFT → reward-model → RL pipeline that aligns a model to human
preferences; the original approach that DPO and GRPO later simplified.

See: Chapter 24 (Beyond Imitation: Why Preference and RL), Chapter 27 (PPO and the Full RL Loop:
Why We Don't Use It Here).

---

### RLOO (REINFORCE Leave-One-Out)

A lightweight online-RL method in the same spirit as **GRPO**: generate several samples per
prompt, score them, and — crucially — like GRPO it needs **no value model / critic**. The
"leave-one-out" part is how it builds each sample's baseline: a given sample's baseline is the
*average reward of the other samples* for that prompt (leaving itself out), so its **advantage** is
"how much better than my peers was I?" That makes it cheaper than **PPO** while staying a genuine
RL method. In TRL it's available as `RLOOTrainer`. Worth knowing as a one-line alternative in the
decision guide alongside GRPO; this book centers GRPO but RLOO is a close cousin.

*One-line definition:* a value-model-free online RL method that baselines each sample against the
average of the other samples for the same prompt; a lighter cousin of GRPO/PPO.

See: Chapter 28 (GRPO: Practical RL With Reward Functions), Chapter 29 (Choosing Your Method).

---

### Rollback

The escape hatch of production ops: instantly reverting to the previous, known-good model when a
freshly deployed one misbehaves. Because every adapter is versioned (see **adapter/dataset
versioning & model registry**), rollback isn't a rebuild — it's pointing the server back at the
prior version tag, which should take seconds. A continual-learning loop *will* eventually ship a
bad round; what keeps that from becoming an incident is that **eval gating** blocks most bad
rounds before deploy, **canary / shadow deploy** catches the rest on a small slice of traffic, and
rollback undoes anything that slips through. Plan the rollback path *before* you need it.

*One-line definition:* reverting serving to the previous known-good model version; the fast,
pre-planned recovery when a new deploy regresses.

See: Chapter 34 (Production Ops: Monitoring, Versioning, Gating, and Rollback).

---

### safetensors

A file format for storing model **weights** developed by Hugging Face. It is safer than the older
PyTorch `.pt` / `.bin` format because it cannot execute arbitrary code during loading (pickle
could). Models downloaded from the Hub and adapter weights saved by Unsloth use safetensors
by default. The file extension is `.safetensors`.

*One-line definition:* a safe, efficient file format for storing model weights; the standard on
the Hugging Face Hub.

---

### Sampling (temperature, `do_sample`)

A way to introduce randomness into generation by sampling from the model's probability
distribution rather than always picking the highest-probability token (**greedy decoding**).
Controlled by the `temperature` parameter: higher temperature (e.g. 1.0–1.5) makes output more
varied and creative; lower temperature (e.g. 0.1) makes output more deterministic and consistent.

For memory extraction you want consistency, not creativity — use `temperature=0.1` or
`do_sample=False` (greedy).

*One-line definition:* generating tokens by sampling from the probability distribution rather than
always picking the top token; `temperature` controls how much randomness to add.

---

### SFT (Supervised Fine-Tuning)

The standard fine-tuning recipe used in this book: show the model labeled input/output pairs and
train it to produce the correct output for each input. "Supervised" because every training example
has a known correct answer (the label). SFT is how we teach the memory-extraction behavior.

Contrast with reinforcement learning from human feedback (RLHF) or direct preference optimization
(DPO), which use preference signals instead of exact labels. SFT is simpler, cheaper, and the
right starting point.

*One-line definition:* fine-tuning on labeled input/output examples where the correct output is
known; the standard recipe for task-specific fine-tuning.

See: Chapter 15 (Your First Fine-Tune with Unsloth), Chapter 9 (The Toolbox).

---

### SFTTrainer

The training class from the **TRL** library that implements **SFT**. It wraps the Hugging Face
`Trainer` and adds conveniences for instruction-following fine-tuning: it handles chat template
formatting, dataset column mapping, and proper loss masking (so the model only learns to predict
the output tokens, not the input prompt tokens). Unsloth integrates with SFTTrainer seamlessly.

*One-line definition:* TRL's pre-built training loop for supervised fine-tuning; handles
formatting, loss masking, and integration with Unsloth.

See: Chapter 15 (Your First Fine-Tune with Unsloth), Chapter 9 (The Toolbox).

---

### Special tokens

Reserved tokens that carry structural meaning rather than word meaning. Examples: `<|im_start|>`
(marks the start of a message in Qwen's **chat template**), `<|endoftext|>` (marks end of
document), `[PAD]` (padding to fill a batch to uniform length). Special tokens are defined in
the `tokenizer_config.json` and are handled automatically by `tokenizer.apply_chat_template()`.

*One-line definition:* reserved tokens with structural roles (marking message boundaries,
sequence ends, padding) rather than lexical meaning.

See: Chapter 5 (Tokens, Context Windows, and Chat Templates).

---

### Step

One forward pass + one backward pass + one optimizer weight update. The atomic unit of training
progress. Step count equals `(dataset_size / batch_size) * epochs / gradient_accumulation_steps`.
When you read training logs, the step counter is the number that ticks up every time a weight
update happens.

*One-line definition:* one complete forward-backward-update cycle; the atomic unit of training.

See: Chapter 7 (How Training Actually Works).

---

### System prompt

The first message in a **chat template** exchange, with `role: "system"`. It sets the model's
behavior for the entire conversation — instructions, persona, output format. For memory
extraction, the system prompt defines the task, the schema, and the expected JSON structure.
After fine-tuning, the system prompt can often be shortened or removed because the behavior is
baked in.

*One-line definition:* the "role: system" message in a chat conversation that instructs the model
on its task and behavior.

---

### Tensor

A multi-dimensional array of numbers — the fundamental data structure in deep learning
(and in **PyTorch**). A single number is a 0-dimensional tensor (scalar). A list of numbers is
a 1D tensor (vector). A matrix is a 2D tensor. Model weight matrices and token ID sequences are
all tensors. In code you will see them as `torch.Tensor` objects.

*One-line definition:* a multi-dimensional numeric array; the data structure PyTorch uses for
everything from token IDs to model weight matrices.

---

### Token

The basic unit of text that a language model reads and generates. Tokens are not exactly words:
a tokenizer splits text into sub-word pieces. The word "preference" might be one token; the word
"preferences" might be two (`"preference"` + `"s"`). Numbers, punctuation, and spaces each have
their own token representations.

A rough rule of thumb: 1 token ≈ 0.75 English words, or 4 characters. A 2048-token **context
window** holds roughly 1,500 words.

*One-line definition:* the atomic unit of text a model processes; roughly a word-piece, ~4
characters on average.

See: Chapter 5 (Tokens, Context Windows, and Chat Templates).

---

### Tokenizer

The component that converts raw text into a sequence of integer **token** IDs (and back). Every
model family has its own tokenizer vocabulary, so you must always use the tokenizer that matches
your model. Unsloth loads the correct tokenizer automatically when you load a model.

*One-line definition:* the component that splits text into tokens and maps them to integer IDs
(and reverses the process for output decoding).

See: Chapter 5 (Tokens, Context Windows, and Chat Templates).

---

### Transformer

The neural network architecture underlying every modern LLM, including Qwen3 and Gemma 3. Key
components: an **embedding** layer, many stacked **attention** + feed-forward **layers**, and an
output head that converts the final layer's representation into a probability over the
**vocabulary**. The transformer's key innovation was **attention**, which lets every token
communicate with every other token in the sequence simultaneously.

*One-line definition:* the neural network architecture at the heart of all modern LLMs; built from
stacked attention + feed-forward layers.

See: Chapter 4 (Transformers and LLMs in 20 Minutes).

---

### TRL (Transformer Reinforcement Learning)

A Hugging Face library that provides high-quality, production-ready training loops for language
model fine-tuning. Despite the name, it is used primarily for **SFT** via its `SFTTrainer` class
(not just reinforcement learning). TRL handles the inner training loop, checkpoint saving,
evaluation, and logging, so you configure it with arguments rather than write the loop yourself.

*One-line definition:* the Hugging Face library that provides the SFTTrainer training loop; handles
the core training cycle, evaluation, and checkpointing.

See: Chapter 9 (The Toolbox).

---

### Underfitting

The opposite of **overfitting**: the model hasn't learned the task well enough. Both training
loss and eval loss remain high. Common causes: too few training examples, too few **epochs**, too
small a **LoRA rank** for the task's complexity, or a **learning rate** too small to make
progress.

*One-line definition:* the failure mode where the model has not learned the task; both training
and eval loss remain high.

See: Chapter 7 (How Training Actually Works), Chapter 19 (When It Goes Wrong).

---

### Unsloth

The primary library used throughout this book for fine-tuning. Unsloth wraps **transformers**,
**PEFT**, **bitsandbytes**, and **accelerate** into a streamlined API and applies hand-tuned CUDA
kernels that reduce **VRAM** usage by 30–50% and speed up training by 2–5× compared to raw
Hugging Face code. For our purposes, the key entry points are `FastLanguageModel.from_pretrained()`
and `FastLanguageModel.get_peft_model()`.

*One-line definition:* an optimized fine-tuning library that wraps the Hugging Face stack with
custom CUDA kernels for lower VRAM and faster training.

See: Chapter 9 (The Toolbox).

---

### Validation split (dev set)

A held-out portion of your dataset — typically 10–15% — that the model never trains on. It is
used to compute **eval loss** and catch **overfitting** early. Never include validation examples
in training data; their entire value comes from being unseen.

*One-line definition:* the portion of the dataset withheld from training, used to measure how well
the model generalizes to unseen examples.

See: Chapter 14 (Cleaning, Splitting, and Sanity-Checking Data).

---

### Value head / critic

The extra component **PPO** bolts onto the model to estimate the **advantage**. While the main
model decides *what to say* (the **policy model**), the value head — also called the **critic** —
sits alongside it and predicts *how much reward to expect* from a given state, i.e. the baseline.
Subtracting that baseline from the actual reward is what turns a raw score into an advantage
("better than expected?"). In TRL this is the `AutoModelForCausalLMWithValueHead` wrapper. The
catch: the critic is itself a model that has to be trained and held in memory, which is a big part
of why PPO is heavy — and exactly the part **GRPO** and **RLOO** throw away, replacing it with the
average score across a group of samples.

*One-line definition:* PPO's learned baseline predictor (the critic) that estimates expected
reward so an advantage can be computed; the costly piece GRPO and RLOO eliminate.

See: Chapter 27 (PPO and the Full RL Loop: Why We Don't Use It Here).

---

### VLLM (vLLM)

A high-throughput inference server for language models. Unlike a simple script that generates one
response at a time, vLLM uses a technique called **PagedAttention** (a memory-management trick
that pools GPU memory across requests instead of reserving a fixed block per request — similar to
how an OS uses virtual memory so multiple programs can share RAM) to handle many concurrent
requests efficiently, making it suitable for production APIs. If you want to serve your
memory-extraction model to multiple users simultaneously, vLLM is the right tool.

To use vLLM, export your fine-tuned model as a merged 16-bit model (see Chapter 21).

*One-line definition:* a high-throughput LLM serving framework using PagedAttention to handle
many concurrent requests efficiently; the choice for production API serving.

See: Chapter 22 (Serving Your Model and Using It in an App).

---

### Vocabulary (vocab)

The complete set of **tokens** a model knows. A typical LLM vocabulary contains 30,000–150,000
tokens (Qwen3 uses 151,936). The vocabulary is defined by the **tokenizer** and is fixed after
pretraining. When the model generates output, it is choosing from this vocabulary at each step.

*One-line definition:* the complete set of tokens a model can recognize and generate; fixed at
pretraining time.

---

### Warmup (`warmup_ratio`, `warmup_steps`)

A training schedule that starts with a very small **learning rate** and gradually ramps it up
over the first few hundred steps before holding at the target rate. Without warmup, large
gradient updates in the first steps (when the LoRA adapter's weights are randomly initialized)
can cause instability or **catastrophic forgetting** of the base model's existing knowledge.

`warmup_ratio=0.05` means warmup runs over the first 5% of total training steps.

*One-line definition:* gradually increasing the learning rate from near-zero at the start of
training to prevent instability from large early gradient steps.

See: Chapter 7 (How Training Actually Works), Chapter 16 (Hyperparameters).

---

### Weight decay

A small penalty added to the **loss** that pushes weights toward zero, preventing any single
weight from growing very large. Combined with **dropout**, it is a standard regularization tool
against **overfitting**. **AdamW** incorporates weight decay; the "W" stands for it. The default
value in most fine-tuning setups is `0.01`.

*One-line definition:* a regularization penalty that keeps weights small and reduces overfitting;
built into the AdamW optimizer.

---

## Code section: numeric reference sheet

When you are staring at training logs and want a quick reminder of what typical numbers look
like, run this script. It prints a reference cheat sheet of the key quantities for our
memory-extraction project.

```python
# glossary_reference.py
# Prints a quick numeric reference sheet for the memory-extraction fine-tuning project.
# Run this any time you want to sanity-check your setup or logs.

# ── Model size reference ──────────────────────────────────────────────────────

model_sizes = {
    "Qwen3-1.7B (4-bit)":  {"params_B": 1.7,  "vram_load_GB": 1.1, "vram_train_GB": 4},
    "Qwen3-4B   (4-bit)":  {"params_B": 4.0,  "vram_load_GB": 2.5, "vram_train_GB": 8},
    "Qwen3-8B   (4-bit)":  {"params_B": 7.6,  "vram_load_GB": 4.5, "vram_train_GB": 14},
    "Gemma3-1B  (4-bit)":  {"params_B": 1.0,  "vram_load_GB": 0.7, "vram_train_GB": 4},
    "Gemma3-4B  (4-bit)":  {"params_B": 4.3,  "vram_load_GB": 2.7, "vram_train_GB": 8},
}

print("=" * 65)
print("MODEL SIZE REFERENCE (all with QLoRA / 4-bit base)")
print("=" * 65)
print(f"{'Model':<26} {'Params':>8} {'Load VRAM':>12} {'Train VRAM':>12}")
print("-" * 65)
for name, info in model_sizes.items():
    # Training VRAM includes: 4-bit base + bf16 adapters + optimizer state
    # These are conservative estimates; Unsloth typically saves 30-40% more
    print(
        f"{name:<26} "
        f"{info['params_B']:>7.1f}B "
        f"{info['vram_load_GB']:>10.1f} GB "
        f"{info['vram_train_GB']:>10d} GB"
    )

# ── LoRA parameter reference ──────────────────────────────────────────────────

print()
print("=" * 65)
print("LORA PARAMETER STARTING POINTS (memory-extraction task)")
print("=" * 65)

lora_params = {
    "r (rank)":           "16   → raise to 32/64 if quality is poor",
    "lora_alpha":         "16   → set equal to r (or 2×r for stronger adapter)",
    "lora_dropout":       "0.05 → 0.0 for >20k examples; 0.1 for <500 examples",
    "target_modules":     "q/k/v/o + gate/up/down (all 7 for structured output)",
    "bias":               "none → rarely changed",
}

for param, guidance in lora_params.items():
    # Right-pad param name for alignment
    print(f"  {param:<20} {guidance}")

# ── Training hyperparameter reference ─────────────────────────────────────────

print()
print("=" * 65)
print("TRAINING HYPERPARAMETER STARTING POINTS")
print("=" * 65)

training_params = {
    "learning_rate":              "2e-4  → cut 5-10x if loss oscillates",
    "num_train_epochs":           "3     → 2-5 for small datasets (<5k rows)",
    "per_device_train_batch_size":"2     → reduce to 1 if OOM",
    "gradient_accumulation_steps":"4     → raise to 8 to simulate larger batch",
    "warmup_ratio":               "0.05  → warmup over first 5% of steps",
    "logging_steps":              "10    → see curve shape; don't set to 500",
}

for param, guidance in training_params.items():
    print(f"  {param:<34} {guidance}")

# ── Healthy loss range reference ──────────────────────────────────────────────

print()
print("=" * 65)
print("LOSS VALUES: WHAT TO EXPECT")
print("=" * 65)

loss_guide = [
    ("Start of training (step 1)",      "2.0 – 3.5",  "Normal; model hasn't learned yet"),
    ("After 10% of steps",              "1.0 – 2.0",  "Should be dropping; warmup ending"),
    ("After 50% of steps",              "0.4 – 0.9",  "Steady descent; check eval loss too"),
    ("End of training (healthy)",       "0.1 – 0.4",  "Model has learned the pattern"),
    ("End of training (suspicious)",    "< 0.05",     "May be overfitting; check eval loss"),
    ("End of training (underfitting)",  "> 1.0",      "Not enough data, epochs, or rank"),
]

print(f"  {'Stage':<36} {'Typical range':>14}   Note")
print("  " + "-" * 61)
for stage, rng, note in loss_guide:
    print(f"  {stage:<36} {rng:>14}   {note}")

print()
print("Rule of thumb: a healthy run sees loss drop by 70-90% from start to end.")
print("Watch eval loss in parallel — a rising eval loss means overfitting.")
```

Running this script prints three reference tables that are useful to have open during any
training run: model VRAM estimates, LoRA parameter guidance, and typical loss ranges. None of
these numbers require memorization — just run the script when you need a reminder.

---

## Common mistakes

**Using the wrong term for the task.** "Fine-tuning" and "training" are often used loosely. In
this book, **training** refers to the general process (including pretraining), while **fine-tuning**
means a second, shorter pass on your own data starting from a pretrained base. "Pretraining" is
the expensive first pass done by the model developer. Using them interchangeably can make
debugging conversations confusing.

**Conflating loss and accuracy.** Loss is a continuous number; accuracy (or F1) is a task-specific
metric. Low loss means the model is predicting tokens well — but that does not automatically mean
it is extracting the right memories. Always pair loss monitoring with a task-level evaluation
(Chapter 18).

**Confusing batch size and effective batch size.** If `per_device_train_batch_size=2` and
`gradient_accumulation_steps=4`, the effective batch size is 8 — not 2. The distinction matters
when comparing training runs or debugging instability.

**Thinking quantization and training precision are the same thing.** The base model is quantized
to 4-bit for storage and loading. The LoRA adapter trains in 16-bit (bf16). These are two
separate and independent settings. Do not try to train the adapter in 4-bit.

**Mixing up validation split and test set.** The validation split is used during training to
detect overfitting and tune hyperparameters. The test set (if you have one) is a completely
held-out set used only for final evaluation after all training decisions are made. Using the
validation split to make decisions about training and then reporting its numbers as "test results"
is a form of data leakage.

**Forgetting that perplexity is just exp(loss).** If a paper reports perplexity and your training
logs report loss, you can convert: `import math; ppl = math.exp(loss)`. A loss of 0.3 is a
perplexity of about 1.35 — very good. A loss of 2.3 is a perplexity of about 10.

---

## Recap

- Every term in this book has a plain-English definition here; use this appendix as a reference
  throughout, not just at the end.
- The memory-extraction task — outputting a JSON array of `{text, type, entities}` objects from
  raw conversation — is the thread connecting almost every term. When a term feels abstract, ask
  "what does this mean for my JSONL training file or my loss curve?"
- Quantization (4-bit) and LoRA (adapter-only training) are the two tricks that make fine-tuning
  feasible on a single consumer GPU. QLoRA stacks them together.
- Loss is the primary number to watch during training. Task metrics (F1, precision, recall) are
  what you check after training to know if the model actually works.
- VRAM, not compute time, is usually the binding constraint for fine-tuning. Know your GPU's
  limit, and reach for gradient checkpointing and smaller batch sizes before reaching for a
  bigger machine.
- When in doubt about any number — VRAM, loss range, LoRA rank — run the reference script in
  this appendix to print a calibrated starting point.
- The appendix is alphabetical: if you know the word, you can find it. If you don't know the
  word, search for it in the chapter where you first encountered it and then come here for the
  clean definition.
- Nothing in this glossary needs to be memorized. It exists so that the main chapters can move
  quickly, and you always have a backstop.

## Next

**Appendix B - Project Layout and Command Cheat-Sheet** — a single-page reference for the
directory structure of the memory-extraction project and every command you need to run, from
installation through serving.
