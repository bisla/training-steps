# Ch5 - Tokens, Context Windows, and Chat Templates

Before you can train a model on your memory-extraction data, the model needs to be able to *read* it. But models don't read text. They read numbers. This chapter explains the pipeline that turns your raw strings into something a model can process — and why getting one step of that pipeline wrong silently breaks your entire fine-tune.

---

## What you'll learn

- What a tokenizer is and why text is split into integer IDs before the model ever sees it
- Why "number of tokens" and "number of words" are different things — and why that gap bites you
- What a context window is, why larger windows cost exponentially more, and what that means for your training budget
- What chat templates are, what special tokens like `<|im_start|>` and `<bos>` actually do, and how Qwen3 and Gemma 3 differ
- How to write runnable code that tokenizes a conversation sample and applies a chat template, so you can verify your data looks exactly right before spending a cent on training

---

## Concepts you need first

### Concept 1: Tokenization — text as a sequence of integer IDs

**Analogy.** Imagine a model is a musician who can only read sheet music, not tablature, not chord charts. Sheet music has a fixed vocabulary of symbols — whole notes, half notes, sharps, rests. To play your song, someone has to transcribe it into that notation first. That transcription step is tokenization.

**One-line definition.** A tokenizer splits your raw text into small chunks called *tokens* and then maps each chunk to a unique integer ID from a fixed vocabulary table.

**Why it matters for us.** Every training example in your memory-extraction dataset is a string. The model never processes that string directly. It processes a list of integers. The tokenizer is the translator between those two worlds. If the tokenizer is wrong, or if you use a different tokenizer at inference time than at training time, the model receives gibberish — like playing the wrong sheet music.

### Concept 2: Subword tokens — why the vocabulary isn't just words

**Analogy.** Imagine trying to build a dictionary with every English word in it. Now add every name, every technical term, every code snippet. You'd need millions of entries. Worse, you'd have no way to handle a word you've never seen. Subword tokenization solves this by breaking words into meaningful pieces — "tokenization" might become "token", "ization". "Unhappiness" might become "un", "happiness". Common short words get their own token; rare long words get split.

**One-line definition.** Subword tokenization uses an algorithm (usually Byte-Pair Encoding, or BPE) to find the set of ~32,000–128,000 chunks that most efficiently covers real text, so that common patterns stay whole and rare patterns get split.

**Why it matters for us.** Your memory extraction output is JSON — curly braces, colons, quoted strings, field names like `"entities"` and `"type"`. Each of those characters and substrings will be tokenized. JSON field names that appear constantly in your training data (like `"memory_text"`) will likely get a single token; complex strings might split unpredictably. Understanding this helps you design your output schema to be tokenizer-friendly (short, common field names).

### Concept 3: Context window — the model's working memory

**Analogy.** Think of the context window as a notepad that the model can read while generating its answer. It can hold a fixed number of tokens — say 8,192 or 32,768. Once you fill the notepad, you can't add more. The model cannot read anything that doesn't fit on the notepad.

**One-line definition.** The context window is the maximum number of tokens the model processes in a single forward pass — it includes your system prompt, the user message, and the assistant's response, all together.

**Why it matters for us.** During fine-tuning, each training row must fit inside the context window. A row is: [system prompt] + [input conversation] + [expected JSON output]. If a row is longer than the window, the trainer either truncates it (silently losing your expected output) or errors out. You need to know your target model's window size and make sure your training rows fit comfortably inside it. The cost argument matters here too: 32k tokens is 16× longer than 2k tokens — but because attention scales quadratically with sequence length, the attention computation grows by roughly 256× compared to a 2k-token row. Long rows are dramatically more expensive than their length alone suggests. Keep your training rows as short as they can be while still conveying the full task.

### Concept 4: Chat templates and special tokens — the model's expected format

**Analogy.** Imagine you're writing a letter to a government office. There's a required format: your name in the top-left, the date below it, a subject line, then the body, then a formal sign-off. If you just hand them a paragraph with no formatting, the clerk doesn't know where your name ends and your request begins. Models are the same: they were trained to expect conversation in a very specific format, with specific marker strings (special tokens) that signal "system prompt starts here," "user turn starts here," "assistant turn starts here." Deviate from that format, and the model's responses degrade — it's reading a letter that wasn't written in the required format.

**One-line definition.** A chat template is the exact string format that wraps each conversation turn, including special tokens like `<|im_start|>system`, `<bos>`, `[INST]`, and `<|eot_id|>`, that tell the model which role is speaking and where turns begin and end.

**Why it matters for us.** The single most common cause of a broken fine-tune — a model that was trained for hours but produces garbage — is a wrong or inconsistent chat template. If your training data uses the template from one model and you load a different base model, the special tokens don't match and the model never learns what you actually wanted.

---

## Tokenizers in practice

Let's make this concrete. We'll load the Qwen3 tokenizer, tokenize a raw string, and look at what comes out.

```python
# Install if you haven't yet:
# pip install transformers tokenizers torch

from transformers import AutoTokenizer

# We load only the tokenizer here — not the full model.
# This downloads a small config file, not multi-gigabyte weights.
# Use the same model ID you'll use for training (see Ch10 for how to choose).
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# A short sample — the kind of text we want to extract memories from.
sample_text = "Alice told Bob she prefers async Python over threading."

# tokenize() returns the integer IDs directly.
token_ids = tokenizer.encode(sample_text)
print("Token IDs:", token_ids)
# Example output: [38949, 5707, 13432, 1057, 19350, 13316, 916, 39123, 13]
# (Your exact IDs will vary — different tokenizers produce different numbers.)

# convert_ids_to_tokens() shows you the actual subword chunks.
tokens = tokenizer.convert_ids_to_tokens(token_ids)
print("Tokens:", tokens)
# Example output: ['Alice', 'Ġtold', 'ĠBob', 'Ġshe', 'Ġprefers', 'Ġasync', 'ĠPython', 'Ġover', 'Ġthreading', '.']
# Note: 'Ġ' is how HuggingFace represents a leading space character inside a subword.

print(f"Word count: {len(sample_text.split())}")   # 9
print(f"Token count: {len(token_ids)}")             # ~10 — close here, but not always
```

Notice that the word count and token count are similar for clean English prose. They diverge significantly for code, JSON, and non-English text:

```python
# JSON output is punchier — let's see how it tokenizes.
json_output = '{"memory_text": "Alice prefers async Python over threading.", "type": "preference", "entities": ["Alice"]}'

json_ids = tokenizer.encode(json_output)
json_tokens = tokenizer.convert_ids_to_tokens(json_ids)

print(f"JSON character count: {len(json_output)}")    # ~98 characters
print(f"JSON token count: {len(json_ids)}")           # ~30–35 tokens — much less than characters
print("JSON tokens:", json_tokens)
# You'll see curly braces, quotes, colons each get their own token or pair up.
# Field names like 'memory_text' may split: ['memory', '_text'] or stay whole.
```

This is useful intelligence for your schema design. If `memory_text` splits into two tokens but `text` stays whole, using the shorter field name slightly reduces training cost. It's a small optimization, but worth knowing.

---

## Counting tokens before training

Before you build your full dataset (covered in Ch12 and Ch13), you need a way to check that rows fit in the context window. Here's a utility function you'll reuse throughout the book:

```python
def count_tokens(tokenizer, text: str) -> int:
    """
    Returns the number of tokens in a string.
    Use this to verify your training rows fit inside the model's context window.
    """
    # add_special_tokens=False because we'll add them via apply_chat_template later.
    return len(tokenizer.encode(text, add_special_tokens=False))

# Qwen3-0.6B has a context window of 32,768 tokens.
# A typical training row for our task (system prompt + short convo + JSON output)
# is roughly 200–500 tokens — well within budget.
CONTEXT_WINDOW = 32_768
MAX_ROW_TOKENS = 2048  # leave headroom; don't fill the window completely

sample_row = (
    "You are a memory extraction assistant. "
    "Extract all facts from the conversation as JSON.\n\n"
    "User: Alice told Bob she prefers async Python over threading.\n\n"
    '{"memory_text": "Alice prefers async Python over threading.", '
    '"type": "preference", "entities": ["Alice"]}'
)

row_token_count = count_tokens(tokenizer, sample_row)
print(f"Row token count: {row_token_count}")   # ~80–120 tokens
print(f"Fits in window: {row_token_count <= MAX_ROW_TOKENS}")  # True
```

---

## Chat templates: the format that cannot be wrong

Now the important part. When you fine-tune, you don't feed the model raw text. You feed it a *formatted conversation* using the exact template the base model was trained with. The `transformers` library stores this template on the tokenizer object and provides a method called `apply_chat_template` that generates the correctly formatted string for you.

Let's look at what Qwen3 actually produces:

```python
# A training example for memory extraction.
# This is structured as a conversation: system prompt, user message, assistant reply.
messages = [
    {
        "role": "system",
        "content": (
            "You are a memory extraction assistant. "
            "Given a conversation, extract all atomic facts, preferences, "
            "decisions, and relationships. "
            "Return a JSON list of objects, each with keys: "
            "memory_text (string), type (string), entities (list of strings)."
        ),
    },
    {
        "role": "user",
        "content": "Alice told Bob she prefers async Python over threading. "
                   "Bob said he'll set up the project with asyncio.",
    },
    {
        "role": "assistant",
        "content": (
            '[\n'
            '  {"memory_text": "Alice prefers async Python over threading.", '
            '"type": "preference", "entities": ["Alice"]},\n'
            '  {"memory_text": "Bob will set up the project with asyncio.", '
            '"type": "decision", "entities": ["Bob"]}\n'
            ']'
        ),
    },
]

# apply_chat_template adds the special tokens and role markers.
# tokenize=False returns a string so we can read it; set tokenize=True for training.
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,         # give us the raw string first so we can inspect it
    add_generation_prompt=False,  # False when the assistant turn is already in the messages
)

print(formatted)
```

For Qwen3, the output looks roughly like this (your exact special token IDs will vary):

```
<|im_start|>system
You are a memory extraction assistant. Given a conversation, extract all atomic facts...
<|im_end|>
<|im_start|>user
Alice told Bob she prefers async Python over threading. Bob said he'll set up the project with asyncio.
<|im_end|>
<|im_start|>assistant
[
  {"memory_text": "Alice prefers async Python over threading.", "type": "preference", "entities": ["Alice"]},
  {"memory_text": "Bob will set up the project with asyncio.", "type": "decision", "entities": ["Bob"]}
]
<|im_end|>
```

The special tokens `<|im_start|>` and `<|im_end|>` are Qwen's way of marking where each turn begins and ends. The string `system`, `user`, `assistant` after `<|im_start|>` tells the model which role is speaking.

Now let's do the same for Gemma 3:

```python
# Load Gemma 3's tokenizer — note: you need a HuggingFace account and
# to accept Gemma's license at huggingface.co/google/gemma-3-1b-it
gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

gemma_formatted = gemma_tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

print(gemma_formatted)
```

Gemma 3's output looks different:

```
<bos><start_of_turn>user
You are a memory extraction assistant...

Alice told Bob she prefers async Python over threading...<end_of_turn>
<start_of_turn>model
[
  {"memory_text": "Alice prefers async Python over threading.", ...}
]
<end_of_turn>
```

A few differences to notice:
- Gemma uses `<bos>` (beginning-of-sequence) at the very start; Qwen handles BOS differently.
- Gemma folds the system prompt into the first user turn — there is no separate `system` role in Gemma's template; it prepends your system content to the user content automatically.
- Gemma uses `<start_of_turn>` / `<end_of_turn>` instead of `<|im_start|>` / `<|im_end|>`.
- Gemma calls the assistant role `model`, not `assistant`.

These differences are not cosmetic. They are baked into the model's weights from pretraining. If you train Gemma on data formatted in Qwen's style, the model will never perform correctly — it's reading the wrong sheet music. Always use `apply_chat_template` with the tokenizer that matches your base model.

---

## Getting the tokenized form for training

During actual training, you need integer IDs, not strings. Here's how to get them:

```python
# tokenize=True returns a dict with 'input_ids' and 'attention_mask'.
# input_ids: the integer IDs the model processes.
# attention_mask: a list of 1s and 0s; 1 = real token, 0 = padding (ignore this token).
tokenized = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt",   # "pt" = PyTorch tensors; what Unsloth expects
)

print("input_ids shape:", tokenized["input_ids"].shape)
# e.g. torch.Size([1, 187]) — batch size 1, 187 tokens

print("Token count for this training row:", tokenized["input_ids"].shape[1])
# Keep this well under your context window limit (32,768 for Qwen3).

# You can also get back a dict instead of tensors for easier inspection:
tokenized_dict = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_dict=True,
)
print("First 10 token IDs:", tokenized_dict["input_ids"][:10])
```

In Ch15, when you run the full training script with Unsloth, the library handles this call for you automatically via its `DataCollatorForSeq2Seq` setup — but it's using exactly this function under the hood. Seeing it now means you'll understand what the training loop is doing to your data.

---

## Common mistakes

**Mistake 1: Mixing tokenizers from different models.**
If you load Qwen3's weights but accidentally apply Gemma's chat template (perhaps copied from a tutorial), your training data will contain special tokens the model doesn't recognize. The loss will appear to go down but the model will produce malformed JSON or repeat the template markers in its output. Fix: always call `apply_chat_template` on the tokenizer you loaded from the same `model_id` you're training.

**Mistake 2: Using `add_generation_prompt=True` in training data.**
`add_generation_prompt=True` appends the opening of an assistant turn (e.g. `<|im_start|>assistant\n`) without closing it, which signals "generate from here." That's correct at inference time when you want the model to continue writing. But in training data, you already have the full assistant response — using `add_generation_prompt=True` here creates a malformed example with a dangling half-turn. Fix: `add_generation_prompt=False` for training rows; `add_generation_prompt=True` only when running inference.

**Mistake 3: Assuming token count equals word count.**
Engineers often estimate dataset size by word count and then hit context window limits in training because the actual token count is higher. JSON curly braces, escaped quotes, field names, and non-ASCII characters all inflate token counts beyond what a word count suggests. Fix: always use `tokenizer.encode()` to measure real token length before you build your full dataset.

**Mistake 4: Ignoring the system prompt's token cost.**
Your system prompt might be 80 words but it appears in *every single training row*. If it's 120 tokens and you have 5,000 training rows, that's 600,000 tokens of context you're paying for — money that could be spent on longer, richer examples. Fix: keep your system prompt tight. "Extract atomic facts from the conversation as JSON objects with keys memory_text, type, entities." says the same thing in half the tokens.

**Mistake 5: Not verifying the formatted output.**
It's easy to pass the wrong arguments to `apply_chat_template` and get a subtly wrong format — for example, missing the closing `<|im_end|>`. If you never print the formatted string and inspect it, you won't catch this until training is done. Fix: always print `apply_chat_template(..., tokenize=False)` on a few samples before starting a training run. Takes thirty seconds; saves hours.

---

## Recap

- A tokenizer converts text into a list of integer token IDs using a fixed vocabulary of ~32k–128k subword chunks.
- Token count is not word count — JSON, code, and rare words inflate token counts relative to plain prose.
- The context window is the maximum total tokens (system prompt + user + assistant) the model can process in one step. Qwen3 and Gemma 3 both support large windows, but compute cost scales quadratically with length, so shorter training rows are cheaper.
- Chat templates wrap each conversation turn with special tokens (`<|im_start|>`, `<bos>`, `<start_of_turn>`) that vary between models. These tokens are not interchangeable.
- Qwen3 and Gemma 3 use different template formats, different role names, and different ways of handling system prompts. Always use `apply_chat_template` from the matching tokenizer.
- Use `add_generation_prompt=False` when building training data; `add_generation_prompt=True` only when generating predictions at inference time.
- Always visually inspect a few formatted examples before starting training. It's the cheapest debugging step in the whole pipeline.

## Next

**Ch6 - LoRA and QLoRA Without the Math Headache** — now that your data is formatted and tokenized, the next chapter explains how to update a tiny fraction of the model's parameters (not all 7 billion of them) to teach it your task, and how 4-bit quantization lets you do that on a single consumer GPU.
