# Ch12 - Data Format: Turning the Task into Training Rows

You have a task. You have some raw conversations. Now you need to answer the most underestimated question in fine-tuning: *what exactly does a training example look like?*

The format matters more than almost anything else. Get it wrong and training will run without errors, your loss curve will look fine, and your model will still fail silently at inference time. This chapter is about getting it right from the start.

---

## What you'll learn

- What the three-part "instruction tuning" format is (system prompt, user message, assistant reply) and why it works
- How to shape one memory-extraction example into a training row
- What JSONL is and why it is the standard format for training data files
- The exact Python dictionary structure that Unsloth and TRL expect
- How to write a builder function that creates, validates, and saves training rows

---

## Concepts you need first

### Instruction tuning vs. raw text training

When you train a model on raw text — say, a giant pile of Wikipedia articles — you are teaching it to predict the next word. That is useful but it does not teach the model to *follow instructions*.

Instruction tuning is the step that turns a raw language model into an assistant. You show it thousands of examples of: here is an instruction, here is an input, here is the correct response. The model learns the pattern: when I see an instruction followed by some text, I should produce an output that looks like those correct responses.

Think of it like training a new employee. Reading every company document gives them background knowledge. But they only learn *how to do the job* when you show them real examples of tasks being done correctly.

For our memory-extraction task, the instruction is always the same: "extract atomic memories from this conversation." The input changes with every example. The correct output is a JSON list of memories.

### The three-message format (chat template)

Modern language models are trained to expect a specific three-part structure for each example:

1. **System message** — the standing instruction. Describes the job, the output format, and any rules. You write this once. It gets prepended to every training example and every inference call.
2. **User message** — the input for this specific example. In our task, this is the conversation chunk.
3. **Assistant message** — the target output. What the model should produce. In our task, this is the JSON list of memories.

Together these three parts are called a **conversation** or a **chat sample**. Under the hood, the tokenizer (the piece of software that converts text to numbers — covered in *Ch5 - Tokens, Context Windows, and Chat Templates*) wraps these parts in special tokens that mark where each speaker starts and stops. The model learns to associate the end of the user turn with the start of the assistant turn.

**Why the system prompt must match at inference time.** During training, the model sees the system prompt on every single example. It learns: "when I see these exact instructions, produce output in this exact format." If you use a different system prompt — or no system prompt — at inference time, you are giving the model an unfamiliar context and quality drops noticeably. Copy-paste your system prompt from training into your inference code verbatim. We will come back to this in *Ch22 - Serving Your Model and Using It in an App*.

### JSONL — one row, one line

JSONL stands for JSON Lines. It is a plain text file where each line is one valid JSON object. No commas between lines. No outer array wrapper. Just one self-contained JSON object per line.

```
{"messages": [...]}   ← line 1, example 1
{"messages": [...]}   ← line 2, example 2
{"messages": [...]}   ← line 3, example 3
```

Why this format? Because you can stream it. Reading a 50,000-row training file one line at a time uses almost no memory, whereas loading a single giant JSON array loads everything at once. The Hugging Face `datasets` library, TRL's `SFTTrainer`, and virtually every ML training tool expect JSONL.

---

## The memory schema, pinned

Before we build training rows, we need to agree on the exact JSON schema for a memory object. We defined this task in *Ch11 - Defining the Task: What "Memory Extraction" Means*. Here is the schema we will use throughout the rest of the book:

```python
# A single memory looks like this:
{
    "text": "Sarah prefers dark roast coffee in the morning",   # The fact, written as a complete sentence
    "type": "preference",                                        # One of: preference | fact | decision | relationship
    "entities": ["Sarah"]                                        # List of named people, places, or things involved
}
```

The assistant message in every training example will be a JSON array of zero or more objects that match this schema. Zero is valid — not every conversation contains memorable facts.

---

## Building one training row

Let's build a single training example end-to-end in Python. We will define the system prompt, a sample conversation, and the target output, then assemble them into the structure TRL expects.

```python
import json

# ── 1. The system prompt ────────────────────────────────────────────────────
# This exact string MUST be used at inference time too.
# It tells the model what job it is doing and what format to produce.
SYSTEM_PROMPT = """You are a memory extraction assistant. Your job is to read a conversation and extract every memorable piece of information as a list of atomic memory objects.

Each memory object must follow this exact JSON schema:
{
  "text": "<the fact, written as a complete, standalone sentence>",
  "type": "<one of: preference | fact | decision | relationship>",
  "entities": ["<list of named people, places, or things involved>"]
}

Rules:
- One fact per memory object. Do not bundle multiple facts into one.
- Write "text" as a sentence someone could read without any surrounding context.
- If there are no memorable facts in the conversation, return an empty list: []
- Return ONLY a valid JSON array. No explanation, no markdown fences, no extra text.
"""

# ── 2. One example conversation (the user message) ──────────────────────────
# This is the raw input the model will see during both training and inference.
USER_MESSAGE = """Alex: I finally switched to using Obsidian for all my notes. Notion was getting too slow.
Jamie: Nice! I've been thinking about it too. Do you use it for work or personal stuff?
Alex: Both. I keep my book notes there too — I'm reading through the entire Dune series again.
Jamie: Love that series. Have you read the ones Brian Herbert wrote?
Alex: Started the first one, didn't finish it. Frank Herbert's originals are just on another level.
"""

# ── 3. The target assistant output ──────────────────────────────────────────
# This is what we WANT the model to produce given the conversation above.
# Each object is atomic — one fact per object, no bundles.
TARGET_MEMORIES = [
    {
        "text": "Alex switched from Notion to Obsidian for note-taking.",
        "type": "decision",
        "entities": ["Alex", "Notion", "Obsidian"]
    },
    {
        "text": "Alex uses Obsidian for both work and personal notes.",
        "type": "fact",
        "entities": ["Alex", "Obsidian"]
    },
    {
        "text": "Alex keeps book notes in Obsidian.",
        "type": "fact",
        "entities": ["Alex", "Obsidian"]
    },
    {
        "text": "Alex found Notion too slow.",
        "type": "preference",
        "entities": ["Alex", "Notion"]
    },
    {
        "text": "Alex is re-reading the entire Dune series.",
        "type": "fact",
        "entities": ["Alex", "Dune"]
    },
    {
        "text": "Alex prefers Frank Herbert's original Dune books over Brian Herbert's continuations.",
        "type": "preference",
        "entities": ["Alex", "Frank Herbert", "Brian Herbert", "Dune"]
    },
    {
        "text": "Alex started but did not finish the first Brian Herbert Dune book.",
        "type": "fact",
        "entities": ["Alex", "Brian Herbert", "Dune"]
    }
]

# ── 4. Assemble the training row ─────────────────────────────────────────────
# TRL's SFTTrainer expects a dict with a "messages" key.
# The value is a list of dicts, each with "role" and "content".
# Roles must be exactly: "system", "user", "assistant"
def build_training_row(system_prompt: str, user_message: str, target_memories: list) -> dict:
    """
    Build one JSONL-ready training row in the format TRL + Unsloth expect.

    Args:
        system_prompt:    The standing instruction string (same for every row).
        user_message:     The conversation text (unique per row).
        target_memories:  A Python list of memory dicts (the desired output).

    Returns:
        A dict with a single key "messages" containing the three-turn conversation.
    """

    # Serialize the memories list to a JSON string.
    # This is what the model will learn to produce as text output.
    # indent=2 makes it slightly easier to read in a file; the model handles both.
    assistant_content = json.dumps(target_memories, indent=2, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": assistant_content}
        ]
    }

# Build the row
row = build_training_row(SYSTEM_PROMPT, USER_MESSAGE, TARGET_MEMORIES)

# Quick sanity check: print it
print(json.dumps(row, indent=2, ensure_ascii=False))
```

Run this and you will see a single, complete training row printed to your terminal. The `messages` list has exactly three items: system, user, assistant.

---

## Validating the row before you save it

One malformed row in a 5,000-row file will either crash training partway through or silently skip the example. Build validation in from the start.

```python
def validate_memory_list(raw_json_string: str) -> tuple[bool, str]:
    """
    Validate that a JSON string is a list of well-formed memory objects.

    Returns:
        (True, "") if valid.
        (False, <error description>) if not.
    """
    # Step 1: can we even parse it as JSON?
    try:
        data = json.loads(raw_json_string)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"

    # Step 2: is it a list?
    if not isinstance(data, list):
        return False, f"Expected a JSON array, got {type(data).__name__}"

    # Step 3: check every item in the list
    valid_types = {"preference", "fact", "decision", "relationship"}

    for i, item in enumerate(data):
        # Each item must be a dict
        if not isinstance(item, dict):
            return False, f"Item {i} is not an object (got {type(item).__name__})"

        # Check required keys
        for key in ("text", "type", "entities"):
            if key not in item:
                return False, f"Item {i} is missing required key '{key}'"

        # "text" must be a non-empty string
        if not isinstance(item["text"], str) or not item["text"].strip():
            return False, f"Item {i}: 'text' must be a non-empty string"

        # "type" must be one of the allowed values
        if item["type"] not in valid_types:
            return False, f"Item {i}: 'type' must be one of {valid_types}, got '{item['type']}'"

        # "entities" must be a list of strings
        if not isinstance(item["entities"], list):
            return False, f"Item {i}: 'entities' must be a list"
        if not all(isinstance(e, str) for e in item["entities"]):
            return False, f"Item {i}: all items in 'entities' must be strings"

    return True, ""


def validate_training_row(row: dict) -> tuple[bool, str]:
    """
    Validate the overall structure of a training row before saving it.
    """
    # Must have the "messages" key
    if "messages" not in row:
        return False, "Row missing 'messages' key"

    messages = row["messages"]

    # Must be a list of exactly 3 items
    if not isinstance(messages, list) or len(messages) != 3:
        return False, f"'messages' must be a list of 3 items, got {len(messages) if isinstance(messages, list) else type(messages)}"

    # Check roles in order
    expected_roles = ["system", "user", "assistant"]
    for i, (msg, expected_role) in enumerate(zip(messages, expected_roles)):
        if msg.get("role") != expected_role:
            return False, f"Message {i}: expected role '{expected_role}', got '{msg.get('role')}'"
        if not isinstance(msg.get("content"), str) or not msg["content"].strip():
            return False, f"Message {i} (role={expected_role}): 'content' must be a non-empty string"

    # Validate the assistant's content as a memory list
    assistant_content = messages[2]["content"]
    ok, err = validate_memory_list(assistant_content)
    if not ok:
        return False, f"Assistant content failed memory validation: {err}"

    return True, ""


# Validate the row we just built
ok, error = validate_training_row(row)
if ok:
    print("Row is valid.")
else:
    print(f"Row is INVALID: {error}")
```

---

## Writing to a JSONL file

Once you have valid rows, you save them one per line. Each line must be a complete, self-contained JSON object — no line breaks inside it.

```python
import pathlib

def save_rows_to_jsonl(rows: list[dict], output_path: str) -> int:
    """
    Save a list of training rows to a JSONL file.
    Each row is validated before writing. Invalid rows are skipped with a warning.

    Args:
        rows:         List of training row dicts.
        output_path:  Path to the output .jsonl file.

    Returns:
        Number of rows successfully written.
    """
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)  # create directories if needed

    written = 0
    skipped = 0

    with path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            ok, error = validate_training_row(row)

            if not ok:
                # Print a warning but keep going — don't crash on one bad row
                print(f"[SKIP] Row {i}: {error}")
                skipped += 1
                continue

            # json.dumps with no indent produces a single-line JSON string.
            # ensure_ascii=False preserves non-ASCII characters (names, accents, etc.)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} rows to {output_path}  ({skipped} skipped)")
    return written


# Save our one example row as a demonstration
rows = [row]  # In Ch13 we'll build thousands of these
save_rows_to_jsonl(rows, "data/raw/memory_extraction_sample.jsonl")
```

---

## Reading the file back — what Unsloth and TRL will see

Let's verify the round-trip. Load the file and confirm the structure looks exactly right:

```python
def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, returning a list of row dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip blank lines
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_num}: could not parse JSON — {e}")
    return rows


loaded = load_jsonl("data/raw/memory_extraction_sample.jsonl")

# Print a summary so we can eyeball it
for i, r in enumerate(loaded):
    msgs = r["messages"]
    print(f"Row {i}:")
    print(f"  system  ({len(msgs[0]['content'])} chars): {msgs[0]['content'][:60]}...")
    print(f"  user    ({len(msgs[1]['content'])} chars): {msgs[1]['content'][:60]}...")
    print(f"  asst    ({len(msgs[2]['content'])} chars): {msgs[2]['content'][:60]}...")

    # Parse the assistant output back into Python to confirm it's valid JSON
    memories = json.loads(msgs[2]["content"])
    print(f"  → {len(memories)} memories extracted")
```

When TRL's `SFTTrainer` loads your JSONL file (via Hugging Face `datasets`), it reads rows in exactly this structure. The tokenizer then applies the model's **chat template** — a set of special tokens specific to Qwen3 or Gemma 3 — to wrap the three messages. You do not need to add those tokens manually; the tokenizer handles it. This is covered in detail in *Ch15 - Your First Fine-Tune with Unsloth (Full Script)*.

---

## What about multi-turn conversations?

The format above has exactly three messages: one system, one user, one assistant. That covers the vast majority of fine-tuning cases, including ours.

You can have more turns (user → assistant → user → assistant → ...) for tasks that require back-and-forth dialogue. For memory extraction, that would be unusual — a user submits a chunk of text, the model returns JSON, done. We will keep the three-message format throughout this book. If your task genuinely needs multi-turn training, the structure is the same `messages` list, just longer; TRL handles it identically.

---

## Common mistakes

**1. Using a different system prompt at inference time.**

This is the single most common cause of "it worked in training but fails in production." The model does not just learn a skill in the abstract; it learns that skill *conditional on the exact phrasing of the system prompt*. Store your system prompt as a constant in a shared module and import it in both your training code and your inference code.

**2. Putting JSON in a markdown code fence in the assistant message.**

It is tempting to write the assistant content as:

```
```json
[{"text": "..."}]
```
```

Do not. Your training rows should contain the raw JSON string, no backticks, no `json` language tag. If you train with fences, the model will produce fences at inference time, and your JSON parser will fail.

**3. Pretty-printing the assistant JSON in the JSONL file line.**

`json.dumps(row, indent=2)` adds newlines inside the row, which breaks the one-row-per-line contract of JSONL. Always use `json.dumps(row)` (no indent) when writing rows to the file. You can use `indent=2` when printing to the terminal for readability.

**4. Empty entities list when entities are clearly present.**

If the conversation mentions "Sarah" and your training row has `"entities": []`, the model learns inconsistent patterns. Be thorough: include every named person, product, or place that appears in the fact. An empty entities list should only appear for truly generic facts with no named actors.

**5. Including non-UTF-8 characters without `ensure_ascii=False`.**

Python's `json.dumps` by default escapes non-ASCII characters as `\uXXXX`. This is fine but inflates file size and makes files harder to read. Pass `ensure_ascii=False` to keep names like "André" or "東京" readable. The Hugging Face `datasets` library handles both fine.

**6. Forgetting to create the output directory.**

`open("data/raw/file.jsonl", "w")` fails if `data/raw/` does not exist. The `save_rows_to_jsonl` function above uses `pathlib.Path.mkdir(parents=True, exist_ok=True)` to create it automatically. Always do this.

---

## Recap

- A training example for instruction tuning is a three-message conversation: system prompt, user message, assistant reply.
- The system prompt describes the task and output format. It must be identical at training time and inference time.
- The assistant message for our memory-extraction task is a raw JSON array of memory objects — no markdown fences, no explanation.
- Each memory object has three fields: `text` (a standalone sentence), `type` (one of four values), and `entities` (list of named things).
- Training data files are saved as JSONL: one JSON object per line, each with a `"messages"` key.
- Validate rows before writing them. One bad row does not break the file, but silent schema drift across thousands of rows will hurt model quality.
- TRL's `SFTTrainer` loads JSONL rows in the `"messages"` format directly; the tokenizer applies chat-template tokens automatically.

## Next

*Ch13 - Creating Your Training Data with Synthetic Generation* — we have the format; now we use a teacher model (via the Anthropic or OpenAI API) to generate thousands of conversation/memory pairs automatically, filling the training set we need to fine-tune.
