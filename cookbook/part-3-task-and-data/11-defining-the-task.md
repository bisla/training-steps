# Ch11 - Defining the Task: What "Memory Extraction" Means

Before you write a single line of training code, you need a precise spec. The spec is the contract. Every training example you build in Ch12 and Ch13, every evaluation in Ch18, every debugging step in Ch19 — all of it flows from this one document. Get it wrong here, and you end up with a model that does *something* loosely related to what you wanted, but not the thing you actually need.

This chapter nails that contract.

---

## What you'll learn

- What "memory extraction" means precisely: exact input format, exact output format, exact rules
- The JSON schema we will use for every training example and every evaluation in this book
- How to recognize a *good* extraction versus a *bad* one
- Three fully worked input → output examples, including the edge case where there is nothing to extract
- How to validate schema compliance in Python so bugs surface early

---

## Concepts you need first

### Structured output

Most people think of an LLM as something that writes prose: paragraphs, essays, chat replies. But a model can be trained to output any text — including valid JSON. When you fine-tune a model to always respond with a JSON array, you are teaching it a *structured output* task. The training examples show it the pattern so many times that it learns: "for this kind of input, the right output is a well-formed JSON object, not a paragraph."

Why does this matter for memory extraction? Because you need to feed the model's output into downstream code. A paragraph is hard to parse. A JSON object with fixed field names is trivial to parse. Structured output is the bridge between "model does something useful" and "you can actually use it in an app."

### Atomicity

A memory is *atomic* when it contains exactly one standalone fact. "Alice likes coffee and hates mornings and lives in Berlin" is not atomic — it is three facts bundled together. If you store it as one memory, you will retrieve all three whenever you search for any one of them, which creates noise. Break it into three separate memories, each of which makes sense on its own, and retrieval becomes precise.

This is exactly the principle behind products like mem0: the value is not in storing text blobs, it is in storing *clean, fine-grained, retrievable facts* that a future model (or search system) can use without re-reading the whole conversation.

---

## The spec: input, output, and rules

### Input

The model receives a chunk of conversational text. This might be:

- A single message ("I'm trying to cut down on sugar")
- A multi-turn chat exchange (user + assistant alternating)
- A block of meeting notes or a journal entry

For this book we represent multi-turn input as plain text with speaker labels, because that is the simplest format that works across models and is easy to synthesize. Here is what our input looks like:

```
User: I just moved to Tokyo last month. Still getting used to the time zone.
Assistant: That's a big move! How are you finding it?
User: I love it so far. I'm vegetarian, so finding good food took some effort at first.
Assistant: Tokyo actually has great vegetarian options once you know where to look.
User: Yeah, my colleague Aiko showed me a few spots. She's been really helpful.
```

Rules for input:
- Speaker labels are `User:` and `Assistant:` (capitalized, followed by a space)
- Each turn is on its own line
- No length limit is enforced at this stage; in practice, keep chunks under the model's context window (see Ch5)

### Output: the JSON schema

The model must always respond with a JSON array. Each element of the array is a memory object. Here is the schema:

```json
[
  {
    "text": "string — the memory, written as a standalone declarative sentence",
    "type": "string — one of: fact | preference | relationship | decision | event",
    "entities": ["string", "..."]
  }
]
```

Field-by-field breakdown:

| Field | Type | What goes here |
|-------|------|----------------|
| `text` | string | The memory itself, written as a complete sentence a future reader can understand without context |
| `type` | string enum | Category of memory (see table below) |
| `entities` | string array | Named things mentioned: people, places, products, orgs |

**Memory types:**

| Type | Meaning | Example |
|------|---------|---------|
| `fact` | A stable, objective fact about a person, place, or thing | "The user lives in Tokyo." |
| `preference` | Something the person likes, dislikes, or prefers | "The user is vegetarian." |
| `relationship` | A social or professional connection | "The user's colleague is named Aiko." |
| `decision` | A choice the person made or is planning to make | "The user decided to adopt a dog." |
| `event` | Something that happened or is scheduled | "The user moved to Tokyo last month." |

If a fact does not fit cleanly, use `fact` as the fallback. The type is a hint for downstream retrieval, not a rigid taxonomy — being approximately right is fine.

### The rules (non-negotiable)

1. **Atomic.** One fact per memory object. Never bundle two facts into one `text` field.
2. **Standalone.** The `text` must make sense without reading the original conversation. "The user likes it there" is bad (what is "there"?). "The user likes living in Tokyo" is good.
3. **No hallucination.** Every memory must be directly supported by the input text. Do not infer, guess, or extrapolate. If the user says "I moved to Tokyo," do not generate "The user speaks Japanese."
4. **Deduplicated.** If the same fact appears twice in the conversation, extract it once.
5. **No filler.** Do not extract pleasantries, generic statements, or things the assistant said as facts about the user.
6. **Empty is valid.** If the input contains no extractable memories, return an empty array: `[]`. Never force a memory out of nothing.

---

## Three fully worked examples

### Example 1: A rich conversation

**Input:**

```
User: Hey, I've been thinking about switching from Notion to Obsidian for my notes.
Assistant: What's driving that?
User: Mostly the offline access. I travel a lot for work and can't always rely on internet.
Assistant: Makes sense. Are you a heavy markdown user?
User: Yeah, I write everything in markdown. I'm a backend engineer so it feels natural.
User: Oh, also — I'm based in Amsterdam. Most of my team is in the US so I do a lot of early morning calls.
```

**Output:**

```json
[
  {
    "text": "The user is considering switching from Notion to Obsidian for note-taking.",
    "type": "decision",
    "entities": ["Notion", "Obsidian"]
  },
  {
    "text": "The user values offline access in their tools because they travel frequently for work.",
    "type": "preference",
    "entities": []
  },
  {
    "text": "The user writes everything in markdown.",
    "type": "preference",
    "entities": []
  },
  {
    "text": "The user is a backend engineer.",
    "type": "fact",
    "entities": []
  },
  {
    "text": "The user is based in Amsterdam.",
    "type": "fact",
    "entities": ["Amsterdam"]
  },
  {
    "text": "Most of the user's team is based in the US.",
    "type": "fact",
    "entities": ["US"]
  },
  {
    "text": "The user does a lot of early morning calls because most of their team is in the US.",
    "type": "event",
    "entities": ["US"]
  }
]
```

Notice: "it feels natural" is not extracted as a memory — it is a passing comment, not a standalone fact. "Can't always rely on internet" is captured indirectly through the preference for offline access, but we don't invent "the user has unreliable internet" as a fact.

---

### Example 2: A short, focused message

**Input:**

```
User: Just booked flights to Lisbon for the conference in September. Flying out on the 14th.
```

**Output:**

```json
[
  {
    "text": "The user booked flights to Lisbon for a conference in September.",
    "type": "event",
    "entities": ["Lisbon"]
  },
  {
    "text": "The user is flying to Lisbon on September 14th.",
    "type": "event",
    "entities": ["Lisbon"]
  }
]
```

Two memories here, not one — the booking and the specific date are both useful independently. A future search for "travel plans" finds the first; a search for "what is the user doing on the 14th" finds the second. Both facts come from the same sentence, but they serve different retrieval needs — splitting them respects the atomicity rule because each is independently useful on its own.

---

### Example 3: No extractable memories (the empty-array case)

**Input:**

```
User: Thanks!
Assistant: You're welcome! Let me know if you need anything else.
User: Will do.
```

**Output:**

```json
[]
```

This is correct and expected. The model must not invent memories. An empty array is a good answer when the input is pleasantries with no factual content.

---

## Python: validating the schema

Here is a utility you can use throughout the rest of the book to check that any string the model produces actually conforms to our schema. We will reuse this in Ch18 when we evaluate the fine-tuned model.

```python
from __future__ import annotations  # makes list[dict] work on Python 3.8 as well as 3.9+

import json

# The exact set of allowed memory types.
# Keep this in sync with your training data — if you add a type here,
# also add examples for it in your dataset.
VALID_TYPES = {"fact", "preference", "relationship", "decision", "event"}


def validate_memory_output(raw_output: str) -> list[dict]:
    """
    Parse and validate a model's raw string output.

    Returns the list of memory dicts if valid.
    Raises ValueError with a clear message if anything is wrong.

    We raise rather than silently return an empty list so that callers
    know the difference between "no memories" and "malformed output".
    """

    # Step 1: try to parse as JSON.
    # If the model outputs prose instead of JSON, this will fail immediately.
    try:
        data = json.loads(raw_output.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Output is not valid JSON: {e}\n\nRaw output was:\n{raw_output}")

    # Step 2: the top-level must be a list (array), not a dict or scalar.
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array at the top level, got {type(data).__name__}.\n"
            "Wrap the output in [ ... ] even if there is only one memory."
        )

    # Step 3: validate each item in the list.
    for i, item in enumerate(data):

        # Each item must be a dict (JSON object).
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a JSON object: {item!r}")

        # Required fields must be present.
        for field in ("text", "type", "entities"):
            if field not in item:
                raise ValueError(f"Item {i} is missing required field '{field}': {item!r}")

        # 'text' must be a non-empty string.
        if not isinstance(item["text"], str) or not item["text"].strip():
            raise ValueError(f"Item {i}: 'text' must be a non-empty string.")

        # 'type' must be one of the allowed values.
        if item["type"] not in VALID_TYPES:
            raise ValueError(
                f"Item {i}: 'type' is '{item['type']}', must be one of {sorted(VALID_TYPES)}."
            )

        # 'entities' must be a list of strings (can be empty).
        if not isinstance(item["entities"], list):
            raise ValueError(f"Item {i}: 'entities' must be a list, got {type(item['entities']).__name__}.")

        for j, entity in enumerate(item["entities"]):
            if not isinstance(entity, str):
                raise ValueError(f"Item {i}, entity {j}: expected a string, got {type(entity).__name__}.")

    # If we get here, the output is valid. Return the parsed list.
    return data


# ── Quick demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # A valid output — should pass cleanly.
    good_output = json.dumps([
        {
            "text": "The user is based in Amsterdam.",
            "type": "fact",
            "entities": ["Amsterdam"]
        },
        {
            "text": "The user is a backend engineer.",
            "type": "fact",
            "entities": []
        }
    ])

    memories = validate_memory_output(good_output)
    print(f"Valid! Extracted {len(memories)} memories.")

    # An empty array — also valid.
    empty_output = "[]"
    memories = validate_memory_output(empty_output)
    print(f"Empty output is valid: {memories}")

    # A broken output — should raise a clear error.
    bad_output = '{"text": "The user likes coffee", "type": "preference"}'  # dict not list
    try:
        validate_memory_output(bad_output)
    except ValueError as e:
        print(f"\nCaught expected error:\n{e}")
```

Running this script as-is requires only the Python standard library — no pip installs needed. You will import `validate_memory_output` again in Ch14 (data cleaning), Ch15 (training loop sanity checks), and Ch18 (evaluation).

---

## What "good" looks like: a checklist

Before you accept any training example or model output, run through this:

- [ ] Is the top-level value a JSON array?
- [ ] Is each memory a standalone sentence (no pronouns like "it" or "there" that require context)?
- [ ] Does each memory contain exactly one fact?
- [ ] Is every fact directly supported by the input text?
- [ ] Is `type` one of the five allowed values?
- [ ] Is `entities` a list (even if empty)?
- [ ] Are there duplicates that should be merged?
- [ ] If the input had no facts, is the output `[]`?

You will turn this checklist into automated assertions in Ch14.

---

## Common mistakes

**Mistake: bundling facts into one text field.**

```json
{ "text": "The user is a vegetarian backend engineer based in Amsterdam.", "type": "fact", "entities": ["Amsterdam"] }
```

This fails the atomicity rule. If someone later searches for "vegetarian preferences," this memory may rank lower because the text is diluted with unrelated facts. Split it into three entries.

**Mistake: relative or context-dependent text.**

```json
{ "text": "The user likes it there.", "type": "preference", "entities": [] }
```

What is "there"? This memory is useless when retrieved without the original conversation. Always write the `text` field as if the reader has never seen the source text.

**Mistake: extracting assistant statements as user facts.**

If the assistant says "Tokyo has great vegetarian options," do not create a memory "Tokyo has great vegetarian options" attributed to the user. The assistant's knowledge is not a user memory. Only extract facts *about* or *stated by* the user.

**Mistake: forcing a memory from vague input.**

```
User: "Yeah, totally."
```

There is no memory here. Return `[]`. Training examples that force a memory from thin air teach the model to hallucinate, which is exactly the failure mode you are trying to avoid.

**Mistake: using a type not in the enum.**

```json
{ "text": "The user scheduled a dentist appointment.", "type": "appointment", "entities": [] }
```

`appointment` is not in our schema. Use `event`. Introducing undocumented types causes schema validation to fail at evaluation time and creates inconsistency in your training data.

---

## Recap

- The task is: given conversational text, produce a JSON array of atomic, standalone memory objects.
- Each memory has three fields: `text` (a full sentence), `type` (one of five values), and `entities` (a list of named things).
- Memories must be atomic (one fact each), standalone (no context required to read them), and grounded (no inference beyond what the text says).
- An empty array `[]` is a correct output when the input contains no extractable facts.
- The `validate_memory_output` function gives you a reusable schema checker you will reach for in every later chapter.
- The five memory types are: `fact`, `preference`, `relationship`, `decision`, `event`.
- This spec is the contract. Every training row, every evaluation metric, and every debugging step derives from it.

## Next

**Ch12 - Data Format: Turning the Task into Training Rows** — takes this spec and shows how to package input/output pairs into the exact JSONL format that Unsloth and TRL expect for fine-tuning.
