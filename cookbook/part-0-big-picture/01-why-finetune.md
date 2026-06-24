# Ch1 - Why Teach a Model Your Own World

You have a powerful language model. It can write code, summarize documents, answer questions, and hold a conversation. So why would you need to teach it anything?

The short answer: it knows the world, but it does not know *your* world.

---

## What you'll learn

- Why stock models fail at domain-specific tasks even when they seem smart
- The three ways to give a model your knowledge — prompting, RAG, and fine-tuning — and when each one is the right call
- What fine-tuning actually buys you that the other two approaches cannot
- Why this book picks fine-tuning for a specific task: extracting structured memories from conversations
- A map of the full journey ahead, part by part

---

## Concepts you need first

### What "training data" means

When a company builds a large language model, they feed it enormous amounts of text — web pages, books, code, forum posts. The model reads all of it and adjusts its internal settings (billions of numbers, called weights) until it gets good at predicting what comes next in a sentence. That process is called training, and the text it learned from is called training data.

The catch: that training data was collected once, from the public internet, before a cutoff date. Everything the model knows comes from that snapshot. Your company's internal Slack threads, your users' chat history, your product's quirky terminology — none of that was in the training data. The model has never seen it.

This is the core problem. A brilliant generalist, but a stranger to your context.

---

## Three ways to give a model your knowledge

Imagine you hired a brilliant new teammate — call her Iris. Iris is a fast learner, extremely well-read, and can handle almost any task. But she just walked in the door. She doesn't know your clients, your codebase, your conventions, or your jargon. You have three options for getting her up to speed.

### Option 1: Tell her everything in the meeting

Before every task, you brief Iris: here are our style rules, here is what the client cares about, here is the relevant background. She reads it, does the task, and it goes well.

This is **prompting**. You stuff all the context you need into the prompt itself (the message you send to the model), and it uses that context to answer correctly.

**When it works well:** You have a small, well-defined piece of knowledge to share. You need it to work today, without any setup.

**Where it breaks down:** You can only fit so much text into a prompt. Most models top out between 8,000 and 128,000 tokens (roughly 6,000–100,000 words). If your domain knowledge is large — thousands of past conversations, a huge document library, a long history of user preferences — it simply won't fit. And even if it does fit, every single API call sends all that context again, which costs money and adds latency.

### Option 2: Give her a searchable filing cabinet

Instead of briefing Iris up front, you give her a filing cabinet full of your documents. Before each task, she searches the cabinet, pulls the three most relevant files, skims them, and uses them to answer.

This is **RAG** — Retrieval-Augmented Generation. You store your knowledge in a database (usually as vector embeddings, which we'll explain in later chapters). At query time, the system retrieves the most relevant chunks and injects them into the prompt.

**When it works well:** Your knowledge base is large, it changes frequently, and you need the model to cite specific facts or documents.

**Where it breaks down:** RAG is a lookup, not understanding. If the task requires deeply *behaving* a certain way — always outputting a specific JSON format, always extracting exactly these fields and no others, always following a particular reasoning style — retrieval won't help. You can retrieve an example of the format, but the model still has to interpret it and follow it correctly every time. That's a skill, not a fact. Skills don't live in a filing cabinet.

### Option 3: Train her until it's second nature

You spend two weeks working side-by-side with Iris. She watches you do the task, she tries it herself, you correct her. After enough practice, she doesn't need the filing cabinet or the briefing — she just *knows* how to do it. It's internalized.

This is **fine-tuning**. You take a pre-trained model and run a second, shorter training pass on a dataset of examples you curate yourself. The model adjusts its internal weights to get good at your specific task. That knowledge is now baked into the model — no retrieval step, no long system prompt required.

**When it works well:** You have a task with a consistent input/output pattern, you want reliable, structured output (like JSON), and you want the model to be fast and token-efficient.

**Where it breaks down:** It requires upfront work to build training data, and it takes compute time to run. It also doesn't update automatically when your knowledge changes — you'd need to fine-tune again.

Here's a quick comparison:

| Approach | Setup effort | Handles large knowledge base | Bakes in behavior | Token cost per call |
|---|---|---|---|---|
| Prompting | None | No | Partially | High |
| RAG | Medium | Yes | No | Medium |
| Fine-tuning | High (upfront) | No | Yes | Low |

Most real systems combine all three. But for this book, we are going to focus on fine-tuning — because the task we care about is a *behavior* problem, not a retrieval problem.

---

## Why fine-tuning for memory extraction?

Here is the task we will build throughout this entire book:

> Given a chunk of conversation — a few chat messages, a meeting transcript, a journal entry — output a JSON list of the key memories buried in it. Each memory should be atomic (one fact per item), standalone (readable without the original context), and categorized.

A sample output looks like this:

```json
[
  {
    "text": "Alice prefers async communication over meetings.",
    "type": "preference",
    "entities": ["Alice"]
  },
  {
    "text": "The project deadline is March 15th.",
    "type": "fact",
    "entities": ["project"]
  },
  {
    "text": "Bob is Alice's manager.",
    "type": "relationship",
    "entities": ["Bob", "Alice"]
  }
]
```

This is exactly what products like [mem0](https://mem0.ai) do under the hood: they listen to conversations and extract structured facts so the AI can remember things about you over time. The Engram idea — models that learn from your context rather than just retrieving it — starts right here.

Now ask yourself: could you do this with prompting alone?

Yes, with a long, carefully-written system prompt. But a stock GPT-4 or Llama model will hallucinate fields, miss subtle memories, add fields you didn't ask for, return malformed JSON, or over-extract (grabbing every sentence as a "memory" rather than just the durable facts). Getting reliable JSON out of a general model requires babysitting the prompt constantly.

Could RAG help? Not really. RAG is for fetching knowledge the model doesn't have. The model already *knows* how to write JSON and what a "preference" means. The problem isn't missing knowledge — it's missing *discipline*. The model hasn't been trained to do this specific task in this specific way.

Fine-tuning solves this. You show the model a few hundred examples of conversations paired with perfect memory extractions, and it learns the exact pattern. After fine-tuning, it reliably outputs clean JSON, extracts the right level of detail, and uses the exact field names you specified. That consistency is the whole point.

And because the behavior is baked into the weights, you don't need a giant system prompt anymore. The fine-tuned model just knows what to do. That means fewer tokens per call — and at scale, that adds up to real cost savings.

---

## The north star: a small model that reliably emits memory JSON

By the end of this book, you will have a fine-tuned model — either Qwen3-1.7B or Gemma 3-1B (both small enough to run on a single consumer GPU or a cheap cloud instance) — that takes a raw conversation as input and returns the structured JSON above, correctly and consistently.

Small matters. A 1–2 billion parameter model fine-tuned for your task often beats a 70B general model on that task, runs 10x faster, and costs 10x less to serve. That is the practical payoff of the fine-tuning bet.

---

## A map of the journey ahead

Here is what this book covers, part by part. We will come back to this map often.

**Part 0 — Big Picture (Chapters 1–3)**
We start where you are right now: understanding *why* this approach exists and where it fits in the landscape. Chapter 2 ("Mental Models: What a Model Actually Is") builds the mental model you need for everything else. Chapter 3 ("Prompting vs RAG vs Fine-Tuning vs Full Training") gives you the decision framework in full.

**Part 1 — Concepts Primer (Chapters 4–7)**
Before you touch any training code, you need to understand four things: how transformers work at a high level, what tokens and context windows are, what LoRA is (the technique that makes fine-tuning cheap), and how training actually works mechanically. Each chapter gives you the Pareto version — the 20% of the concept that gives 80% of the understanding — no math required.

**Part 2 — Setup and Tools (Chapters 8–10)**
We get your environment ready. What GPU you need (and what you can rent cheaply), how to install the Unsloth ecosystem, and how to choose between Qwen3 and Gemma 3 as your base model.

**Part 3 — Task and Data (Chapters 11–14)**
This is where most fine-tuning projects succeed or fail. We define the memory-extraction task precisely, design the data format, generate synthetic training examples using a strong model as a teacher, and clean and split the dataset. Data quality is the highest-leverage thing you can do.

**Part 4 — Training (Chapters 15–17)**
We write and run the actual fine-tuning script using Unsloth. We cover the hyperparameters you need to understand, and we show you how to watch training as it happens — what the loss curve tells you and when to stop.

**Part 5 — Evaluation and Iteration (Chapters 18–20)**
A model that compiles is not the same as a model that works. We write an evaluation harness for memory extraction, walk through a debugging playbook for when results are bad, and show the iteration loop that takes you from mediocre to good.

**Part 6 — Deploy and Beyond (Chapters 21–23)**
We export the trained model, serve it with a local API, and integrate it into a simple Python app. The final chapter points toward the Engram vision: continual learning, where the model keeps improving as it sees more data.

**Appendices**
A full glossary of every term used, a command cheat-sheet, a troubleshooting guide for common errors, and a cost/time reference with a go-live checklist.

---

## A quick code taste: what you're building toward

You don't need to understand all of this yet. But it is useful to see the destination before the journey. By the end of the book, your usage code will look something like this:

```python
# This is what the end result looks like.
# By Ch22 you'll understand every line of this.

from unsloth import FastLanguageModel  # Load our fine-tuned model efficiently
import json

# Load the model we trained (we'll build this in Part 4)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./my-memory-extractor",  # our saved fine-tuned model
    max_seq_length=2048,                 # how many tokens it can handle at once
    load_in_4bit=True,                   # use less VRAM by compressing weights
)

# A raw conversation — this is the input
conversation = """
Alice: I've been thinking about switching to async standups.
Bob: Makes sense, I know you hate early mornings.
Alice: Ha, yeah. Also the Q2 demo is locked in for April 3rd.
Bob: Got it. Should I tell the design team?
Alice: Please do — they need at least two weeks.
"""

# Ask the model to extract memories (this is the prompt format we'll design in Ch12)
prompt = f"""Extract all memories from this conversation as a JSON array.
Each memory must have: text, type (fact/preference/relationship/decision), entities.

Conversation:
{conversation}

Memories:"""

# Tokenize and run — the model has learned to output clean JSON
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # 'pt' = PyTorch tensors — the format the model expects; 'cuda' = NVIDIA GPU — we cover GPU setup and alternatives in Part 2
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)  # temperature controls randomness — lower (closer to 0) = more deterministic, higher = more creative

# Decode the output back to text
result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Pull out just the JSON part (everything after "Memories:")
json_str = result_text.split("Memories:")[-1].strip()

# Parse it — the fine-tuned model reliably outputs valid JSON
memories = json.loads(json_str)

# Print each memory cleanly
for m in memories:
    print(f"[{m['type'].upper()}] {m['text']}")
    print(f"  Entities: {', '.join(m['entities'])}\n")
```

Running this would print something like:

```
[PREFERENCE] Alice prefers async standups over synchronous ones.
  Entities: Alice

[FACT] The Q2 demo is scheduled for April 3rd.
  Entities: Q2 demo

[FACT] The design team needs at least two weeks of lead time.
  Entities: design team

[DECISION] Bob will notify the design team about the April 3rd demo.
  Entities: Bob, design team
```

That is the goal. A small model, running locally or on a cheap GPU, that takes messy human conversation and returns clean, structured, useful data. Let's build it.

---

## Common mistakes

**Mistake: Reaching for fine-tuning too soon.**
Fine-tuning is not free. It takes time to build a dataset, time to run training, and time to evaluate results. Before you commit, try a well-engineered prompt first. If a prompt with five good examples (few-shot prompting — where you include 2–5 worked input/output examples directly in your prompt so the model learns the pattern on the fly) gets you 80% of the way there, you may not need to fine-tune at all. This book is for the cases where you need that last 20% — or where you need to run the model 10,000 times a day and can't afford the token cost of big prompts.

**Mistake: Thinking fine-tuning updates the model's knowledge.**
Fine-tuning teaches the model *how* to behave, not *what to know*. If you fine-tune on memory-extraction examples, the model will get great at extracting memories. It will not learn the facts from those conversations. Knowledge lives in the training data of the original pre-training run, not in fine-tuning. If you need the model to know new facts, use RAG.

**Mistake: Expecting a fine-tuned model to generalize to every task.**
A fine-tuned model is a specialist. The memory-extraction model we build will be excellent at memory extraction and noticeably worse at tasks it wasn't trained on (like writing poetry or debugging code). That is a fair trade for most production use cases — you want a reliable specialist, not an unreliable generalist.

**Mistake: Skipping data quality in favor of getting to training faster.**
The number-one reason fine-tuned models fail is bad training data. A model trained on inconsistent, mislabeled, or low-quality examples will produce inconsistent, bad output. We spend two full chapters on data (Chapter 13 on synthetic generation and Chapter 14 on cleaning and validation) precisely because this is where the leverage is.

---

## Recap

- Stock language models are trained once on public data — they don't know your domain, your users, or your context.
- There are three ways to give a model your knowledge: prompting (stuff it in the message), RAG (retrieve and inject at query time), and fine-tuning (bake it into the weights).
- Prompting is fast but limited by context window size and inconsistent behavior. RAG is great for large, changing knowledge bases. Fine-tuning is best for teaching consistent *behavior* and *output format*.
- Memory extraction — pulling structured JSON facts from raw conversations — is a behavior problem, not a retrieval problem. Fine-tuning is the right tool.
- A small fine-tuned model (1–2B parameters) can outperform a large general model on a narrow task, while being faster and cheaper to serve.
- This book walks you from zero ML knowledge to a deployed, fine-tuned memory-extraction model, using Qwen3 or Gemma 3 with the Unsloth ecosystem.

## Next

**Ch2 - Mental Models: What a Model Actually Is** — before we write a single line of training code, you need a working mental model of what's actually happening inside the weights.
