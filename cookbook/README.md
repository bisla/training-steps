# The Domain Fine-Tuning Handbook
### Fine-Tuning Small Language Models for Your Own Domain

A practical, ground-up guide for Python developers with no prior machine learning background. By the end you will have fine-tuned a real model — Qwen or Gemma — to extract structured memory facts from natural conversation.

---

## Who this is for

You write Python. You have not trained a neural network before. You may have heard the words "LoRA", "loss curve", and "gradient" and felt nothing click. That is exactly the starting point this book is written from.

No math prerequisites. No GPU farm required. A single consumer GPU or a rented cloud instance is enough to complete every exercise.

---

## What domain fine-tuning is for

Domain fine-tuning means teaching a small, cheap model a skill that is specific to *your*
world — something a big general-purpose model does inconsistently, slowly, or expensively.
A few examples of what people build this way:

- A **company "brain"** that pulls decisions, owners, and commitments out of meeting and chat transcripts.
- A **personal recall feature** that turns someone's notes and conversations into durable, queryable memories.
- A **domain extractor** that lifts structured fields out of clinical notes, contracts, or support tickets.
- Any **reliable structured-output task** where prompting a large model just is not consistent enough at scale.

## The running example: a memory extractor

To keep things concrete, every chapter builds toward one runnable system: a **memory
extractor** that reads a conversation and writes out structured, atomic facts — the kind a
personal AI assistant needs to remember between sessions.

```
Input:  "Alex switched from Notion to Obsidian because Notion got too slow."
Output: [{"text": "Alex switched from Notion to Obsidian.",
          "type": "decision", "entities": ["Alex", "Notion", "Obsidian"]}]
```

You will generate the training data, fine-tune the model, evaluate it, debug it, and ship it
— and then go further: align it with preference and RL, and run it as a continually
improving system. The memory extractor is the example; the technique is the point.

> **In a hurry?** [Chapter 0 — The Speedrun](quickstart/00-speedrun.md) takes you end to end
> — synthetic data, fine-tune, and serve — in an afternoon for under $30. The rest of the
> book then explains *why* each step works and how to make it production-grade.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Any recent install works |
| `pip` / virtual env | `venv` or `conda` |
| ~20 GB disk | For model weights and datasets |
| GPU (optional but faster) | NVIDIA with CUDA; CPU fallback works for small runs |
| An OpenAI or Anthropic key | Used only for synthetic data generation in Ch 13 |

---

## How to read this book

If you just want a win, start with **Chapter 0 — The Speedrun**: it takes you end to end (synthetic data → fine-tune → serve) in an afternoon for under $30. Then read Parts 0 and 1 straight through — they are short and load the mental models everything else depends on. Part 2 is setup; follow it on your own machine. Parts 3–6 are the hands-on core: do them in order, running the code as you go. Part 7 (Preference & RL) and Part 8 (Continuous Learning as a System) are the advanced build-out — reach for them once a plain fine-tune is working and deployed. The appendices are reference material; jump to them when you hit an error or need a term defined.

---

## Table of Contents

### Quickstart

| | |
|---|---|
| [Chapter 0](quickstart/00-speedrun.md) | The Speedrun: A Working Fine-Tune This Afternoon for Under $30 |

### Part 0 — The Big Picture

| | |
|---|---|
| [Chapter 1](part-0-big-picture/01-why-finetune.md) | Why Teach a Model Your Own World |
| [Chapter 2](part-0-big-picture/02-mental-models.md) | Mental Models: What a Model Actually Is |
| [Chapter 3](part-0-big-picture/03-landscape-when-to-use-what.md) | Prompting vs RAG vs Fine-Tuning vs Full Training |

### Part 1 — Concepts Primer

| | |
|---|---|
| [Chapter 4](part-1-concepts-primer/04-transformers-in-20-min.md) | Transformers and LLMs in 20 Minutes |
| [Chapter 5](part-1-concepts-primer/05-tokenization-context-chat-templates.md) | Tokens, Context Windows, and Chat Templates |
| [Chapter 6](part-1-concepts-primer/06-lora-qlora-explained.md) | LoRA and QLoRA Without the Math Headache |
| [Chapter 7](part-1-concepts-primer/07-how-training-works.md) | How Training Actually Works (Loss, Gradients, Epochs) |

### Part 2 — Setup and Tools

| | |
|---|---|
| [Chapter 8](part-2-setup-tools/08-hardware-and-environment.md) | Hardware, GPUs, and Setting Up Your Environment |
| [Chapter 9](part-2-setup-tools/09-the-toolbox.md) | The Toolbox: Unsloth, Transformers, TRL, PEFT, and Friends |
| [Chapter 10](part-2-setup-tools/10-choosing-base-model.md) | Choosing Your Base Model: Qwen vs Gemma |

### Part 3 — Task and Data

| | |
|---|---|
| [Chapter 11](part-3-task-and-data/11-defining-the-task.md) | Defining the Task: What "Memory Extraction" Means |
| [Chapter 12](part-3-task-and-data/12-data-format-and-schema.md) | Data Format: Turning the Task into Training Rows |
| [Chapter 13](part-3-task-and-data/13-synthetic-data-generation.md) | Creating Your Training Data with Synthetic Generation |
| [Chapter 14](part-3-task-and-data/14-data-prep-and-splits.md) | Cleaning, Splitting, and Sanity-Checking Data |

### Part 4 — Training

| | |
|---|---|
| [Chapter 15](part-4-training/15-first-finetune-unsloth.md) | Your First Fine-Tune with Unsloth (Full Script) |
| [Chapter 16](part-4-training/16-hyperparameters.md) | Hyperparameters: Which Knobs to Turn and When |
| [Chapter 17](part-4-training/17-monitoring-training.md) | Watching Training: Loss Curves and When to Stop |

### Part 5 — Evaluation and Iteration

| | |
|---|---|
| [Chapter 18](part-5-eval-iteration/18-evaluation.md) | Did It Actually Work? Evaluating Memory Extraction |
| [Chapter 19](part-5-eval-iteration/19-debugging-bad-results.md) | When It Goes Wrong: A Debugging Playbook |
| [Chapter 20](part-5-eval-iteration/20-iterating.md) | Iterating: From a Mediocre Model to a Good One |

### Part 6 — Deploy and Beyond

| | |
|---|---|
| [Chapter 21](part-6-deploy-beyond/21-saving-merging-exporting.md) | Saving, Merging, and Exporting Your Model |
| [Chapter 22](part-6-deploy-beyond/22-serving-and-integration.md) | Serving Your Model and Using It in an App |
| [Chapter 23](part-6-deploy-beyond/23-toward-continual-learning.md) | Continual Learning and Scaling Up |

### Part 7 — Preference & RL

| | |
|---|---|
| [Chapter 24](part-7-preference-rl/24-why-preference.md) | Beyond Imitation: Why Preference and RL |
| [Chapter 25](part-7-preference-rl/25-reward-modeling.md) | Rewards: Functions and Reward Models |
| [Chapter 26](part-7-preference-rl/26-dpo.md) | DPO: Learning Directly From Preference Pairs |
| [Chapter 27](part-7-preference-rl/27-ppo-why-not.md) | PPO and the Full RL Loop: Why We Don't Use It Here |
| [Chapter 28](part-7-preference-rl/28-grpo.md) | GRPO: Practical RL With Reward Functions |
| [Chapter 29](part-7-preference-rl/29-choosing-a-method.md) | Choosing Your Method: SFT vs DPO vs KTO/ORPO vs GRPO vs PPO |

### Part 8 — Continuous Learning as a System

| | |
|---|---|
| [Chapter 30](part-8-continual-learning/30-the-loop-architecture.md) | The Continual Learning Loop as a System |
| [Chapter 31](part-8-continual-learning/31-data-selection-curation.md) | Selecting and Curating Data That Actually Helps |
| [Chapter 32](part-8-continual-learning/32-how-much-how-often.md) | How Much Data, and How Often to Retrain |
| [Chapter 33](part-8-continual-learning/33-catastrophic-forgetting.md) | Catastrophic Forgetting Over Many Rounds |
| [Chapter 34](part-8-continual-learning/34-production-ops.md) | Production Ops: Monitoring, Versioning, Gating, and Rollback |

### Appendices

| | |
|---|---|
| [Appendix A](appendices/A-glossary.md) | Glossary of Every Term Used |
| [Appendix B](appendices/B-project-layout-and-commands.md) | Project Layout and Command Cheat-Sheet |
| [Appendix C](appendices/C-troubleshooting.md) | Troubleshooting Common Errors |
| [Appendix D](appendices/D-cost-time-and-checklist.md) | Cost, Time, and a Go-Live Checklist |

---

## Building the book

See [`build/build.sh`](build/build.sh) for instructions to render this book as HTML (via mdBook) or as PDF/EPUB (via Pandoc). The [`build/book.toml`](build/book.toml) contains the mdBook configuration.

---

## License

MIT. Use the code freely; attribution appreciated.
