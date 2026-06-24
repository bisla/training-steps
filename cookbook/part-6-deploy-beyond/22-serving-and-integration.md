# Ch22 - Serving Your Model and Using It in an App

You trained a model. You evaluated it in *Ch18 - Did It Actually Work? Evaluating Memory Extraction*. You debugged the rough edges in *Ch19 - When It Goes Wrong: A Debugging Playbook*. Now you want to actually use it.

This chapter closes the loop. We will take the fine-tuned model you saved in *Ch21 - Saving, Merging, and Exporting Your Model* and get it answering real requests. There are three core paths we walk in full — Ollama, vLLM, and in-process transformers — plus two production alternatives worth knowing about (Hugging Face TGI and a managed cloud endpoint). We will walk the three, survey the alternatives, and wire up a clean Python function — `extract_memories(text) -> list` — that you can drop into any application.

---

## What you'll learn

- The serving options and when to pick each one: Ollama (quick local test), vLLM (production-ready API), in-process transformers (batch jobs), plus two alternatives — Hugging Face TGI and a managed cloud endpoint
- How to load your exported GGUF file with Ollama and call it with one `curl` command
- How to spin up a vLLM server that speaks the OpenAI API format
- How to run the model directly in Python without a server at all
- When TGI is a better production server than vLLM, and when paying for a managed endpoint beats running your own
- How to write a `extract_memories()` client function that works against any OpenAI-compatible backend
- How throughput, ops effort, and rough cost trade off across all the options

---

## Concepts you need first

### Inference vs. training

Training is the expensive, one-time process of adjusting a model's weights. Inference is just *using* the model: you hand it some text, it produces a response. Inference is much cheaper — you are not computing gradients, not storing optimizer state, not running backward passes. A model that needed a 24 GB GPU to train can often run inference on 8 GB, especially after quantization.

### A model server

A model server loads the model weights into GPU memory once and then waits for requests. Each request sends text in, gets a response back, and the weights stay loaded the whole time. This is important: loading weights takes 10–30 seconds. If you loaded them fresh for every request, your app would be unusably slow. A server solves this by keeping the model resident.

The server exposes an HTTP API. Your Python app sends a POST request, the server runs inference, and returns the result. This is the same pattern you use when calling OpenAI's API — the difference is the server is running on your machine, on your GPU.

### The OpenAI API format

OpenAI published an API spec that has become the de-facto standard for LLM servers. It looks like this: you POST a JSON body with a `model` field and a `messages` list to `/v1/chat/completions`, and you get back a JSON response with a `choices` list containing the model's reply. Most open-source serving tools (vLLM, Ollama, LM Studio) implement this exact format, which means the same Python code works against all of them.

### GGUF — the portable weight format

GGUF is a file format for storing model weights in a compact, quantized form. Think of it as the MP3 of model weights: you trade a tiny bit of quality for a big reduction in file size and memory usage. A 7B-parameter model that takes ~14 GB as raw float16 weights fits into ~4–5 GB as a 4-bit quantized GGUF. You exported this file in *Ch21 - Saving, Merging, and Exporting Your Model*. Ollama and llama.cpp use GGUF directly.

---

## The system prompt: the one constant

Before we look at any serving option, anchor this: **the system prompt you used during training must be used verbatim at inference time.** We established this rule in *Ch12 - Data Format: Turning the Task into Training Rows* and it deserves repeating here because it is the most common cause of a working model appearing broken in production.

Store it in a shared module and import it everywhere. Here is the canonical version from this book:

```python
# memory_prompt.py
# Import this constant in both your training code AND your serving code.
# Never rewrite it inline — even one changed word can degrade output quality.

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
```

Save this as `memory_prompt.py` in your project root. Every code block in this chapter imports from it.

---

## Option 1: Ollama — quick local testing in five minutes

**When to use this:** You want to run the model on your laptop right now, try a few prompts interactively, and confirm it works before going further. Ollama is not designed for high-throughput production use, but it is the fastest path from "I have a GGUF file" to "it is answering requests."

### Install Ollama

Download from [https://ollama.com](https://ollama.com) — there is a macOS installer, a Linux one-liner, and a Windows preview. After installing, the `ollama` command is available in your terminal.

### Create a Modelfile

Ollama needs a small text file called a `Modelfile` that tells it where your weights live and what system prompt to use.

```
# Modelfile  (save this in your project root, next to your .gguf file)
#
# IMPORTANT: Find your actual GGUF filename first — Unsloth may have used a
# different name depending on your model and quantization settings in Ch21.
# Run this in your terminal before editing the FROM line:
#
#   ls ./models/
#
# Look for a file ending in .gguf (e.g. unsloth.Q4_K_M.gguf or similar).
# Copy that exact filename into the FROM line below.

FROM ./models/memory-extractor-q4.gguf

# Set the system prompt that matches your training data exactly.
# Ollama bakes this into the model's context on every call.
SYSTEM """You are a memory extraction assistant. Your job is to read a conversation and extract every memorable piece of information as a list of atomic memory objects.

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

# Tell Ollama how many tokens the model can handle in one context window.
# 4096 is a safe default for Qwen3/Gemma3 7B variants.
PARAMETER num_ctx 4096
```

### Load the model

```bash
# Build the Ollama model from your Modelfile.
# "memory-extractor" is just a local name — you can call it whatever you want.
ollama create memory-extractor -f Modelfile

# Verify it appears in the list of local models.
ollama list
```

### Test it with curl

> **Two endpoints, one server.** Ollama actually exposes two different HTTP APIs on port 11434:
>
> - `/api/chat` — Ollama's own native format. The request shape is slightly different (it uses `"stream": false` at the top level, and the response wraps the reply in `"message": {"content": "..."}`).
> - `/v1/chat/completions` — the OpenAI-compatible format. The request and response shapes match the OpenAI spec exactly, which is what the Python `openai` client library in the next section sends.
>
> Always use `/v1/chat/completions` when you want your curl tests to mirror what your Python code does. We use that endpoint below.

```bash
# One curl call — this is the full round-trip test.
# We use the OpenAI-compatible endpoint (/v1/chat/completions) so this request
# matches exactly what the Python client code sends in the next section.
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "memory-extractor",
    "messages": [
      {
        "role": "user",
        "content": "Alex: I finally switched to using Obsidian for all my notes. Notion was getting too slow.\nJamie: Nice! Do you use it for work or personal?\nAlex: Both. Also reading through the Dune series again."
      }
    ],
    "temperature": 0.1
  }'
```

You should get back a JSON response with a `choices[0].message.content` field containing a JSON array of memories. If you see prose or an error, check that the Modelfile path is correct and that `ollama list` shows the model.

**Typical performance on a modern laptop (M2 Mac or a machine with a mid-range NVIDIA GPU):** 10–25 tokens per second for a 7B Q4 model. A single short conversation (100–200 tokens input) will return in roughly 3–8 seconds. That is fine for manual testing, not fine for a production app handling hundreds of requests.

---

## Option 2: vLLM — production-ready API with real throughput

**When to use this:** You need your app to handle multiple requests, you have a dedicated GPU machine (cloud or local), and you want a drop-in replacement for the OpenAI API. vLLM is the right tool here.

vLLM uses a technique called **PagedAttention** to serve many requests in parallel far more efficiently than naive sequential inference. For a 7B model on a single A100 (80 GB), you can typically handle 20–50 concurrent requests. On a single A10G (24 GB), expect 5–15 concurrent requests. These are rough ballparks — actual throughput depends on input/output lengths.

### Install vLLM

> **Before you install — platform requirements.** vLLM runs on **Linux only** with a **CUDA-capable NVIDIA GPU**. It does not support Windows or macOS (macOS has no CUDA; Windows support is experimental and not recommended for production). You need CUDA 11.8 or 12.1+ installed on your system, and at least 16 GB of VRAM for a 7B model in bfloat16. If you are on a Mac or Windows machine, use the Ollama path above for local testing, and run vLLM on a Linux cloud instance (e.g. a Colab Pro+ runtime, a RunPod A10G, or any cloud VM with an NVIDIA GPU) when you need production throughput. Tested with vLLM >= 0.4.

```bash
# vLLM requires Python 3.9+, Linux, and CUDA 11.8 or 12.1+.
# Use the merged HuggingFace model from Ch21 here,
# not the GGUF — vLLM works with HuggingFace format weights directly.
pip install vllm
```

### Start the server

```bash
# Replace the path with your merged HuggingFace model directory from Ch21.
# --served-model-name is the name your client code will use in the "model" field.
# --max-model-len caps the context window; set it to what your training used.
# --dtype bfloat16 is the right precision for Qwen3 and Gemma3 models.

# Recommended syntax for vLLM 0.4+ — use "vllm serve":
vllm serve ./models/memory-extractor-merged \
    --served-model-name memory-extractor \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --port 8000

# If "vllm serve" is not found on your PATH (older vLLM installs), use the
# module-path form as a fallback:
# python -m vllm.entrypoints.openai.api_server \
#     --model ./models/memory-extractor-merged \
#     --served-model-name memory-extractor \
#     --max-model-len 4096 \
#     --dtype bfloat16 \
#     --port 8000
```

The server prints a line like `INFO:     Uvicorn running on http://0.0.0.0:8000` when it is ready. It takes 20–60 seconds to load the weights. After that, it stays loaded and handles requests until you kill it.

### Test it with curl

Because vLLM implements the OpenAI API format, the curl call looks almost identical to what you would send to OpenAI — just with a different base URL:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "memory-extractor",
    "messages": [
      {
        "role": "system",
        "content": "You are a memory extraction assistant. Your job is to read a conversation and extract every memorable piece of information as a list of atomic memory objects.\n\nEach memory object must follow this exact JSON schema:\n{\n  \"text\": \"<the fact, written as a complete, standalone sentence>\",\n  \"type\": \"<one of: preference | fact | decision | relationship>\",\n  \"entities\": [\"<list of named people, places, or things involved>\"]\n}\n\nRules:\n- One fact per memory object. Do not bundle multiple facts into one.\n- Write \"text\" as a sentence someone could read without any surrounding context.\n- If there are no memorable facts in the conversation, return an empty list: []\n- Return ONLY a valid JSON array. No explanation, no markdown fences, no extra text.\n"
      },
      {
        "role": "user",
        "content": "Alex: I finally switched to using Obsidian for all my notes. Notion was getting too slow."
      }
    ],
    "temperature": 0.1,
    "max_tokens": 1024
  }'
```

Notice we pass the system prompt explicitly here because vLLM does not have a Modelfile concept. The system prompt goes in the `messages` list as a `"role": "system"` entry — exactly as it appeared in your training data.

---

## Option 3: In-process transformers — batch jobs without a server

**When to use this:** You have a batch of 500 conversations to process overnight, you do not want the overhead of an HTTP server, and you want to run the model directly inside your Python script. This is the simplest setup with the lowest moving parts.

The downside: your Python process holds the GPU while it is running. You cannot easily share the GPU with other processes. Use this for one-off batch processing, data pipelines, or evaluation scripts — not for serving live app traffic.

```python
# batch_extract.py
# Runs the fine-tuned model directly in Python — no server required.
# Requires: pip install torch transformers accelerate

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from memory_prompt import SYSTEM_PROMPT  # the shared constant from earlier


def load_model(model_path: str):
    """
    Load the merged HuggingFace model and tokenizer from disk.

    model_path: path to the merged model directory saved in Ch21.
    Returns (model, tokenizer) ready for inference.

    This takes 20-60 seconds and uses ~14 GB VRAM for a 7B bfloat16 model.
    Call it once at startup — do not reload per request.
    """
    print(f"Loading model from {model_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,   # bfloat16 matches training precision
        device_map="auto",             # automatically places layers on available GPUs
    )
    model.eval()  # switch off dropout; not needed for inference
    print("Model loaded.")
    return model, tokenizer


def run_inference(model, tokenizer, conversation_text: str) -> str:
    """
    Run one inference call. Returns the raw string the model produced.

    We keep this function separate from JSON parsing so you can inspect
    the raw output if something goes wrong.
    """

    # Build the messages list exactly as it looked in training.
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": conversation_text},
    ]

    # apply_chat_template converts the messages list into the token sequence
    # the model expects. add_generation_prompt=True appends the assistant-turn
    # opening tokens, telling the model "now it is your turn to respond."
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():   # no_grad disables gradient tracking — saves memory during inference
        output_ids = model.generate(
            input_ids,
            max_new_tokens=1024,    # cap the response length
            do_sample=False,        # greedy decoding: always pick the single most-likely
                                    # next token. No randomness at all — the safest choice
                                    # for structured JSON output where you want the same
                                    # answer every time. (do_sample=True would enable
                                    # random sampling, which requires a temperature setting
                                    # and can occasionally produce subtly broken JSON.)
            pad_token_id=tokenizer.eos_token_id,  # suppress a harmless warning
        )

    # Decode only the newly generated tokens (not the prompt we fed in).
    # output_ids[0] is the full sequence; input_ids.shape[-1] is the prompt length.
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return raw_output.strip()
```

---

## Two more options worth knowing about

The three paths above cover most situations. But two alternatives come up often enough that you should know they exist and when to reach for them. The good news: both speak the OpenAI API format, so the `extract_memories()` client you write later works against them unchanged — you only change the base URL.

### Option 4: Hugging Face TGI — the other production server

**When to use this:** You want production-grade serving like vLLM, but you are already living in the Hugging Face ecosystem (you push models to the Hub, you deploy on Hugging Face Inference Endpoints, or your ops team already runs TGI containers). TGI — Text Generation Inference — is Hugging Face's battle-tested serving stack. It ships as a Docker image, supports continuous batching and tensor parallelism much like vLLM, and exposes an OpenAI-compatible `/v1/chat/completions` endpoint.

Think of vLLM and TGI as two competing espresso machines: both pull a great shot, the throughput is in the same ballpark, and which one you buy mostly comes down to which kitchen it fits into. vLLM tends to lead on raw throughput and is the more common open-source default; TGI tends to win on operational polish and Hub integration. For a single 7B model you will not notice a dramatic difference — pick the one your infrastructure already favors.

```bash
# Run TGI as a Docker container. It downloads the model and starts an
# OpenAI-compatible server on port 8080.
# Mount a local directory so TGI can read the merged HuggingFace model from Ch21.
# (You can also pass a Hub repo id instead of a local path.)

docker run --gpus all --shm-size 1g -p 8080:80 \
  -v $PWD/models:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id /data/memory-extractor-merged \
  --max-total-tokens 4096
```

Once it is up, the client code is identical to vLLM — just point `base_url` at `http://localhost:8080/v1`. TGI does not have a Modelfile concept either, so you pass the pinned system prompt in the `messages` list exactly as you do for vLLM. Tested against TGI 2.x.

### Option 5: A managed cloud endpoint — least ops, pay per use

**When to use this:** You do not want to manage a GPU at all. No CUDA drivers, no Docker, no `vllm serve` process to keep alive, no machine to patch. You upload (or point at) your model, and a provider runs it for you behind an HTTPS URL. This is the right call when your traffic is spiky, when you are a solo developer who would rather pay than operate, or when you want to ship before standing up real infrastructure.

The landscape moves fast, but the categories are stable. **Serverless GPU hosts** (Modal, Replicate, RunPod Serverless, Baseten) spin a GPU up on demand, run your model, and spin it down — you pay per second, often only while a request is in flight. **Managed inference endpoints** (Hugging Face Inference Endpoints, or the major clouds' model-serving products) keep a dedicated instance warm behind a stable URL. Most expose an OpenAI-compatible API or a thin wrapper you can adapt in a few lines.

The tradeoff is intuition-first simple: you trade money and a little control for almost zero operational burden. Two cost patterns to keep straight:

- **Per-second serverless** bills only while a request runs. Cheap for spiky or low-volume traffic, but the first request after idle pays a **cold start** — the GPU and weights must load, which can add 10–60 seconds of latency. Fine for background jobs; rough for an interactive app unless you keep one instance warm.
- **Always-on managed endpoint** bills by the hour whether or not traffic arrives — typically in the rough range of **$0.50–$2 per GPU-hour** for the kind of mid-range GPU (a 16–24 GB card) that comfortably serves a 7B model. Predictable and warm, but you pay for idle time.

If you are running thousands of requests an hour, all day, a managed endpoint or your own vLLM box is cheaper per call. If you serve a few hundred requests in bursts, serverless usually wins. Either way, your application code does not change — it is still an OpenAI client pointed at a URL.

---

## The client function: `extract_memories()`

Now we tie everything together. The goal is one clean function that any part of your application can call, regardless of which backend is running underneath.

We will write two versions: one that hits an HTTP server (works for both Ollama and vLLM), and one that calls the in-process model directly. Both return the same type: a Python list of memory dicts.

### HTTP version (Ollama or vLLM)

```python
# client_http.py
# Works against both Ollama (port 11434) and vLLM (port 8000).
# Requires: pip install openai
#
# Why openai? The openai Python library works against any OpenAI-compatible
# server, not just OpenAI's own API. It handles retries, streaming, and
# response parsing so we do not have to write raw requests.

import json
from openai import OpenAI
from memory_prompt import SYSTEM_PROMPT


def make_client(base_url: str, api_key: str = "not-needed") -> OpenAI:
    """
    Create an OpenAI client pointed at a local server.

    base_url examples:
        "http://localhost:8000/v1"   — vLLM
        "http://localhost:11434/v1"  — Ollama (it also speaks the OpenAI format)

    api_key is required by the library but ignored by local servers —
    any non-empty string works.
    """
    return OpenAI(base_url=base_url, api_key=api_key)


def extract_memories(
    text: str,
    client: OpenAI,
    model_name: str = "memory-extractor",
    temperature: float = 0.1,
) -> list[dict]:
    """
    Extract atomic memories from a conversation or text snippet.

    Args:
        text:        The raw conversation text. Speaker labels optional.
        client:      An OpenAI client from make_client().
        model_name:  Must match the --served-model-name flag you passed to vLLM
                     or the name you gave Ollama in the Modelfile.
        temperature: Keep this low (0.0–0.2) for JSON output — you want
                     deterministic structure, not creative variation.

    Returns:
        A list of memory dicts, each with "text", "type", and "entities".
        Returns an empty list [] if the model finds nothing to extract.

    Raises:
        ValueError if the model returns something that cannot be parsed as JSON.
    """

    # Call the server. This is a blocking HTTP request.
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ],
        temperature=temperature,
        max_tokens=1024,
    )

    # Pull out the text the model generated.
    raw = response.choices[0].message.content.strip()

    # Parse it as JSON. If the model followed its training, this should always work.
    # If it does not, the ValueError bubbles up with the raw output attached so
    # you can see exactly what went wrong.
    try:
        memories = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model returned non-JSON output.\n"
            f"JSON error: {e}\n"
            f"Raw output was:\n{raw}"
        )

    # Basic type check before we return — better to fail loudly here than
    # silently pass a dict or string to downstream code that expects a list.
    if not isinstance(memories, list):
        raise ValueError(
            f"Expected a JSON array, got {type(memories).__name__}.\nRaw: {raw}"
        )

    return memories


# ── Example usage ────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Point at vLLM. To use Ollama instead, change the port to 11434.
    client = make_client("http://localhost:8000/v1")

    # The same conversation we have been using throughout this book.
    conversation = """
User: I just moved to Tokyo last month. Still getting used to the time zone.
Assistant: That's a big move! How are you finding it?
User: I love it so far. I'm vegetarian, so finding good food took some effort at first.
Assistant: Tokyo actually has great vegetarian options once you know where to look.
User: Yeah, my colleague Aiko showed me a few spots. She's been really helpful.
""".strip()

    memories = extract_memories(conversation, client)

    print(f"Extracted {len(memories)} memories:\n")
    for m in memories:
        print(f"  [{m['type']}] {m['text']}")
        if m["entities"]:
            print(f"           entities: {m['entities']}")
```

Running this against the Tokyo conversation from *Ch11 - Defining the Task: What "Memory Extraction" Means* should produce output like:

```
Extracted 4 memories:

  [fact] The user moved to Tokyo last month.
           entities: ['Tokyo']
  [preference] The user is vegetarian.
           entities: []
  [fact] The user's colleague is named Aiko.
           entities: ['Aiko']
  [fact] Aiko helped the user find vegetarian-friendly restaurants in Tokyo.
           entities: ['Aiko', 'Tokyo']
```

### In-process version (batch jobs)

```python
# client_inprocess.py
# Use this when you want to process many conversations in one Python script
# without running a server. The model stays loaded in memory for the entire run.

import json
from memory_prompt import SYSTEM_PROMPT

# Import the loader and inference functions from the batch script above.
from batch_extract import load_model, run_inference


def extract_memories_batch(
    texts: list[str],
    model_path: str,
) -> list[list[dict]]:
    """
    Extract memories from a list of conversation texts.

    Loads the model once, then processes every text in sequence.
    Returns a list-of-lists: one inner list per input text.

    Args:
        texts:       List of conversation strings to process.
        model_path:  Path to the merged HuggingFace model directory.

    Returns:
        List of memory lists. Index i of the output corresponds to texts[i].
        If a text produces a parse error, that slot contains an empty list and
        a warning is printed — we do not abort the whole batch for one bad output.
    """

    # Load once — this is the expensive step (~30 seconds, ~14 GB VRAM).
    model, tokenizer = load_model(model_path)

    results = []

    for i, text in enumerate(texts):
        raw = run_inference(model, tokenizer, text)

        # Try to parse. On failure, log and insert an empty list so the
        # rest of the batch keeps going.
        try:
            memories = json.loads(raw)
            if not isinstance(memories, list):
                raise ValueError(f"Expected list, got {type(memories).__name__}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WARN] Text {i}: parse failed — {e}")
            print(f"       Raw output: {raw[:200]}")
            memories = []

        results.append(memories)

        # Progress indicator for long batches — prints every 10 items.
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(texts)}")

    return results


# ── Example usage ────────────────────────────────────────────────────────────

if __name__ == "__main__":

    conversations = [
        "User: I'm training for a half marathon in October. About 20 miles a week right now.",
        "User: Thanks! Assistant: You're welcome! User: Great.",   # should produce []
        "User: Switched to vim last week after years on VSCode. Not going back.",
    ]

    all_memories = extract_memories_batch(
        texts=conversations,
        model_path="./models/memory-extractor-merged",
    )

    for i, memories in enumerate(all_memories):
        print(f"\nConversation {i}: {len(memories)} memories")
        for m in memories:
            print(f"  [{m['type']}] {m['text']}")
```

**Approximate throughput for the in-process batch approach on a single A10G (24 GB):** roughly 2–5 short conversations per second. For 500 conversations, expect 2–5 minutes. For 10,000 conversations, budget about an hour.

---

## Latency, cost, and batching

Here is a practical comparison of the three options. All numbers are rough ballparks for a 7B model:

| | Ollama | vLLM | In-process |
|---|---|---|---|
| **Setup time** | 5 min | 15 min | 5 min |
| **Single request latency** | 3–8 s (laptop GPU) | 0.5–2 s (server GPU) | 2–6 s |
| **Concurrent requests** | 1 | 10–50 | 1 |
| **Good for** | Local testing | Live app traffic | Batch processing |
| **GPU requirement** | 8 GB VRAM (Q4) | 16–24 GB VRAM | 14–24 GB VRAM |
| **Cloud cost (A10G)** | n/a | ~$0.0003 per call | ~$0.0001 per call |

And here is the wider picture, including the two production alternatives, scored on the four things that actually drive the decision — when to reach for it, throughput, how much operations work it costs you, and rough money. Treat every number as an order-of-magnitude range, not a quote:

| Option | When to use | Throughput | Ops effort | Rough cost |
|---|---|---|---|---|
| **Ollama** | Dev / single user / local test | Low (1 request at a time) | Almost none | Free on your laptop |
| **vLLM** | Self-hosted production; the workhorse | High (10–50 concurrent) | Medium — you run the GPU box | ~$0.5–2/GPU-hr if rented |
| **TGI** | Production inside the HF ecosystem | High (comparable to vLLM) | Medium — Docker + GPU box | ~$0.5–2/GPU-hr if rented |
| **In-process** | One-off batch jobs, eval scripts | Low–medium (sequential) | Low — no server | ~$0.5–2/GPU-hr while running |
| **Managed / serverless** | Least ops; spiky or low volume | Provider-scaled | Lowest — no machine to run | Per-second, or ~$0.5–2/GPU-hr always-on |

A few practical notes:

**Temperature matters for JSON.** Always use `temperature=0.1` or lower for this task. Higher temperatures make the model creative, which means it produces creative JSON — invalid JSON. `temperature=0` is theoretically the most deterministic but some inference backends implement it differently. `0.1` is a safe, consistent choice.

**`max_tokens` should be generous but bounded.** A 7B model extracting memories from a short conversation rarely needs more than 512 tokens of output. Set `max_tokens=1024` as a ceiling to avoid runaway generation while leaving headroom for longer inputs.

**Batching at the HTTP layer.** If you are processing many conversations through vLLM, send them as concurrent requests rather than sequential ones. vLLM's scheduler groups concurrent requests into efficient GPU batches automatically. You can use Python's `asyncio` + the `openai` library's async client, or simply use a thread pool:

```python
# parallel_client.py
# Process multiple conversations concurrently using a thread pool.
# vLLM batches these requests internally for much better GPU utilization.

import concurrent.futures
from client_http import make_client, extract_memories


def extract_memories_parallel(
    texts: list[str],
    base_url: str = "http://localhost:8000/v1",
    max_workers: int = 10,
) -> list[list[dict]]:
    """
    Send multiple extraction requests to vLLM concurrently.

    max_workers controls how many parallel HTTP requests are in-flight at once.
    For vLLM, 10–20 workers is a good starting point for a single GPU.
    Too many workers causes request queuing at the server with no benefit.

    Returns a list aligned with the input texts list.
    """
    client = make_client(base_url)

    # We use a dict to preserve ordering — ThreadPoolExecutor may return
    # futures in any order depending on completion time.
    results = [None] * len(texts)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        # Submit all requests at once.
        future_to_index = {
            pool.submit(extract_memories, text, client): i
            for i, text in enumerate(texts)
        }

        # Collect results as they complete.
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[WARN] Text {idx} failed: {e}")
                results[idx] = []

    return results
```

With 10 workers against a vLLM server on an A10G, you can typically process 500 short conversations in under 2 minutes — compared to 10+ minutes if you sent them one at a time.

---

## Closing the loop: the running example end-to-end

Here is the complete pipeline, from raw conversation to stored memories, using everything this book has built:

```python
# end_to_end.py
# The full memory pipeline: raw text → extract → validate → store.
# This is what a real integration looks like.

import json
from client_http import make_client, extract_memories

# ---------------------------------------------------------------------------
# Inline schema validator — originally introduced in Ch11.
# We reproduce the essential logic here so this file runs without Ch11 open.
# In a real project, move this to a shared utils module (e.g. memory_schema.py)
# and import it with: from memory_schema import validate_memory_output
#
# validate_memory_output(raw_json: str) -> list[dict]
#   Takes the raw JSON string the model produced, parses it, and checks that
#   every memory object has the required fields with the right types.
#   Raises ValueError with a clear message on any violation.
#   Returns the parsed list on success.

VALID_TYPES = {"preference", "fact", "decision", "relationship"}

def validate_memory_output(raw_json: str) -> list[dict]:
    """Parse and validate a JSON string of memory objects against the book schema."""
    try:
        memories = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output is not valid JSON: {e}\nRaw: {raw_json}")

    if not isinstance(memories, list):
        raise ValueError(f"Expected a JSON array, got {type(memories).__name__}")

    for i, mem in enumerate(memories):
        if not isinstance(mem, dict):
            raise ValueError(f"Memory {i} is not a dict: {mem}")
        for field in ("text", "type", "entities"):
            if field not in mem:
                raise ValueError(f"Memory {i} is missing required field '{field}': {mem}")
        if mem["type"] not in VALID_TYPES:
            raise ValueError(
                f"Memory {i} has invalid type '{mem['type']}'. "
                f"Must be one of: {sorted(VALID_TYPES)}"
            )
        if not isinstance(mem["entities"], list):
            raise ValueError(f"Memory {i} 'entities' must be a list: {mem}")

    return memories
# ---------------------------------------------------------------------------


def process_conversation(text: str, client, user_id: str) -> list[dict]:
    """
    Full pipeline for one conversation:
      1. Extract memories via the served model.
      2. Validate the output against our schema.
      3. Attach metadata (who this memory belongs to).
      4. Return the enriched memory list for storage.

    In a real app, step 4 would write to a vector database or key-value store.
    This mirrors the architecture products like mem0 use.
    """

    # Step 1: Extract.
    raw_memories = extract_memories(text, client)

    # Step 2: Validate each memory against the schema from Ch11.
    # validate_memory_output raises ValueError on any schema violation.
    raw_json = json.dumps(raw_memories)
    validated = validate_memory_output(raw_json)

    # Step 3: Attach metadata so we know whose memories these are.
    enriched = []
    for mem in validated:
        enriched.append({
            **mem,           # text, type, entities from the model
            "user_id": user_id,
            "source": "chat",
        })

    return enriched


if __name__ == "__main__":
    client = make_client("http://localhost:8000/v1")

    # The conversation from Ch11 — the very first example in this book.
    conversation = """
User: I just moved to Tokyo last month. Still getting used to the time zone.
Assistant: That's a big move! How are you finding it?
User: I love it so far. I'm vegetarian, so finding good food took some effort at first.
Assistant: Tokyo actually has great vegetarian options once you know where to look.
User: Yeah, my colleague Aiko showed me a few spots. She's been really helpful.
""".strip()

    memories = process_conversation(conversation, client, user_id="user_42")

    print(f"Stored {len(memories)} memories for user_42:\n")
    for m in memories:
        print(json.dumps(m, indent=2))
```

This is the pattern behind products like mem0. A conversation comes in. The fine-tuned model extracts atomic facts. Each fact is validated against the schema. The facts are stored, indexed, and later retrieved to give the assistant memory across sessions. You have now built every layer of that stack.

---

## What this chapter deliberately does *not* cover

Getting the model answering requests is one thing; running it as a dependable service is another. This chapter stops at "it serves correct output." The harder production questions — **monitoring** what the model does in the wild, **versioning** which model is live, doing **canary** or shadow deploys so a bad model only sees a sliver of traffic, and being able to **roll back** in seconds when something regresses — all live in *Ch34 - Production Ops*. Reach for that chapter the moment this model carries traffic you care about.

And serving is not the end of the story. Once real conversations are flowing through the model, you are sitting on the most valuable training data you will ever get: examples from your actual users. Turning that stream into a model that keeps getting better — the continual-improvement loop — is the subject of **Part 8 - Continuous Learning**. This chapter hands you a model that works; Part 8 is how you keep it working as your world changes.

---

## Common mistakes

**Mistake: forgetting the system prompt at inference time.**

The model's fine-tuning wired it to produce structured JSON *given the specific instructions it saw in training*. Without that system prompt, the model has no reason to produce JSON at all — it will likely produce a helpful prose summary instead. Always pass the system prompt. Store it as a constant and import it; never retype it.

**Mistake: using `temperature=1.0` for JSON output.**

At high temperature, the model sometimes generates `[{"text": "...", "tyep": "fact"` — a plausible-looking but invalid JSON fragment. Keep temperature at `0.1` or below. You want boring, correct JSON, not creative prose.

**Mistake: not handling the empty-array case.**

The model will correctly return `[]` when there is nothing to extract. Your calling code must handle this gracefully — a zero-length list is not an error. Crashing when `len(memories) == 0` is a common bug.

**Mistake: calling `load_model()` on every request in the in-process approach.**

Loading model weights takes 20–60 seconds and uses a burst of GPU memory during the load. If you call `load_model()` per request, your "serving" layer will be 30 seconds slow and will likely OOM (out-of-memory) if two requests overlap. Load once at process startup, keep the model object in memory, and call `run_inference()` per request.

**Mistake: pointing at the wrong model path in vLLM.**

vLLM needs the merged HuggingFace model directory (the one you saved with `model.save_pretrained()` in *Ch21 - Saving, Merging, and Exporting Your Model*). It cannot load LoRA adapter checkpoints directly. If you pass a LoRA adapter path, vLLM will either crash or silently load the base model without your fine-tuning. Confirm the directory contains `config.json` and `model.safetensors` (or shards of it), not just `adapter_config.json`.

**Mistake: using a different `model_name` in the client than the `--served-model-name` on the server.**

vLLM will return a `404 model not found` error if the names do not match. Keep `--served-model-name memory-extractor` in your server startup command and `model_name="memory-extractor"` in your client code. Pick one name and stick with it everywhere.

---

## Recap

- There are three core serving options: Ollama (GGUF, local test, minimal setup), vLLM (HuggingFace format, OpenAI-compatible API, production throughput), and in-process transformers (batch jobs, no server needed). Two production alternatives round it out: Hugging Face TGI (a vLLM-class server that fits the HF ecosystem) and a managed/serverless endpoint (least ops, pay per use).
- The system prompt from training must be used verbatim at inference time. Store it as a shared constant and import it; never rewrite it inline.
- `extract_memories(text, client) -> list[dict]` is the one function the rest of your app needs. It abstracts away which backend is running.
- For JSON-structured output, use `temperature=0.1` or lower. High temperature produces creatively broken JSON.
- vLLM handles concurrent requests efficiently; use a thread pool to send parallel requests and let vLLM batch them on the GPU.
- An empty list `[]` is a valid, correct model output — handle it without error.
- The full pipeline is: extract → validate schema → attach metadata → store. This is the architecture that products like mem0 are built on.

## Next

*Ch23 - Continual Learning and Scaling Up* — now that the model is serving real traffic and accumulating real data, we look at how to keep improving it: collecting production feedback, building a continual fine-tuning loop, and the path toward models that genuinely internalize your users' world.
