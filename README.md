<div align="center">

# 🧠 TalentGraph AI — Local LLM Playground

**A lightweight, zero-framework Python playground for running and experimenting with local LLMs via Ollama.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20Inference-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)
[![Requests](https://img.shields.io/badge/HTTP-Requests-2CA5E0?style=for-the-badge)](https://requests.readthedocs.io/)
[![Day 1](https://img.shields.io/badge/Milestone-Day%201%20Complete-22C55E?style=for-the-badge)](#)

</div>

---

## 🎯 Project Objective

Build a clean, understandable Python application that communicates with locally downloaded Large Language Models—**without OpenAI, Gemini, or heavy orchestration frameworks**—to understand the core engineering behind local inference.

### What This Playground Covers

| Concept | Engineering Implementation |
| :--- | :--- |
| **Local LLM Inference** | Communicates directly with Ollama's local REST server (`http://localhost:11434/api/chat`) |
| **Parameters & Quantization** | Runs compact 4-bit quantized models (`llama3.2:1b` and `qwen2.5:0.5b`) on consumer hardware |
| **Prompt Engineering** | Separates the `system` persona/instructions from `user` and `assistant` turns |
| **Sampling Control** | Configures `temperature` via Ollama's `options` payload to test deterministic vs. creative outputs |
| **Stateless APIs & Memory** | Maintains a client-side `history` list with a sliding window (`--max-turns`) |
| **Persistence** | Saves conversations and hyperparameters to timestamped JSON files (`saved_chats/`) |
| **Model Benchmarking** | Compares two models side by side on the exact same prompt, measuring throughput (`tok/s`) and latency |

---

## 🏗️ Architecture & Visual Flow

### System Overview

```mermaid
flowchart LR
    User(["👤 User (Terminal)"])
    CLI["💬 chat_cli.py<br/>REPL + History + /commands"]
    CMP["⚖️ compare_models.py<br/>Side-by-Side Benchmark"]
    Client["🔌 ollama_client.py<br/>OllamaClient (requests)"]
    Server["🦙 Ollama REST Server<br/>localhost:11434/api/chat"]
    M1[("llama3.2:1b<br/>~1.3 GB (Q4)")]
    M2[("qwen2.5:0.5b<br/>~397 MB (Q4)")]

    User <--> CLI
    User <--> CMP
    CLI --> Client
    CMP --> Client
    Client <-- "JSON over HTTP POST" --> Server
    Server <--> M1
    Server <--> M2
```

### How Multi-Turn Conversation Memory Works

Because LLM REST APIs are **stateless**, the Python client tracks the conversation and re-sends the context window on every turn:

```mermaid
sequenceDiagram
    actor User
    participant CLI as chat_cli.py
    participant Client as OllamaClient
    participant Ollama as Ollama (/api/chat)

    User->>CLI: "My name is Alex."
    CLI->>Client: [system, user("My name is Alex.")]
    Client->>Ollama: POST /api/chat (temperature)
    Ollama-->>Client: assistant("Nice to meet you, Alex.")
    Client-->>CLI: reply + prompt_eval_count
    CLI-->>User: Display reply & update history

    User->>CLI: "What is my name?"
    CLI->>Client: [system, user, assistant, user("What is my name?")]
    Client->>Ollama: POST /api/chat (full history)
    Ollama-->>Client: assistant("Your name is Alex.")
    Client-->>CLI: reply + updated token count
    CLI-->>User: Display reply
```

---

## 📁 Project Structure

```text
Local_LLM_playground/
├── ollama_client.py     # Lightweight REST client wrapping Ollama's /api/chat endpoint
├── chat_cli.py          # Interactive CLI chat loop with system prompts, temperature, history & JSON save
├── compare_models.py    # Side-by-side comparison & throughput benchmark across two local models
├── requirements.txt     # Minimal dependency list (requests)
├── .gitignore           # Excludes __pycache__, virtualenvs, and local saved_chats/
└── README.md            # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
1. Install **[Ollama](https://ollama.com/download)** and ensure the background daemon is running (`http://localhost:11434`).
2. Pull the small instruct models used in this playground:
   ```bash
   ollama pull llama3.2:1b
   ollama pull qwen2.5:0.5b
   ```
3. Install Python dependencies (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### 1. Test the Python Client (`ollama_client.py`)
Runs a single test prompt and prints response throughput metrics:
```bash
python ollama_client.py
```

### 2. Interactive Chat (`chat_cli.py`)
Start the CLI chat loop with optional flags:
```bash
python chat_cli.py --system "You are a concise Python tutor." --temperature 0.2 --max-turns 10
```

**CLI Flags:**
- `--system`: Initial system prompt (if omitted, prompts interactively at startup).
- `-t`, `--temperature`: Sampling temperature `>= 0.0` (default: `0.7`).
- `--max-turns`: Sliding window of recent conversation turns to keep (`0` = unlimited, default: `10`).

**In-Chat Commands:**
| Command | Action |
| :--- | :--- |
| `/system <prompt>` | Update the active system prompt mid-session (or view current if empty) |
| `/temp <value>` | Change sampling temperature live (e.g., `/temp 0.0` vs `/temp 1.5`) |
| `/history` | Print all messages currently stored in conversation history |
| `/clear` | Clear conversation history while keeping system prompt and temperature |
| `/save [name]` | Save conversation + metadata as JSON in `saved_chats/` |
| `exit` or `quit` | Exit the chat cleanly |

### 3. Compare Two Models (`compare_models.py`)
Send the exact same prompt and hyperparameters to two models and compare output + speed:
```bash
python compare_models.py --prompt "Explain quantization in LLMs in two short sentences." --temperature 0.0
```
*(You can also pass custom models with `--models <model_a> <model_b>`.)*

**Example Output:**
```text
=== Model Comparison ===
System prompt: You are a helpful, concise AI assistant.
Temperature:   0.0
User prompt:   Explain quantization in LLMs in two short sentences.

--- Model: llama3.2:1b ---
...
[Metrics] tokens=82 | speed=87.4 tok/s | total_wall_time=6.85s

--- Model: qwen2.5:0.5b ---
...
[Metrics] tokens=50 | speed=194.2 tok/s | total_wall_time=3.44s
```

---

## 🛠️ Tech Stack

- **Runtime:** [Ollama](https://ollama.com/)
- **Language:** Python (standard library: `argparse`, `json`, `pathlib`, `datetime`, `time`, `typing`)
- **HTTP Client:** `requests`
