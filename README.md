<<<<<<< HEAD
### AI Research Assistant
A CLI + API tool that takes a topic or a set of documents and produces structured research reports. It demonstrates the skills needed for AI/ML platforms engineer.

**Prerequisites:**

Make sure you have Python 3.10+ installed:
```
python3 --version
```

**Step 1:**
Navigate to the project (ai-research-assistant).

**Step 2: Create and activate the virtual environment.**
```
python3 -m venv .venv
source .venv/bin/activate
```

**Step 3: Install dependencies**
```
pip install -e ".[dev]"
```

**Step 4: Set up your API key.**
```
cp .env.example .env
```

> NOTE: ANTHROPIC_API_KEY is needed, contact the owner of this repo for this key.

**Step 5: Run the tests (optional but good to verify).**
```
pytest tests/ -v
```

**Step 6: Start the assistant.**
```
research
```
=======
# AI Research Assistant

A production-quality research assistant built on the [Anthropic Claude API](https://docs.anthropic.com/).
Demonstrates streaming chat, tool use, RAG, multi-step agents, prompt caching, and a FastAPI REST interface — all in Python.

---

## Features

| Feature | Description |
|---------|-------------|
| **Streaming chat** | Multi-turn conversations with real-time token streaming |
| **Tool use** | Calculator, web search (DuckDuckGo), and date/time — called autonomously |
| **RAG pipeline** | Ingest `.txt`, `.md`, `.pdf` files → chunk → embed → query via ChromaDB |
| **Prompt caching** | System prompt and document context cached server-side to cut costs |
| **Memory** | Auto-summarises long conversations to stay within context limits |
| **Session persistence** | Save and reload conversations as JSON |
| **Multi-step agent** | Plan → Execute → Reflect loop that produces a structured markdown report |
| **REST API** | FastAPI with Server-Sent Events streaming for all major features |
| **Cost tracking** | Per-session USD cost breakdown including cache savings |

---

## Project Structure

```
ai-research-assistant/
├── src/assistant/
│   ├── config.py          # Env-based configuration
│   ├── client.py          # Anthropic SDK wrapper (streaming, retry, usage)
│   ├── cost.py            # USD cost calculator per model
│   ├── tools/             # Tool use / function calling
│   │   ├── calculator.py  # Safe AST-based math evaluator
│   │   ├── datetime_tool.py
│   │   ├── search.py      # DuckDuckGo web search
│   │   ├── document_search.py  # RAG-backed document search
│   │   └── registry.py    # Tool registry + Anthropic schema builder
│   ├── rag/               # Retrieval-Augmented Generation
│   │   ├── loader.py      # File loader (.txt, .md, .pdf)
│   │   ├── chunker.py     # Token-based overlapping chunker (tiktoken)
│   │   ├── store.py       # ChromaDB vector store
│   │   └── pipeline.py    # End-to-end ingest + search interface
│   ├── memory/            # Conversation memory
│   │   ├── summarizer.py  # Auto-compress old turns when history grows long
│   │   └── persistence.py # Save/load sessions to JSON
│   ├── agent/             # Multi-step research agent
│   │   ├── planner.py     # Decompose goal → step list (JSON)
│   │   ├── executor.py    # Run one step with tools
│   │   ├── reflector.py   # Evaluate findings, identify gaps
│   │   └── loop.py        # Orchestrate full loop, emit typed events
│   ├── api/               # FastAPI REST interface
│   │   ├── app.py         # App factory, CORS, lifespan
│   │   ├── deps.py        # Shared dependencies (Depends)
│   │   ├── models.py      # Pydantic request/response models
│   │   ├── sse.py         # Server-Sent Event helpers
│   │   ├── server.py      # Uvicorn entry point
│   │   └── routes/
│   │       ├── chat.py        # POST /chat
│   │       ├── research.py    # POST /research
│   │       └── documents.py   # POST/GET/DELETE /documents
│   └── cli.py             # Rich interactive REPL
├── tests/                 # 87 unit tests (pytest)
├── data/
│   └── sample_docs/       # Sample documents for RAG demo
├── demo.py                # End-to-end demo script
└── pyproject.toml
```

---

## Setup

**Requirements:** Python 3.10+, an [Anthropic API key](https://console.anthropic.com)

```bash
# 1. Clone and navigate
cd ai-research-assistant

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-...

# 5. Run tests
pytest tests/ -v

# 6. Start the CLI
research

# — or start the API —
serve
```

---

## CLI Usage

```
research
```

```
You> What is retrieval-augmented generation?
You> /ingest data/sample_docs
You> /research How do LLM agents work?
You> /usage
You> /save my-session
You> /help
```

### Commands

| Command | Description |
|---------|-------------|
| `/research <goal>` | Run the full agent loop on a research goal |
| `/ingest <path>` | Load a file or directory into the document store |
| `/sources` | List all ingested documents |
| `/cleardb` | Wipe the document store |
| `/memory` | Show token count and memory stats |
| `/save [name]` | Save session to disk |
| `/load [path]` | Load a saved session |
| `/usage` | Show token usage and estimated cost |
| `/tools` | List available tools |
| `/clear` | Clear conversation history |
| `/quit` | Exit |

---

## REST API

```bash
serve
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### Endpoints

#### `POST /chat` — Streaming chat
```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is RAG?"}]}'
```

SSE events: `chunk` · `done` · `error`

#### `POST /research` — Research agent
```bash
curl -N -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"goal": "How do transformer attention mechanisms work?"}'
```

SSE events: `plan` · `step_started` · `step_done` · `reflection` · `extra_steps` · `report` · `done`

#### `POST /documents/ingest` — Upload a document
```bash
curl -X POST http://localhost:8000/documents/ingest \
  -F "file=@my_document.pdf"
```

#### `GET /documents/sources` — List ingested files
#### `DELETE /documents` — Clear document store
#### `GET /health` — Health check

---

## How It Works

### RAG Pipeline

```
File  →  Loader  →  Chunker (tiktoken)  →  Embeddings  →  ChromaDB
                                                               ↓
Query  →  Embed query  →  Top-K similarity search  →  Context  →  LLM
```

### Agent Loop

```
Goal
 └─► Planner  ─► [Step 1, Step 2, ..., Step N]
                        │
                 Executor (per step)
                        │  ← tool calls (search, calculator, etc.)
                        ▼
                 Reflector — sufficient?
                        │  No → add extra steps (max 2 rounds)
                        │  Yes ↓
                 Report synthesis  →  Markdown report
```

### Prompt Caching

Every API call sends the system prompt as a `cache_control: ephemeral` block.
After the first call, Anthropic serves the cached prompt at 90% lower cost.
Document context is cached as a second block, updated when files are ingested.

---

## Running the Demo

```bash
python demo.py
```

Walks through all features interactively — suitable for a live interview demo.

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `anthropic` | Claude API — chat, streaming, tool use, caching |
| `chromadb` | Local vector database for RAG |
| `tiktoken` | Token counting for chunking |
| `duckduckgo-search` | Web search (no API key required) |
| `fastapi` + `uvicorn` | REST API with SSE streaming |
| `rich` | CLI formatting |
| `pytest` | 87 unit tests |
>>>>>>> 1963b90 (Add eval metrics to project.)
