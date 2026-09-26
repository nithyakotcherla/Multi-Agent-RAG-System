# ⚖️ RAG Regulators — Kaggle AI Assistants Hackathon
---

## Project Overview

**RAG Regulators** is a production-quality **Multi-Agent RAG** (Retrieval-Augmented Generation) system that answers questions about three major AI/data-governance frameworks:

- 🇪🇺 **GDPR** — General Data Protection Regulation
- 🤖 **EU AI Act** — European Union Artificial Intelligence Act
- 🛡️ **NIST AI RMF** — NIST AI Risk Management Framework (48-page document)

The system evolves from a standard RAG pipeline into a fully **agentic architecture**, featuring live web fact-checking, dynamic PDF ingestion, PDF source highlighting, human-in-the-loop review, and an optional Signal Messenger bot interface.

---

## Architecture

```
User Query
    │
    ▼
┌────────────────────────────────────────────────────────┐
│                 Gradio Web UI  (app/app.py)             │
│                                                        │
│  ┌─── Stage 1: Embed ──────────────────────────────┐  │
│  │  text-embedding-3-small  (OpenAI)               │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌─── Stage 2: Hybrid Retrieval ───────────────────┐  │
│  │  FAISS (semantic) + BM25Okapi (keyword)         │  │
│  │       └── Reciprocal Rank Fusion (RRF)          │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌─── Stage 3: LLM Re-rank ────────────────────────┐  │
│  │  gpt-4o-mini listwise re-ranking (2×k → k)      │  │
│  └─────────────────────────────────────────────────┘  │
│  ┌─── Stage 4: Dynamic Prompt + Generate ──────────┐  │
│  │  Question-type detection → tailored system      │  │
│  │  prompt → gpt-4o-mini answer generation         │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
│  [Optional] Agentic Mode (Enable Agentic checkbox)     │
│  ┌─────────────────────────────────────────────────┐  │
│  │  Agent 1: Internal Researcher (RAG corpus)      │  │
│  │  Agent 2: External Fact-Checker (DuckDuckGo)   │  │  ← via MCP server
│  │  Agent 3: Synthesizer (merge + HITL review)    │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
│  PDF Highlighting → macOS Preview auto-open            │
└────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Hybrid Search + LLM Re-ranking (`scripts/query_rag.py`)

| Stage | Method | Detail |
|---|---|---|
| Semantic search | FAISS | `text-embedding-3-small`, cosine similarity |
| Keyword search | BM25Okapi | `rank-bm25`, tokenised word matching |
| Fusion | Reciprocal Rank Fusion | Merges both ranked lists |
| Re-ranking | GPT listwise | `gpt-4o-mini` orders 2k candidates → top k |
| Generation | Dynamic prompt | 5-type detector: `yes_no`, `factual`, `comparison`, `listing`, `open_ended` |

### 2. Multi-Agent Pipeline (`app/agents.py`)

Three async agents orchestrated inside the Gradio UI:

- **Agent 1 — Internal Researcher**: Deep FAISS + BM25 + re-rank search over the indexed corpus.
- **Agent 2 — External Fact-Checker**: Asks `gpt-4o-mini` to generate an optimal web query, then calls the MCP `search_web` tool (DuckDuckGo) to fetch live results and fact-check the internal answer.
- **Agent 3 — Synthesizer**: Merges internal corpus findings with live web findings into a single grounded response, clearly attributing each source.

### 3. MCP Server (`app/mcp_server.py`)

A **FastMCP** HTTP server (port 8001) exposing four tools:

| Tool | Description |
|---|---|
| `search_web` | Live DuckDuckGo search via `duckduckgo_search` |
| `load_pdf_to_database` | Extract, chunk (500 tok / 100 tok overlap) and queue a new PDF into `data/live_facts.json` |
| `add_to_database` | Directly write an arbitrary fact to the live-facts queue |
| `create_markdown_report` | Append the synthesized answer to a `.md` report file |

The MCP server is **auto-started** by `app.py` as a background subprocess — no manual terminal needed.

### 4. PDF Source Highlighter (`scripts/pdf_highlighter.py`)

After every query, retrieved passages are highlighted directly in the source PDFs:
- Extracts **only the relevant pages** into a small temporary PDF (macOS Preview opens at page 1 automatically — no page navigation needed).
- Uses **word-level `SequenceMatcher`** to handle cleaned vs. raw PDF text mismatches.
- Falls back to a yellow margin bar if no word match is found.
- On macOS, the highlighted PDF opens in **Preview** instantly.

### 5. Human-in-the-Loop (HITL)

When Agentic Mode is enabled, a review panel appears after each synthesized answer:
- **Approve & Save Markdown** — calls MCP `create_markdown_report` to persist the answer.
- **Update Vector DB & Save** — calls MCP `add_to_database` to queue the web findings for re-indexing, then saves the report.
- **Reject Draft** — discards the output.

The **🔄 Re-index Live Facts** button embeds and merges all queued facts into the live in-memory FAISS + BM25 index without a server restart.

### 6. Dynamic PDF Ingestion

Users can upload any PDF via the Gradio UI → MCP extracts and chunks it → queued into `data/live_facts.json` → one click re-indexes into the live corpus.

### 7. Signal Messenger Bot (`app/signal_bot.py`)

Full agentic RAG over Signal, powered by `signal-cli`:
- **Native polling** — no Docker, no REST wrapper.
- **Per-user agentic toggle**: send `/agentic on` to enable web search, `/agentic off` for corpus-only mode.
- Automatically routes messages → Agent 1 → (optional) Agents 2 & 3 → sends the reply back on Signal.

### 8. Cost Tracker (`scripts/cost_tracker.py`)

Every OpenAI API call (embeddings + chat) is tracked in `cost_tracker.json` with a running total printed to the console:
```
💰 This call: $0.000293 | Team total: $0.0719 / $5.00
```

---

## Repository Structure

```
Kaggle_week_Rag_Regulators/
├── README.md
├── requirements.txt
├── .gitignore
├── .env                         # ← OPENAI_API_KEY (git-ignored)
│
├── app/
│   ├── app.py                   # 🚀 MAIN ENTRY POINT — Gradio UI + auto MCP launch
│   ├── agents.py                # Async agent definitions (Researcher, Fact-Checker, Synthesizer)
│   ├── mcp_server.py            # FastMCP server (tools: search, PDF, DB, report)
│   ├── main.py                  # Alternative CLI entry for multi-agent pipeline
│   └── signal_bot.py            # Signal Messenger bot (signal-cli polling)
│
├── scripts/
│   ├── query_rag.py             # Core RAG: FAISS + BM25 + RRF + LLM re-rank + dynamic prompt
│   ├── extract.py               # PDF text extraction & chunking (PyMuPDF)
│   ├── index_data.py            # Build FAISS index from chunks.json
│   ├── pdf_highlighter.py       # Highlight retrieved passages in source PDFs
│   ├── cost_tracker.py          # API cost tracking utility
│   ├── eval_runner.py           # Evaluation runner (generation accuracy + retrieval hit rate)
│   ├── run_eval.py              # Simple evaluation script
│   ├── run_eval2.py             # Extended evaluation script
│   ├── retrieval_checker.py     # Per-query retrieval inspection tool
│   ├── FAILURE_LOG.md           # Documented failure cases & fixes
│   └── EXTRACTION_REPORT.md    # PDF extraction statistics
│
└── data/                        # Git-ignored — generated locally
    ├── GDPR.pdf
    ├── EU AI ACT.pdf
    ├── risk managment 48 pages.pdf
    ├── chunks.json              # Chunked text with metadata (2 758 chunks)
    ├── my_index.faiss           # FAISS vector index
    ├── live_facts.json          # Dynamic facts queue (appended via MCP)
    ├── final_report.md          # Generated HITL reports
    ├── questions.json           # Evaluation question set (v1)
    ├── questions2.json          # Evaluation question set (v2, extended)
    └── eval_results*.json       # Evaluation output files
```

---

## Getting Started

### Prerequisites

- **Python 3.12** (required for `fastmcp`)
- **[uv](https://docs.astral.sh/uv/)** package manager (recommended) or `pip`
- An **OpenAI API key**

### 1. Clone & Set Up Environment

```bash
# Create and activate virtual environment (Python 3.12)
uv venv --python 3.12
source .venv/bin/activate    # macOS / Linux

# Install all dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```ini
OPENAI_API_KEY=sk-proj-...
```

### 3. Build the Vector Index (first time only)

```bash
# Step 1 — Extract and chunk the PDFs
python scripts/extract.py

# Step 2 — Build the FAISS index
python scripts/index_data.py
```

This generates `data/chunks.json` (2 758 chunks) and `data/my_index.faiss`.

### 4. Launch the Application

```bash
python app/app.py
```

This command:
1. **Auto-starts the MCP server** on port 8001 (background subprocess).
2. **Loads** the FAISS index + BM25 index into memory.
3. **Launches** the Gradio web interface at `http://127.0.0.1:7860`.

> No second terminal or manual FastMCP command needed.

---

## Usage Guide

### Standard Mode (fast, corpus-only)

1. Open `http://127.0.0.1:7860`
2. Type a question and click **Send**
3. The pipeline embeds → retrieves (hybrid) → re-ranks → generates the answer
4. Source passages are highlighted in the orignal PDFs and opened in macOS Preview

### Agentic Mode (web fact-checking)

1. Tick **Enable Agentic Workflow (Web Fact-Checking)**
2. Ask a question — the three agents run in sequence
3. The HITL panel appears — review the synthesized answer and choose:
   - **Approve & Save Markdown** — saves to `data/final_report.md`
   - **Update Vector DB & Save** — queues web findings for re-indexing + saves report
   - **Reject Draft** — discards

### Dynamic PDF Ingestion

1. Open the **Load a New PDF** accordion
2. Upload any PDF → the MCP server chunks it into `data/live_facts.json`
3. Click **🔄 Re-index Live Facts into RAG** — the new content is instantly searchable

### Retrieval Parameters

| Control | Default | Effect |
|---|---|---|
| `k — chunks retrieved` slider | 8 | Number of passages passed to the LLM (1–12) |
| Auto-open PDF checkbox | ✅ on | Opens highlighted PDF in Preview after each query |

---

## Evaluation

```bash
# Run evaluation against the standard question set
python scripts/eval_runner.py

# Run against the extended question set (v2)
python scripts/eval_runner.py --questions data/questions2.json --output data/eval_results2.json

# Inspect retrieval quality for a single query
python scripts/retrieval_checker.py "What are the fines under GDPR?"

# Side-by-side comparison of retrieval settings
python scripts/query_rag.py "What are the GDPR fines?" --compare k=3 k=8
python scripts/query_rag.py "What are the GDPR fines?" --compare rerank=false rerank=true
```

---

## Budget

| Item | Detail |
|---|---|
| Team budget | **$5.00** |
| Embedding model | `text-embedding-3-small` — $0.02 / 1M tokens |
| Chat model | `gpt-4o-mini` — $0.15 / 1M input · $0.60 / 1M output |
| Live spend | Logged in `cost_tracker.json` (git-ignored) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Vector search | `faiss-cpu 1.13.0` |
| Keyword search | `rank-bm25 0.2.2` |
| LLM & Embeddings | `openai 2.31.0` (`gpt-4o-mini`, `text-embedding-3-small`) |
| PDF processing | `PyMuPDF 1.26.5` |
| Web UI | `gradio 6.12.0` |
| Agentic tools | `fastmcp 3.2.4` |
| Web search | `duckduckgo-search 8.1.1` |
| Env management | `python-dotenv 1.2.1` |
| Tokenisation | `tiktoken 0.12.0` |
