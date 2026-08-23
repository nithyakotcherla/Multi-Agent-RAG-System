"""Gradio 6.x frontend for RAG Regulators — backed by the real FAISS + OpenAI pipeline."""

import os
import sys
import json
import subprocess
import time
import socket
import atexit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import gradio as gr
from scripts.query_rag import load_index_and_chunks, build_bm25, retrieve, generate_answer, get_embeddings
from scripts.pdf_highlighter import create_highlighted_pdfs, render_highlighted_pages_html
import numpy as np
import faiss
import re

# Since this file is named app.py, running it makes Python treat "app" as this very file.
# To avoid the ModuleNotFoundError, we import directly from 'agents' since we are already in the app folder.
from agents import agent_1_internal_researcher, agent_2_external_fact_checker, agent_3_synthesizer

try:
    from fastmcp import Client
except ImportError:
    Client = None

MCP_PORT = 8001
# fastmcp serves at /mcp (no trailing slash needed by the client)
MCP_URL = f"http://localhost:{MCP_PORT}/mcp"
_mcp_proc = None


def _is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if something is already listening on the given port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _get_fastmcp_bin() -> str:
    """Return the fastmcp binary path from the active venv, falling back to PATH."""
    # Look beside the current Python interpreter (inside the venv)
    candidate = os.path.join(os.path.dirname(sys.executable), "fastmcp")
    if os.path.isfile(candidate):
        return candidate
    return "fastmcp"  # fall back to whatever is on PATH


def start_mcp_server():
    """Launch mcp_server.py via `fastmcp run` in a background subprocess."""
    global _mcp_proc
    if _is_port_open(MCP_PORT):
        print(f"✅ MCP server already running on port {MCP_PORT}.")
        return
    mcp_script = os.path.join(PROJECT_ROOT, "app", "mcp_server.py")
    fastmcp_bin = _get_fastmcp_bin()
    cmd = [
        fastmcp_bin, "run",
        mcp_script + ":mcp",
        "--transport", "http",
        "--host", "127.0.0.1",
        "--port", str(MCP_PORT),
    ]
    print(f"🚀 Starting MCP server — binary: {fastmcp_bin}")
    _mcp_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Wait up to 40 seconds — heavier imports (PyMuPDF) need more boot time
    for i in range(80):
        time.sleep(0.5)
        # Check if the process crashed before the port opened
        if _mcp_proc.poll() is not None:
            out = _mcp_proc.stdout.read() if _mcp_proc.stdout else ""
            print(f"❌ MCP server process exited early (code {_mcp_proc.returncode}):\n{out}")
            return
        if _is_port_open(MCP_PORT):
            print(f"✅ MCP server is up on port {MCP_PORT} (after {(i+1)*0.5:.1f}s).")
            return
    print("⚠️  MCP server did not start in time — agentic mode may fail.")


def _stop_mcp_server():
    if _mcp_proc and _mcp_proc.poll() is None:
        _mcp_proc.terminate()
        print("MCP server stopped.")


atexit.register(_stop_mcp_server)
start_mcp_server()

APP_TITLE = "⚖️ RAG Regulators"
EXAMPLE_QUESTIONS = [
    "What are the fines under GDPR?",
    "What makes an AI system high-risk under the EU AI Act?",
    "What are the four functions of the NIST AI RMF?",
    "How do the fines compare between the EU AI Act and GDPR?",
]

# Load index and chunks once at startup (no API call needed)
print("Loading FAISS index and chunks...")
INDEX, CHUNKS = load_index_and_chunks()
BM25 = build_bm25(CHUNKS)
print(f"✅ Loaded {len(CHUNKS)} chunks — hybrid retrieval + LLM re-ranking ready.")

LIVE_FACTS_PATH = os.path.join(PROJECT_ROOT, "data", "live_facts.json")
_indexed_fact_count = 0  # track how many live facts are already embedded


def reindex_live_facts() -> str:
    """Read any new rows from live_facts.json and merge them into the live FAISS + BM25 index."""
    global INDEX, CHUNKS, BM25, _indexed_fact_count

    if not os.path.exists(LIVE_FACTS_PATH):
        return "⚠️ No live_facts.json found yet. Add facts first via 'Update Vector DB queue & Save'."

    # Read all lines and only process the ones we haven't indexed yet
    with open(LIVE_FACTS_PATH, "r", encoding="utf-8") as f:
        all_lines = [l.strip() for l in f if l.strip()]

    new_lines = all_lines[_indexed_fact_count:]
    if not new_lines:
        return "✅ Nothing new to index — all live facts are already in the search index."

    new_chunks = []
    for line in new_lines:
        try:
            fact = json.loads(line)
            new_chunks.append({
                "text": fact.get("text", ""),
                "source": fact.get("source", "Live Web Search"),
                "page": "live",
                "score": 0.0
            })
        except json.JSONDecodeError:
            continue

    if not new_chunks:
        return "⚠️ Found new lines but couldn't parse any valid facts."

    # Embed the new chunks
    texts = [c["text"] for c in new_chunks]
    try:
        embeddings = get_embeddings(texts)
    except Exception as e:
        return f"❌ Embedding failed: {e}"

    vectors = np.array(embeddings).astype("float32")
    faiss.normalize_L2(vectors)

    # Add to the live in-memory FAISS index
    INDEX.add(vectors)

    # Append to the live chunks list so retrieval can look them up
    CHUNKS.extend(new_chunks)

    # Rebuild BM25 to include the new chunks
    BM25 = build_bm25(CHUNKS)

    _indexed_fact_count += len(new_chunks)
    return f"✅ Re-indexed {len(new_chunks)} new fact(s) — total corpus now has {len(CHUNKS)} chunks. Your next query will use the updated knowledge!"


def format_sources(results: list) -> str:
    """Return a markdown-formatted source block."""
    if not results:
        return ""
    lines = ["\n\n---\n📚 **Sources**"]
    for r in results:
        lines.append(
            f"- **{r['source']}** — page {r['page']} "
            f"*(relevance: {r['score']:.2f})*"
        )
    return "\n".join(lines)


def _open_pdfs_in_viewer(highlighted: list) -> None:
    """Open each highlighted PDF in Preview (macOS).
    The PDF contains only the retrieved pages, so Preview opens at page 1
    which is already the highlighted content — no page navigation needed.
    """
    if sys.platform != "darwin":
        return
    for pdf_path, _src, _first_page in highlighted:
        try:
            r = subprocess.run(["open", "-a", "Preview", pdf_path], capture_output=True)
            if r.returncode != 0:
                print(f"[pdf] open failed (code {r.returncode}): {r.stderr.decode()}")
            else:
                print(f"[pdf] Opened highlighted extract for '{_src}' in Preview")
        except Exception as e:
            print(f"[pdf] Could not open file: {e}")


def _pdf_source_panel_html(retrieved: list, highlighted: list) -> str:
    """Build an HTML summary of which sources were retrieved and highlighted."""
    if not retrieved:
        return ""

    # Collect unique (source, page) pairs in order
    seen: set[tuple] = set()
    rows: list[str] = []
    for chunk in retrieved:
        src = chunk.get("source", "?")
        pg = chunk.get("page", "?")
        key = (src, pg)
        if key not in seen:
            seen.add(key)
            score = chunk.get("score", 0.0)
            rows.append(
                f"<tr><td style='padding:3px 10px;'>{src}</td>"
                f"<td style='padding:3px 10px; text-align:center;'>p.{pg}</td>"
                f"<td style='padding:3px 10px; text-align:center;'>{score:.2f}</td></tr>"
            )

    highlight_note = ""
    if highlighted:
        opened = ", ".join(s for _, s, _ in highlighted)
        highlight_note = (
            f"<p style='margin:8px 0 0; color:#1e7e34;'>"
            f"✅ Highlighted PDF opened: <strong>{opened}</strong></p>"
        )

    return f"""
<div style='border:1px solid #d0d7de; border-radius:6px; padding:12px;
            background:#f6f8fa; font-size:0.85rem;'>
  <strong>📚 Retrieved passages</strong>
  <table style='border-collapse:collapse; margin-top:6px; width:100%;'>
    <thead>
      <tr style='background:#e9ecef;'>
        <th style='padding:4px 10px; text-align:left;'>Source</th>
        <th style='padding:4px 10px;'>Page</th>
        <th style='padding:4px 10px;'>Score</th>
      </tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  {highlight_note}
</div>"""


def _status_bar_html(stage: str) -> str:
    """
    Returns an HTML progress indicator for embedding → searching → generating.
    stage: 'embedding' | 'searching' | 'generating' | '' (hidden)
    """
    if not stage:
        return ""

    steps = [("embedding", "Embedding"), ("searching", "Searching"), ("generating", "Generating")]
    order = [s[0] for s in steps]
    active_idx = order.index(stage) if stage in order else -1

    parts: list[str] = []
    for i, (key, label) in enumerate(steps):
        if i < active_idx:
            # Completed — green checkmark
            parts.append(
                f'<span style="display:inline-flex;align-items:center;gap:5px;'
                f'color:#16a34a;font-weight:600;">'
                f'<svg width="14" height="14" viewBox="0 0 14 14" fill="none" style="flex-shrink:0">'
                f'<circle cx="7" cy="7" r="6" fill="#16a34a"/>'
                f'<path d="M4 7.2l2 2 4-4" stroke="white" stroke-width="1.5" '
                f'stroke-linecap="round" stroke-linejoin="round"/></svg>{label}</span>'
            )
        elif i == active_idx:
            # Active — spinning ring
            parts.append(
                f'<span style="display:inline-flex;align-items:center;gap:6px;'
                f'color:#1e3a5f;font-weight:700;">'
                f'<span style="width:14px;height:14px;border:2px solid #c7d7f0;'
                f'border-top-color:#1e3a5f;border-radius:50%;'
                f'animation:rag-spin 0.75s linear infinite;'
                f'display:inline-block;flex-shrink:0;"></span>{label}</span>'
            )
        else:
            # Pending — empty ring
            parts.append(
                f'<span style="display:inline-flex;align-items:center;gap:5px;color:#94a3b8;">'
                f'<span style="width:14px;height:14px;border:2px solid #e2e8f0;border-radius:50%;'
                f'display:inline-block;flex-shrink:0;"></span>{label}</span>'
            )
        if i < len(steps) - 1:
            parts.append('<span style="color:#cbd5e1;margin:0 6px;font-size:0.7rem;">▶</span>')

    inner = "".join(parts)
    return (
        '<style>@keyframes rag-spin{to{transform:rotate(360deg)}}</style>'
        '<div style="display:flex;align-items:center;gap:2px;padding:9px 14px;'
        'background:linear-gradient(90deg,#f0f4ff 0%,#f8f9fb 100%);'
        'border:1px solid #c7d7f0;border-radius:8px;font-size:0.84rem;'
        "font-family:'IBM Plex Sans',ui-sans-serif,sans-serif;"
        'box-shadow:0 1px 3px rgba(30,58,95,0.07);margin-bottom:2px;">'
        + inner
        + '</div>'
    )


def _normalize_history(history: list) -> list:
    """Convert Gradio 6 ChatMessage objects to plain dicts for safe manipulation."""
    normalized = []
    for msg in history:
        if isinstance(msg, dict):
            normalized.append({"role": msg["role"], "content": msg["content"]})
        else:
            # Gradio 6 ChatMessage objects use attribute access
            normalized.append({"role": msg.role, "content": msg.content})
    return normalized


async def rag_chat(message: str, history: list, use_agentic: bool, auto_open_pdf: bool, k: int = 8):
    """Run the real RAG pipeline. history is Gradio 6 messages format."""
    _no_pdf = ("", gr.update(visible=False))
    _empty_pages = ""   # placeholder until highlighted pages are ready

    if not message.strip():
        yield history, "", "", "", gr.update(), *_no_pdf, "", _empty_pages
        return

    # Normalize history — Gradio 6 may pass ChatMessage objects (not dicts)
    history = _normalize_history(history)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "⏳ Processing your question…"})

    print(f"[rag_chat] query={message!r}  k={int(k)}")

    # ── Stage 1: Embedding ────────────────────────────────────────────────────
    yield history, "", "", "", gr.update(visible=False), *_no_pdf, _status_bar_html("embedding"), _empty_pages
    try:
        query_vec = np.array(get_embeddings([message])).astype("float32")
    except Exception as e:
        history[-1]["content"] = f"❌ Embedding error: {e}"
        yield history, "", "", "", gr.update(visible=False), *_no_pdf, "", _empty_pages
        return

    # ── Stage 2: Searching ────────────────────────────────────────────────────
    history[-1]["content"] = "🔍 Searching internal corpus…"
    yield history, "", "", "", gr.update(visible=False), *_no_pdf, _status_bar_html("searching"), _empty_pages
    try:
        retrieved, highlight_candidates = retrieve(
            message, INDEX, CHUNKS, bm25=BM25, k=int(k),
            rerank=True, query_vec=query_vec, return_candidates=True
        )
    except Exception as e:
        history[-1]["content"] = f"❌ Retrieval error: {e}"
        yield history, "", "", "", gr.update(visible=False), *_no_pdf, "", _empty_pages
        return

    # --- PDF highlighting (after search, before generate) ---
    # Use the broader pre-rerank candidate pool so more unique pages are included;
    # the reranked `retrieved` is used only for the answer.
    highlighted = create_highlighted_pdfs(highlight_candidates)
    print(f"[pdf] highlighted={[(s, pg) for _, s, pg in highlighted]}, auto_open={auto_open_pdf}")
    if auto_open_pdf and highlighted:
        _open_pdfs_in_viewer(highlighted)
    panel_html = _pdf_source_panel_html(retrieved, highlighted if auto_open_pdf else [])
    _pdf_outputs = (panel_html, gr.update(visible=bool(panel_html)))
    pages_html = render_highlighted_pages_html(highlighted)

    # ── Stage 3: Generating ───────────────────────────────────────────────────
    history[-1]["content"] = "✏️ Generating answer…"
    yield history, "", "", "", gr.update(visible=False), *_pdf_outputs, _status_bar_html("generating"), pages_html
    try:
        internal_ans = generate_answer(message, retrieved)
    except Exception as e:
        history[-1]["content"] = f"❌ Generation error: {e}"
        yield history, "", "", "", gr.update(visible=False), *_pdf_outputs, "", pages_html
        return

    if not use_agentic or Client is None:
        reply = internal_ans + format_sources(retrieved)
        history[-1]["content"] = reply
        yield history, "", "", "", gr.update(visible=False), *_pdf_outputs, "", pages_html
        return

    # ── Agentic mode ─────────────────────────────────────────────────────────
    history[-1]["content"] = internal_ans + "\n\n---\n*🌐 Connecting to web agent for fact-checking...*"
    yield history, "", "", "", gr.update(visible=False), *_pdf_outputs, "", pages_html

    try:
        mcp_client = Client(MCP_URL)
        async with mcp_client:
            history[-1]["content"] = internal_ans + "\n\n---\n*🌐 Web Agent is researching...*"
            yield history, "", "", "", gr.update(visible=False), *_pdf_outputs, "", pages_html

            external_answer = await agent_2_external_fact_checker(message, internal_ans, mcp_client)

            history[-1]["content"] = internal_ans + f"\n\n---\n*🌐 Web findings summary:*\n{external_answer}\n\n*🤖 Synthesizing final response...*"
            yield history, "", "", "", gr.update(visible=False), *_pdf_outputs, "", pages_html

            final_response = await agent_3_synthesizer(message, internal_ans, external_answer)
            reply = final_response + format_sources(retrieved)
            history[-1]["content"] = reply
            yield history, "", final_response, external_answer, gr.update(visible=True), *_pdf_outputs, "", pages_html
    except Exception as e:
        reply = internal_ans + format_sources(retrieved) + f"\n\n*(Error connecting to Web Agent: {e}. Is the fastmcp server running?)*"
        history[-1]["content"] = reply
        yield history, "", "", "", gr.update(visible=False), *_pdf_outputs, "", pages_html

async def on_approve(filename: str, final_response: str):
    if not Client: return "FastMCP Client missing"
    try:
        mcp_client = Client(MCP_URL)
        async with mcp_client:
            res = await mcp_client.call_tool("create_markdown_report", {"filename": filename, "content": final_response})
        return res
    except Exception as e:
        return f"Error: {e}"

async def on_update_db(filename: str, final_response: str, external_answer: str):
    if not Client: return "FastMCP Client missing"
    try:
        mcp_client = Client(MCP_URL)
        async with mcp_client:
            res1 = await mcp_client.call_tool("add_to_database", {"text": external_answer, "source": "Live Web Search MCP"})
            res2 = await mcp_client.call_tool("create_markdown_report", {"filename": filename, "content": final_response})
        return f"{res1} | {res2}"
    except Exception as e:
        return f"Error: {e}"

def on_reject():
    return "Draft rejected. Output not saved.", gr.update(visible=False)


async def on_upload_pdf(file) -> str:
    """Called when user uploads a PDF. Sends it to the MCP load_pdf_to_database tool."""
    if file is None:
        return "⚠️ No file selected."
    if not Client:
        return "❌ FastMCP Client not available."
    pdf_path = file.name if hasattr(file, "name") else str(file)
    try:
        mcp_client = Client(MCP_URL)
        async with mcp_client:
            result = await mcp_client.call_tool("load_pdf_to_database", {"pdf_path": pdf_path})
        return str(result)
    except Exception as e:
        return f"❌ Error loading PDF via MCP: {e}"


theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#f8f9fb",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e2e6ea",
    block_shadow="0 1px 4px rgba(0,0,0,0.06)",
    button_primary_background_fill="#1e3a5f",
    button_primary_background_fill_hover="#274d80",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#eef1f6",
    button_secondary_text_color="#1e3a5f",
)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE, theme=theme) as demo:
        # ── Full-width header ─────────────────────────────────────────────────
        gr.HTML("""
        <div style="
            display:flex; align-items:center; gap:14px;
            padding:20px 0 8px; border-bottom:2px solid #e2e6ea; margin-bottom:8px;
        ">
            <div style="font-size:2.4rem; line-height:1;">⚖️</div>
            <div>
                <h1 style="margin:0; font-size:1.6rem; font-weight:700;
                           color:#1e3a5f; letter-spacing:-0.5px;">
                    RAG Regulators
                </h1>
                <p style="margin:2px 0 0; font-size:0.85rem; color:#64748b;">
                    Grounded Q&amp;A on GDPR · EU AI Act · NIST AI RMF
                </p>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False):
            # ── Left column: chat interface ───────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="",
                    show_label=False,
                    height=480,
                    avatar_images=(
                        None,
                        "https://em-content.zobj.net/source/twitter/376/balance-scale_2696-fe0f.png",
                    ),
                )

                status_bar = gr.HTML(value="", elem_id="rag-status-bar")

                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask a question about GDPR, the EU AI Act, or the NIST AI RMF…",
                        show_label=False,
                        scale=8,
                        container=False,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

                with gr.Row():
                    use_agents_checkbox = gr.Checkbox(label="Enable Agentic Workflow (Web Fact-Checking)", value=False)
                    auto_open_pdf_checkbox = gr.Checkbox(label="Also open PDF in Preview (macOS)", value=False)
                    k_slider = gr.Slider(
                        minimum=1, maximum=12, step=1, value=8,
                        label="k — chunks retrieved",
                        info="How many passages to retrieve and pass to the LLM",
                    )

                current_final_response = gr.State(value="")
                current_external_answer = gr.State(value="")

                with gr.Group(visible=False) as hitl_panel:
                    gr.Markdown("### 🛠️ Human-in-the-Loop Actions")
                    with gr.Row():
                        filename_input = gr.Textbox(label="Save Filename", value="data/final_report.md", scale=2)
                        hitl_status = gr.Textbox(label="Action Status", interactive=False, scale=3)
                    with gr.Row():
                        approve_btn = gr.Button("Approve & Save Markdown", variant="primary")
                        update_db_btn = gr.Button("Update Vector DB queue & Save", variant="secondary")
                        reject_btn = gr.Button("Reject Draft", variant="stop")

                with gr.Group(visible=False) as pdf_sources_panel:
                    gr.Markdown("### 🔍 Source Documents")
                    pdf_sources_html = gr.HTML(value="")

                with gr.Accordion("📄 Load a New PDF into the Database (via MCP)", open=False):
                    gr.Markdown(
                        "Upload a PDF file. The MCP server will extract its text, chunk it, "
                        "and queue it in `data/live_facts.json`. "
                        "Then click **🔄 Re-index** below to instantly make it searchable."
                    )
                    with gr.Row():
                        pdf_upload = gr.File(
                            label="Upload PDF",
                            file_types=[".pdf"],
                            scale=3,
                        )
                        pdf_status = gr.Textbox(label="PDF Load Status", interactive=False, scale=4)
                    pdf_upload_btn = gr.Button("📥 Extract & Queue PDF into Database", variant="primary")

                with gr.Row():
                    reindex_btn = gr.Button("🔄 Re-index Live Facts into RAG", variant="secondary", size="sm")
                    reindex_status = gr.Textbox(label="Re-index Status", interactive=False, scale=4)

                with gr.Accordion("💡 Example questions", open=False):
                    gr.Examples(
                        examples=EXAMPLE_QUESTIONS,
                        inputs=msg_box,
                        label="",
                    )

                gr.HTML("""
                <p style="text-align:center; font-size:0.78rem; color:#94a3b8; margin-top:12px;">
                    Powered by FAISS + OpenAI gpt-4o-mini · RAG Regulators Team
                </p>
                """)

            # ── Right column: inline PDF viewer ──────────────────────────────
            with gr.Column(scale=2, min_width=320):
                gr.HTML("""
                <div style="font-size:0.9rem;font-weight:700;color:#1e3a5f;
                            padding:6px 0 10px;border-bottom:1px solid #e2e6ea;
                            margin-bottom:8px;">
                    📑 Source Pages
                </div>
                """)
                pdf_pages_viewer = gr.HTML(
                    value=(
                        "<div style='"
                        "height:calc(100vh - 180px);min-height:300px;"
                        "overflow-y:auto;overflow-x:hidden;"
                        "padding:60px 20px;box-sizing:border-box;"
                        "color:#94a3b8;font-size:0.85rem;text-align:center;'>"
                        "Highlighted source pages will appear here after your query."
                        "</div>"
                    ),
                )

        # ── Wire up events ────────────────────────────────────────────────────
        _rag_inputs  = [msg_box, chatbot, use_agents_checkbox, auto_open_pdf_checkbox, k_slider]
        _rag_outputs = [chatbot, msg_box, current_final_response, current_external_answer,
                        hitl_panel, pdf_sources_html, pdf_sources_panel, status_bar,
                        pdf_pages_viewer]

        send_btn.click(rag_chat, _rag_inputs, _rag_outputs)
        msg_box.submit(rag_chat, _rag_inputs, _rag_outputs)

        approve_btn.click(on_approve, [filename_input, current_final_response], [hitl_status])
        update_db_btn.click(on_update_db, [filename_input, current_final_response, current_external_answer], [hitl_status])
        reject_btn.click(on_reject, [], [hitl_status, hitl_panel])
        pdf_upload_btn.click(on_upload_pdf, [pdf_upload], [pdf_status])
        reindex_btn.click(reindex_live_facts, [], [reindex_status])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1")
