from fastmcp import FastMCP
from ddgs import DDGS
import json
import os
import sys
from pathlib import Path

# Resolve project root so lazy imports work correctly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mcp = FastMCP("RAG Web Search Server")


@mcp.tool
def search_web(query: str, max_results: int = 5) -> str:
    """Searches the web for recent information using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error searching web: {e}"


@mcp.tool
def create_markdown_report(filename: str, content: str) -> str:
    """Appends the given output content to a .md file."""
    if not filename.endswith(".md"):
        filename += ".md"
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(content + "\n\n---\n\n")
        return f"Report successfully appended to {filename}"
    except Exception as e:
        return f"Error writing file: {e}"


@mcp.tool
def load_pdf_to_database(pdf_path: str) -> str:
    """
    Converts a PDF file to readable text using PyMuPDF, chunks it,
    and appends all chunks to data/live_facts.json for re-indexing.
    After calling this, click 'Re-index Live Facts into RAG' in the UI.
    """
    # Lazy imports — only loaded when this tool is actually used
    # This keeps the MCP server startup fast for all other tools
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from scripts.extract import extract_pdf  # noqa: PLC0415
    from scripts.index_data import chunk_text  # noqa: PLC0415

    path = Path(pdf_path)
    if not path.exists():
        return f"Error: File not found at '{pdf_path}'"
    if path.suffix.lower() != ".pdf":
        return f"Error: '{pdf_path}' is not a PDF file."

    try:
        pages = extract_pdf(path)
    except Exception as e:
        return f"Error extracting PDF: {e}"

    queue_path = os.path.join(_PROJECT_ROOT, "data", "live_facts.json")
    chunk_count = 0
    try:
        with open(queue_path, "a", encoding="utf-8") as f:
            for page in pages:
                if not page["text"].strip():
                    continue  # skip empty pages
                chunks = chunk_text(page["text"], chunk_size=500, overlap=100)
                for chunk in chunks:
                    fact = {
                        "text": chunk,
                        "source": f"{path.name} (p.{page['page']})"
                    }
                    f.write(json.dumps(fact) + "\n")
                    chunk_count += 1
    except Exception as e:
        return f"Error writing to live_facts.json: {e}"

    return (
        f"✅ PDF '{path.name}' processed successfully!\n"
        f"   - {len(pages)} pages extracted\n"
        f"   - {chunk_count} chunks queued in data/live_facts.json\n"
        f"   Click '🔄 Re-index Live Facts into RAG' in the UI to make them searchable."
    )


@mcp.tool
def add_to_database(text: str, source: str) -> str:
    """Writes new facts back into the vector DB queue dynamically."""
    try:
        fact = {"text": text, "source": source}
        queue_path = os.path.join(_PROJECT_ROOT, "data", "live_facts.json")
        with open(queue_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(fact) + "\n")
        return "Fact successfully added to database queue (data/live_facts.json)."
    except Exception as e:
        return f"Error adding to database: {e}"


if __name__ == "__main__":
    # To run: fastmcp run app/mcp_server.py:mcp --transport http --port 8001
    mcp.run()
