"""
PDF highlighting utility for RAG Regulators.

After retrieval, call create_highlighted_pdfs(chunks) to get annotated copies
of the source PDF(s) with the retrieved passages highlighted in yellow.

Key design decision
-------------------
We do NOT save the full source PDF (200+ pages).  Instead we extract ONLY the
pages that contain retrieved passages into a new, small PDF.  When macOS
Preview opens that file it starts at page 1 — which is already the first
highlighted page — so no AppleScript page-navigation is needed.

Text-matching strategy
----------------------
extract.py cleans the corpus text (joins hyphenated breaks, joins mid-sentence
lines, strips repeated headers).  Exact phrase search against the raw PDF
therefore fails whenever those edits applied.  We use word-level
SequenceMatcher on the raw PDF word list vs the chunk word list instead, so
words that were removed during cleaning are simply skipped.
"""

import os
import re
import time
import base64
import tempfile
from difflib import SequenceMatcher
from typing import Optional

import pymupdf

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

_HIGHLIGHT_TMPDIR = tempfile.mkdtemp(prefix="rag_highlights_")
MIN_MATCH = 5   # minimum consecutive matching words to place a highlight


# ── Public API ────────────────────────────────────────────────────────────────

def create_highlighted_pdfs(
    retrieved_chunks: list,
) -> list[tuple[str, str, int]]:
    """
    For each source PDF referenced in retrieved_chunks, create a small PDF
    containing only the retrieved pages with yellow highlight annotations.

    Returns [(highlighted_pdf_path, source_filename, first_page_0idx=0), ...]
    ordered by chunk-count descending.  first_page_0idx is always 0 because
    the extracted PDF starts at the first relevant page.
    """
    if not retrieved_chunks:
        return []

    source_map: dict[str, list[dict]] = {}
    for chunk in retrieved_chunks:
        src = chunk.get("source", "")
        if src:
            source_map.setdefault(src, []).append(chunk)

    results = []
    for src in sorted(source_map, key=lambda s: len(source_map[s]), reverse=True):
        pdf_path = _find_pdf(src)
        if not pdf_path:
            continue
        out = _build_highlighted_pdf(pdf_path, source_map[src], src)
        if out:
            results.append((out, src, 0))   # 0 → Preview opens at page 1 = highlighted content

    return results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_pdf(source_name: str) -> Optional[str]:
    exact = os.path.join(DATA_DIR, source_name)
    if os.path.exists(exact):
        return exact
    try:
        lo = source_name.lower()
        for fname in os.listdir(DATA_DIR):
            if fname.lower() == lo and fname.lower().endswith(".pdf"):
                return os.path.join(DATA_DIR, fname)
    except OSError:
        pass
    return None


def _build_highlighted_pdf(
    pdf_path: str, chunks: list[dict], source_name: str
) -> Optional[str]:
    """
    1. Open the source PDF.
    2. Collect the distinct page indices referenced by the chunks.
    3. Add yellow highlights on those pages (word-level matching).
    4. Insert those pages — with their new annotations — into a fresh small PDF.
    5. Save to a timestamped temp file and return its path.
    """
    try:
        src_doc = pymupdf.open(pdf_path)
        total = len(src_doc)

        # Collect & sort the 0-indexed page numbers we need
        page_indices: list[int] = []
        for chunk in chunks:
            try:
                pg0 = int(chunk.get("page", 1)) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= pg0 < total:
                page_indices.append(pg0)

        page_indices = sorted(set(page_indices))
        if not page_indices:
            src_doc.close()
            return None

        # Add highlights directly on the source document pages
        any_hit = False
        for chunk in chunks:
            try:
                pg0 = int(chunk.get("page", 1)) - 1
            except (TypeError, ValueError):
                continue
            if not (0 <= pg0 < total):
                continue
            page = src_doc[pg0]
            hit = _highlight_words(page, chunk.get("text", ""))
            if not hit:
                _draw_page_marker(page)   # yellow bar fallback
                hit = True
            any_hit = any_hit or hit

        if not any_hit:
            src_doc.close()
            return None

        # Build a new PDF containing only the highlighted pages
        out_doc = pymupdf.open()
        for pg0 in page_indices:
            out_doc.insert_pdf(src_doc, from_page=pg0, to_page=pg0)

        src_doc.close()

        # Unique filename so Preview never caches a stale version
        safe = source_name.replace(" ", "_").replace("/", "_")
        ts = int(time.time() * 1000)
        out_path = os.path.join(_HIGHLIGHT_TMPDIR, f"highlighted_{safe}_{ts}.pdf")
        out_doc.save(out_path, garbage=4, deflate=True)
        out_doc.close()
        return out_path

    except Exception as exc:
        print(f"[pdf_highlighter] Failed to highlight '{source_name}': {exc}")
        return None


def _normalize(word: str) -> str:
    return re.sub(r"[^\w\d]", "", word.lower())


def _highlight_words(page: pymupdf.Page, chunk_text: str) -> bool:
    """
    Word-level highlight via SequenceMatcher.
    Matches blocks of ≥ MIN_MATCH consecutive words between the raw PDF page
    and the (cleaned) chunk text, then merges per-word rects into line rects.
    """
    raw_words = page.get_text("words")
    if not raw_words:
        return False

    pdf_norm, pdf_rects = [], []
    for w in raw_words:
        n = _normalize(w[4])
        if n:
            pdf_norm.append(n)
            pdf_rects.append((w[0], w[1], w[2], w[3]))

    if not pdf_norm:
        return False

    chunk_norm = [_normalize(t) for t in re.findall(r"[\w\d]+", chunk_text)]
    chunk_norm = [w for w in chunk_norm if w]
    if not chunk_norm:
        return False

    matcher = SequenceMatcher(None, pdf_norm, chunk_norm, autojunk=False)
    hit_rects = []
    for pdf_start, _chunk_start, size in matcher.get_matching_blocks():
        if size >= MIN_MATCH:
            hit_rects.extend(pdf_rects[pdf_start: pdf_start + size])

    if not hit_rects:
        return False

    for merged in _merge_by_line(hit_rects):
        try:
            annot = page.add_highlight_annot(pymupdf.Rect(*merged))
            annot.set_colors(stroke=[1.0, 0.85, 0.0])
            annot.update()
        except Exception:
            pass

    return True


def _merge_by_line(
    rects: list[tuple], tol: float = 3.0
) -> list[tuple]:
    """Combine word-level rects on the same text line into one rect per line."""
    if not rects:
        return []
    rects_sorted = sorted(rects, key=lambda r: (r[1], r[0]))
    groups: list[list[tuple]] = [[rects_sorted[0]]]
    for r in rects_sorted[1:]:
        if abs(r[1] - groups[-1][-1][1]) <= tol:
            groups[-1].append(r)
        else:
            groups.append([r])
    return [
        (min(r[0] for r in g), min(r[1] for r in g),
         max(r[2] for r in g), max(r[3] for r in g))
        for g in groups
    ]


def _draw_page_marker(page: pymupdf.Page) -> None:
    """Thin yellow bar at the top — fallback when word matching finds nothing."""
    try:
        r = page.rect
        strip = pymupdf.Rect(r.x0 + 10, r.y0 + 10, r.x1 - 10, r.y0 + 22)
        page.draw_rect(strip, color=(1.0, 0.85, 0.0), fill=(1.0, 0.85, 0.0))
    except Exception:
        pass


# ── UI rendering ──────────────────────────────────────────────────────────────

def render_highlighted_pages_html(
    highlighted: list[tuple[str, str, int]],
    dpi: int = 150,
) -> str:
    """
    Render every page from each highlighted PDF as an inline base64 PNG and
    return an HTML string ready for a gr.HTML component.

    Parameters
    ----------
    highlighted : list of (pdf_path, source_name, first_page_0idx)
        As returned by create_highlighted_pdfs().
    dpi : int
        Render resolution.  150 is crisp enough for reading on-screen.
    """
    if not highlighted:
        return ""

    sections: list[str] = []
    for pdf_path, src_name, _first_page in highlighted:
        try:
            doc = pymupdf.open(pdf_path)
            page_imgs: list[str] = []
            for page in doc:
                mat = pymupdf.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                b64 = base64.b64encode(pix.tobytes("png")).decode()
                page_imgs.append(
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="width:100%;border:1px solid #d0d7de;border-radius:4px;'
                    f'margin-bottom:6px;display:block;" />'
                )
            doc.close()
            if page_imgs:
                sections.append(
                    f'<div style="margin-bottom:14px;">'
                    f'<div style="font-size:0.78rem;color:#64748b;font-weight:600;'
                    f'margin-bottom:4px;word-break:break-all;">📄 {src_name}</div>'
                    + "".join(page_imgs)
                    + "</div>"
                )
        except Exception as exc:
            print(f"[pdf_highlighter] Failed to render '{src_name}': {exc}")

    if not sections:
        return ""

    return (
        # Outer wrapper — fixed height with its own scrollbar so it works
        # regardless of how Gradio nests its component divs.
        '<div style="'
        'height:calc(100vh - 180px);'
        'min-height:300px;'
        'overflow-y:auto;'
        'overflow-x:hidden;'
        'padding:4px 6px 4px 2px;'
        'box-sizing:border-box;'
        '">'
        '<div style="font-size:0.9rem;font-weight:700;color:#1e3a5f;'
        'margin-bottom:10px;">📑 Highlighted Source Pages</div>'
        + "".join(sections)
        + "</div>"
    )
