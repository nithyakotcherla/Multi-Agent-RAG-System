#!/usr/bin/env python3
"""
PDF Text Extraction Script
Extracts clean text from a folder of PDFs and outputs:
  - corpus.json  (all pages)
  - sample.json  (first 10 pages)
  - Corpus statistics printed to stdout
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pymupdf  # PyMuPDF


# ── Text cleanup helpers ────────────────────────────────────────────────────

def clean_text(raw: str) -> str:
    """
    Clean raw PDF-extracted text:
      1. Normalise unicode whitespace
      2. Remove repeated headers/footers (lines that appear on almost every page)
      3. Fix mid-sentence line breaks
      4. Collapse excessive blank lines and spaces
    """
    text = raw

    # Replace non-breaking spaces and other unicode whitespace with regular space
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")  # zero-width space
    text = text.replace("\ufeff", "")  # BOM

    # Remove form-feed characters
    text = text.replace("\f", "")

    # Fix hyphenated line breaks (word- \n continuation)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Join lines that are mid-sentence (line does not end with sentence-ending
    # punctuation or a colon, and the next line starts with a lowercase letter)
    text = re.sub(
        r"([a-zA-Z,;])\s*\n\s*([a-z])",
        r"\1 \2",
        text,
    )

    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces into one (but keep newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Strip leading/trailing whitespace on each line
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Strip leading/trailing whitespace overall
    text = text.strip()

    return text


def detect_repeated_lines(pages_text: list[str], threshold: float = 0.6) -> set[str]:
    """
    Detect lines that appear on more than `threshold` fraction of pages.
    These are likely headers or footers.
    Returns a set of such lines (stripped and lowered for comparison).
    """
    if len(pages_text) < 4:
        return set()

    # Count how often each line appears across pages
    line_counts: dict[str, int] = {}
    for page_text in pages_text:
        # Take only the first 3 and last 3 lines of each page
        lines = page_text.split("\n")
        candidate_lines = lines[:3] + lines[-3:]
        seen_on_page: set[str] = set()
        for line in candidate_lines:
            normalised = line.strip().lower()
            if len(normalised) > 3 and normalised not in seen_on_page:
                seen_on_page.add(normalised)
                line_counts[normalised] = line_counts.get(normalised, 0) + 1

    total_pages = len(pages_text)
    repeated = set()
    for line, count in line_counts.items():
        if count / total_pages >= threshold:
            repeated.add(line)

    return repeated


def remove_repeated_lines(text: str, repeated: set[str]) -> str:
    """Remove lines whose normalised form appears in `repeated`."""
    if not repeated:
        return text
    lines = text.split("\n")
    filtered = [
        line for line in lines
        if line.strip().lower() not in repeated
    ]
    return "\n".join(filtered)


# ── PDF extraction ───────────────────────────────────────────────────────────

def extract_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract text from every page of a PDF and return a list of page records.
    """
    doc = pymupdf.open(str(pdf_path))
    filename = pdf_path.name

    # First pass: extract raw text to detect repeated headers/footers
    raw_pages: list[str] = []
    for page in doc:
        raw_pages.append(page.get_text("text"))

    repeated = detect_repeated_lines(raw_pages)

    # Second pass: clean and build records
    records: list[dict] = []
    for page_num, raw in enumerate(raw_pages, start=1):
        text = clean_text(raw)
        text = remove_repeated_lines(text, repeated)
        text = text.strip()

        records.append({
            "source": filename,
            "page": page_num,
            "char_count": len(text),
            "text": text,
        })

    doc.close()
    return records


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract text from a folder of PDFs into corpus.json"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the folder containing PDF files",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="data",
        help="Output directory for corpus.json and sample.json (default: data)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Find all PDF files
    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in '{input_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF file(s) in '{input_dir}':\n")
    for f in pdf_files:
        print(f"  • {f.name}")
    print()

    # Extract text from each PDF
    corpus: list[dict] = []
    for pdf_path in pdf_files:
        print(f"Extracting: {pdf_path.name} ... ", end="", flush=True)
        pages = extract_pdf(pdf_path)
        corpus.extend(pages)
        print(f"{len(pages)} pages")

    print()

    # ── Save outputs ──────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / "corpus.json"
    sample_path = output_dir / "sample.json"

    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"Saved full corpus   → {corpus_path}  ({len(corpus)} pages)")

    sample = corpus[:10]
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"Saved sample        → {sample_path}  ({len(sample)} pages)")

    # ── Print corpus stats ────────────────────────────────────────────────
    num_docs = len(pdf_files)
    total_pages = len(corpus)
    total_chars = sum(p["char_count"] for p in corpus)
    avg_chars = total_chars / total_pages if total_pages else 0
    empty_pages = [p for p in corpus if p["char_count"] < 50]
    num_empty = len(empty_pages)

    print("\n" + "=" * 55)
    print("                  CORPUS STATISTICS")
    print("=" * 55)
    print(f"  Documents (PDFs)          : {num_docs}")
    print(f"  Total pages               : {total_pages}")
    print(f"  Total characters          : {total_chars:,}")
    print(f"  Avg characters per page   : {avg_chars:,.0f}")
    print(f"  Empty/near-empty (<50ch)  : {num_empty}")
    print("=" * 55)

    if empty_pages:
        print("\n⚠  Empty / near-empty pages:")
        for p in empty_pages:
            preview = p["text"][:60].replace("\n", " ") if p["text"] else "(empty)"
            print(f"    {p['source']}  p.{p['page']}  ({p['char_count']} chars)  → {preview}")

    print("\nDone ✓")


if __name__ == "__main__":
    main()
