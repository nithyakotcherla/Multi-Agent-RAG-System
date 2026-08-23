# Extraction Report

## Corpus Overview

| Metric | Value |
|---|---|
| Documents (PDFs) | 3 |
| Total pages | 280 |
| Total characters | 1,045,540 |
| Avg characters per page | 3,734 |
| Empty/near-empty pages (<50 chars) | 0 |

### Documents processed

| Document | Pages |
|---|---|
| `GDPR.pdf` (EU GDPR — Regulation 2016/679) | 88 |
| `EU AI ACT.pdf` (EU Artificial Intelligence Act) | 144 |
| `risk managment 48 pages.pdf` (NIST AI Risk Management Framework) | 48 |

---

## Issues Encountered & How They Were Handled

### 1. Repeated Headers / Footers

**Issue:** All three PDFs are official regulatory/standards documents that include recurring page numbers, document titles, and official journal references at the top and/or bottom of every page.

**Solution:** Implemented an automatic detection algorithm that identifies lines appearing on more than 60% of pages (within the first 3 and last 3 lines of each page). These lines are removed from the final text. This effectively strips repeated headers and footers without affecting body content.

### 2. Mid-sentence Line Breaks

**Issue:** PDFs store text line-by-line as rendered on the page, so sentences are often broken mid-flow. For example:
```
This regulation applies to the
processing of personal data wholly
or partly by automated means.
```

**Solution:** Applied regex-based line joining: when a line ends with an alphabetic character or comma and the next line starts with a lowercase letter, the newline is replaced with a space. Hyphenated word breaks (e.g., `regula-\ntion`) are also rejoined.

### 3. Unicode / Special Whitespace Characters

**Issue:** Some pages contained non-breaking spaces (`\u00a0`), zero-width spaces (`\u200b`), and BOM characters (`\ufeff`), which can cause issues in downstream NLP processing.

**Solution:** All such characters are normalized or removed during the cleanup step.

### 4. Extra Blank Lines / Whitespace

**Issue:** Raw extracted text often contained multiple consecutive blank lines and excessive spaces, especially around section headings, footnotes, and tables of contents.

**Solution:** Multiple blank lines are collapsed to a single blank line. Multiple spaces are collapsed to a single space. Leading/trailing whitespace is stripped from each line.

### 5. Tables

**Issue:** Some pages (especially in the EU AI Act) contain structured tables. PyMuPDF extracts table content as plain text, which can lose the tabular structure.

**Observation:** Spot-checking showed that table content remained readable as plain text, though column alignment is lost. For the purposes of RAG text retrieval, this is acceptable — the key information (terms, definitions, requirements) is preserved even without formatting. If table extraction becomes critical for downstream tasks, switching to `pdfplumber` for table-heavy sections would be recommended.

### 6. Scanned / Image-based Pages

**Observation:** None of the PDFs in this corpus are scanned documents — all contain selectable text. No OCR was needed.

---

## Quality Assessment

- **Text readability:** ✅ Spot-checked pages from all three documents. Text is clean, readable, and preserves paragraph structure.
- **Completeness:** ✅ All 280 pages extracted. Zero empty/near-empty pages.
- **Metadata accuracy:** ✅ Each entry correctly records source filename, page number, and character count.

## Library Used

- **PyMuPDF** (`pymupdf==1.27.2.2`) — chosen for speed and reliable general-purpose text extraction from text-based regulatory PDFs.
