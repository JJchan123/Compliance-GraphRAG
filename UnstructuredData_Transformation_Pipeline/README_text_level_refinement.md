# MinerU `text_level` Refinement Agent

An AI-powered pipeline that corrects `text_level` tags in MinerU's `_content_list.json` output using **Gemini 2.5 Flash** VLM-based Table of Contents extraction and fuzzy matching.

## Problem

MinerU's OCR/layout detection produces `_content_list.json` files where `text_level: 1` tags are often inaccurate:

| Issue | Example |
|---|---|
| **False Positives** | `"TABLE OF CONTENTS"`, `"Example:"`, clause numbers like `"5.19"` incorrectly tagged as headings |
| **False Negatives** | Real section headings in the document body missing `text_level` entirely |
| **TOC Page Leakage** | TOC entries (pages 1–5) tagged as body headings |

## Solution

A 3-stage pipeline with human-in-the-loop verification:

```
PDF ──► Gemini 2.5 Flash ──► TOC JSON ──► Human Review ──► Fuzzy Match ──► Corrected JSON
```

### Stage 1: VLM TOC Extraction
- Sends the native PDF to **Gemini 2.5 Flash**
- Extracts top-level section headings and TOC page boundaries
- Returns structured JSON with `toc_pages` and `sections`

### Stage 2: Human-in-the-Loop Review
- Saves the LLM result to `toc_ground_truth.json` for inspection
- User reviews, optionally edits, and confirms before proceeding
- Acts as a quality gate to catch any LLM extraction errors

### Stage 3: Fuzzy Match & Correct
- Uses the confirmed ground truth to correct `content_list.json`:
  - **Remove TOC page tags** — all `text_level` on TOC pages are stripped
  - **Remove false positives** — body blocks with `text_level: 1` that don't match any ground truth section
  - **Add false negatives** — body blocks matching ground truth but missing `text_level`
- Two-pass fuzzy matching (direct ratio + token-set ratio) via `rapidfuzz`

## Output

| File | Description |
|---|---|
| `*_corrected.json` | Corrected `content_list.json` with fixed `text_level` tags |
| `*_correction_log.json` | Audit log of all changes (added, removed, kept) with match scores |
| `toc_ground_truth.json` | Confirmed TOC extraction used as ground truth |

## Dependencies

| Package | Purpose |
|---|---|
| `google-genai` | Gemini 2.5 Flash API client |
| `rapidfuzz` | Fast fuzzy string matching |
| `PyMuPDF (fitz)` | PDF handling (already installed) |

### Install

```bash
pip install google-genai rapidfuzz
```

## Configuration

Edit the configuration cell (Cell 2) in the notebook:

```python
GEMINI_API_KEY = 'your-api-key'          # or set GEMINI_API_KEY env var
GEMINI_MODEL   = 'gemini-2.5-flash'

PDF_PATH  = r'path\to\document.pdf'
JSON_PATH = r'path\to\document_content_list.json'
OUTPUT_DIR = None                         # None = save alongside input

FUZZY_THRESHOLD = 80                      # Minimum fuzzy match score (0-100)
```

## Notebook Structure

| Cell | Type | Description |
|---:|---|---|
| 1 | Markdown | Title & pipeline overview |
| 2 | Code | Imports & configuration |
| 3 | Code | Load `content_list.json` & display statistics |
| 4 | Markdown | Stage 1 description |
| 5 | Code | VLM TOC extraction (Gemini 2.5 Flash) |
| 6 | Markdown | Human-in-the-loop instructions |
| 7 | Code | Review, save, and confirm TOC result |
| 8 | Markdown | Stage 2 description |
| 9 | Code | Fuzzy matching & correction engine |
| 10 | Code | Save corrected JSON & correction log |
| 11 | Code | Validation spot-check & cross-reference |

## Usage

1. Set your `GEMINI_API_KEY` and file paths in Cell 2
2. Run Cells 1–5 to extract TOC from the PDF
3. Review the LLM output in Cell 7 — type `yes` to confirm or `edit` after modifying the saved JSON
4. Run Cells 9–11 to apply corrections and validate

## Ground Truth JSON Format

The VLM produces (and the human confirms) this structure:

```json
{
  "toc_pages": [1, 2, 3, 4, 5],
  "sections": [
    {"section_id": null, "title": "PART I: INTRODUCTION AND OVERVIEW"},
    {"section_id": "1",  "title": "Introduction"},
    {"section_id": "2",  "title": "Overview of the PDPA"},
    {"section_id": "6",  "title": "Organisations"}
  ]
}
```

- `toc_pages`: 0-indexed page numbers containing the Table of Contents
- `section_id`: section number as string, or `null` for Part titles / Annexes
- `title`: clean heading text (no dot leaders or page numbers)

## Matching Logic

1. **Normalize** — lowercase, collapse whitespace, strip trailing dots/page numbers
2. **Pass 1 (strict)** — `fuzz.ratio` ≥ `FUZZY_THRESHOLD` (default 80)
3. **Pass 2 (loose)** — `fuzz.token_set_ratio` ≥ 90, with length-ratio guard > 0.5 to prevent short-text false matches
4. Section ID variants are also generated (e.g., matching `"6 Organisations"` against `"Organisations"`)
