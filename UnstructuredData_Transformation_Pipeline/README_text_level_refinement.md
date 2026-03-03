# MinerU `text_level` Refinement + CSV Export Pipeline

This folder contains two notebooks that turn MinerU `_content_list.json` into:
1) a **corrected** `content_list.json` with reliable `text_level: 1` headings, and  
2) a **GraphRAG-ready CSV** aggregated by section headings.

## What problem this solves

MinerU OCR/layout output often has `text_level: 1` issues:

- **False positives**: non-headings tagged as headings (examples, random short lines, etc.)
- **False negatives**: real body headings missing `text_level`
- **Non-text leakage**: some non-text blocks may incorrectly carry `text_level`
- **TOC duplication**: TOC headings can appear again in the body (same title), causing duplicated sections

This pipeline uses a VLM-extracted TOC (human-verified) as **ground truth** to correct headings.

---

## Notebooks

### 1) `mineru_text_level_refinement.ipynb` — Correct `text_level`

**Inputs**
- A PDF (used only for TOC extraction)
- MinerU `_content_list.json` (the file to correct)
- `GEMINI_API_KEY` environment variable

**Steps**
1. **Load MinerU JSON** and print statistics (blocks, pages, existing `text_level` blocks).
2. **Stage 1: Gemini TOC extraction**
   - Sends native PDF bytes to `gemini-2.5-flash`
   - Extracts **top-level** TOC sections as:
     ```json
     { "sections": [ { "section_id": "string or null", "title": "string" } ] }
     ```
3. **Human-in-the-loop gate**
   - Saves the extracted TOC to `toc_ground_truth.json`
   - User confirms (`yes`) or edits (`edit`) before correction continues
4. **Stage 2: Fuzzy match + correction**
   - Builds match targets from ground truth:
     - `title`
     - `"{section_id} {title}"` (only if `section_id` exists)  
       This improves matching when body headings include numeric prefixes like `"6 Organisations"`.
   - For each block in `content_list.json`:
     - If `type != "text"` and it has `text_level`, remove `text_level`
     - If `type == "text"`:
       - **Keep** `text_level` if the block matches a ground-truth heading
       - **Remove** `text_level` if it does not match (false positive)
       - **Add** `text_level = 1` if it matches but is missing (false negative)

**Matching**
- Normalization: lowercase + collapse whitespace (the “strip dot leaders/page numbers” rule is currently disabled in code)
- Similarity: `rapidfuzz.fuzz.ratio`
- Threshold: `FUZZY_THRESHOLD = 95` (strict)

> Note: The optional “TOC pages removal” and the “token_set_ratio pass” exist in the notebook but are currently commented out.

**Outputs**
- **Corrected JSON is written by overwriting `JSON_PATH`** (in-place overwrite)
- `correction_log.json` is saved next to `JSON_PATH`
- `toc_ground_truth.json` is saved next to `JSON_PATH` (for audit / reuse)

---

### 2) `mineru_to_csv.ipynb` — Aggregate corrected sections into CSV

**Purpose**
Convert corrected `content_list.json` into a section-level CSV for GraphRAG ingestion.

**Key logic**
- Treat `text_level == 1` blocks as **section boundaries**
- Use a **last-occurrence heuristic** to skip TOC duplicates:
  - headings with identical normalized text appear multiple times
  - the **last** occurrence is assumed to be the real body heading
- Aggregate all following text blocks until the next heading

**Outputs**
- A CSV with columns:
  - `text` (section heading + section body)
  - `doc_title`, `version`, `author` (from PDF metadata)
  - `section_title` (the heading)

---

## Dependencies

- `google-genai` (Gemini client)
- `rapidfuzz` (string similarity)
- `PyMuPDF` / `fitz` (PDF metadata)

Install (Windows / PowerShell):

```
pip install google-genai rapidfuzz pymupdf
```

---

## Configuration

### Environment variable
Set your Gemini API key:

**PowerShell**
```
$env:GEMINI_API_KEY="YOUR_KEY"
```

### Notebook variables
In `mineru_text_level_refinement.ipynb` adjust:
- `DOC_CODE`
- `PDF_PATH`
- `JSON_PATH`
- thresholds: `FUZZY_THRESHOLD`, etc.

In `mineru_to_csv.ipynb` adjust:
- `INPUT_JSON` (use the corrected file)
- `INPUT_PDF`
- `OUTPUT_CSV`

---

## Recommended run order

1. Run **`mineru_text_level_refinement.ipynb`**
   - Confirm / edit `toc_ground_truth.json`
   - Produce corrected `content_list.json` (in-place overwrite) + `correction_log.json`
2. Run **`mineru_to_csv.ipynb`**
   - Produce aggregated CSV for GraphRAG

---

## Notes / gotchas

- The refinement notebook currently only matches blocks where `block["type"] == "text"`.
- If your headings in body text include dot leaders/page numbers, consider enabling the commented normalization rule that strips trailing dots/page numbers.
- If OCR causes small word drops/reordering, consider enabling the commented `token_set_ratio` pass.
- TOC page-based stripping is present but disabled; current deduplication is primarily handled later by the CSV stage via the “last occurrence wins” rule.