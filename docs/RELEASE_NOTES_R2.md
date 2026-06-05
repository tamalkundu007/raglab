# RAGLab R2 — Release Notes

**Version:** 0.2.0 · **Theme:** Advanced Chunking + Cloud Storage · **Date:** June 2026  
**Builds on:** R1 (Full Shell + Core Pipeline)

---

## Summary

Release 2 activates the chunker and storage layers that were scaffolded as "Coming Soon" stubs in R1. Five document-type chunkers, one meta-strategy, and two cloud storage backends — all wired, tested, and registered in their respective factories. The Control Panel UI already had all knobs visible; R2 flips them live.

---

## Stats

| Metric | Value |
|--------|-------|
| New tests | 195 |
| Total tests passing | 564 |
| Tests skipped | 3 (tiktoken BPE in sandboxed CI) |
| Infra required to run tests | Zero |
| raglab-chunkers version | 0.3.0 |
| storage-service version | 0.2.0 |
| New chunkers activated | 6 (5 document-type + 1 meta-strategy) |
| New storage backends activated | 2 (S3 + Azure Blob) |

---

## What Shipped

### raglab-chunkers v0.3.0

All chunkers implement `BaseChunker`, register in `ChunkerFactory`, and call `_boundary.split_into_windows()` — zero reimplementation of the boundary backtracking algorithm.

**PDFChunker** (`chunker_type: "pdf"`)
- PyMuPDF (fitz) text extraction, page-by-page
- `respect_page_boundary=True`: `split_into_windows()` runs within each page — chunks never cross PDF page breaks
- `page_number` injected into `ChunkModel.metadata` when `page_metadata=True`
- `chunk_pdf_bytes()` for raw bytes input; `chunk()` for pre-extracted text fallback
- Parameters: `chunk_size`, `chunk_overlap`, `boundary_enforcement`, `boundary_chars`, `tokenizer`, `min_chunk_size`, `respect_page_boundary`, `page_metadata`

**DOCXChunker** (`chunker_type: "docx"`)
- python-docx paragraph parsing with heading style detection (`_is_heading()`, `_heading_level()`)
- `preserve_headings=True`: paragraphs grouped under their nearest heading as structural units, `split_into_windows()` within each
- `include_heading_in_chunk=True`: heading text prepended to every chunk in that section
- `heading` + `heading_level` injected into `ChunkModel.metadata`
- `chunk_docx_bytes()` for raw bytes input
- Parameters: `chunk_size`, `chunk_overlap`, `boundary_enforcement`, `tokenizer`, `min_chunk_size`, `preserve_headings`, `include_heading_in_chunk`

**MarkdownChunker** (`chunker_type: "markdown"`)
- ATX header parsing via regex (`#{1,6}`)
- `_split_by_headers()` segments text at configured `header_levels` (default `[1,2,3]`)
- Preamble before first header preserved as a section with `header=None`
- `include_header_in_chunk=True`: header line prepended to section text before windowing
- `header` + `header_level` injected into metadata
- `split_on_headers=False`: single-stream token fallback
- Parameters: `chunk_size`, `chunk_overlap`, `split_on_headers`, `header_levels`, `include_header_in_chunk`, `boundary_enforcement`, `tokenizer`, `min_chunk_size`

**HTMLChunker** (`chunker_type: "html"`)
- BeautifulSoup semantic tag extraction (`p`, `article`, `section`, `div`, `li`, `td`, `h1`–`h6`, etc.)
- Greedy node binning: concatenate nodes until `chunk_size_fallback` token budget fills
- Oversized single nodes split directly with `split_into_windows()`
- `strip_scripts_styles=True`: removes `<script>`, `<style>`, `<noscript>` elements
- `include_tag_attrs=True`: source tag name(s) injected into chunk metadata
- Fallback: no matching tags → `get_text()` on full soup, then window
- Parameters: `split_tags`, `include_tag_attrs`, `strip_scripts_styles`, `chunk_size_fallback`, `overlap_fallback`, `boundary_enforcement`, `boundary_chars`, `tokenizer`, `min_chunk_size`

**ExcelChunker** (`chunker_type: "excel"`)
- Three sheet strategies:
  - `"row"` (default): group rows in `row_grouping` batches, represent each as `Header: Value | Header: Value` format — column context always travels with data
  - `"column"`: one chunk per column with all values listed vertically
  - `"cell"`: one chunk per non-empty cell (`Header: Value`)
- `header_rows`: first N rows treated as column names, prepended to every data row
- `include_sheet_name=True`: `Sheet: <name>` prefix on every chunk
- `chunk_excel_bytes()` for raw xlsx bytes via pandas + openpyxl
- Skips `nan`/`None`/empty values in all strategies
- Parameters: `sheet_strategy`, `header_rows`, `row_grouping`, `include_sheet_name`, `tokenizer`, `chunk_size`, `chunk_overlap`

**HybridChunker** (`chunker_type: "hybrid"`) — Meta-strategy
- Two-pass pipeline:
  - **Pass 1 (structural)**: delegates to `source_chunker` (text/markdown/html) for document structure
  - **Pass 2 (token_window)**: applies `split_into_windows()` within any unit exceeding `chunk_size`
- Units within budget → `hybrid_pass: "structural"` in metadata
- Units exceeding budget → `hybrid_pass: "token_window"` in metadata
- `structural_first=False`: bypass Pass 1, pure token windowing (A/B baseline)
- `source_config`: dict forwarded to source chunker constructor (e.g. `header_levels`)
- `max_unit_tokens`: controls structural unit size (forwarded as `chunk_size` to source chunker)
- **Naming distinction (tested):** `HybridChunker` = structural + token windowing. `HybridRetriever` (R3) = dense + sparse fusion. Completely separate concepts, confirmed in code and tests.

### Factory updates

`ChunkerFactory._REGISTRY` now includes all 6 R2 types as active. `pdf_images` and `table_stitch` remain stubs (`R2-extended` scope). `available()` handles non-enum string keys (`"hybrid"`) via `_ACTIVE_STRINGS`.

### storage-service v0.2.0

**BaseStorageBackend** — four-method abstract interface: `upload()`, `download()`, `delete()`, `exists()`. All raise `StorageError` — callers never see raw SDK exceptions. Key is always a path-like string, never a URL.

**LocalStorageBackend** (R1 active)
- Configurable root directory
- Path traversal guard: `resolve()` + prefix check → `StorageError`
- Idempotent `delete()` (no-op if key missing)
- Nested directory creation on `upload()`
- URI format: `local://<absolute-path>`

**S3StorageBackend** (R2 active)
- `boto3` at module level — fully patchable in tests (zero AWS calls)
- Prefix support: `_full_key()` prepends configured prefix to all keys
- `NoSuchKey`/`404` `ClientError` → `StorageError('not found')`
- URI format: `s3://<bucket>/<key>`
- Config: `bucket`, `region`, `prefix`

**AzureBlobStorageBackend** (R2 active)
- `azure-storage-blob` at module level — fully patchable in tests
- Dual auth: `connection_string` or `account_name` + `account_key`
- `upload_blob(overwrite=True)` — idempotent overwrites
- `ResourceNotFoundError` → `StorageError('not found')`
- Idempotent `delete()` (swallows `ResourceNotFoundError`)
- URI format: `https://<account>.blob.core.windows.net/<container>/<blob>`
- Config: `container`, `connection_string`, `account_name`, `account_key`, `prefix`

**GCSStorageBackend** — stub, all methods raise `NotImplementedFeatureError("R7")`

**StorageFactory** — `create()`, `available()`. Same pattern as `ChunkerFactory` and `RetrieverFactory`.

**HTTP endpoints** (all under `/storage`):
- `POST /storage/upload/{key:path}` — base64-encoded body → uri + size + backend
- `GET  /storage/download/{key:path}` — raw binary response
- `DELETE /storage/{key:path}` — 204, idempotent
- `GET  /storage/exists/{key:path}` — `{exists: bool, backend: str}`
- `GET  /storage/backends` — full registry with active flags

---

## Design Decisions

**Reuse rule enforced:** Every R2 chunker calls `_boundary.split_into_windows()` for token+boundary splitting. Zero reimplementation. The structural pass (headings, pages, tags, sheets) is each chunker's own logic; the algorithm is shared.

**Column context in ExcelChunker:** The `row` strategy produces `Header: Value | Header: Value` text rather than raw CSV rows. A retrieved chunk `"Name: Alice | Age: 30 | City: London"` is self-contained — the LLM never sees `"Alice, 30, London"` without knowing what those values mean.

**Heading context in DOCXChunker/MarkdownChunker:** `include_heading_in_chunk=True` prepends the heading to every chunk in that section. A retrieved chunk from the middle of a long section retains its section context for downstream generation.

**StorageFactory credential discipline:** All cloud credentials come from env vars / pydantic-settings with `RAGLAB_` prefix. No credential is ever accepted as a plain function argument or hardcoded anywhere.

**HybridChunker lazy source init:** Source chunker is instantiated via `ChunkerFactory.create()` on first call, not in `__init__()`, to prevent circular imports at module load time (`factory.py` imports `HybridChunker`; `HybridChunker` would import `factory`).

---

## Quick Start — R2 Chunkers

```python
from raglab_chunkers import ChunkerFactory

# PDF with page boundary respect
chunker = ChunkerFactory.create("pdf", config={
    "tokenizer": "word_count",
    "chunk_size": 500,
    "respect_page_boundary": True,
    "page_metadata": True,
})
with open("report.pdf", "rb") as f:
    chunks = chunker.chunk_pdf_bytes(f.read(), doc_id="report-001")

# Markdown with H1/H2 structural splits
chunker = ChunkerFactory.create("markdown", config={
    "tokenizer": "word_count",
    "chunk_size": 500,
    "header_levels": [1, 2],
    "include_header_in_chunk": True,
})
chunks = chunker.chunk(markdown_text, doc_id="doc-001")

# Hybrid: markdown structure + token windowing within sections
chunker = ChunkerFactory.create("hybrid", config={
    "source_chunker": "markdown",
    "max_unit_tokens": 1000,
    "chunk_size": 300,
    "source_config": {"header_levels": [1, 2]},
})
chunks = chunker.chunk(markdown_text, doc_id="doc-001")

# Excel with row strategy (column context preserved)
chunker = ChunkerFactory.create("excel", config={
    "sheet_strategy": "row",
    "row_grouping": 5,
    "include_sheet_name": True,
})
with open("data.xlsx", "rb") as f:
    chunks = chunker.chunk_excel_bytes(f.read(), doc_id="sheet-001")
```

---

## 7-Release Roadmap

| Release | Theme | Status |
|---------|-------|--------|
| R1 | Full Shell + Core Pipeline | ✅ Done |
| **R2** | **Advanced Chunking + Cloud Storage** | ✅ **Done** |
| R3 | Advanced Retrievers (BM25/Hybrid/MMR/Re-ranker) + CI/CD + Cloud Deploy | 🔜 Next |
| R4 | GraphRAG (NetworkX + leidenalg + graph retrieval) | 🔜 Planned |
| R5 | Caching + Performance (Redis semantic cache) | 🔜 Planned |
| R6 | Observability / LLMOps (evaluation metrics + tracing) | 🔜 Planned |
| R7 | Auth + Multi-tenancy + GCS | 🔜 Planned |

---

*Built by [Tamal Kundu](https://tamalkundu.com) · Kundu Corp · June 2026*
