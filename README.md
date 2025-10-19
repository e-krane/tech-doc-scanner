# Technical Document Conversion Agent

A high-performance document conversion agent powered by [Docling](https://github.com/DS4SD/docling) that converts technical documents to markdown with smart chunking optimized for RAG pipelines.

## Features

- **Multi-Format Support**: PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, HTML, HTM, MD, TXT
- **GPU Acceleration**: Automatic device detection (CUDA/MPS/CPU) for fast processing
- **Partial Parsing**: Convert specific page ranges for faster previews and section extraction
- **Smart Chunking**: Token-aware, structure-preserving chunking with Docling's HybridChunker
- **Figure Extraction**: Automatically extract images, tables, and equations
- **Document Optimizer**: Analyzes documents to determine optimal processing settings
- **Batch Processing**: Process entire directories with progress tracking
- **Multiple Output Formats**: Export chunks as JSON, JSONL, Markdown, or CSV

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/e-krane/tech-doc-scanner.git
cd tech-doc-scanner

# Install dependencies
uv sync

# Activate the environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

### Using pip

```bash
pip install -e .
```

## Quick Start

### Convert a Single Document

```bash
python main.py convert document.pdf
```

This will:
- Convert the PDF to markdown
- Extract figures to `output/figures/`
- Save markdown to `output/markdown/`

### Batch Convert Documents

```bash
python main.py batch documents/ -o output/
```

Process all PDF files in a directory with progress tracking.

### Python API

```python
from pathlib import Path
from src.config import ConverterConfig
from src.converter import TechDocConverter

# Create converter
config = ConverterConfig(
    output_dir=Path("output"),
    use_gpu=True,
    save_figures=True
)
converter = TechDocConverter(config)

# Convert full document
result = converter.convert_and_save(Path("document.pdf"))

if result.success:
    print(f"Converted {result.page_count} pages in {result.conversion_time:.2f}s")
    print(f"Markdown saved")
    print(f"Figures: {result.figure_count}")
```

### Partial Document Parsing

Convert only specific pages for faster processing:

```python
from pathlib import Path
from src.converter import TechDocConverter
from src.config import ConverterConfig

converter = TechDocConverter(ConverterConfig())

# Convert specific page range (pages 5-10)
result = converter.convert(
    Path("document.pdf"),
    page_range=(5, 10)
)

# Convert first N pages (preview mode)
result = converter.convert(
    Path("document.pdf"),
    max_pages=5
)

# Quick single-page preview
result = converter.convert(
    Path("document.pdf"),
    page_range=(1, 1)
)

print(f"Converted {result.page_count} pages in {result.conversion_time:.2f}s")
```

**Use Cases for Partial Parsing:**
- Quick document previews before full conversion
- Extract specific chapters or sections
- Test conversion settings on representative pages
- Process large documents incrementally
- Reduce processing time for exploratory analysis

## Detailed Examples

See the `examples/` directory for complete working examples:
- `basic_conversion.py` - Simple document conversion
- `batch_with_chunks.py` - Batch processing with chunking
- `custom_pipeline.py` - Advanced pipeline with optimization
- `partial_parsing.py` - Partial document parsing examples

### With Smart Chunking

```python
from pathlib import Path
from src.config import ConverterConfig, ChunkerConfig
from src.converter import TechDocConverter
from src.chunker import DocumentChunker

# Convert document
converter = TechDocConverter(ConverterConfig())
result = converter.convert(Path("document.pdf"))

# Chunk the document
chunker_config = ChunkerConfig(max_tokens=512)
chunker = DocumentChunker(chunker_config)

chunks = chunker.chunk_document(
    docling_doc=result.docling_doc,
    title=result.file_path.stem,
    source=str(result.file_path)
)

# Export chunks
chunker.export_chunks(chunks, Path("output/chunks.jsonl"), format="jsonl")

print(f"Created {len(chunks)} chunks")
```

### With Document Optimizer

```python
from pathlib import Path
from src.config import OptimizerConfig, ConverterConfig
from src.optimizer import DocumentOptimizer
from src.converter import TechDocConverter

# Analyze document
optimizer = DocumentOptimizer(OptimizerConfig())
analysis = optimizer.analyze(Path("document.pdf"))

print(f"Needs OCR: {analysis.needs_ocr}")
print(f"Has tables: {analysis.has_tables}")
print(f"Language: {analysis.language}")
print(f"Estimated time: {analysis.estimated_time:.1f}s")

# Get optimized config
config_dict = optimizer.optimize_converter_config(Path("document.pdf"))
config = ConverterConfig(**config_dict)

# Convert with optimized settings
converter = TechDocConverter(config)
result = converter.convert_and_save(Path("document.pdf"))
```

## CLI Commands

### Convert Command

Convert a single document to markdown:

```bash
python main.py convert [OPTIONS] FILE_PATH

Options:
  -o, --output-dir PATH   Output directory (default: output)
  --no-figures           Disable figure extraction
  --profile              Enable performance profiling
  -v, --verbose          Enable verbose logging
  -q, --quiet            Suppress output except errors
  --log-file PATH        Log to file
```

Example:

```bash
python main.py convert research_paper.pdf -o results/ --profile
```

### Batch Command

Process all documents in a directory:

```bash
python main.py batch [OPTIONS] INPUT_DIR

Options:
  -o, --output-dir PATH   Output directory (default: output)
  --no-figures           Disable figure extraction
  -p, --pattern TEXT     File pattern to match (default: *.pdf)
  -v, --verbose          Enable verbose logging
```

Example:

```bash
python main.py batch documents/ -p "*.pdf" -o converted/
```

## Configuration

### ConverterConfig

Control document conversion behavior:

```python
from src.config import ConverterConfig
from pathlib import Path

config = ConverterConfig(
    output_dir=Path("output"),      # Where to save outputs
    use_gpu=True,                   # Enable GPU acceleration
    device="auto",                  # "auto", "cpu", "cuda", "mps"
    num_threads=4,                  # Thread count for processing
    do_ocr=True,                    # Enable OCR for scanned docs
    do_table_structure=True,        # Extract table structure
    do_cell_matching=True,          # Match cells in tables
    save_figures=True,              # Extract figures
    enable_profiling=False          # Performance profiling (basic timing)
)
```

**Configuration Presets:**

For **Born-Digital Technical Papers**:
```python
config = ConverterConfig(
    use_gpu=True,
    do_ocr=False,              # Not needed for born-digital PDFs
    do_table_structure=True,   # Extract data tables
    save_figures=True          # Extract diagrams and plots
)
```

For **Scanned Documents**:
```python
config = ConverterConfig(
    use_gpu=True,
    do_ocr=True,               # Required for scanned documents
    do_table_structure=True,
    save_figures=True
)
```

For **Large Documents** (100+ pages):
```python
config = ConverterConfig(
    use_gpu=True,
    num_threads=8,             # Increase for better performance
    save_figures=False         # Disable to speed up processing
)
```

### ChunkerConfig

Control document chunking behavior:

```python
from src.config import ChunkerConfig

config = ChunkerConfig(
    max_tokens=512,                 # Max tokens per chunk
    merge_peers=True,               # Merge small adjacent chunks
    tokenizer_model="sentence-transformers/all-MiniLM-L6-v2"
)
```

### OptimizerConfig

Control document analysis:

```python
from src.config import OptimizerConfig

config = OptimizerConfig(
    enable_language_detection=True,  # Detect document language
    ocr_quality_threshold=0.8,       # OCR quality threshold
    sample_pages=3                   # Pages to sample for analysis
)
```

## Output Formats

### Markdown

Standard markdown with:
- Preserved document structure
- Figure references
- Table formatting
- Section headings

### Chunks (JSONL)

```json
{
  "content": "Chapter 1: Introduction\n\nThis document describes...",
  "contextualized_content": "# Chapter 1\n## Introduction\n\nThis document describes...",
  "index": 0,
  "token_count": 245,
  "metadata": {
    "title": "Technical Manual",
    "source": "manual.pdf",
    "headings": ["Chapter 1", "Introduction"],
    "page": 1
  }
}
```

### Figures

Extracted as PNG files with:
- Sequential numbering (`figure_001.png`, `figure_002.png`)
- README.md index with captions
- Organized by document name

## Supported Formats

| Format | Extension | OCR Support | Table Extraction | Figure Extraction |
|--------|-----------|-------------|------------------|-------------------|
| PDF    | `.pdf`    | ✓           | ✓                | ✓                 |
| Word   | `.docx`, `.doc` | ✓     | ✓                | ✓                 |
| PowerPoint | `.pptx`, `.ppt` | ✓   | ✓                | ✓                 |
| Excel  | `.xlsx`, `.xls` | ✓     | ✓                | ✓                 |
| HTML   | `.html`, `.htm` | ✗     | ✓                | ✓                 |
| Markdown | `.md`   | ✗           | ✗                | ✗                 |
| Text   | `.txt`    | ✗           | ✗                | ✗                 |

## Architecture

```
tech-doc-scanner/
├── src/
│   ├── config.py       # Configuration dataclasses (ConverterConfig, ChunkerConfig, OptimizerConfig)
│   ├── converter.py    # Document conversion with Docling (GPU acceleration, partial parsing)
│   ├── chunker.py      # Smart chunking with HybridChunker (token-aware, structure-preserving)
│   ├── optimizer.py    # Document analysis & auto-optimization (OCR detection, language, tables)
│   ├── utils.py        # Logging, statistics, and utilities
│   └── cli.py          # Command-line interface with Click & Rich
├── tests/              # Comprehensive test suite (94 tests, 100% pass rate)
│   ├── test_config.py
│   ├── test_converter.py
│   ├── test_chunker.py
│   ├── test_optimizer.py
│   ├── test_cli.py
│   └── test_utils.py
├── examples/           # Practical usage examples
│   ├── basic_conversion.py      # Simple conversion
│   ├── batch_with_chunks.py     # Batch processing with chunking
│   ├── custom_pipeline.py       # Advanced pipeline with optimizer
│   ├── partial_parsing.py       # Partial document parsing
│   └── README.md                # Examples documentation
├── documents/          # Sample documents for testing
├── output/             # Default output directory
│   ├── markdown/       # Converted markdown files
│   ├── figures/        # Extracted figures (organized by document)
│   └── chunks/         # Chunked documents (JSONL/JSON/CSV)
└── main.py             # CLI entry point
```

### Key Components

**TechDocConverter** (`src/converter.py`):
- Docling integration with GPU acceleration (CUDA/MPS/CPU auto-detection)
- Partial parsing support (page_range, max_pages)
- Multi-format support (PDF, DOCX, PPTX, XLSX, HTML, MD, TXT)
- Figure extraction (images, tables, equations)
- Error handling and logging

**DocumentChunker** (`src/chunker.py`):
- HybridChunker integration for structure-aware chunking
- Token counting with transformers models
- Contextualized content generation for RAG
- Multiple export formats (JSONL, JSON, Markdown, CSV)

**DocumentOptimizer** (`src/optimizer.py`):
- Document analysis (file size, page count, language detection)
- OCR needs detection (RapidOCR quality assessment)
- Table detection (layout analysis)
- Automatic configuration optimization
- Processing time estimation

**CLI** (`src/cli.py`):
- Convert command (single document)
- Batch command (directory processing)
- Rich progress bars and formatting
- Flexible logging options

## Development

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_converter.py -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/

# Type checking
uv run mypy src/
```

## Performance

### Full Document Conversion

Benchmark on NASA technical paper (PDF, 20 pages, aerospace engineering content):

| Configuration | Time | Speed | GPU Usage |
|---------------|------|-------|-----------|
| GPU (CUDA) | 132s | 6.6s/page | 85% |
| CPU only | ~440s | ~22s/page | 0% |

### Partial Document Parsing

Performance comparison on the same 20-page document:

| Pages | Method | Time | Speed | Speedup |
|-------|--------|------|-------|---------|
| 20 | Full conversion | 132s | 6.6s/page | 1x |
| 5 | Partial (1-5) | 35s | 7.0s/page | 3.8x faster |
| 6 | Partial (10-15) | 40s | 6.7s/page | 3.3x faster |
| 1 | Single page | 4s | 4.0s/page | 33x faster |

**Key Insights:**
- Partial parsing provides near-linear time reduction
- Single-page conversion ideal for quick previews (~4 seconds)
- 5-page conversion completes in ~35 seconds (perfect for testing)
- GPU acceleration provides ~3.3x speedup over CPU
- Per-page processing time consistent across ranges (~6-7s/page)

## API Reference

### TechDocConverter

Main converter class for document processing.

#### `convert(file_path, page_range=None, max_pages=None) -> ConversionResult`

Convert a document to markdown format.

**Parameters:**
- `file_path` (Path): Path to the document file
- `page_range` (tuple[int, int] | None): Optional tuple (start_page, end_page) for partial conversion. Pages are 1-indexed. Example: `(1, 5)` converts pages 1-5.
- `max_pages` (int | None): Optional maximum number of pages to convert from start. Example: `max_pages=10` converts first 10 pages.

**Returns:**
- `ConversionResult`: Object containing markdown, DoclingDocument, metadata, and success status

**Example:**
```python
# Full document
result = converter.convert(Path("doc.pdf"))

# Pages 5-10
result = converter.convert(Path("doc.pdf"), page_range=(5, 10))

# First 5 pages
result = converter.convert(Path("doc.pdf"), max_pages=5)
```

#### `convert_and_save(file_path) -> ConversionResult`

Convert document and automatically save markdown and figures.

#### `save_markdown(result, output_path=None) -> Path`

Save markdown from conversion result to file.

#### `extract_figures(result) -> Path`

Extract figures from DoclingDocument to directory.

### DocumentChunker

Smart chunking for RAG pipelines.

#### `chunk_document(docling_doc, title=None, source=None, metadata=None) -> list[Chunk]`

Create token-aware chunks from DoclingDocument.

#### `export_chunks(chunks, output_path, format="jsonl")`

Export chunks in specified format (jsonl, json, md, csv).

### DocumentOptimizer

Analyze documents and optimize processing settings.

#### `analyze(file_path) -> DocumentAnalysis`

Analyze document characteristics (OCR needs, tables, language, size).

#### `optimize_converter_config(file_path) -> dict`

Get optimized ConverterConfig parameters based on document analysis.

## Troubleshooting

### GPU Not Detected

If GPU acceleration isn't working:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode
config = ConverterConfig(use_gpu=False)
```

### Out of Memory

For large documents:

```python
# Use partial parsing to process in sections
result1 = converter.convert(Path("large.pdf"), page_range=(1, 50))
result2 = converter.convert(Path("large.pdf"), page_range=(51, 100))

# Or disable GPU and reduce threads
config = ConverterConfig(
    use_gpu=False,
    num_threads=2,
    save_figures=False
)
```

### Slow Processing

If conversion is taking too long:

```python
# 1. Use partial parsing for preview
preview = converter.convert(Path("doc.pdf"), page_range=(1, 1))

# 2. Disable figure extraction
config = ConverterConfig(save_figures=False)

# 3. Use document optimizer to auto-configure
optimizer = DocumentOptimizer(OptimizerConfig())
analysis = optimizer.analyze(Path("doc.pdf"))
print(f"Estimated time: {analysis.estimated_time:.1f}s")
```

### OCR Quality Issues

If OCR results are poor:

```python
# Ensure OCR is enabled for scanned documents
config = ConverterConfig(
    do_ocr=True,
    use_gpu=True  # GPU accelerates OCR significantly
)
```

### Figure Extraction Issues

Known issue with Docling API changes - figures may not extract in some versions:

```python
# Workaround: Disable figure extraction if it causes errors
config = ConverterConfig(save_figures=False)
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass (`uv run pytest`)
5. Submit a pull request

## License

[Add your license here]

## Acknowledgments

- Built with [Docling](https://github.com/DS4SD/docling) by IBM Research
- Uses [transformers](https://huggingface.co/transformers/) for token counting
- CLI powered by [Click](https://click.palletsprojects.com/) and [Rich](https://rich.readthedocs.io/)

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{tech_doc_scanner,
  title={Technical Document Conversion Agent},
  author={Your Name},
  year={2025},
  url={https://github.com/e-krane/tech-doc-scanner}
}
```
