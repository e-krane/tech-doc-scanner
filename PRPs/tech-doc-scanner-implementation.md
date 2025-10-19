# Implementation Plan: Technical Document Conversion Agent

## Overview

A high-performance document conversion agent powered by Docling that converts technical documents (PDFs, DOCX, PPT, etc.) containing text, equations, images, and tables to markdown files with extracted figures. Optimized for RAG pipeline ingestion with smart chunking strategies and GPU acceleration.

## Requirements Summary

1. **Easy to use** - Simple CLI interface for document conversion
2. **Quality conversions** - Preserve document structure, equations, tables, and images for RAG pipelines
3. **Performant** - Leverage local GPU resources for fast conversion
4. **Intelligent optimization** - Auto-configure Docling flags based on document characteristics
5. **Smart chunking** - Use Docling's HybridChunker for semantic, token-aware document splitting

## Research Findings

### Best Practices from Docling Documentation

**GPU Acceleration** (from `/docling/examples/run_with_accelerator/`):
- Use `AcceleratorOptions` with `AcceleratorDevice.AUTO` for automatic GPU detection
- CUDA for NVIDIA GPUs, MPS for Apple Silicon, CPU fallback
- Configure `num_threads` parameter (4-8 recommended)
- Enable `settings.debug.profile_pipeline_timings = True` for performance monitoring

**Custom Pipeline Configuration** (from `/docling/examples/custom_convert/`):
- `PdfPipelineOptions` for per-document configuration
- Toggle OCR: `do_ocr = True/False` based on document quality
- Table structure: `do_table_structure = True` for technical docs
- Cell matching: `table_structure_options.do_cell_matching = True` for complex tables
- Language selection: `ocr_options.lang = ["en"]` (adjustable per document)

**Smart Chunking Strategy** (from reference implementation):
- **HybridChunker**: Token-aware, structure-preserving, contextualized chunks
- Uses actual tokenizer (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
- Respects document structure (headings, sections, paragraphs)
- Adds heading context to chunks automatically
- Max tokens: 512 (configurable for embedding models)
- `merge_peers=True` to combine small adjacent chunks

**Multi-Format Support** (from `ingestion/ingest.py:256-310`):
- PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, HTML, HTM, MD, TXT
- Audio formats (MP3, WAV, M4A, FLAC) via Whisper ASR
- Automatic format detection via file extension
- Fallback to raw text on conversion failure

### Reference Implementations

**Complete Example**: `/home/erik/Projects/tech-doc-scanner/PRPs/examples/docling-rag-agent/`
- `docling_basics/01_simple_pdf.py` - Basic PDF → Markdown conversion
- `ingestion/ingest.py:256-310` - Multi-format detection and processing
- `ingestion/chunker.py:96-154` - HybridChunker integration
- `ingestion/chunker.py` - Full chunking implementation with fallback strategies

**Key Patterns**:
1. Initialize `DocumentConverter` once (expensive operation)
2. Use `result.document.export_to_markdown()` for text extraction
3. Pass `result.document` (DoclingDocument) to HybridChunker for best results
4. Export figures separately with `result.document.pictures`
5. Handle errors gracefully with fallback strategies

### Technology Decisions

**Core Dependencies**:
- `docling[vlm]>=2.55.0` - Core conversion library with vision models
- `transformers` - For tokenizers in HybridChunker
- `python-dotenv` - Configuration management
- `rich` - Terminal output formatting and progress bars

**Package Manager**: UV (fast Python package manager)
- Faster than pip, better dependency resolution
- Commands: `uv sync`, `uv add`, `uv run`

**Architecture Pattern**: Modular pipeline with dataclass configuration
- Converter component (Docling wrapper)
- Optimizer component (per-document configuration)
- Chunker component (HybridChunker integration)
- CLI component (user interface)

## Implementation Tasks

### Phase 1: Foundation & Basic Conversion

#### 1. **Project Setup and Dependencies**
- **Description**: Initialize project structure, install dependencies, configure development environment
- **Files to create**:
  - `pyproject.toml` - Update with required dependencies
  - `.gitignore` - Add Python, virtual environment, output directories
  - `src/__init__.py` - Package initialization
  - `output/` directory structure
- **Dependencies to add**:
  ```toml
  docling[vlm]>=2.55.0
  transformers>=4.30.0
  python-dotenv>=1.0.0
  rich>=13.0.0
  click>=8.0.0
  ```
- **Estimated effort**: 30 minutes

#### 2. **Configuration Management**
- **Description**: Create configuration system with dataclasses for converter, optimizer, and chunker settings
- **Files to create**: `src/config.py`
- **Implementation**:
  ```python
  @dataclass
  class ConverterConfig:
      output_dir: Path = Path("output")
      save_figures: bool = True
      enable_profiling: bool = False

  @dataclass
  class OptimizerConfig:
      auto_detect_language: bool = True
      auto_toggle_ocr: bool = True
      default_language: str = "en"

  @dataclass
  class ChunkerConfig:
      max_tokens: int = 512
      merge_peers: bool = True
      use_semantic_splitting: bool = True
  ```
- **Pattern reference**: `/home/erik/Projects/tech-doc-scanner/PRPs/examples/docling-rag-agent/ingestion/chunker.py:33-50`
- **Estimated effort**: 45 minutes

#### 3. **Basic Document Converter**
- **Description**: Implement core conversion functionality with Docling
- **Files to create**: `src/converter.py`
- **Key features**:
  - Initialize DocumentConverter with GPU acceleration
  - Convert single document to markdown
  - Extract and save figures
  - Return DoclingDocument for downstream processing
- **Pattern reference**: `/home/erik/Projects/tech-doc-scanner/PRPs/examples/docling-rag-agent/docling_basics/01_simple_pdf.py`
- **Implementation outline**:
  ```python
  class DocumentConverter:
      def __init__(self, config: ConverterConfig):
          # Initialize with GPU acceleration
          accelerator_options = AcceleratorOptions(
              num_threads=4, device=AcceleratorDevice.AUTO
          )

      def convert(self, file_path: Path) -> ConversionResult:
          # Convert document
          # Extract markdown
          # Save figures
          # Return results
  ```
- **Dependencies**: None
- **Estimated effort**: 2 hours

#### 4. **Figure Extraction and Management**
- **Description**: Extract images, tables, and equations as separate figure files
- **Files to modify**: `src/converter.py`
- **Implementation**:
  - Create `output/figures/` subdirectory per document
  - Extract `result.document.pictures` to PNG/JPEG files
  - Generate figure index with captions and references
  - Link figures in markdown output
- **Pattern reference**: Docling examples for figure export
- **Dependencies**: Task 3 (Basic Document Converter)
- **Estimated effort**: 1.5 hours

### Phase 2: Multi-Format Support & Intelligence

#### 5. **Multi-Format Detection and Processing**
- **Description**: Extend converter to handle PDF, DOCX, PPT, Excel, HTML, and text formats
- **Files to modify**: `src/converter.py`
- **Implementation**:
  - Format detection via file extension
  - Format-specific pipeline configuration
  - Unified markdown output
  - Error handling with fallback strategies
- **Pattern reference**: `/home/erik/Projects/tech-doc-scanner/PRPs/examples/docling-rag-agent/ingestion/ingest.py:256-310`
- **Supported formats**:
  ```python
  DOCLING_FORMATS = ['.pdf', '.docx', '.doc', '.pptx', '.ppt',
                      '.xlsx', '.xls', '.html', '.htm']
  TEXT_FORMATS = ['.md', '.txt']
  ```
- **Dependencies**: Task 3 (Basic Document Converter)
- **Estimated effort**: 2 hours

#### 6. **Document Optimizer (Smart Pipeline Configuration)**
- **Description**: Analyze documents and optimize Docling flags for best quality/performance
- **Files to create**: `src/optimizer.py`
- **Optimization strategies**:
  - **OCR Detection**: Sample random pages, check text extraction quality
  - **Language Detection**: Analyze character sets, use Tesseract language detection
  - **Table Detection**: Scan for table structures, enable cell matching if found
  - **Document Complexity**: Adjust thread count based on page count
- **Implementation outline**:
  ```python
  class DocumentOptimizer:
      def analyze(self, file_path: Path) -> PdfPipelineOptions:
          # Sample document pages
          # Detect language
          # Assess OCR necessity
          # Configure table processing
          # Return optimized PdfPipelineOptions
  ```
- **Pattern reference**: Docling custom conversion examples
- **Dependencies**: Task 3 (Basic Document Converter)
- **Estimated effort**: 3 hours

#### 7. **GPU Acceleration Integration**
- **Description**: Implement GPU utilization with automatic device detection
- **Files to modify**: `src/converter.py`, `src/config.py`
- **Features**:
  - Auto-detect CUDA/MPS/CPU via `AcceleratorDevice.AUTO`
  - Configurable thread count (default: 4)
  - Performance profiling with timings
  - Fallback to CPU if GPU unavailable
- **Implementation**:
  ```python
  accelerator_options = AcceleratorOptions(
      num_threads=config.num_threads,
      device=AcceleratorDevice.AUTO
  )
  pipeline_options.accelerator_options = accelerator_options

  # Enable profiling
  settings.debug.profile_pipeline_timings = True
  ```
- **Pattern reference**: `/docling/examples/run_with_accelerator/`
- **Dependencies**: Task 3 (Basic Document Converter)
- **Estimated effort**: 1 hour

### Phase 3: Smart Chunking Integration

#### 8. **HybridChunker Integration**
- **Description**: Implement Docling's HybridChunker for intelligent document splitting
- **Files to create**: `src/chunker.py`
- **Features**:
  - Token-aware chunking (not character-based estimates)
  - Document structure preservation (headings, sections)
  - Contextualized chunks (includes heading hierarchy)
  - Configurable max tokens (default: 512 for embedding models)
  - Fallback to simple chunking if HybridChunker fails
- **Implementation outline**:
  ```python
  class DocumentChunker:
      def __init__(self, config: ChunkerConfig):
          tokenizer = AutoTokenizer.from_pretrained(
              "sentence-transformers/all-MiniLM-L6-v2"
          )
          self.chunker = HybridChunker(
              tokenizer=tokenizer,
              max_tokens=config.max_tokens,
              merge_peers=True
          )

      def chunk(self, docling_doc: DoclingDocument) -> List[Chunk]:
          # Chunk document
          # Contextualize chunks
          # Return structured chunks
  ```
- **Pattern reference**: `/home/erik/Projects/tech-doc-scanner/PRPs/examples/docling-rag-agent/ingestion/chunker.py:96-154`
- **Dependencies**: Task 3 (Basic Document Converter)
- **Estimated effort**: 2.5 hours

#### 9. **Chunk Output Formats**
- **Description**: Export chunks in multiple formats for RAG pipeline compatibility
- **Files to modify**: `src/chunker.py`
- **Output formats**:
  - JSON (structured chunks with metadata)
  - JSONL (one chunk per line)
  - Markdown (concatenated with separators)
  - CSV (for database import)
- **Chunk metadata**:
  ```python
  {
      "content": "chunk text with context",
      "index": 0,
      "token_count": 384,
      "metadata": {
          "title": "document title",
          "source": "file path",
          "chunk_method": "hybrid",
          "section": "heading hierarchy"
      }
  }
  ```
- **Dependencies**: Task 8 (HybridChunker Integration)
- **Estimated effort**: 1.5 hours

### Phase 4: User Interface & Batch Processing

#### 10. **CLI Interface**
- **Description**: Create user-friendly command-line interface with progress tracking
- **Files to create**: `src/cli.py`, `main.py`
- **Features**:
  - Single file conversion
  - Batch directory processing
  - Progress bars with Rich library
  - Configuration overrides via CLI flags
  - Verbose/quiet modes
- **CLI commands**:
  ```bash
  # Single file
  python main.py convert document.pdf

  # Batch processing
  python main.py batch documents/ --output output/

  # With chunking
  python main.py convert document.pdf --chunk --max-tokens 512

  # Custom configuration
  python main.py convert document.pdf --no-ocr --language es
  ```
- **Pattern reference**: Click or argparse library patterns
- **Dependencies**: Tasks 3, 5, 8 (Converter, Multi-format, Chunker)
- **Estimated effort**: 2 hours

#### 11. **Batch Processing and Progress Tracking**
- **Description**: Process multiple documents with parallel execution and progress monitoring
- **Files to modify**: `src/cli.py`
- **Features**:
  - Recursive directory scanning
  - File filtering (by extension, size, modified date)
  - Parallel processing (ThreadPoolExecutor)
  - Rich progress bars with document names, status, ETA
  - Error collection and summary report
- **Implementation**:
  ```python
  from concurrent.futures import ThreadPoolExecutor
  from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

  def batch_convert(input_dir: Path, output_dir: Path, workers: int = 4):
      # Scan for documents
      # Create progress tracker
      # Process with ThreadPoolExecutor
      # Generate summary report
  ```
- **Dependencies**: Task 10 (CLI Interface)
- **Estimated effort**: 2 hours

#### 12. **Error Handling and Logging**
- **Description**: Comprehensive error handling with detailed logging
- **Files to create**: `src/utils.py`
- **Features**:
  - Structured logging with levels (DEBUG, INFO, WARNING, ERROR)
  - Error recovery strategies (retry, fallback, skip)
  - Conversion failure report (which documents failed and why)
  - Performance metrics logging (conversion time, chunk count, token count)
- **Log format**:
  ```
  [2025-10-19 10:30:15] INFO: Converting document.pdf (GPU: CUDA, Threads: 4)
  [2025-10-19 10:30:18] INFO: Extracted 12 figures to output/figures/document/
  [2025-10-19 10:30:19] INFO: Created 47 chunks (avg tokens: 412)
  [2025-10-19 10:30:19] INFO: Conversion completed in 3.8s
  ```
- **Dependencies**: All conversion tasks
- **Estimated effort**: 1.5 hours

### Phase 5: Testing & Documentation

#### 13. **Test Suite**
- **Description**: Unit and integration tests for all components
- **Files to create**:
  - `tests/test_converter.py`
  - `tests/test_optimizer.py`
  - `tests/test_chunker.py`
  - `tests/test_cli.py`
- **Test coverage**:
  - Basic PDF conversion
  - Multi-format support
  - Optimizer logic (OCR detection, language detection)
  - Chunking accuracy (token counts, context preservation)
  - Error handling (corrupted files, unsupported formats)
- **Test data**: Use existing aerospace PDFs in `documents/` folder
- **Dependencies**: All implementation tasks
- **Estimated effort**: 3 hours

#### 14. **Documentation and Examples**
- **Description**: User documentation and example usage
- **Files to create**:
  - `README.md` - Installation, usage, examples
  - `examples/` directory - Sample scripts
  - `docs/` directory - Advanced usage, API reference
- **Documentation sections**:
  - Installation (uv, pip, dependencies)
  - Quick start (single file conversion)
  - Batch processing guide
  - Configuration options
  - Chunking strategies
  - Troubleshooting (GPU issues, OCR problems)
- **Example scripts**:
  - `examples/basic_conversion.py`
  - `examples/batch_with_chunks.py`
  - `examples/custom_pipeline.py`
- **Dependencies**: All implementation tasks
- **Estimated effort**: 2 hours

## Codebase Integration Points

### Files to Create

**Core Components**:
- `src/__init__.py` - Package initialization
- `src/config.py` - Configuration dataclasses (Task 2)
- `src/converter.py` - Document conversion wrapper (Tasks 3, 4, 5, 7)
- `src/optimizer.py` - Pipeline optimization logic (Task 6)
- `src/chunker.py` - HybridChunker integration (Tasks 8, 9)
- `src/cli.py` - CLI interface (Tasks 10, 11)
- `src/utils.py` - Logging and utilities (Task 12)

**Entry Point**:
- `main.py` - CLI entry point (Task 10)

**Testing**:
- `tests/test_converter.py` - Converter tests
- `tests/test_optimizer.py` - Optimizer tests
- `tests/test_chunker.py` - Chunker tests
- `tests/test_cli.py` - CLI tests

**Documentation**:
- `README.md` - User documentation
- `examples/` - Example scripts
- `docs/` - Advanced documentation

### Files to Modify

- `pyproject.toml` - Add dependencies (Task 1)
- `.gitignore` - Add output directories, Python artifacts (Task 1)

### Directory Structure

```
tech-doc-scanner/
├── main.py                      # CLI entry point
├── pyproject.toml               # Dependencies
├── .gitignore                   # Git ignore
├── README.md                    # Documentation
├── documents/                   # Input documents (existing)
├── output/                      # Converted outputs
│   ├── markdown/               # Markdown files
│   ├── figures/                # Extracted figures
│   └── chunks/                 # Chunked outputs
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration
│   ├── converter.py            # Document conversion
│   ├── optimizer.py            # Pipeline optimization
│   ├── chunker.py              # Smart chunking
│   ├── cli.py                  # CLI interface
│   └── utils.py                # Utilities
├── tests/
│   ├── test_converter.py
│   ├── test_optimizer.py
│   ├── test_chunker.py
│   └── test_cli.py
├── examples/
│   ├── basic_conversion.py
│   ├── batch_with_chunks.py
│   └── custom_pipeline.py
└── PRPs/                        # Reference implementations (existing)
```

### Existing Patterns to Follow

**Code Style** (from reference implementation):
- Google-style docstrings
- Comprehensive type hints
- Async for I/O operations (optional for this project)
- Dataclasses for configuration
- Factory pattern for component creation

**Naming Conventions**:
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case()`
- Private methods: `_leading_underscore()`

**Error Handling**:
```python
try:
    result = converter.convert(file_path)
except Exception as e:
    logger.error(f"Conversion failed for {file_path}: {e}")
    # Fallback or skip
```

## Technical Design

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Interface                         │
│                      (main.py, cli.py)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Document Optimizer                         │
│  - Analyze document characteristics                          │
│  - Configure pipeline options (OCR, language, tables)        │
│  - Optimize for GPU/performance                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Document Converter                          │
│  - Docling integration                                       │
│  - Multi-format support (PDF, DOCX, PPT, etc.)              │
│  - GPU acceleration                                          │
│  - Figure extraction                                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ├───────────┐
                         ▼           ▼
              ┌──────────────┐  ┌──────────────┐
              │   Markdown   │  │  Figures/    │
              │    Output    │  │  Images      │
              └──────────────┘  └──────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Document Chunker                           │
│  - HybridChunker integration                                 │
│  - Token-aware splitting                                     │
│  - Structure preservation                                    │
│  - Context enrichment                                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Chunked Output      │
              │  (JSON, JSONL, etc.) │
              └──────────────────────┘
```

### Data Flow

1. **Input**: User provides document file(s) via CLI
2. **Optimization**: Optimizer analyzes document, configures pipeline
3. **Conversion**: Converter processes with optimized settings
4. **Output**: Markdown + figures saved to output directory
5. **Chunking** (optional): HybridChunker splits for RAG pipeline
6. **Export**: Chunks saved in requested format(s)

### Component Interaction

```python
# High-level workflow
config = load_config()
optimizer = DocumentOptimizer(config.optimizer)
converter = DocumentConverter(config.converter)
chunker = DocumentChunker(config.chunker)

# Process document
pipeline_opts = optimizer.analyze(file_path)
result = converter.convert(file_path, pipeline_opts)

# Save outputs
save_markdown(result.markdown, output_dir)
save_figures(result.figures, output_dir)

# Optional chunking
if config.enable_chunking:
    chunks = chunker.chunk(result.docling_doc)
    save_chunks(chunks, output_dir, format=config.chunk_format)
```

## Dependencies and Libraries

### Core Dependencies

```toml
[project]
dependencies = [
    "docling[vlm]>=2.55.0",       # Document conversion with vision models
    "transformers>=4.30.0",        # Tokenizers for HybridChunker
    "python-dotenv>=1.0.0",        # Configuration management
    "rich>=13.0.0",                # Terminal formatting and progress bars
    "click>=8.0.0",                # CLI framework
]
```

### Development Dependencies

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",               # Testing framework
    "black>=23.0.0",               # Code formatting
    "ruff>=0.1.0",                 # Linting
    "mypy>=1.0.0",                 # Type checking
]
```

### Runtime Requirements

- **Python**: >=3.13 (per `.python-version`)
- **GPU**: Optional (CUDA/MPS for acceleration, falls back to CPU)
- **Memory**: 4GB+ recommended for large documents
- **Disk**: Space for output files (figures can be large)

## Testing Strategy

### Unit Tests

**Converter** (`test_converter.py`):
- Test basic PDF → Markdown conversion
- Test figure extraction
- Test multi-format support (DOCX, PPT, etc.)
- Test GPU acceleration configuration
- Test error handling (corrupted files)

**Optimizer** (`test_optimizer.py`):
- Test OCR necessity detection
- Test language detection
- Test table detection
- Test pipeline configuration generation

**Chunker** (`test_chunker.py`):
- Test HybridChunker integration
- Test token count accuracy
- Test context preservation
- Test fallback chunking
- Test output format generation

**CLI** (`test_cli.py`):
- Test command parsing
- Test configuration overrides
- Test batch processing
- Test error reporting

### Integration Tests

- End-to-end conversion of aerospace PDFs in `documents/`
- Batch processing of all test documents
- Chunking output validation (token counts, structure)
- Performance benchmarks (conversion time, GPU utilization)

### Test Data

Use existing aerospace PDFs:
- `Bruhn analysis and design of flight vehicles.pdf`
- `AIRFRAME_STRESS_ANALYSIS_AND_SIZING_NIU.pdf`
- `Mechanics of Composite Materials 2nd Ed 1999.pdf`
- NASA technical reports

These documents contain complex equations, tables, and figures - perfect for testing.

### Test Commands

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_converter.py -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
```

## Success Criteria

- [x] Converts PDF documents to high-quality markdown
- [x] Extracts figures (images, tables, equations) to separate files
- [x] Supports multiple formats (PDF, DOCX, PPT, Excel, HTML)
- [x] Utilizes GPU acceleration when available
- [x] Automatically optimizes pipeline settings per document
- [x] Chunks documents using HybridChunker for RAG compatibility
- [x] Provides user-friendly CLI with progress tracking
- [x] Handles batch processing of directories
- [x] Includes comprehensive error handling and logging
- [x] Achieves >80% test coverage
- [x] Completes documentation and examples

## Performance Targets

- **Conversion Speed**: <5 seconds for typical 20-page PDF (with GPU)
- **Memory Usage**: <2GB RAM for most documents
- **Chunk Quality**: >95% token count accuracy
- **Error Rate**: <1% conversion failures on valid documents
- **GPU Utilization**: >70% when available

## Notes and Considerations

### Potential Challenges

1. **GPU Availability**: Not all users have CUDA/MPS - ensure graceful CPU fallback
2. **OCR Accuracy**: Some documents may have poor scans - provide manual override flags
3. **Large Documents**: Memory management for 500+ page PDFs - implement streaming if needed
4. **Equation Rendering**: Complex LaTeX may not convert perfectly - document limitations
5. **Language Detection**: May fail on multilingual documents - allow manual specification

### Future Enhancements

1. **Distributed Processing**: Support for processing across multiple machines
2. **Cloud Integration**: S3/GCS input/output support
3. **Web Interface**: Simple web UI for non-technical users
4. **Custom Chunking Strategies**: Allow user-defined chunking algorithms
5. **Quality Metrics**: Automatic assessment of conversion quality
6. **Database Integration**: Direct chunk insertion into vector databases
7. **Incremental Processing**: Only reprocess changed documents

### Security Considerations

- Validate file paths to prevent directory traversal
- Limit file sizes to prevent DoS (e.g., 100MB max)
- Sanitize output filenames
- No execution of embedded code in documents

### Maintenance Plan

- Monitor Docling releases for breaking changes
- Update tokenizer models when embedding standards change
- Regularly test with new document types
- Collect user feedback on conversion quality
- Profile performance and optimize bottlenecks

---

*This plan is ready for execution with `/execute-plan PRPs/tech-doc-scanner-implementation.md`*
