# Document Conversion Application - Planning Document

## Project Overview

A simple yet efficient document conversion application that uses IBM's Docling library to convert technical PDF documents to Markdown format with intelligent, structure-aware chunking for downstream processing (e.g., RAG applications, knowledge bases).

## Technology Stack

### Core Technologies

1. **Docling** (v2.x+)
   - IBM's open-source document processing library
   - Primary conversion engine for PDF to Markdown
   - Built-in support for smart chunking strategies
   - MIT licensed

2. **Python 3.11+**
   - Required for Docling compatibility
   - Modern async/await support for potential batch processing

3. **PyTorch**
   - Dependency for Docling's AI models
   - Supports CPU, CUDA (NVIDIA GPUs), and MPS (Apple Silicon)

### Key Docling Components

1. **DocumentConverter**
   - Main conversion interface
   - Configurable pipeline options
   - Support for batch processing

2. **HybridChunker**
   - Combines hierarchical (document-structure-based) and tokenization-aware chunking
   - Recommended for technical documents
   - Maintains semantic coherence and context

3. **Pipeline Options**
   - `PdfPipelineOptions` for PDF-specific configuration
   - `AcceleratorOptions` for hardware optimization

## Architecture Design

### High-Level Flow

```
Input PDF(s) → DocumentConverter → DoclingDocument → HybridChunker → Markdown Chunks → Output
```

### Core Components

1. **Input Handler**
   - File path validation
   - Batch file discovery
   - Support for single file or directory processing

2. **Converter Service**
   - Docling DocumentConverter initialization
   - Configuration management
   - Conversion orchestration

3. **Chunking Service**
   - HybridChunker configuration
   - Tokenizer alignment (for RAG readiness)
   - Metadata preservation

4. **Output Manager**
   - Markdown file generation
   - Chunk metadata serialization
   - File organization

## Configuration Strategy

### Optimal Docling Flags for Technical Documents

Based on research, the following configuration provides the best balance of quality and performance:

#### PdfPipelineOptions

```python
pipeline_options = PdfPipelineOptions(
    # Core Features
    do_table_structure=True,          # Critical for technical docs
    do_ocr=False,                     # Only enable if scanned PDFs
    
    # Performance
    do_code_enrichment=False,         # Enable if code blocks important
    do_formula_enrichment=False,      # Enable for math-heavy documents
    do_picture_description=False,     # Disable for performance
    do_picture_classification=False,  # Disable for performance
    
    # Table Processing
    table_structure_options=TableStructureOptions(
        do_cell_matching=True,        # Better table quality
        mode=TableFormerMode.ACCURATE # vs FAST - choose based on needs
    ),
    
    # Backend Selection
    # Note: dlparse_v2 is recommended (default), pypdfium2 for speed but lower quality
)
```
These flags should be set based on a precursor analysis of the document.

#### AcceleratorOptions

```python
accelerator_options = AcceleratorOptions(
    device=AcceleratorDevice.AUTO,    # Auto-detect GPU/MPS/CPU
    num_threads=8,                     # Adjust based on CPU cores
)
```

### Performance Optimization

1. **For Speed Priority**
   - `do_table_structure=True` with `mode=TableFormerMode.FAST`
   - `do_ocr=False` (unless needed)
   - `do_cell_matching=False` (if table accuracy not critical)
   - Use GPU/MPS acceleration when available

2. **For Quality Priority**
   - `do_table_structure=True` with `mode=TableFormerMode.ACCURATE`
   - `do_ocr=True` (if scanned PDFs)
   - `do_cell_matching=True`
   - Enable formula/code enrichment if relevant

3. **Benchmarks (from Docling Technical Report)**
   - CPU-only: ~0.49 sec/page
   - With CUDA GPU: ~0.49 sec/page (current optimization)
   - With MPS (Apple Silicon): ~2 min for 10 papers

### Chunking Configuration

#### HybridChunker Setup

```python
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

# Align with your embedding model
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 512  # Typical for many embedding models

tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
    max_tokens=MAX_TOKENS,
)

chunker = HybridChunker(
    tokenizer=tokenizer,
    merge_peers=True,  # Merge small chunks with same context
)
```

#### Chunking Strategy Benefits

1. **Document-Aware Splitting**
   - Respects sections, paragraphs, lists
   - Preserves document hierarchy
   - Maintains table integrity

2. **Token-Aware Refinement**
   - Splits oversized chunks
   - Merges undersized chunks
   - Aligns with embedding model tokenizer

3. **Metadata Preservation**
   - Headings/captions attached to chunks
   - Page information retained
   - Document structure metadata

## Deployment Considerations

### Environment Setup

1. **Virtual Environment**
   - Isolated Python environment
   - Specific PyTorch version for accelerator support

2. **Model Caching**
   - Models auto-download on first use
   - Cache location: HuggingFace cache directory
   - Pre-download option: `docling-tools download --all`

3. **Hardware Requirements**
   - Minimum: 8GB RAM, 4 CPU cores
   - Recommended: 16GB RAM, 8 CPU cores, GPU/MPS
   - Storage: ~2-3GB for models

### Scalability

1. **Single File Processing**
   - Simple CLI interface
   - Direct file input/output

2. **Batch Processing**
   - Directory traversal
   - Parallel processing potential (ThreadedPdfPipelineOptions)
   - Progress tracking

3. **Future Extensions**
   - Web API wrapper
   - Queue-based processing
   - Database integration for metadata

## Output Format

### Markdown Files

- One markdown file per input PDF
- Naming convention: `{original_name}.md`

### Chunk Files

- JSON or individual markdown files per chunk
- Metadata included:
  - Source document
  - Page numbers
  - Headings/captions
  - Token count
  - Chunk index

### Example Structure

```
output/
├── document1.md              # Full markdown
├── document1_chunks/
│   ├── chunk_000.md
│   ├── chunk_001.md
│   └── metadata.json
└── document2.md
```

## Error Handling

1. **File Access Errors**
   - Invalid paths
   - Permission issues
   - Corrupt PDFs

2. **Conversion Errors**
   - Timeout handling (`document_timeout` option)
   - Partial conversion recovery
   - Logging strategy

3. **Resource Constraints**
   - Memory management
   - Batch size adjustment
   - Graceful degradation

## Testing Strategy

1. **Unit Tests**
   - Configuration validation
   - File I/O operations
   - Chunk generation logic

2. **Integration Tests**
   - End-to-end conversion
   - Various PDF types (text-based, scanned, tables, formulas)
   - Performance benchmarks

3. **Test Documents**
   - Simple text PDFs
   - Complex layouts with tables
   - Multi-column technical papers
   - Scanned documents

## Success Metrics

1. **Conversion Quality**
   - Table structure preservation
   - Reading order accuracy
   - Metadata extraction completeness

2. **Performance**
   - Pages per second
   - Memory usage
   - Chunk generation speed

3. **Usability**
   - Simple CLI interface
   - Clear error messages
   - Comprehensive logging

## Future Enhancements

1. **Additional Format Support**
   - DOCX, PPTX (Docling supports these)
   - HTML output option

2. **Advanced Chunking**
   - Custom chunking strategies
   - Semantic chunking experiments

3. **RAG Integration**
   - Direct vector database integration
   - Embedding generation
   - Retrieval testing utilities

4. **Performance Optimization**
   - Streaming processing for large files
   - Distributed processing support
   - Advanced GPU utilization
