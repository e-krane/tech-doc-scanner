# Examples

This directory contains practical examples demonstrating how to use the Technical Document Conversion Agent.

## Available Examples

### 1. Basic Conversion (`basic_conversion.py`)

Demonstrates the simplest way to convert a document to markdown.

**Features:**
- Single document conversion
- Figure extraction
- Basic configuration

**Usage:**
```bash
python examples/basic_conversion.py
```

**What it does:**
1. Loads a PDF document
2. Converts it to markdown
3. Extracts figures
4. Saves outputs to the output directory

### 2. Batch with Chunks (`batch_with_chunks.py`)

Shows how to process multiple documents and create RAG-optimized chunks.

**Features:**
- Batch processing of multiple PDFs
- Smart chunking with HybridChunker
- JSONL export for vector databases

**Usage:**
```bash
python examples/batch_with_chunks.py
```

**What it does:**
1. Finds all PDFs in the documents/ directory
2. Converts each to markdown
3. Creates token-aware chunks
4. Exports chunks in JSONL format

### 3. Custom Pipeline (`custom_pipeline.py`)

Demonstrates advanced usage with automatic optimization.

**Features:**
- Document analysis and optimization
- Custom processing pipeline
- Multiple output formats
- Performance comparison

**Usage:**
```bash
python examples/custom_pipeline.py
```

**What it does:**
1. Analyzes document characteristics (OCR needs, tables, language)
2. Automatically configures optimal settings
3. Converts with optimized configuration
4. Creates chunks for RAG
5. Exports in multiple formats (MD, JSONL, figures)

## Getting Started

### Prerequisites

Make sure you have installed the project dependencies:

```bash
uv sync
source .venv/bin/activate
```

### Prepare Your Documents

1. Create a `documents/` directory in the project root:
```bash
mkdir -p documents
```

2. Add your PDF files to the documents/ directory

3. Update the file paths in the examples if needed

### Run an Example

```bash
# Run basic conversion
python examples/basic_conversion.py

# Run batch processing with chunking
python examples/batch_with_chunks.py

# Run custom pipeline with optimization
python examples/custom_pipeline.py
```

## Example Outputs

### Markdown Output

```markdown
# Document Title

## Chapter 1: Introduction

This section introduces the main concepts...

### 1.1 Background

The historical context...

## Chapter 2: Methods

### 2.1 Data Collection

![Figure 1](../figures/document/figure_001.png)

| Parameter | Value | Unit |
|-----------|-------|------|
| Temperature | 25 | °C |
```

### JSONL Chunks Output

```json
{"content": "Chapter 1: Introduction\n\nThis section...", "contextualized_content": "# Chapter 1\n## Introduction\n\nThis section...", "index": 0, "token_count": 245, "metadata": {"title": "Document", "source": "document.pdf", "page": 1}}
{"content": "Chapter 2: Methods...", "contextualized_content": "# Chapter 2\n## Methods...", "index": 1, "token_count": 312, "metadata": {"title": "Document", "source": "document.pdf", "page": 5}}
```

## Configuration Tips

### For Technical Papers

```python
config = ConverterConfig(
    use_gpu=True,              # Enable for faster processing
    do_ocr=False,              # Usually not needed for born-digital PDFs
    do_table_structure=True,   # Extract data tables
    save_figures=True          # Extract diagrams and plots
)
```

### For Scanned Documents

```python
config = ConverterConfig(
    use_gpu=True,
    do_ocr=True,               # Required for scanned documents
    do_table_structure=True,
    save_figures=True
)
```

### For Large Documents

```python
config = ConverterConfig(
    use_gpu=True,
    num_threads=8,             # Increase for better performance
    enable_profiling=True      # Monitor performance
)
```

## Chunking for RAG

### Optimal Settings for Different Embedding Models

**OpenAI ada-002** (max 8191 tokens):
```python
ChunkerConfig(max_tokens=512, merge_peers=True)
```

**sentence-transformers** (typically 512 tokens):
```python
ChunkerConfig(max_tokens=384, merge_peers=True)
```

**Long-context models** (e.g., Cohere):
```python
ChunkerConfig(max_tokens=1024, merge_peers=True)
```

## Troubleshooting

### "File not found" errors

Update the file paths in the example scripts:
```python
document = Path("path/to/your/document.pdf")
```

### GPU not being used

Check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

If CUDA is not available, the converter will automatically fall back to CPU.

### Out of memory errors

Try reducing the number of threads:
```python
config = ConverterConfig(
    use_gpu=False,  # or reduce GPU usage
    num_threads=2
)
```

## Next Steps

- Read the main [README](../README.md) for more detailed documentation
- Check the [tests/](../tests/) directory for more usage examples
- Explore the [src/](../src/) directory to understand the implementation

## Contributing Examples

Have a useful example? Please contribute!

1. Create a new example script
2. Document it clearly
3. Add it to this README
4. Submit a pull request
