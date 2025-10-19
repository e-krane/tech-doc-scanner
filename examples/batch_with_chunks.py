#!/usr/bin/env python3
"""
Batch Conversion with Chunking Example
=======================================

This example demonstrates batch processing of multiple documents
with automatic chunking optimized for RAG pipelines.
"""

from pathlib import Path
from src.config import ConverterConfig, ChunkerConfig
from src.converter import TechDocConverter
from src.chunker import DocumentChunker


def process_documents(input_dir: Path, output_dir: Path):
    """Process all PDF documents in a directory and create chunks."""

    # Configure converter
    converter_config = ConverterConfig(
        output_dir=output_dir,
        use_gpu=True,
        save_figures=True
    )
    converter = TechDocConverter(converter_config)

    # Configure chunker
    chunker_config = ChunkerConfig(
        max_tokens=512,              # Max tokens per chunk (fits most embedding models)
        merge_peers=True,            # Merge small adjacent chunks
        tokenizer_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    chunker = DocumentChunker(chunker_config)

    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files to process\n")

    # Process each document
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")

        # Convert document
        result = converter.convert(pdf_file)

        if not result.success:
            print(f"  ✗ Conversion failed: {result.error}\n")
            continue

        print(f"  ✓ Converted {result.page_count} pages in {result.conversion_time:.2f}s")

        # Chunk the document
        if result.docling_doc is not None:
            chunks = chunker.chunk_document(
                docling_doc=result.docling_doc,
                title=pdf_file.stem,
                source=str(pdf_file),
                metadata={
                    "page_count": result.page_count,
                    "file_size_mb": pdf_file.stat().st_size / (1024 * 1024)
                }
            )

            print(f"  ✓ Created {len(chunks)} chunks")

            # Export chunks to JSONL
            chunks_file = output_dir / "chunks" / f"{pdf_file.stem}_chunks.jsonl"
            chunker.export_chunks(chunks, chunks_file, format="jsonl")

            print(f"  ✓ Saved chunks to: {chunks_file}\n")
        else:
            print(f"  ! No DoclingDocument available, skipping chunking\n")

    print(f"\nProcessing complete!")
    print(f"Output directory: {output_dir}")


def main():
    # Configure paths
    input_dir = Path("documents")  # Directory containing PDF files
    output_dir = Path("output")

    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("Please create a 'documents/' directory and add PDF files.")
        return

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chunks").mkdir(exist_ok=True)

    # Process documents
    process_documents(input_dir, output_dir)


if __name__ == "__main__":
    main()
