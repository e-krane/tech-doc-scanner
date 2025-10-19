#!/usr/bin/env python3
"""
Custom Pipeline with Optimizer Example
=======================================

This example demonstrates using the document optimizer to analyze
documents and apply optimal processing settings automatically.
"""

from pathlib import Path
from src.config import ConverterConfig, OptimizerConfig, ChunkerConfig
from src.converter import TechDocConverter
from src.optimizer import DocumentOptimizer
from src.chunker import DocumentChunker


def process_with_optimization(document_path: Path, output_dir: Path):
    """Process a document with automatic optimization."""

    print(f"Processing: {document_path.name}\n")

    # Step 1: Analyze document
    print("Step 1: Analyzing document...")
    optimizer_config = OptimizerConfig(
        enable_language_detection=True,
        ocr_quality_threshold=0.8,
        sample_pages=3
    )
    optimizer = DocumentOptimizer(optimizer_config)

    analysis = optimizer.analyze(document_path)

    print(f"  Document size: {analysis.file_size_mb:.2f} MB")
    print(f"  Page count: {analysis.page_count}")
    print(f"  Language: {analysis.language}")
    print(f"  Needs OCR: {'Yes' if analysis.needs_ocr else 'No'}")
    print(f"  Has tables: {'Yes' if analysis.has_tables else 'No'}")
    print(f"  Complex layout: {'Yes' if analysis.has_complex_layout else 'No'}")
    print(f"  Estimated time: {analysis.estimated_time:.1f}s\n")

    # Step 2: Get optimized configuration
    print("Step 2: Creating optimized configuration...")
    config_dict = optimizer.optimize_converter_config(document_path)

    converter_config = ConverterConfig(
        output_dir=output_dir,
        **config_dict,
        use_gpu=True,            # Enable GPU if available
        save_figures=True,
        enable_profiling=True    # Enable profiling to verify optimization
    )

    print(f"  OCR enabled: {converter_config.do_ocr}")
    print(f"  Table extraction: {converter_config.do_table_structure}")
    print(f"  Cell matching: {converter_config.do_cell_matching}\n")

    # Step 3: Convert document
    print("Step 3: Converting document...")
    converter = TechDocConverter(converter_config)
    result = converter.convert(document_path)

    if not result.success:
        print(f"  ✗ Conversion failed: {result.error}")
        return

    print(f"  ✓ Conversion successful!")
    print(f"    Pages: {result.page_count}")
    print(f"    Time: {result.conversion_time:.2f}s")
    print(f"    Estimated vs Actual: {analysis.estimated_time:.1f}s vs {result.conversion_time:.2f}s\n")

    # Step 4: Chunk for RAG
    print("Step 4: Creating chunks for RAG...")
    chunker_config = ChunkerConfig(
        max_tokens=512,
        merge_peers=True
    )
    chunker = DocumentChunker(chunker_config)

    if result.docling_doc:
        chunks = chunker.chunk_document(
            docling_doc=result.docling_doc,
            title=document_path.stem,
            source=str(document_path),
            metadata={
                "language": analysis.language,
                "page_count": result.page_count,
                "ocr_used": config_dict["do_ocr"]
            }
        )

        print(f"  ✓ Created {len(chunks)} chunks")

        # Show chunk statistics
        token_counts = [c.token_count for c in chunks]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

        print(f"    Average tokens/chunk: {avg_tokens:.0f}")
        print(f"    Min tokens: {min(token_counts) if token_counts else 0}")
        print(f"    Max tokens: {max(token_counts) if token_counts else 0}\n")

        # Step 5: Export in multiple formats
        print("Step 5: Exporting outputs...")

        # Save markdown
        markdown_file = converter.save_markdown(result)
        print(f"  ✓ Markdown: {markdown_file}")

        # Save chunks as JSONL (for vector DB ingestion)
        chunks_jsonl = output_dir / "chunks" / f"{document_path.stem}.jsonl"
        chunker.export_chunks(chunks, chunks_jsonl, format="jsonl")
        print(f"  ✓ Chunks (JSONL): {chunks_jsonl}")

        # Save chunks as markdown (for human review)
        chunks_md = output_dir / "chunks" / f"{document_path.stem}_chunks.md"
        chunker.export_chunks(chunks, chunks_md, format="md")
        print(f"  ✓ Chunks (MD): {chunks_md}")

        # Extract figures
        if converter_config.save_figures and result.docling_doc:
            figures_dir = converter.extract_figures(result)
            if figures_dir:
                print(f"  ✓ Figures: {figures_dir}")

    print(f"\n✓ Processing complete!")


def main():
    # Example 1: Single document
    document = Path("document.pdf")  # Change to your document

    if not document.exists():
        print(f"Error: Document not found: {document}")
        print("Please update the 'document' path to point to your PDF file.")
        return

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chunks").mkdir(exist_ok=True)

    process_with_optimization(document, output_dir)

    # Example 2: Batch processing with optimization
    print("\n" + "="*60)
    print("Batch Processing Example")
    print("="*60 + "\n")

    documents_dir = Path("documents")

    if documents_dir.exists():
        pdf_files = list(documents_dir.glob("*.pdf"))

        if pdf_files:
            print(f"Found {len(pdf_files)} documents to process\n")

            for i, pdf_file in enumerate(pdf_files, 1):
                print(f"\n[{i}/{len(pdf_files)}] " + "="*50)
                process_with_optimization(pdf_file, output_dir)
        else:
            print("No PDF files found in documents/ directory")
    else:
        print("documents/ directory not found - skipping batch example")


if __name__ == "__main__":
    main()
