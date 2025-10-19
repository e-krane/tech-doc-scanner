#!/usr/bin/env python3
"""
Basic Document Conversion Example
==================================

This example demonstrates the basic usage of the document converter
to convert a single document to markdown.
"""

from pathlib import Path
from src.config import ConverterConfig
from src.converter import TechDocConverter


def main():
    # Define input and output paths
    input_file = Path("document.pdf")  # Change to your document path
    output_dir = Path("output")

    # Create converter configuration
    config = ConverterConfig(
        output_dir=output_dir,
        use_gpu=True,              # Enable GPU acceleration if available
        save_figures=True,          # Extract figures from document
        do_ocr=True,               # Enable OCR for scanned documents
        do_table_structure=True,   # Extract table structure
        enable_profiling=False     # Disable performance profiling
    )

    # Initialize converter
    print(f"Initializing converter...")
    converter = TechDocConverter(config)

    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("Please update the input_file path to point to your document.")
        return

    # Convert document
    print(f"\nConverting: {input_file.name}")
    result = converter.convert_and_save(input_file)

    # Check result
    if result.success:
        print(f"\n✓ Conversion successful!")
        print(f"  Pages processed: {result.page_count}")
        print(f"  Processing time: {result.conversion_time:.2f}s")

        if result.figure_count > 0:
            print(f"  Figures extracted: {result.figure_count}")
            print(f"  Figures directory: {result.figures_dir}")

        # Show output location
        markdown_file = output_dir / "markdown" / f"{input_file.stem}.md"
        print(f"\n  Markdown output: {markdown_file}")

        # Show preview of markdown (first 500 chars)
        if markdown_file.exists():
            preview = markdown_file.read_text()[:500]
            print(f"\n  Preview:\n{preview}...\n")
    else:
        print(f"\n✗ Conversion failed: {result.error}")


if __name__ == "__main__":
    main()
