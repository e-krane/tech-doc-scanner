#!/usr/bin/env python3
"""
Partial Document Parsing Example
=================================

This example demonstrates how to parse only specific pages or sections
of large documents for faster processing and previewing.
"""

from pathlib import Path
from src.config import ConverterConfig
from src.converter import TechDocConverter


def example_page_range(converter, doc_path):
    """Convert specific page range from a document."""
    print("\n" + "="*60)
    print("Example 1: Convert Specific Page Range (pages 1-5)")
    print("="*60)

    result = converter.convert(
        doc_path,
        page_range=(1, 5)  # Convert pages 1-5 only
    )

    if result.success:
        print(f"✓ Converted pages 1-5")
        print(f"  Pages processed: {result.page_count}")
        print(f"  Time: {result.conversion_time:.2f}s")
        print(f"  Markdown length: {len(result.markdown)} chars")

        # Save with descriptive name
        output_file = converter.config.output_dir / "markdown" / f"{doc_path.stem}_pages_1-5.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(result.markdown)
        print(f"  Saved to: {output_file}")
    else:
        print(f"✗ Conversion failed: {result.error}")


def example_first_n_pages(converter, doc_path, n=10):
    """Convert first N pages of a document."""
    print("\n" + "="*60)
    print(f"Example 2: Convert First {n} Pages")
    print("="*60)

    result = converter.convert(
        doc_path,
        max_pages=n  # Convert first N pages
    )

    if result.success:
        print(f"✓ Converted first {n} pages")
        print(f"  Pages processed: {result.page_count}")
        print(f"  Time: {result.conversion_time:.2f}s")

        # Save with descriptive name
        output_file = converter.config.output_dir / "markdown" / f"{doc_path.stem}_first_{n}_pages.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(result.markdown)
        print(f"  Saved to: {output_file}")
    else:
        print(f"✗ Conversion failed: {result.error}")


def example_middle_section(converter, doc_path):
    """Convert a middle section of a document."""
    print("\n" + "="*60)
    print("Example 3: Convert Middle Section (pages 10-15)")
    print("="*60)

    result = converter.convert(
        doc_path,
        page_range=(10, 15)  # Convert pages 10-15
    )

    if result.success:
        print(f"✓ Converted pages 10-15")
        print(f"  Pages processed: {result.page_count}")
        print(f"  Time: {result.conversion_time:.2f}s")

        # Show preview
        print(f"\n  Preview (first 300 chars):")
        print("  " + "-"*56)
        preview = result.markdown[:300].replace('\n', '\n  ')
        print(f"  {preview}...")
    else:
        print(f"✗ Conversion failed: {result.error}")


def example_preview_first_page(converter, doc_path):
    """Quick preview by converting just the first page."""
    print("\n" + "="*60)
    print("Example 4: Quick Preview (first page only)")
    print("="*60)

    result = converter.convert(
        doc_path,
        page_range=(1, 1)  # Just first page
    )

    if result.success:
        print(f"✓ Quick preview generated")
        print(f"  Time: {result.conversion_time:.2f}s")
        print(f"\n  First Page Content:")
        print("  " + "-"*56)
        # Show full first page content (limited to 1000 chars for display)
        content = result.markdown[:1000].replace('\n', '\n  ')
        print(f"  {content}")
        if len(result.markdown) > 1000:
            print("  ...")
    else:
        print(f"✗ Conversion failed: {result.error}")


def compare_performance(converter, doc_path):
    """Compare performance of partial vs full conversion."""
    print("\n" + "="*60)
    print("Example 5: Performance Comparison")
    print("="*60)

    import time

    # Convert first 5 pages
    start = time.time()
    result_partial = converter.convert(doc_path, max_pages=5)
    time_partial = time.time() - start

    # Note: Not converting full doc here to save time
    # Just showing what the comparison would look like

    if result_partial.success:
        print(f"\n  Partial conversion (5 pages):")
        print(f"    Time: {time_partial:.2f}s")
        print(f"    Markdown: {len(result_partial.markdown)} chars")

        # Estimate full document time based on partial
        if result_partial.page_count > 0:
            time_per_page = time_partial / result_partial.page_count
            print(f"    Time per page: {time_per_page:.2f}s")

            # You could estimate total time for full doc:
            # estimated_full_time = time_per_page * total_pages
            print(f"\n  ✓ Partial parsing is ideal for:")
            print(f"    - Quick document previews")
            print(f"    - Testing conversion settings")
            print(f"    - Extracting specific sections")
            print(f"    - Large documents (100+ pages)")


def main():
    # Configuration
    doc_path = Path("documents/19980147983.pdf")  # 20-page document

    if not doc_path.exists():
        print(f"Error: Document not found: {doc_path}")
        print("Please update the doc_path to point to a PDF file.")
        return

    output_dir = Path("output_partial")
    output_dir.mkdir(exist_ok=True)

    # Create converter with fast settings
    config = ConverterConfig(
        output_dir=output_dir,
        use_gpu=True,
        save_figures=False,  # Disable for speed
        enable_profiling=False
    )

    converter = TechDocConverter(config)

    print("\n" + "="*60)
    print("Partial Document Parsing Examples")
    print("="*60)
    print(f"\nDocument: {doc_path.name}")
    print(f"Output directory: {output_dir}")

    # Run examples
    example_preview_first_page(converter, doc_path)
    example_page_range(converter, doc_path)
    example_first_n_pages(converter, doc_path, n=10)
    example_middle_section(converter, doc_path)
    compare_performance(converter, doc_path)

    print("\n" + "="*60)
    print("✓ All examples completed!")
    print("="*60)

    print(f"\nUse Cases for Partial Parsing:")
    print("  1. Preview large documents before full conversion")
    print("  2. Extract specific chapters or sections")
    print("  3. Test conversion settings on representative pages")
    print("  4. Process documents incrementally (batch by section)")
    print("  5. Reduce processing time for large document sets")


if __name__ == "__main__":
    main()
