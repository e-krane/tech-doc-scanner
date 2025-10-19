"""
Document converter module using Docling.

This module provides the DocumentConverter class that handles conversion
of technical documents to markdown format with GPU acceleration support.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

from docling.document_converter import DocumentConverter as DoclingConverter
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import settings
from docling_core.types.doc import DoclingDocument

from .config import ConverterConfig

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Result of document conversion."""

    markdown: str
    docling_doc: DoclingDocument
    file_path: Path
    success: bool = True
    error: Optional[str] = None
    conversion_time: float = 0.0
    page_count: int = 0
    figure_count: int = 0
    figures_dir: Optional[Path] = None


class TechDocConverter:
    """
    Technical document converter using Docling with GPU acceleration.

    This converter handles single document conversion to markdown format,
    extracting the DoclingDocument for downstream processing (chunking).

    Features:
    - GPU acceleration with automatic device detection
    - Performance profiling support
    - Configurable OCR and table processing
    - Error handling with detailed logging

    Example:
        >>> from src.config import ConverterConfig
        >>> from src.converter import TechDocConverter
        >>>
        >>> config = ConverterConfig()
        >>> converter = TechDocConverter(config)
        >>> result = converter.convert(Path("document.pdf"))
        >>> print(result.markdown[:100])
    """

    def __init__(self, config: ConverterConfig):
        """
        Initialize the document converter.

        Args:
            config: Converter configuration with GPU, OCR, and table settings
        """
        self.config = config
        self._docling_converter: Optional[DoclingConverter] = None
        self._initialize_converter()

    def _initialize_converter(self):
        """Initialize Docling converter with GPU acceleration and pipeline options."""
        logger.info("Initializing Docling converter")

        # Configure accelerator (GPU/CPU)
        accelerator_device = self._get_accelerator_device()
        accelerator_options = AcceleratorOptions(
            num_threads=self.config.num_threads,
            device=accelerator_device
        )

        logger.info(
            f"Accelerator configured: device={accelerator_device}, "
            f"threads={self.config.num_threads}"
        )

        # Configure PDF pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.config.do_ocr
        pipeline_options.do_table_structure = self.config.do_table_structure
        pipeline_options.table_structure_options.do_cell_matching = self.config.do_cell_matching
        pipeline_options.accelerator_options = accelerator_options

        logger.info(
            f"Pipeline options: OCR={self.config.do_ocr}, "
            f"tables={self.config.do_table_structure}, "
            f"cell_matching={self.config.do_cell_matching}"
        )

        # Enable profiling if configured
        if self.config.enable_profiling:
            settings.debug.profile_pipeline_timings = True
            logger.info("Performance profiling enabled")

        # Create Docling converter
        from docling.document_converter import PdfFormatOption

        self._docling_converter = DoclingConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        logger.info("Docling converter initialized successfully")

    def _get_accelerator_device(self) -> AcceleratorDevice:
        """
        Get the accelerator device based on configuration.

        Returns:
            AcceleratorDevice enum value (AUTO, CPU, CUDA, MPS)
        """
        if not self.config.use_gpu:
            logger.info("GPU disabled by configuration, using CPU")
            return AcceleratorDevice.CPU

        device = self.config.device.lower()
        device_map = {
            "auto": AcceleratorDevice.AUTO,
            "cpu": AcceleratorDevice.CPU,
            "cuda": AcceleratorDevice.CUDA,
            "mps": AcceleratorDevice.MPS,
        }

        accelerator = device_map.get(device, AcceleratorDevice.AUTO)

        if device == "auto":
            logger.info("Auto-detecting GPU device")

        return accelerator

    def convert(self, file_path: Path) -> ConversionResult:
        """
        Convert a document to markdown format.

        Args:
            file_path: Path to the document file

        Returns:
            ConversionResult with markdown, DoclingDocument, and metadata

        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: If conversion fails
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            return ConversionResult(
                markdown="",
                docling_doc=None,
                file_path=file_path,
                success=False,
                error=error_msg
            )

        logger.info(f"Converting document: {file_path.name}")

        try:
            import time
            start_time = time.time()

            # Convert document
            result = self._docling_converter.convert(file_path)

            # Extract conversion time
            conversion_time = time.time() - start_time

            # Get timing info if profiling is enabled
            if self.config.enable_profiling and hasattr(result, 'timings'):
                pipeline_time = result.timings.get("pipeline_total", {}).get("times", [])
                if pipeline_time:
                    conversion_time = sum(pipeline_time)

            # Export to markdown
            markdown = result.document.export_to_markdown()

            # Get page count if available
            page_count = 0
            if hasattr(result.document, 'pages'):
                page_count = len(result.document.pages)

            logger.info(
                f"Conversion successful: {file_path.name} "
                f"({page_count} pages, {conversion_time:.2f}s)"
            )

            return ConversionResult(
                markdown=markdown,
                docling_doc=result.document,
                file_path=file_path,
                success=True,
                conversion_time=conversion_time,
                page_count=page_count
            )

        except Exception as e:
            error_msg = f"Conversion failed for {file_path.name}: {e}"
            logger.error(error_msg, exc_info=True)
            return ConversionResult(
                markdown="",
                docling_doc=None,
                file_path=file_path,
                success=False,
                error=error_msg
            )

    def save_markdown(self, result: ConversionResult, output_path: Optional[Path] = None) -> Path:
        """
        Save markdown to file.

        Args:
            result: ConversionResult from convert()
            output_path: Optional output path. If None, uses config.output_dir

        Returns:
            Path to the saved markdown file

        Raises:
            ValueError: If conversion result is not successful
        """
        if not result.success:
            raise ValueError(f"Cannot save failed conversion: {result.error}")

        if output_path is None:
            # Generate output path from input filename
            output_filename = result.file_path.stem + ".md"
            output_path = self.config.output_dir / "markdown" / output_filename

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write markdown file
        output_path.write_text(result.markdown, encoding='utf-8')

        logger.info(f"Markdown saved to: {output_path}")

        return output_path

    def extract_figures(self, result: ConversionResult) -> Path:
        """
        Extract figures (images, tables, equations) from DoclingDocument.

        Args:
            result: ConversionResult from convert()

        Returns:
            Path to the figures directory

        Raises:
            ValueError: If conversion result is not successful
        """
        if not result.success:
            raise ValueError(f"Cannot extract figures from failed conversion: {result.error}")

        if not self.config.save_figures:
            logger.info("Figure extraction disabled by configuration")
            return None

        # Create figures directory for this document
        doc_name = result.file_path.stem
        figures_dir = self.config.output_dir / "figures" / doc_name
        figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting figures to: {figures_dir}")

        figure_count = 0

        # Extract pictures from DoclingDocument
        if hasattr(result.docling_doc, 'pictures') and result.docling_doc.pictures:
            for i, picture in enumerate(result.docling_doc.pictures):
                try:
                    # Generate figure filename
                    figure_filename = f"figure_{i+1:03d}.png"
                    figure_path = figures_dir / figure_filename

                    # Get the image data
                    if hasattr(picture, 'get_image'):
                        # Docling 2.x API
                        image = picture.get_image()
                    elif hasattr(picture, 'image'):
                        # Alternative API
                        image = picture.image
                    else:
                        logger.warning(f"Cannot extract image from picture {i+1}")
                        continue

                    # Save the image
                    if image:
                        image.save(figure_path)
                        figure_count += 1
                        logger.debug(f"Saved figure: {figure_filename}")

                except Exception as e:
                    logger.error(f"Failed to save figure {i+1}: {e}")

        # Create figure index
        if figure_count > 0:
            self._create_figure_index(figures_dir, figure_count, result.docling_doc)

        logger.info(f"Extracted {figure_count} figures")

        return figures_dir if figure_count > 0 else None

    def _create_figure_index(self, figures_dir: Path, figure_count: int, docling_doc: DoclingDocument):
        """
        Create an index file listing all extracted figures.

        Args:
            figures_dir: Directory containing figures
            figure_count: Number of figures extracted
            docling_doc: DoclingDocument for extracting captions
        """
        index_path = figures_dir / "README.md"

        index_content = f"# Figures Index\n\n"
        index_content += f"Total figures: {figure_count}\n\n"

        for i in range(1, figure_count + 1):
            figure_filename = f"figure_{i:03d}.png"
            index_content += f"## Figure {i}\n\n"
            index_content += f"![Figure {i}](./{figure_filename})\n\n"

            # Try to extract caption if available
            if hasattr(docling_doc, 'pictures') and i <= len(docling_doc.pictures):
                picture = docling_doc.pictures[i-1]
                if hasattr(picture, 'caption') and picture.caption:
                    index_content += f"**Caption**: {picture.caption}\n\n"

        index_path.write_text(index_content, encoding='utf-8')
        logger.debug(f"Created figure index: {index_path}")

    def convert_and_save(self, file_path: Path) -> ConversionResult:
        """
        Convert document and save both markdown and figures.

        This is a convenience method that combines convert(), save_markdown(),
        and extract_figures().

        Args:
            file_path: Path to the document file

        Returns:
            ConversionResult with all outputs saved
        """
        # Convert document
        result = self.convert(file_path)

        if not result.success:
            return result

        # Save markdown
        markdown_path = self.save_markdown(result)
        logger.info(f"Saved markdown: {markdown_path}")

        # Extract figures if enabled
        if self.config.save_figures:
            figures_dir = self.extract_figures(result)
            result.figures_dir = figures_dir
            if figures_dir:
                result.figure_count = len(list(figures_dir.glob("figure_*.png")))

        return result
