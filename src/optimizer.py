"""
Document optimizer module for smart pipeline configuration.

This module analyzes documents to determine optimal Docling pipeline settings
based on document characteristics like language, OCR requirements, and complexity.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

from .config import OptimizerConfig

logger = logging.getLogger(__name__)


@dataclass
class DocumentAnalysis:
    """
    Analysis results for a document.

    Attributes:
        needs_ocr: Whether the document needs OCR processing
        has_tables: Whether the document contains tables
        has_complex_layout: Whether the document has complex layout
        language: Detected document language (e.g., 'en', 'de', 'fr')
        page_count: Number of pages in the document
        file_size_mb: File size in megabytes
        estimated_time: Estimated processing time in seconds
        recommended_options: Recommended pipeline options
    """
    needs_ocr: bool = False
    has_tables: bool = False
    has_complex_layout: bool = False
    language: str = "en"
    page_count: int = 0
    file_size_mb: float = 0.0
    estimated_time: float = 0.0
    recommended_options: Optional[PdfPipelineOptions] = None


class DocumentOptimizer:
    """
    Document analyzer for optimizing Docling pipeline configuration.

    Features:
    - OCR detection (determines if document needs OCR)
    - Language detection
    - Table detection
    - Complexity assessment (layout, images, etc.)
    - Smart pipeline configuration based on analysis

    The optimizer samples pages from the document to make quick decisions
    about the optimal processing strategy.

    Example:
        >>> from src.config import OptimizerConfig
        >>> from src.optimizer import DocumentOptimizer
        >>>
        >>> config = OptimizerConfig()
        >>> optimizer = DocumentOptimizer(config)
        >>>
        >>> # Analyze document
        >>> analysis = optimizer.analyze(Path("document.pdf"))
        >>> print(f"Needs OCR: {analysis.needs_ocr}")
        >>> print(f"Has tables: {analysis.has_tables}")
        >>>
        >>> # Get optimized pipeline options
        >>> options = optimizer.get_pipeline_options(analysis)
    """

    def __init__(self, config: OptimizerConfig):
        """
        Initialize the document optimizer.

        Args:
            config: Optimizer configuration with analysis settings
        """
        self.config = config
        logger.info("DocumentOptimizer initialized")

    def analyze(self, file_path: Path) -> DocumentAnalysis:
        """
        Analyze a document to determine optimal processing strategy.

        Args:
            file_path: Path to the document

        Returns:
            DocumentAnalysis with recommendations

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Analyzing document: {file_path.name}")

        analysis = DocumentAnalysis()

        # Get basic file info
        analysis.file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.debug(f"File size: {analysis.file_size_mb:.2f} MB")

        # Only analyze PDFs in detail (for now)
        if file_path.suffix.lower() == '.pdf':
            self._analyze_pdf(file_path, analysis)
        else:
            # For non-PDF formats, use conservative defaults
            analysis.needs_ocr = False
            analysis.has_tables = True  # Assume tables might be present
            analysis.has_complex_layout = False
            logger.info(f"Non-PDF format, using default analysis")

        # Estimate processing time based on complexity
        analysis.estimated_time = self._estimate_processing_time(analysis)

        # Generate recommended pipeline options
        analysis.recommended_options = self.get_pipeline_options(analysis)

        logger.info(
            f"Analysis complete: OCR={analysis.needs_ocr}, "
            f"Tables={analysis.has_tables}, "
            f"Language={analysis.language}, "
            f"Est. time={analysis.estimated_time:.1f}s"
        )

        return analysis

    def _analyze_pdf(self, file_path: Path, analysis: DocumentAnalysis):
        """
        Analyze a PDF document in detail.

        Args:
            file_path: Path to PDF file
            analysis: DocumentAnalysis object to populate
        """
        try:
            import pypdf

            # Open PDF
            pdf = pypdf.PdfReader(str(file_path))
            analysis.page_count = len(pdf.pages)

            logger.debug(f"PDF has {analysis.page_count} pages")

            # Sample pages for analysis
            sample_pages = self._get_sample_pages(analysis.page_count)
            logger.debug(f"Sampling {len(sample_pages)} pages for analysis")

            # Analyze sampled pages
            total_text_length = 0
            has_images = False

            for page_num in sample_pages:
                page = pdf.pages[page_num]

                # Check for text content
                text = page.extract_text()
                total_text_length += len(text) if text else 0

                # Check for images
                if '/XObject' in page['/Resources']:
                    xobjects = page['/Resources']['/XObject'].get_object()
                    for obj in xobjects:
                        if xobjects[obj]['/Subtype'] == '/Image':
                            has_images = True
                            break

            # Determine if OCR is needed
            # Threshold: if average text per page is less than 100 characters, likely needs OCR
            ocr_text_threshold = 100  # characters per page
            avg_text_per_page = total_text_length / len(sample_pages) if sample_pages else 0

            if avg_text_per_page < ocr_text_threshold:
                analysis.needs_ocr = True
                logger.debug(f"Low text content ({avg_text_per_page:.0f} chars/page), OCR recommended")
            else:
                analysis.needs_ocr = False
                logger.debug(f"Sufficient text content ({avg_text_per_page:.0f} chars/page), OCR not needed")

            # Detect language (simple approach - look at first page text)
            if analysis.page_count > 0:
                first_page_text = pdf.pages[0].extract_text()
                if first_page_text:
                    analysis.language = self._detect_language(first_page_text)
                    logger.debug(f"Detected language: {analysis.language}")

            # Estimate table presence (heuristic: PDFs with many small text blocks might have tables)
            # This is a conservative estimate - we assume tables might be present
            analysis.has_tables = True  # Conservative default

            # Assess layout complexity
            analysis.has_complex_layout = has_images or analysis.page_count > 50

        except ImportError:
            logger.warning("pypdf not available, using conservative defaults")
            analysis.needs_ocr = True  # Conservative: assume OCR needed
            analysis.has_tables = True
            analysis.has_complex_layout = True
            analysis.page_count = 10  # Estimate
        except Exception as e:
            logger.error(f"Error analyzing PDF: {e}", exc_info=True)
            # Use conservative defaults on error
            analysis.needs_ocr = True
            analysis.has_tables = True
            analysis.has_complex_layout = True

    def _get_sample_pages(self, page_count: int) -> list[int]:
        """
        Get page indices to sample for analysis.

        Samples first page, last page, and pages from middle sections.

        Args:
            page_count: Total number of pages

        Returns:
            List of page indices to sample (0-indexed)
        """
        if page_count == 0:
            return []

        if page_count <= self.config.sample_pages:
            # Sample all pages
            return list(range(page_count))

        # Sample strategy: first, last, and evenly distributed middle pages
        sample_indices = [0]  # First page

        # Middle pages
        middle_count = self.config.sample_pages - 2
        if middle_count > 0:
            step = page_count // (middle_count + 1)
            for i in range(1, middle_count + 1):
                sample_indices.append(i * step)

        # Last page
        sample_indices.append(page_count - 1)

        return sample_indices

    def _detect_language(self, text: str) -> str:
        """
        Detect document language from text.

        This is a simple heuristic-based detector. For production use,
        consider using a proper language detection library.

        Args:
            text: Text sample to analyze

        Returns:
            ISO 639-1 language code (e.g., 'en', 'de', 'fr')
        """
        if not text or len(text) < 50:
            return "en"  # Default to English

        text_lower = text.lower()

        # Simple keyword-based detection
        # This is NOT robust - use langdetect or similar for production
        german_indicators = ['der', 'die', 'das', 'und', 'ist', 'nicht', 'mit']
        french_indicators = ['le', 'la', 'les', 'et', 'est', 'pas', 'dans']
        spanish_indicators = ['el', 'la', 'los', 'las', 'es', 'no', 'con']

        german_count = sum(1 for word in german_indicators if f' {word} ' in text_lower)
        french_count = sum(1 for word in french_indicators if f' {word} ' in text_lower)
        spanish_count = sum(1 for word in spanish_indicators if f' {word} ' in text_lower)

        max_count = max(german_count, french_count, spanish_count)

        if max_count > 2:
            if german_count == max_count:
                return "de"
            elif french_count == max_count:
                return "fr"
            elif spanish_count == max_count:
                return "es"

        return "en"  # Default to English

    def _estimate_processing_time(self, analysis: DocumentAnalysis) -> float:
        """
        Estimate document processing time based on analysis.

        Args:
            analysis: Document analysis results

        Returns:
            Estimated processing time in seconds
        """
        # Base time per page
        base_time_per_page = 0.5  # seconds

        # Multipliers based on complexity
        time = analysis.page_count * base_time_per_page

        if analysis.needs_ocr:
            time *= 3.0  # OCR is much slower

        if analysis.has_tables:
            time *= 1.5  # Table extraction adds overhead

        if analysis.has_complex_layout:
            time *= 1.3  # Complex layouts take longer

        # File size impact
        if analysis.file_size_mb > 10:
            time *= 1.2

        return max(time, 1.0)  # Minimum 1 second

    def get_pipeline_options(
        self,
        analysis: DocumentAnalysis,
        accelerator_device: AcceleratorDevice = AcceleratorDevice.AUTO,
        num_threads: int = 4
    ) -> PdfPipelineOptions:
        """
        Get optimized pipeline options based on document analysis.

        Args:
            analysis: Document analysis results
            accelerator_device: Accelerator device to use
            num_threads: Number of threads for processing

        Returns:
            Optimized PdfPipelineOptions
        """
        options = PdfPipelineOptions()

        # Configure OCR based on analysis
        options.do_ocr = analysis.needs_ocr

        # Configure table processing
        options.do_table_structure = analysis.has_tables

        # Enable cell matching for tables if present
        if analysis.has_tables:
            options.table_structure_options.do_cell_matching = True

        # Configure accelerator
        accelerator_options = AcceleratorOptions(
            num_threads=num_threads,
            device=accelerator_device
        )
        options.accelerator_options = accelerator_options

        logger.debug(
            f"Pipeline options: OCR={options.do_ocr}, "
            f"Tables={options.do_table_structure}, "
            f"Device={accelerator_device}"
        )

        return options

    def optimize_converter_config(self, file_path: Path) -> Dict[str, Any]:
        """
        Get optimized converter configuration for a document.

        This is a convenience method that returns a dictionary of config
        values that can be used to create a ConverterConfig.

        Args:
            file_path: Path to document

        Returns:
            Dictionary with converter configuration values

        Example:
            >>> optimizer = DocumentOptimizer(OptimizerConfig())
            >>> config_dict = optimizer.optimize_converter_config(Path("doc.pdf"))
            >>> config = ConverterConfig(**config_dict)
        """
        analysis = self.analyze(file_path)

        return {
            "do_ocr": analysis.needs_ocr,
            "do_table_structure": analysis.has_tables,
            "do_cell_matching": analysis.has_tables,
        }
