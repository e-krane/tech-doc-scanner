"""
Unit tests for document converter.

Tests focus on:
- Converter initialization
- Configuration handling
- Error handling for missing files
- ConversionResult dataclass
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.config import ConverterConfig
from src.converter import TechDocConverter, ConversionResult


class TestConversionResult:
    """Test ConversionResult dataclass."""

    def test_successful_result(self):
        """Test creating successful conversion result."""
        result = ConversionResult(
            markdown="# Test",
            docling_doc=Mock(),
            file_path=Path("test.pdf"),
            success=True,
            conversion_time=5.0,
            page_count=10
        )

        assert result.success is True
        assert result.markdown == "# Test"
        assert result.conversion_time == 5.0
        assert result.page_count == 10
        assert result.error is None

    def test_failed_result(self):
        """Test creating failed conversion result."""
        result = ConversionResult(
            markdown="",
            docling_doc=None,
            file_path=Path("test.pdf"),
            success=False,
            error="File not found"
        )

        assert result.success is False
        assert result.error == "File not found"
        assert result.markdown == ""
        assert result.docling_doc is None

    def test_default_values(self):
        """Test ConversionResult default values."""
        result = ConversionResult(
            markdown="",
            docling_doc=None,
            file_path=Path("test.pdf")
        )

        assert result.success is True
        assert result.error is None
        assert result.conversion_time == 0.0
        assert result.page_count == 0
        assert result.figure_count == 0
        assert result.figures_dir is None


class TestTechDocConverterInit:
    """Test TechDocConverter initialization."""

    @patch('src.converter.DoclingConverter')
    def test_initialization_with_default_config(self, mock_docling):
        """Test converter initialization with default config."""
        config = ConverterConfig()
        converter = TechDocConverter(config)

        assert converter.config == config
        assert mock_docling.called

    @patch('src.converter.DoclingConverter')
    def test_initialization_with_custom_config(self, mock_docling):
        """Test converter initialization with custom config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConverterConfig(
                output_dir=Path(tmpdir),
                num_threads=8,
                use_gpu=False
            )
            converter = TechDocConverter(config)

            assert converter.config.num_threads == 8
            assert converter.config.use_gpu is False

    @patch('src.converter.DoclingConverter')
    def test_device_configuration_cpu(self, mock_docling):
        """Test device configuration with CPU."""
        config = ConverterConfig(use_gpu=False)
        converter = TechDocConverter(config)

        device = converter._get_accelerator_device()
        # Should return CPU device when use_gpu is False
        from docling.datamodel.accelerator_options import AcceleratorDevice
        assert device == AcceleratorDevice.CPU

    @patch('src.converter.DoclingConverter')
    def test_device_configuration_auto(self, mock_docling):
        """Test device configuration with auto detection."""
        config = ConverterConfig(use_gpu=True, device="auto")
        converter = TechDocConverter(config)

        device = converter._get_accelerator_device()
        from docling.datamodel.accelerator_options import AcceleratorDevice
        assert device == AcceleratorDevice.AUTO

    @patch('src.converter.DoclingConverter')
    def test_device_configuration_cuda(self, mock_docling):
        """Test device configuration with CUDA."""
        config = ConverterConfig(use_gpu=True, device="cuda")
        converter = TechDocConverter(config)

        device = converter._get_accelerator_device()
        from docling.datamodel.accelerator_options import AcceleratorDevice
        assert device == AcceleratorDevice.CUDA

    @patch('src.converter.DoclingConverter')
    def test_device_configuration_mps(self, mock_docling):
        """Test device configuration with MPS (Apple Silicon)."""
        config = ConverterConfig(use_gpu=True, device="mps")
        converter = TechDocConverter(config)

        device = converter._get_accelerator_device()
        from docling.datamodel.accelerator_options import AcceleratorDevice
        assert device == AcceleratorDevice.MPS


class TestTechDocConverterFileHandling:
    """Test file handling and error cases."""

    @patch('src.converter.DoclingConverter')
    def test_convert_nonexistent_file(self, mock_docling):
        """Test that converting non-existent file returns error result."""
        config = ConverterConfig()
        converter = TechDocConverter(config)

        non_existent = Path("/tmp/nonexistent_file_12345.pdf")
        result = converter.convert(non_existent)

        assert result.success is False
        assert result.error is not None
        assert "File not found" in result.error
        assert result.markdown == ""
        assert result.docling_doc is None

    @patch('src.converter.DoclingConverter')
    def test_convert_path_conversion(self, mock_docling):
        """Test that string paths are converted to Path objects."""
        config = ConverterConfig()
        converter = TechDocConverter(config)

        # Test with string path (should convert to Path and check existence)
        result = converter.convert("/tmp/nonexistent_string_path.pdf")

        assert result.file_path == Path("/tmp/nonexistent_string_path.pdf")
        assert result.success is False

    @patch('src.converter.DoclingConverter')
    def test_save_markdown_failed_conversion(self, mock_docling):
        """Test that saving markdown from failed conversion raises error."""
        config = ConverterConfig()
        converter = TechDocConverter(config)

        result = ConversionResult(
            markdown="",
            docling_doc=None,
            file_path=Path("test.pdf"),
            success=False,
            error="Conversion failed"
        )

        with pytest.raises(ValueError, match="Cannot save failed conversion"):
            converter.save_markdown(result)

    @patch('src.converter.DoclingConverter')
    def test_save_markdown_successful_conversion(self, mock_docling):
        """Test saving markdown from successful conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConverterConfig(output_dir=Path(tmpdir))
            converter = TechDocConverter(config)

            result = ConversionResult(
                markdown="# Test Document\n\nContent here.",
                docling_doc=Mock(),
                file_path=Path("test.pdf"),
                success=True
            )

            output_path = converter.save_markdown(result)

            assert output_path.exists()
            assert output_path.suffix == ".md"
            assert output_path.read_text() == "# Test Document\n\nContent here."

    @patch('src.converter.DoclingConverter')
    def test_save_markdown_custom_path(self, mock_docling):
        """Test saving markdown to custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConverterConfig()
            converter = TechDocConverter(config)

            result = ConversionResult(
                markdown="# Custom Path Test",
                docling_doc=Mock(),
                file_path=Path("test.pdf"),
                success=True
            )

            custom_path = Path(tmpdir) / "custom.md"
            output_path = converter.save_markdown(result, output_path=custom_path)

            assert output_path == custom_path
            assert output_path.exists()
            assert output_path.read_text() == "# Custom Path Test"

    @patch('src.converter.DoclingConverter')
    def test_extract_figures_failed_conversion(self, mock_docling):
        """Test that extracting figures from failed conversion raises error."""
        config = ConverterConfig()
        converter = TechDocConverter(config)

        result = ConversionResult(
            markdown="",
            docling_doc=None,
            file_path=Path("test.pdf"),
            success=False,
            error="Conversion failed"
        )

        with pytest.raises(ValueError, match="Cannot extract figures from failed conversion"):
            converter.extract_figures(result)

    @patch('src.converter.DoclingConverter')
    def test_extract_figures_disabled(self, mock_docling):
        """Test that figure extraction returns None when disabled."""
        config = ConverterConfig(save_figures=False)
        converter = TechDocConverter(config)

        result = ConversionResult(
            markdown="# Test",
            docling_doc=Mock(),
            file_path=Path("test.pdf"),
            success=True
        )

        figures_dir = converter.extract_figures(result)
        assert figures_dir is None

    @patch('src.converter.DoclingConverter')
    def test_extract_figures_no_pictures(self, mock_docling):
        """Test extracting figures when document has no pictures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConverterConfig(output_dir=Path(tmpdir))
            converter = TechDocConverter(config)

            # Mock document with no pictures
            mock_doc = Mock()
            mock_doc.pictures = []

            result = ConversionResult(
                markdown="# Test",
                docling_doc=mock_doc,
                file_path=Path("test.pdf"),
                success=True
            )

            figures_dir = converter.extract_figures(result)
            assert figures_dir is None


class TestTechDocConverterConfiguration:
    """Test converter configuration options."""

    @patch('src.converter.DoclingConverter')
    def test_ocr_enabled(self, mock_docling):
        """Test OCR is enabled by default."""
        config = ConverterConfig(do_ocr=True)
        converter = TechDocConverter(config)

        assert converter.config.do_ocr is True

    @patch('src.converter.DoclingConverter')
    def test_ocr_disabled(self, mock_docling):
        """Test OCR can be disabled."""
        config = ConverterConfig(do_ocr=False)
        converter = TechDocConverter(config)

        assert converter.config.do_ocr is False

    @patch('src.converter.DoclingConverter')
    def test_table_structure_enabled(self, mock_docling):
        """Test table structure processing is enabled by default."""
        config = ConverterConfig(do_table_structure=True)
        converter = TechDocConverter(config)

        assert converter.config.do_table_structure is True

    @patch('src.converter.DoclingConverter')
    def test_profiling_disabled_by_default(self, mock_docling):
        """Test profiling is disabled by default."""
        config = ConverterConfig()
        converter = TechDocConverter(config)

        assert converter.config.enable_profiling is False

    @patch('src.converter.DoclingConverter')
    def test_profiling_enabled(self, mock_docling):
        """Test profiling can be enabled."""
        config = ConverterConfig(enable_profiling=True)
        converter = TechDocConverter(config)

        assert converter.config.enable_profiling is True
