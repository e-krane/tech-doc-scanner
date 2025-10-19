"""
Test that all modules can be imported successfully.

This ensures the basic module structure is correct and dependencies are available.
"""

import pytest


class TestModuleImports:
    """Test that all modules can be imported."""

    def test_import_config(self):
        """Test importing config module."""
        from src import config
        assert hasattr(config, 'ConverterConfig')
        assert hasattr(config, 'OptimizerConfig')
        assert hasattr(config, 'ChunkerConfig')
        assert hasattr(config, 'AppConfig')
        assert hasattr(config, 'load_config')

    def test_import_utils(self):
        """Test importing utils module."""
        from src import utils
        assert hasattr(utils, 'setup_logging')
        assert hasattr(utils, 'ConversionStats')

    def test_import_converter(self):
        """Test importing converter module."""
        from src import converter
        assert hasattr(converter, 'TechDocConverter')
        assert hasattr(converter, 'ConversionResult')

    def test_import_cli(self):
        """Test importing CLI module."""
        from src import cli
        assert hasattr(cli, 'cli')
        assert hasattr(cli, 'convert')
        assert hasattr(cli, 'batch')

    def test_import_all_configs(self):
        """Test importing all config classes directly."""
        from src.config import (
            ConverterConfig,
            OptimizerConfig,
            ChunkerConfig,
            AppConfig,
            load_config
        )
        # Just verify they can be imported
        assert ConverterConfig is not None
        assert OptimizerConfig is not None
        assert ChunkerConfig is not None
        assert AppConfig is not None
        assert callable(load_config)

    def test_import_all_utils(self):
        """Test importing all utility functions directly."""
        from src.utils import setup_logging, ConversionStats
        assert callable(setup_logging)
        assert ConversionStats is not None

    def test_import_converter_components(self):
        """Test importing converter components directly."""
        from src.converter import TechDocConverter, ConversionResult
        assert TechDocConverter is not None
        assert ConversionResult is not None

    def test_docling_imports(self):
        """Test that docling dependencies can be imported."""
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.accelerator_options import AcceleratorDevice
            assert DocumentConverter is not None
            assert AcceleratorDevice is not None
        except ImportError as e:
            pytest.skip(f"Docling not installed or available: {e}")

    def test_click_imports(self):
        """Test that Click dependency is available."""
        try:
            import click
            from click.testing import CliRunner
            assert click is not None
            assert CliRunner is not None
        except ImportError as e:
            pytest.fail(f"Click not installed: {e}")

    def test_rich_imports(self):
        """Test that Rich dependency is available."""
        try:
            from rich.console import Console
            from rich.progress import Progress
            assert Console is not None
            assert Progress is not None
        except ImportError as e:
            pytest.fail(f"Rich not installed: {e}")
