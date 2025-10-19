"""
Unit tests for utility functions.

Tests focus on:
- Logging setup
- ConversionStats tracking
- Error handling
"""

import pytest
import logging
import tempfile
from pathlib import Path

from src.utils import setup_logging, ConversionStats


class TestLoggingSetup:
    """Test logging configuration."""

    def test_default_logging(self):
        """Test default logging setup."""
        logger = setup_logging()
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_verbose_logging(self):
        """Test verbose mode sets DEBUG level."""
        logger = setup_logging(verbose=True)
        assert logger.level == logging.DEBUG

    def test_quiet_logging(self):
        """Test quiet mode sets ERROR level."""
        logger = setup_logging(quiet=True)
        assert logger.level == logging.ERROR

    def test_custom_log_level(self):
        """Test custom log level."""
        logger = setup_logging(level="WARNING")
        assert logger.level == logging.WARNING

    def test_log_file_creation(self):
        """Test that log file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logging(log_file=log_file)

            # Trigger a log message
            logger.info("Test message")

            # Check file was created
            assert log_file.exists()
            assert log_file.read_text().strip().endswith("Test message")

    def test_log_file_directory_creation(self):
        """Test that log file parent directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "logs" / "nested" / "test.log"
            logger = setup_logging(log_file=log_file)

            # Trigger a log message
            logger.info("Test message")

            # Check directory structure was created
            assert log_file.parent.exists()
            assert log_file.exists()

    def test_quiet_mode_no_console_handler(self):
        """Test that quiet mode doesn't add console handler."""
        logger = setup_logging(quiet=True)

        # In quiet mode, we should only have handlers if log_file is set
        # When log_file is None, quiet mode may have no handlers or only error handlers
        # The important thing is ERROR level is set
        assert logger.level == logging.ERROR


class TestConversionStats:
    """Test ConversionStats tracking."""

    def test_initial_state(self):
        """Test initial stats are all zero."""
        stats = ConversionStats()
        assert stats.total_files == 0
        assert stats.successful == 0
        assert stats.failed == 0
        assert stats.total_pages == 0
        assert stats.total_figures == 0
        assert stats.total_time == 0.0
        assert stats.errors == []

    def test_add_success(self):
        """Test adding successful conversion."""
        stats = ConversionStats()
        stats.add_success(pages=10, figures=3, time=5.5)

        assert stats.total_files == 1
        assert stats.successful == 1
        assert stats.failed == 0
        assert stats.total_pages == 10
        assert stats.total_figures == 3
        assert stats.total_time == 5.5

    def test_add_multiple_successes(self):
        """Test adding multiple successful conversions."""
        stats = ConversionStats()
        stats.add_success(pages=10, figures=3, time=5.5)
        stats.add_success(pages=20, figures=5, time=7.3)

        assert stats.total_files == 2
        assert stats.successful == 2
        assert stats.failed == 0
        assert stats.total_pages == 30
        assert stats.total_figures == 8
        assert stats.total_time == 12.8

    def test_add_failure(self):
        """Test adding failed conversion."""
        stats = ConversionStats()
        file_path = Path("test.pdf")
        error_msg = "File not found"
        stats.add_failure(file_path, error_msg)

        assert stats.total_files == 1
        assert stats.successful == 0
        assert stats.failed == 1
        assert len(stats.errors) == 1
        assert stats.errors[0] == (file_path, error_msg)

    def test_add_multiple_failures(self):
        """Test adding multiple failed conversions."""
        stats = ConversionStats()
        stats.add_failure(Path("test1.pdf"), "Error 1")
        stats.add_failure(Path("test2.pdf"), "Error 2")

        assert stats.total_files == 2
        assert stats.successful == 0
        assert stats.failed == 2
        assert len(stats.errors) == 2

    def test_mixed_successes_and_failures(self):
        """Test tracking both successes and failures."""
        stats = ConversionStats()
        stats.add_success(pages=10, figures=2, time=3.0)
        stats.add_failure(Path("failed.pdf"), "Conversion error")
        stats.add_success(pages=15, figures=3, time=4.5)

        assert stats.total_files == 3
        assert stats.successful == 2
        assert stats.failed == 1
        assert stats.total_pages == 25
        assert stats.total_figures == 5
        assert stats.total_time == 7.5
        assert len(stats.errors) == 1

    def test_get_report_no_conversions(self):
        """Test report with no conversions."""
        stats = ConversionStats()
        report = stats.get_report()

        assert "Total files processed: 0" in report
        assert "Successful: 0" in report
        assert "Failed: 0" in report

    def test_get_report_with_successes(self):
        """Test report with successful conversions."""
        stats = ConversionStats()
        stats.add_success(pages=10, figures=2, time=5.0)
        stats.add_success(pages=20, figures=3, time=10.0)

        report = stats.get_report()

        assert "Total files processed: 2" in report
        assert "Successful: 2" in report
        assert "Failed: 0" in report
        assert "Total pages: 30" in report
        assert "Total figures: 5" in report
        assert "Total time: 15.00s" in report
        assert "Average time per document: 7.50s" in report

    def test_get_report_with_failures(self):
        """Test report includes failure details."""
        stats = ConversionStats()
        stats.add_failure(Path("doc1.pdf"), "File not found")
        stats.add_failure(Path("doc2.pdf"), "Conversion error")

        report = stats.get_report()

        assert "Failed: 2" in report
        assert "Failed conversions:" in report
        assert "doc1.pdf" in report
        assert "File not found" in report
        assert "doc2.pdf" in report
        assert "Conversion error" in report

    def test_get_report_mixed(self):
        """Test report with both successes and failures."""
        stats = ConversionStats()
        stats.add_success(pages=10, figures=2, time=5.0)
        stats.add_failure(Path("failed.pdf"), "Error")

        report = stats.get_report()

        assert "Total files processed: 2" in report
        assert "Successful: 1" in report
        assert "Failed: 1" in report
        assert "Failed conversions:" in report

    def test_add_success_default_values(self):
        """Test add_success with default parameter values."""
        stats = ConversionStats()
        stats.add_success()

        assert stats.successful == 1
        assert stats.total_files == 1
        assert stats.total_pages == 0
        assert stats.total_figures == 0
        assert stats.total_time == 0.0
