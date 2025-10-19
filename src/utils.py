"""
Utility functions for logging, error handling, and common operations.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    verbose: bool = False,
    quiet: bool = False
) -> logging.Logger:
    """
    Setup application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        verbose: Enable verbose logging (DEBUG level)
        quiet: Suppress all output except errors

    Returns:
        Root logger instance
    """
    # Determine log level
    if verbose:
        level = "DEBUG"
    elif quiet:
        level = "ERROR"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always DEBUG to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


class ConversionStats:
    """Track conversion statistics."""

    def __init__(self):
        self.total_files = 0
        self.successful = 0
        self.failed = 0
        self.total_pages = 0
        self.total_figures = 0
        self.total_time = 0.0
        self.errors = []

    def add_success(self, pages: int = 0, figures: int = 0, time: float = 0.0):
        """Record a successful conversion."""
        self.successful += 1
        self.total_files += 1
        self.total_pages += pages
        self.total_figures += figures
        self.total_time += time

    def add_failure(self, file_path: Path, error: str):
        """Record a failed conversion."""
        self.failed += 1
        self.total_files += 1
        self.errors.append((file_path, error))

    def get_report(self) -> str:
        """Generate a summary report."""
        report = [
            "\n" + "="*60,
            "Conversion Summary",
            "="*60,
            f"Total files processed: {self.total_files}",
            f"Successful: {self.successful}",
            f"Failed: {self.failed}",
            f"Total pages: {self.total_pages}",
            f"Total figures: {self.total_figures}",
            f"Total time: {self.total_time:.2f}s",
        ]

        if self.successful > 0:
            avg_time = self.total_time / self.successful
            report.append(f"Average time per document: {avg_time:.2f}s")

        if self.errors:
            report.append("\nFailed conversions:")
            for file_path, error in self.errors:
                report.append(f"  - {file_path.name}: {error}")

        report.append("="*60)
        return "\n".join(report)
