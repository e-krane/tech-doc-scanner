"""
Configuration management for the document conversion agent.

This module defines dataclass configurations for all components:
- ConverterConfig: Document conversion settings
- OptimizerConfig: Pipeline optimization settings
- ChunkerConfig: Document chunking settings
- AppConfig: Application-wide configuration
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ConverterConfig:
    """Configuration for document conversion."""

    output_dir: Path = Path("output")
    save_figures: bool = True
    enable_profiling: bool = False
    num_threads: int = 4

    # GPU acceleration settings
    use_gpu: bool = True  # Auto-detect GPU, fallback to CPU
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # Document processing
    do_ocr: bool = True
    do_table_structure: bool = True
    do_cell_matching: bool = True

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Ensure output_dir is a Path object
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

        # Create output directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "markdown").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)

        # Validate num_threads
        if self.num_threads < 1:
            raise ValueError("num_threads must be at least 1")

        # Validate device
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}")


@dataclass
class OptimizerConfig:
    """Configuration for document optimization."""

    auto_detect_language: bool = True
    auto_toggle_ocr: bool = True
    auto_detect_tables: bool = True
    default_language: str = "en"

    # OCR quality threshold (0.0-1.0)
    # If text extraction quality is above this, disable OCR
    ocr_quality_threshold: float = 0.8

    # Sampling settings for document analysis
    sample_pages: int = 3  # Number of pages to sample for analysis
    max_pages_to_analyze: int = 10  # Max pages to analyze for optimization

    def __post_init__(self):
        """Validate configuration."""
        if not (0.0 <= self.ocr_quality_threshold <= 1.0):
            raise ValueError("ocr_quality_threshold must be between 0.0 and 1.0")

        if self.sample_pages < 1:
            raise ValueError("sample_pages must be at least 1")


@dataclass
class ChunkerConfig:
    """Configuration for document chunking."""

    max_tokens: int = 512
    merge_peers: bool = True
    use_semantic_splitting: bool = True
    preserve_structure: bool = True

    # Tokenizer model for token counting
    tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunk size constraints (for fallback chunking)
    chunk_size: int = 1000  # Target characters per chunk
    chunk_overlap: int = 200  # Character overlap between chunks
    max_chunk_size: int = 2000  # Maximum chunk size
    min_chunk_size: int = 100  # Minimum chunk size

    def __post_init__(self):
        """Validate configuration."""
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")

        if self.max_chunk_size < self.chunk_size:
            raise ValueError("max_chunk_size must be >= chunk_size")


@dataclass
class AppConfig:
    """Application-wide configuration."""

    converter: ConverterConfig = field(default_factory=ConverterConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)

    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    # CLI settings
    verbose: bool = False
    quiet: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables.

        Environment variables:
        - OUTPUT_DIR: Output directory path
        - NUM_THREADS: Number of processing threads
        - USE_GPU: Enable GPU acceleration (true/false)
        - MAX_TOKENS: Maximum tokens per chunk
        - LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)

        Returns:
            AppConfig instance with settings from environment
        """
        converter = ConverterConfig(
            output_dir=Path(os.getenv("OUTPUT_DIR", "output")),
            num_threads=int(os.getenv("NUM_THREADS", "4")),
            use_gpu=os.getenv("USE_GPU", "true").lower() == "true",
            enable_profiling=os.getenv("ENABLE_PROFILING", "false").lower() == "true",
        )

        optimizer = OptimizerConfig(
            default_language=os.getenv("DEFAULT_LANGUAGE", "en"),
        )

        chunker = ChunkerConfig(
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
        )

        config = cls(
            converter=converter,
            optimizer=optimizer,
            chunker=chunker,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            verbose=os.getenv("VERBOSE", "false").lower() == "true",
        )

        return config

    def __post_init__(self):
        """Validate application configuration."""
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")

        # Normalize log level to uppercase
        self.log_level = self.log_level.upper()

        # Ensure log_file is a Path if provided
        if self.log_file and not isinstance(self.log_file, Path):
            self.log_file = Path(self.log_file)


def load_config() -> AppConfig:
    """Load application configuration from environment variables.

    Returns:
        AppConfig instance with settings from environment or defaults
    """
    return AppConfig.from_env()
