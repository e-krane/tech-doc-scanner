"""
Unit tests for configuration management.

Tests focus on:
- Config validation (invalid values should raise errors)
- Default values
- Environment variable parsing
- Directory creation
"""

import pytest
import os
import tempfile
from pathlib import Path

from src.config import (
    ConverterConfig,
    OptimizerConfig,
    ChunkerConfig,
    AppConfig,
    load_config
)


class TestConverterConfig:
    """Test ConverterConfig validation and initialization."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConverterConfig()
        assert config.output_dir == Path("output")
        assert config.save_figures is True
        assert config.enable_profiling is False
        assert config.num_threads == 4
        assert config.use_gpu is True
        assert config.device == "auto"

    def test_output_dir_creation(self):
        """Test that output directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_output"
            config = ConverterConfig(output_dir=output_dir)

            assert output_dir.exists()
            assert (output_dir / "markdown").exists()
            assert (output_dir / "figures").exists()

    def test_invalid_num_threads(self):
        """Test that num_threads < 1 raises error."""
        with pytest.raises(ValueError, match="num_threads must be at least 1"):
            ConverterConfig(num_threads=0)

        with pytest.raises(ValueError, match="num_threads must be at least 1"):
            ConverterConfig(num_threads=-1)

    def test_invalid_device(self):
        """Test that invalid device raises error."""
        with pytest.raises(ValueError, match="device must be one of"):
            ConverterConfig(device="invalid")

        with pytest.raises(ValueError, match="device must be one of"):
            ConverterConfig(device="gpu")

    def test_valid_devices(self):
        """Test all valid device options."""
        for device in ["auto", "cpu", "cuda", "mps"]:
            config = ConverterConfig(device=device)
            assert config.device == device

    def test_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = ConverterConfig(output_dir="test_path")
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("test_path")


class TestOptimizerConfig:
    """Test OptimizerConfig validation."""

    def test_default_config(self):
        """Test default optimizer configuration."""
        config = OptimizerConfig()
        assert config.auto_detect_language is True
        assert config.auto_toggle_ocr is True
        assert config.default_language == "en"
        assert config.ocr_quality_threshold == 0.8
        assert config.sample_pages == 3

    def test_invalid_ocr_threshold(self):
        """Test that ocr_quality_threshold must be 0.0-1.0."""
        with pytest.raises(ValueError, match="ocr_quality_threshold must be between 0.0 and 1.0"):
            OptimizerConfig(ocr_quality_threshold=1.5)

        with pytest.raises(ValueError, match="ocr_quality_threshold must be between 0.0 and 1.0"):
            OptimizerConfig(ocr_quality_threshold=-0.1)

    def test_valid_ocr_threshold(self):
        """Test valid ocr_quality_threshold values."""
        config1 = OptimizerConfig(ocr_quality_threshold=0.0)
        assert config1.ocr_quality_threshold == 0.0

        config2 = OptimizerConfig(ocr_quality_threshold=1.0)
        assert config2.ocr_quality_threshold == 1.0

        config3 = OptimizerConfig(ocr_quality_threshold=0.5)
        assert config3.ocr_quality_threshold == 0.5

    def test_invalid_sample_pages(self):
        """Test that sample_pages must be at least 1."""
        with pytest.raises(ValueError, match="sample_pages must be at least 1"):
            OptimizerConfig(sample_pages=0)


class TestChunkerConfig:
    """Test ChunkerConfig validation."""

    def test_default_config(self):
        """Test default chunker configuration."""
        config = ChunkerConfig()
        assert config.max_tokens == 512
        assert config.merge_peers is True
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200

    def test_invalid_max_tokens(self):
        """Test that max_tokens must be at least 1."""
        with pytest.raises(ValueError, match="max_tokens must be at least 1"):
            ChunkerConfig(max_tokens=0)

        with pytest.raises(ValueError, match="max_tokens must be at least 1"):
            ChunkerConfig(max_tokens=-1)

    def test_invalid_chunk_overlap(self):
        """Test that chunk_overlap must be less than chunk_size."""
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            ChunkerConfig(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            ChunkerConfig(chunk_size=100, chunk_overlap=200)

    def test_invalid_min_chunk_size(self):
        """Test that min_chunk_size must be positive."""
        with pytest.raises(ValueError, match="min_chunk_size must be positive"):
            ChunkerConfig(min_chunk_size=0)

        with pytest.raises(ValueError, match="min_chunk_size must be positive"):
            ChunkerConfig(min_chunk_size=-1)

    def test_invalid_max_chunk_size(self):
        """Test that max_chunk_size must be >= chunk_size."""
        with pytest.raises(ValueError, match="max_chunk_size must be >= chunk_size"):
            ChunkerConfig(chunk_size=1000, max_chunk_size=500)

    def test_valid_chunk_config(self):
        """Test valid chunk configuration."""
        config = ChunkerConfig(
            chunk_size=500,
            chunk_overlap=100,
            min_chunk_size=50,
            max_chunk_size=1000
        )
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100


class TestAppConfig:
    """Test AppConfig validation and environment loading."""

    def test_default_config(self):
        """Test default application configuration."""
        config = AppConfig()
        assert isinstance(config.converter, ConverterConfig)
        assert isinstance(config.optimizer, OptimizerConfig)
        assert isinstance(config.chunker, ChunkerConfig)
        assert config.log_level == "INFO"
        assert config.verbose is False
        assert config.quiet is False

    def test_invalid_log_level(self):
        """Test that invalid log level raises error."""
        with pytest.raises(ValueError, match="log_level must be one of"):
            AppConfig(log_level="INVALID")

    def test_valid_log_levels(self):
        """Test all valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = AppConfig(log_level=level)
            assert config.log_level == level

    def test_log_level_case_normalization(self):
        """Test that log level is normalized to uppercase."""
        config = AppConfig(log_level="debug")
        assert config.log_level == "DEBUG"

        config = AppConfig(log_level="InFo")
        assert config.log_level == "INFO"

    def test_log_file_path_conversion(self):
        """Test that log_file is converted to Path."""
        config = AppConfig(log_file="test.log")
        assert isinstance(config.log_file, Path)
        assert config.log_file == Path("test.log")

    def test_from_env_defaults(self):
        """Test loading config from environment with no env vars set."""
        # Clear relevant env vars
        for key in ["OUTPUT_DIR", "NUM_THREADS", "USE_GPU", "MAX_TOKENS", "LOG_LEVEL"]:
            os.environ.pop(key, None)

        config = AppConfig.from_env()
        assert config.converter.output_dir == Path("output")
        assert config.converter.num_threads == 4
        assert config.converter.use_gpu is True
        assert config.chunker.max_tokens == 512
        assert config.log_level == "INFO"

    def test_from_env_custom_values(self):
        """Test loading config from environment with custom values."""
        # Set environment variables
        os.environ["OUTPUT_DIR"] = "custom_output"
        os.environ["NUM_THREADS"] = "8"
        os.environ["USE_GPU"] = "false"
        os.environ["MAX_TOKENS"] = "1024"
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["VERBOSE"] = "true"

        try:
            config = AppConfig.from_env()
            assert config.converter.output_dir == Path("custom_output")
            assert config.converter.num_threads == 8
            assert config.converter.use_gpu is False
            assert config.chunker.max_tokens == 1024
            assert config.log_level == "DEBUG"
            assert config.verbose is True
        finally:
            # Clean up
            for key in ["OUTPUT_DIR", "NUM_THREADS", "USE_GPU", "MAX_TOKENS", "LOG_LEVEL", "VERBOSE"]:
                os.environ.pop(key, None)

    def test_load_config(self):
        """Test load_config function."""
        config = load_config()
        assert isinstance(config, AppConfig)
