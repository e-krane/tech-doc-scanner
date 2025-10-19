"""
Unit tests for CLI interface.

Tests focus on:
- CLI command parsing
- Help text
- Option validation
- Basic command structure
"""

import pytest
from click.testing import CliRunner

from src.cli import cli


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_cli_help(self):
        """Test that CLI help is displayed."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])

        assert result.exit_code == 0
        assert "Technical Document Conversion Agent" in result.output
        assert "convert" in result.output
        assert "batch" in result.output

    def test_cli_verbose_flag(self):
        """Test verbose flag is recognized."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--verbose', '--help'])

        assert result.exit_code == 0
        # Verbose flag should be accepted without error

    def test_cli_quiet_flag(self):
        """Test quiet flag is recognized."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--quiet', '--help'])

        assert result.exit_code == 0
        # Quiet flag should be accepted without error

    def test_cli_log_file_option(self):
        """Test log-file option is recognized."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--log-file', 'test.log', '--help'])

        assert result.exit_code == 0
        # Log file option should be accepted without error


class TestConvertCommand:
    """Test convert command."""

    def test_convert_help(self):
        """Test convert command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['convert', '--help'])

        assert result.exit_code == 0
        assert "Convert a single document" in result.output
        assert "--output-dir" in result.output or "-o" in result.output
        assert "--no-figures" in result.output
        assert "--profile" in result.output

    def test_convert_missing_file_argument(self):
        """Test convert command without file argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ['convert'])

        assert result.exit_code != 0
        # Should fail when no file is provided

    def test_convert_nonexistent_file(self):
        """Test convert command with non-existent file."""
        runner = CliRunner()
        result = runner.invoke(cli, ['convert', '/tmp/nonexistent_test_file.pdf'])

        # Click should validate file existence
        assert result.exit_code != 0

    def test_convert_with_output_dir(self):
        """Test convert command accepts output-dir option."""
        runner = CliRunner()
        # Using help to test option parsing without needing a real file
        result = runner.invoke(cli, ['convert', '--output-dir', 'custom_output', '--help'])

        assert result.exit_code == 0

    def test_convert_with_no_figures_flag(self):
        """Test convert command with no-figures flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ['convert', '--no-figures', '--help'])

        assert result.exit_code == 0

    def test_convert_with_profile_flag(self):
        """Test convert command with profile flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ['convert', '--profile', '--help'])

        assert result.exit_code == 0


class TestBatchCommand:
    """Test batch command."""

    def test_batch_help(self):
        """Test batch command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['batch', '--help'])

        assert result.exit_code == 0
        assert "Convert all documents" in result.output
        assert "--output-dir" in result.output or "-o" in result.output
        assert "--no-figures" in result.output
        assert "--pattern" in result.output or "-p" in result.output

    def test_batch_missing_directory_argument(self):
        """Test batch command without directory argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ['batch'])

        assert result.exit_code != 0
        # Should fail when no directory is provided

    def test_batch_nonexistent_directory(self):
        """Test batch command with non-existent directory."""
        runner = CliRunner()
        result = runner.invoke(cli, ['batch', '/tmp/nonexistent_directory_12345'])

        # Click should validate directory existence
        assert result.exit_code != 0

    def test_batch_with_output_dir(self):
        """Test batch command accepts output-dir option."""
        runner = CliRunner()
        result = runner.invoke(cli, ['batch', '--output-dir', 'custom_output', '--help'])

        assert result.exit_code == 0

    def test_batch_with_pattern(self):
        """Test batch command accepts pattern option."""
        runner = CliRunner()
        result = runner.invoke(cli, ['batch', '--pattern', '*.docx', '--help'])

        assert result.exit_code == 0

    def test_batch_with_no_figures_flag(self):
        """Test batch command with no-figures flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ['batch', '--no-figures', '--help'])

        assert result.exit_code == 0

    def test_batch_empty_directory(self):
        """Test batch command with empty directory."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create empty directory
            import os
            os.mkdir('empty_dir')

            result = runner.invoke(cli, ['batch', 'empty_dir'])

            # Should succeed but report no files found
            assert result.exit_code == 0


class TestCLIIntegration:
    """Test CLI integration and option combinations."""

    def test_verbose_and_quiet_together(self):
        """Test using both verbose and quiet flags."""
        runner = CliRunner()
        # Both flags should be accepted (quiet typically takes precedence)
        result = runner.invoke(cli, ['--verbose', '--quiet', '--help'])

        assert result.exit_code == 0

    def test_convert_with_multiple_options(self):
        """Test convert command with multiple options."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            '--verbose',
            'convert',
            '--output-dir', 'output',
            '--no-figures',
            '--profile',
            '--help'
        ])

        assert result.exit_code == 0

    def test_batch_with_multiple_options(self):
        """Test batch command with multiple options."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            '--quiet',
            'batch',
            '--output-dir', 'output',
            '--pattern', '*.pdf',
            '--no-figures',
            '--help'
        ])

        assert result.exit_code == 0
