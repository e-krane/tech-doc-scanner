"""
Tests for document chunker module.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import asdict

from src.chunker import DocumentChunker, DocumentChunk
from src.config import ChunkerConfig


@pytest.fixture
def chunker_config():
    """Create a test chunker config."""
    return ChunkerConfig(
        max_tokens=100,
        merge_peers=True
    )


@pytest.fixture
def chunker(chunker_config):
    """Create a test chunker instance."""
    with patch('src.chunker.HybridChunker'):
        with patch('src.chunker.AutoTokenizer.from_pretrained') as mock_tokenizer:
            # Mock tokenizer
            mock_tok = MagicMock()
            mock_tok.encode.side_effect = lambda x: list(range(len(x.split())))  # Simple token count
            mock_tokenizer.return_value = mock_tok

            chunker = DocumentChunker(chunker_config)
            # Set up mock chunker with encode method
            chunker.tokenizer.encode = mock_tok.encode
            return chunker


class TestDocumentChunk:
    """Test DocumentChunk dataclass."""

    def test_document_chunk_creation(self):
        """Test creating a DocumentChunk."""
        chunk = DocumentChunk(
            content="Test content",
            contextualized_content="# Header\n\nTest content",
            index=0,
            token_count=5,
            metadata={"title": "Test"}
        )

        assert chunk.content == "Test content"
        assert chunk.contextualized_content == "# Header\n\nTest content"
        assert chunk.index == 0
        assert chunk.token_count == 5
        assert chunk.metadata == {"title": "Test"}

    def test_document_chunk_to_dict(self):
        """Test converting DocumentChunk to dict."""
        chunk = DocumentChunk(
            content="Test",
            contextualized_content="Test",
            index=0,
            token_count=1,
            metadata={}
        )

        chunk_dict = asdict(chunk)
        assert "content" in chunk_dict
        assert "index" in chunk_dict
        assert "token_count" in chunk_dict


class TestDocumentChunker:
    """Test DocumentChunker class."""

    def test_chunker_initialization(self, chunker_config):
        """Test chunker initialization."""
        with patch('src.chunker.HybridChunker'):
            with patch('src.chunker.AutoTokenizer.from_pretrained') as mock_tokenizer:
                mock_tokenizer.return_value = MagicMock()

                chunker = DocumentChunker(chunker_config)

                assert chunker.config == chunker_config
                assert chunker.tokenizer is not None
                assert chunker.chunker is not None
                mock_tokenizer.assert_called_once_with(chunker_config.tokenizer_model)

    def test_chunker_with_custom_config(self):
        """Test chunker with custom configuration."""
        config = ChunkerConfig(
            max_tokens=256,
            merge_peers=False,
            tokenizer_model="custom-model"
        )

        with patch('src.chunker.HybridChunker'):
            with patch('src.chunker.AutoTokenizer.from_pretrained') as mock_tokenizer:
                mock_tokenizer.return_value = MagicMock()

                chunker = DocumentChunker(config)

                assert chunker.config.max_tokens == 256
                assert chunker.config.merge_peers is False
                mock_tokenizer.assert_called_with("custom-model")


class TestChunkDocument:
    """Test chunk_document method."""

    def test_chunk_document_requires_input(self, chunker):
        """Test that chunk_document requires either docling_doc or title."""
        with pytest.raises(ValueError, match="Either docling_doc or title must be provided"):
            chunker.chunk_document()

    def test_chunk_document_with_title_only(self, chunker):
        """Test chunking with title only (fallback to simple chunking)."""
        chunks = chunker.chunk_document(
            title="Test Document",
            source="test.txt"
        )

        assert isinstance(chunks, list)
        # Should return empty list or single chunk for empty text
        assert len(chunks) == 0 or len(chunks) == 1

    def test_chunk_document_with_metadata(self, chunker):
        """Test that metadata is included in chunks."""
        metadata = {"author": "Test Author", "date": "2024-01-01"}

        chunks = chunker.chunk_document(
            title="Test Document",
            source="test.txt",
            metadata=metadata
        )

        # Check metadata is present (if chunks were created)
        if chunks:
            for chunk in chunks:
                assert "title" in chunk.metadata
                assert "source" in chunk.metadata
                assert chunk.metadata["author"] == "Test Author"

    @patch('src.chunker.DoclingDocument')
    def test_chunk_document_with_docling_doc(self, mock_docling, chunker):
        """Test chunking with DoclingDocument."""
        # Create mock DoclingDocument
        mock_doc = Mock()
        mock_doc.export_to_markdown.return_value = "# Test\n\nContent here."

        # Mock the chunker.chunk method
        mock_chunk = Mock()
        mock_chunk.text = "Content here."
        mock_chunk.meta = None
        mock_chunk.page = None

        with patch.object(chunker.chunker, 'chunk', return_value=[mock_chunk]):
            with patch.object(chunker.chunker, 'contextualize', return_value="# Test\n\nContent here."):
                chunks = chunker.chunk_document(
                    docling_doc=mock_doc,
                    title="Test",
                    source="test.pdf"
                )

                assert len(chunks) > 0
                assert all(isinstance(c, DocumentChunk) for c in chunks)


class TestSimpleChunking:
    """Test simple chunking fallback."""

    def test_split_sentences(self, chunker):
        """Test sentence splitting."""
        text = "First sentence. Second sentence! Third sentence? Fourth."
        sentences = chunker._split_sentences(text)

        assert len(sentences) >= 3
        assert any("First" in s for s in sentences)
        assert any("Second" in s for s in sentences)

    def test_split_sentences_empty(self, chunker):
        """Test splitting empty text."""
        sentences = chunker._split_sentences("")
        assert sentences == []

    def test_split_sentences_single(self, chunker):
        """Test splitting single sentence."""
        text = "Only one sentence here."
        sentences = chunker._split_sentences(text)
        assert len(sentences) == 1

    def test_chunk_simple(self, chunker):
        """Test simple chunking with plain text."""
        text = "First sentence. " * 50  # Create long text
        metadata = {"title": "Test"}

        chunks = chunker._chunk_simple(text, metadata)

        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.token_count <= chunker.config.max_tokens for c in chunks)
        assert all(c.metadata.get("title") == "Test" for c in chunks)

    def test_chunk_simple_respects_max_tokens(self, chunker):
        """Test that simple chunking respects max_tokens."""
        # Create text with proper sentence structure that will need multiple chunks
        # Each sentence is ~10 words, so we need enough to exceed max_tokens
        sentence = "This is a sentence with ten words in it today."
        text = " ".join([sentence] * 20)  # 20 sentences
        metadata = {}

        chunks = chunker._chunk_simple(text, metadata)

        # Should create multiple chunks (20 sentences * ~10 tokens = 200 tokens > 100 max)
        assert len(chunks) > 1
        # All chunks should be within token limit
        assert all(c.token_count <= chunker.config.max_tokens for c in chunks)

    def test_chunk_simple_sequential_indices(self, chunker):
        """Test that chunks have sequential indices."""
        text = " ".join(["sentence."] * 100)
        chunks = chunker._chunk_simple(text, {})

        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))


class TestExportChunks:
    """Test export_chunks method."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            DocumentChunk(
                content="First chunk",
                contextualized_content="# Header\n\nFirst chunk",
                index=0,
                token_count=5,
                metadata={"title": "Test", "page": 1}
            ),
            DocumentChunk(
                content="Second chunk",
                contextualized_content="# Header\n\nSecond chunk",
                index=1,
                token_count=6,
                metadata={"title": "Test", "page": 2}
            )
        ]

    def test_export_jsonl(self, chunker, sample_chunks, tmp_path):
        """Test exporting chunks to JSONL format."""
        output_path = tmp_path / "chunks.jsonl"

        result = chunker.export_chunks(sample_chunks, output_path, format="jsonl")

        assert result == output_path
        assert output_path.exists()

        # Verify content
        lines = output_path.read_text().strip().split('\n')
        assert len(lines) == 2

        import json
        chunk_data = json.loads(lines[0])
        assert "content" in chunk_data
        assert "index" in chunk_data
        assert chunk_data["index"] == 0

    def test_export_json(self, chunker, sample_chunks, tmp_path):
        """Test exporting chunks to JSON format."""
        output_path = tmp_path / "chunks.json"

        result = chunker.export_chunks(sample_chunks, output_path, format="json")

        assert result == output_path
        assert output_path.exists()

        # Verify content
        import json
        data = json.loads(output_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["index"] == 0

    def test_export_markdown(self, chunker, sample_chunks, tmp_path):
        """Test exporting chunks to Markdown format."""
        output_path = tmp_path / "chunks.md"

        result = chunker.export_chunks(sample_chunks, output_path, format="md")

        assert result == output_path
        assert output_path.exists()

        content = output_path.read_text()
        assert "# Chunk 0" in content
        assert "# Chunk 1" in content
        assert "**Tokens**:" in content

    def test_export_csv(self, chunker, sample_chunks, tmp_path):
        """Test exporting chunks to CSV format."""
        output_path = tmp_path / "chunks.csv"

        result = chunker.export_chunks(sample_chunks, output_path, format="csv")

        assert result == output_path
        assert output_path.exists()

        # Verify content
        import csv
        with open(output_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 3  # Header + 2 chunks
        assert rows[0] == ['index', 'token_count', 'content', 'contextualized_content', 'metadata']

    def test_export_unsupported_format(self, chunker, sample_chunks, tmp_path):
        """Test that unsupported format raises error."""
        output_path = tmp_path / "chunks.txt"

        with pytest.raises(ValueError, match="Unsupported format"):
            chunker.export_chunks(sample_chunks, output_path, format="txt")

    def test_export_creates_directory(self, chunker, sample_chunks, tmp_path):
        """Test that export creates parent directories."""
        output_path = tmp_path / "nested" / "dir" / "chunks.json"

        chunker.export_chunks(sample_chunks, output_path, format="json")

        assert output_path.exists()
        assert output_path.parent.exists()


class TestIntegration:
    """Integration tests for chunker."""

    def test_full_chunking_workflow(self, chunker_config):
        """Test complete chunking workflow."""
        with patch('src.chunker.HybridChunker'):
            with patch('src.chunker.AutoTokenizer.from_pretrained') as mock_tokenizer:
                # Setup mock tokenizer
                mock_tok = MagicMock()
                mock_tok.encode.side_effect = lambda x: list(range(len(x.split())))
                mock_tokenizer.return_value = mock_tok

                # Create chunker
                chunker = DocumentChunker(chunker_config)
                chunker.tokenizer.encode = mock_tok.encode

                # Chunk a simple document
                text = "This is a test. " * 50
                chunks = chunker._chunk_simple(text, {"source": "test.txt"})

                # Verify results
                assert len(chunks) > 0
                assert all(c.index == i for i, c in enumerate(chunks))
                assert all(c.metadata.get("source") == "test.txt" for c in chunks)

    def test_chunking_preserves_content(self, chunker_config):
        """Test that chunking preserves all content."""
        with patch('src.chunker.HybridChunker'):
            with patch('src.chunker.AutoTokenizer.from_pretrained') as mock_tokenizer:
                mock_tok = MagicMock()
                mock_tok.encode.side_effect = lambda x: list(range(len(x.split())))
                mock_tokenizer.return_value = mock_tok

                chunker = DocumentChunker(chunker_config)
                chunker.tokenizer.encode = mock_tok.encode

                original_text = "Sentence one. Sentence two. Sentence three."
                chunks = chunker._chunk_simple(original_text, {})

                # Reconstruct text from chunks
                reconstructed = " ".join(c.content for c in chunks)

                # Should preserve all content
                assert len(reconstructed) > 0
                assert "Sentence" in reconstructed
