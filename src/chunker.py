"""
Document chunker module using Docling's HybridChunker.

This module provides token-aware, structure-preserving chunking for RAG pipelines.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument

from .config import ChunkerConfig

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    A chunk of document content with metadata.

    Attributes:
        content: The chunk text content
        contextualized_content: Content with heading hierarchy prepended
        index: Chunk index in the document
        token_count: Number of tokens in contextualized content
        metadata: Additional metadata (headings, page numbers, etc.)
    """
    content: str
    contextualized_content: str
    index: int
    token_count: int
    metadata: Dict[str, Any]


class DocumentChunker:
    """
    Document chunker using Docling's HybridChunker with token awareness.

    Features:
    - Token-aware chunking with configurable max_tokens
    - Document structure preservation (headings, sections)
    - Contextualized chunks with heading hierarchy
    - Fallback to simple chunking for text-only documents

    The HybridChunker respects document structure (headings, paragraphs, tables)
    while ensuring chunks stay within token limits for embedding models.

    Example:
        >>> from src.config import ChunkerConfig
        >>> from src.chunker import DocumentChunker
        >>> from src.converter import TechDocConverter
        >>>
        >>> config = ChunkerConfig(max_tokens=512)
        >>> chunker = DocumentChunker(config)
        >>>
        >>> # After converting a document
        >>> result = converter.convert(Path("document.pdf"))
        >>> chunks = chunker.chunk_document(
        ...     docling_doc=result.docling_doc,
        ...     title=result.file_path.stem
        ... )
        >>> print(f"Created {len(chunks)} chunks")
    """

    def __init__(self, config: ChunkerConfig):
        """
        Initialize the document chunker.

        Args:
            config: Chunker configuration with max_tokens and other settings
        """
        self.config = config

        # Initialize tokenizer for token counting
        logger.info(f"Loading tokenizer: {config.tokenizer_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model)

        # Initialize HybridChunker
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=config.max_tokens,
            merge_peers=config.merge_peers
        )

        logger.info(
            f"DocumentChunker initialized: max_tokens={config.max_tokens}, "
            f"merge_peers={config.merge_peers}"
        )

    def chunk_document(
        self,
        docling_doc: Optional[DoclingDocument] = None,
        title: str = "",
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Chunk a document using HybridChunker or fallback to simple chunking.

        Args:
            docling_doc: DoclingDocument from conversion (None for text-only docs)
            title: Document title
            source: Source file path or identifier
            metadata: Additional metadata to include in chunks

        Returns:
            List of DocumentChunk objects with content and metadata

        Raises:
            ValueError: If neither docling_doc nor title is provided
        """
        if docling_doc is None and not title:
            raise ValueError("Either docling_doc or title must be provided")

        base_metadata = metadata or {}
        base_metadata.update({
            "title": title,
            "source": source
        })

        # Use HybridChunker if DoclingDocument is available
        if docling_doc is not None:
            return self._chunk_with_hybrid(docling_doc, base_metadata)
        else:
            # Fallback to simple chunking for text-only documents
            logger.info("DoclingDocument not available, using simple chunking")
            return self._chunk_simple(title, base_metadata)

    def _chunk_with_hybrid(
        self,
        docling_doc: DoclingDocument,
        base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Chunk using Docling's HybridChunker with structure awareness.

        Args:
            docling_doc: DoclingDocument with structure information
            base_metadata: Base metadata to include in all chunks

        Returns:
            List of DocumentChunk objects
        """
        logger.info("Chunking document with HybridChunker")

        chunks = []

        try:
            # Generate chunks using HybridChunker
            chunk_iter = self.chunker.chunk(dl_doc=docling_doc)

            for i, chunk in enumerate(chunk_iter):
                # Get contextualized text (includes heading hierarchy)
                contextualized_text = self.chunker.contextualize(chunk=chunk)

                # Get plain text content (without heading context)
                plain_text = chunk.text if hasattr(chunk, 'text') else str(chunk)

                # Count actual tokens in contextualized version
                token_count = len(self.tokenizer.encode(contextualized_text))

                # Extract chunk metadata
                chunk_metadata = base_metadata.copy()

                # Add heading information if available
                if hasattr(chunk, 'meta') and chunk.meta:
                    if hasattr(chunk.meta, 'headings'):
                        chunk_metadata['headings'] = chunk.meta.headings
                    if hasattr(chunk.meta, 'doc_items'):
                        chunk_metadata['doc_items'] = [
                            str(item) for item in chunk.meta.doc_items
                        ]

                # Add page information if available
                if hasattr(chunk, 'page') and chunk.page is not None:
                    chunk_metadata['page'] = chunk.page

                # Create DocumentChunk
                doc_chunk = DocumentChunk(
                    content=plain_text,
                    contextualized_content=contextualized_text,
                    index=i,
                    token_count=token_count,
                    metadata=chunk_metadata
                )

                chunks.append(doc_chunk)

                logger.debug(
                    f"Chunk {i}: {token_count} tokens, "
                    f"{len(plain_text)} chars"
                )

            logger.info(f"Created {len(chunks)} chunks with HybridChunker")

        except Exception as e:
            logger.error(f"HybridChunker failed: {e}", exc_info=True)
            # Fallback to simple chunking
            logger.info("Falling back to simple chunking")
            return self._chunk_simple(
                docling_doc.export_to_markdown(),
                base_metadata
            )

        return chunks

    def _chunk_simple(
        self,
        text: str,
        base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Simple fallback chunking by token count without structure awareness.

        This is used when DoclingDocument is not available (e.g., for plain text files)
        or when HybridChunker fails.

        Args:
            text: Plain text to chunk
            base_metadata: Base metadata to include in all chunks

        Returns:
            List of DocumentChunk objects
        """
        logger.info("Using simple token-based chunking")

        chunks = []

        # Split text into sentences (simple approach)
        sentences = self._split_sentences(text)

        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            # Count tokens in sentence
            sentence_tokens = len(self.tokenizer.encode(sentence))

            # Check if adding this sentence would exceed max_tokens
            if current_tokens + sentence_tokens > self.config.max_tokens and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    DocumentChunk(
                        content=chunk_text,
                        contextualized_content=chunk_text,  # No context in simple mode
                        index=chunk_index,
                        token_count=current_tokens,
                        metadata=base_metadata.copy()
                    )
                )

                # Start new chunk
                current_chunk = [sentence]
                current_tokens = sentence_tokens
                chunk_index += 1
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add final chunk if not empty
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                DocumentChunk(
                    content=chunk_text,
                    contextualized_content=chunk_text,
                    index=chunk_index,
                    token_count=current_tokens,
                    metadata=base_metadata.copy()
                )
            )

        logger.info(f"Created {len(chunks)} chunks with simple chunking")

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences (simple approach).

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting on common punctuation
        # Note: This is a basic approach; for production, consider using
        # a proper sentence tokenizer like nltk.sent_tokenize

        import re

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def export_chunks(
        self,
        chunks: List[DocumentChunk],
        output_path: Path,
        format: str = "jsonl"
    ) -> Path:
        """
        Export chunks to file in various formats.

        Args:
            chunks: List of DocumentChunk objects
            output_path: Path to output file
            format: Output format (jsonl, json, md, csv)

        Returns:
            Path to the exported file

        Raises:
            ValueError: If format is not supported
        """
        import json

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            # JSON Lines format (one JSON object per line)
            with open(output_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    chunk_dict = {
                        "content": chunk.content,
                        "contextualized_content": chunk.contextualized_content,
                        "index": chunk.index,
                        "token_count": chunk.token_count,
                        "metadata": chunk.metadata
                    }
                    f.write(json.dumps(chunk_dict, ensure_ascii=False) + '\n')

        elif format == "json":
            # Single JSON array
            chunks_data = [
                {
                    "content": chunk.content,
                    "contextualized_content": chunk.contextualized_content,
                    "index": chunk.index,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata
                }
                for chunk in chunks
            ]
            output_path.write_text(
                json.dumps(chunks_data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )

        elif format == "md":
            # Markdown format with metadata headers
            md_content = []
            for chunk in chunks:
                md_content.append(f"# Chunk {chunk.index}\n")
                md_content.append(f"**Tokens**: {chunk.token_count}\n")
                if chunk.metadata.get('headings'):
                    md_content.append(f"**Headings**: {' > '.join(chunk.metadata['headings'])}\n")
                md_content.append(f"\n{chunk.contextualized_content}\n")
                md_content.append("\n---\n\n")

            output_path.write_text(''.join(md_content), encoding='utf-8')

        elif format == "csv":
            # CSV format
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['index', 'token_count', 'content', 'contextualized_content', 'metadata'])
                for chunk in chunks:
                    writer.writerow([
                        chunk.index,
                        chunk.token_count,
                        chunk.content,
                        chunk.contextualized_content,
                        json.dumps(chunk.metadata)
                    ])

        else:
            raise ValueError(f"Unsupported format: {format}. Supported: jsonl, json, md, csv")

        logger.info(f"Exported {len(chunks)} chunks to {output_path} ({format} format)")

        return output_path
