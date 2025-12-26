# document_processor.py - FIXED
"""Advanced document loading and processing with metadata extraction"""
import logging
import re
from typing import List, Dict, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    import fitz  # PyMuPDF - better PDF parsing

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedPDFLoader:
    """Enhanced PDF loader with metadata extraction"""

    def __init__(self, file_path: str, extract_metadata: bool = True):
        self.file_path = file_path
        self.extract_metadata = extract_metadata

    def load(self) -> List[Document]:
        """Load PDF with enhanced metadata"""
        documents = []

        if PYMUPDF_AVAILABLE:
            documents = self._load_with_pymupdf()
        else:
            documents = self._load_with_langchain()

        # Post-process to clean text
        for doc in documents:
            doc.page_content = self._clean_text(doc.page_content)

        return documents

    def _load_with_pymupdf(self) -> List[Document]:
        """Load using PyMuPDF for better quality"""
        documents = []

        try:
            pdf_document = fitz.open(self.file_path)

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]

                # Extract text with layout preservation
                text = page.get_text("text", sort=True)

                # Extract metadata
                metadata = {
                    "source": self.file_path,
                    "page": page_num + 1,
                    "total_pages": len(pdf_document),
                }

                if self.extract_metadata:
                    # Extract page-level metadata
                    metadata.update({
                        "file_name": Path(self.file_path).name,
                        "page_width": page.rect.width,
                        "page_height": page.rect.height,
                    })

                    # Try to extract section/heading
                    section = self._extract_section_from_text(text)
                    if section:
                        metadata["section"] = section

                if text.strip():  # Only add non-empty pages
                    documents.append(Document(
                        page_content=text,
                        metadata=metadata
                    ))

            pdf_document.close()
            logger.info(f"✓ Loaded {len(documents)} pages from {Path(self.file_path).name} using PyMuPDF")

        except Exception as e:
            logger.error(f"PyMuPDF loading failed: {e}")
            return self._load_with_langchain()

        return documents

    def _load_with_langchain(self) -> List[Document]:
        """Fallback to LangChain loader"""
        try:
            loader = PyPDFLoader(self.file_path)
            documents = loader.load()

            # Enhance metadata
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    "page": i + 1,
                    "file_name": Path(self.file_path).name,
                })

                if self.extract_metadata:
                    section = self._extract_section_from_text(doc.page_content)
                    if section:
                        doc.metadata["section"] = section

            logger.info(f"✓ Loaded {len(documents)} pages from {Path(self.file_path).name} using PyPDFLoader")
            return documents

        except Exception as e:
            logger.error(f"Failed to load {self.file_path}: {e}")
            raise

    def _extract_section_from_text(self, text: str) -> Optional[str]:
        """Extract section heading from text"""
        lines = text.strip().split('\n')

        # Look for heading patterns in first few lines
        for line in lines[:5]:
            line = line.strip()

            # Common heading patterns
            if re.match(r'^(SECTION|Section|CHAPTER|Chapter)\s+\d+', line, re.IGNORECASE):
                return line

            # Numbered sections
            if re.match(r'^\d+\.\s+[A-Z]', line):
                return line

            # All caps short lines (likely headings)
            if line.isupper() and 5 < len(line) < 60:
                return line

        return None

    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)

        # Remove page numbers (common patterns)
        text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        return text.strip()


class DocumentProcessor:
    """Advanced document processor with smart chunking"""

    def __init__(self, config):
        self.config = config
        self.splitter = self._create_splitter()

    def _create_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create advanced text splitter - FIXED to handle both old and new config"""
        # Handle both old flat config and new nested config
        if hasattr(self.config, 'chunking'):
            # New nested config
            chunk_size = self.config.chunking.chunk_size
            chunk_overlap = self.config.chunking.chunk_overlap
            separators = self.config.chunking.separators
        else:
            # Old flat config (backward compatibility)
            chunk_size = getattr(self.config, 'chunk_size', 1500)
            chunk_overlap = getattr(self.config, 'chunk_overlap', 300)
            separators = getattr(self.config, 'separators', [
                "\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""
            ])

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )

    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document with metadata"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()

        # Get extract_metadata setting from config (handle both formats)
        extract_metadata = getattr(self.config, 'extract_metadata', True)

        try:
            if suffix == ".pdf":
                loader = EnhancedPDFLoader(
                    str(path),
                    extract_metadata=extract_metadata
                )
                docs = loader.load()

            elif suffix == ".docx":
                loader = Docx2txtLoader(str(path))
                docs = loader.load()

                # Add metadata
                for i, doc in enumerate(docs):
                    doc.metadata.update({
                        "file_name": path.name,
                        "page": i + 1,
                    })

            elif suffix == ".txt":
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                docs = [Document(
                    page_content=content,
                    metadata={
                        "source": str(path),
                        "file_name": path.name,
                    }
                )]
            else:
                raise ValueError(f"Unsupported file type: {suffix}")

            # Log statistics
            total_chars = sum(len(doc.page_content) for doc in docs)
            logger.info(f"✓ {path.name}: {len(docs)} pages, {total_chars:,} characters")

            return docs

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise

    def load_documents(self, folder: str) -> List[Document]:
        """Load all documents with metadata"""
        folder_path = Path(folder)
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder}")

        documents = []
        supported_extensions = {".pdf", ".docx", ".txt"}

        files = [
            f for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        logger.info(f"Found {len(files)} documents to process")

        for file_path in files:
            try:
                docs = self.load_document(str(file_path))
                documents.extend(docs)
            except Exception as e:
                logger.warning(f"Skipped {file_path.name}: {e}")

        total_chars = sum(len(doc.page_content) for doc in documents)
        logger.info(
            f"Successfully loaded {len(documents)} pages "
            f"from {len(files)} files ({total_chars:,} total characters)"
        )

        return documents

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents with metadata preservation"""
        if not docs:
            return []

        chunks = []

        for doc in docs:
            # Split the document
            doc_chunks = self.splitter.split_documents([doc])

            # Enhance chunk metadata
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(doc_chunks),
                })

                # Add snippet for reference
                chunk.metadata["snippet"] = chunk.page_content[:100] + "..."

            chunks.extend(doc_chunks)

        logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")

        # Log chunk statistics
        if chunks:
            chunk_sizes = [len(c.page_content) for c in chunks]
            logger.info(
                f"Chunk sizes - Min: {min(chunk_sizes)}, "
                f"Max: {max(chunk_sizes)}, "
                f"Avg: {sum(chunk_sizes) // len(chunk_sizes)}"
            )

        return chunks

    def get_document_text(self, file_path: str) -> str:
        """Get full text from document"""
        docs = self.load_document(file_path)
        return "\n\n".join(doc.page_content for doc in docs)

    def get_document_with_metadata(self, file_path: str) -> List[Dict]:
        """Get document with full metadata for analysis"""
        docs = self.load_document(file_path)

        return [
            {
                "page": doc.metadata.get("page", 0),
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ]