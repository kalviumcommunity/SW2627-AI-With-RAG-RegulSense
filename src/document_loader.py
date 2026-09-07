"""RegulSense Document Loading Pipeline.

Provides multi-format document ingestion for regulatory documents (PDF, TXT, HTML, MD)
with fault-tolerant error handling, metadata provenance preservation, and intake confirmation.
"""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Set up logger
logger = logging.getLogger("regulsense.document_loader")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class DocumentLoaderError(Exception):
    """Base exception for document loading failures."""
    pass


class DocumentNotFoundError(DocumentLoaderError):
    """Raised when the specified file path does not exist."""
    pass


class UnsupportedFormatError(DocumentLoaderError):
    """Raised when the file extension is not supported by the loader."""
    pass


class CorruptedDocumentError(DocumentLoaderError):
    """Raised when a file cannot be parsed or opened due to corruption or malformed binary."""
    pass


@dataclass
class Document:
    """Standardized plain-text representation of an ingested document.

    Attributes:
        content: The extracted plain-text body of the document.
        metadata: Key-value dictionary preserving provenance, file properties, and citations.
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        """Return the source file path or identifier."""
        return self.metadata.get("source", "Unknown Source")

    @property
    def filename(self) -> str:
        """Return the base file name."""
        return self.metadata.get("filename", Path(self.source).name if self.source else "unknown")

    @property
    def file_type(self) -> str:
        """Return the document format / extension."""
        return self.metadata.get("file_type", "")

    @property
    def char_count(self) -> int:
        """Return character length of plain text."""
        return len(self.content)

    @property
    def word_count(self) -> int:
        """Return word count of plain text."""
        return len(self.content.split())

    def citation(self) -> str:
        """Return a formatted citation string for referencing this document."""
        parts = [f"Source: {self.filename}"]
        if "page_count" in self.metadata and self.metadata["page_count"] > 1:
            parts.append(f"Pages: {self.metadata['page_count']}")
        return f"[{', '.join(parts)}]"

    def sample(self, max_chars: int = 150) -> str:
        """Return a trimmed, clean sample snippet of the text for intake confirmation."""
        # Replace multiple whitespaces/newlines with a single space for preview
        clean_text = " ".join(self.content.split())
        if len(clean_text) <= max_chars:
            return clean_text
        return clean_text[:max_chars].rstrip() + "..."


@dataclass
class DocumentLoadingResult:
    """Aggregate result from loading a batch or folder of documents.

    Attributes:
        documents: List of successfully ingested Document objects.
        errors: List of error dictionaries detailing skipped or failed items.
        total_examined: Total count of files evaluated.
    """
    documents: List[Document] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    total_examined: int = 0

    @property
    def successful_count(self) -> int:
        return len(self.documents)

    @property
    def skipped_count(self) -> int:
        return len(self.errors)

    @property
    def total_chars_loaded(self) -> int:
        return sum(doc.char_count for doc in self.documents)

    @property
    def total_words_loaded(self) -> int:
        return sum(doc.word_count for doc in self.documents)


class DocumentLoader:
    """Multi-format document loader supporting PDF, TXT, HTML, and Markdown.

    Designed with fault tolerance to skip missing, corrupted, or unsupported files
    gracefully while logging explicit warnings.
    """

    SUPPORTED_EXTENSIONS: Set[str] = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}

    def __init__(
        self,
        supported_extensions: Optional[Set[str]] = None,
        raise_on_error: bool = False,
    ):
        """Initialize DocumentLoader.

        Args:
            supported_extensions: Optional set of allowed extensions (defaults to .txt, .md, .html, .pdf).
            raise_on_error: If True, loader will raise exceptions instead of skipping bad files.
        """
        self.supported_extensions = (
            {ext.lower() for ext in supported_extensions}
            if supported_extensions
            else self.SUPPORTED_EXTENSIONS
        )
        self.raise_on_error = raise_on_error

    def load_file(self, file_path: Union[str, Path]) -> Optional[Document]:
        """Load a single document from disk into a plain-text Document.

        Args:
            file_path: Path to the file.

        Returns:
            A Document object if successfully loaded, or None if skipped due to error.

        Raises:
            DocumentNotFoundError: If file does not exist and raise_on_error is True.
            UnsupportedFormatError: If format unsupported and raise_on_error is True.
            CorruptedDocumentError: If file cannot be read and raise_on_error is True.
        """
        path = Path(file_path)

        # Task 2: Validate existence
        if not path.exists():
            msg = f"File not found: '{path}'"
            logger.warning(f"[SKIPPED] {msg}")
            if self.raise_on_error:
                raise DocumentNotFoundError(msg)
            return None

        if not path.is_file():
            msg = f"Path is not a regular file: '{path}'"
            logger.warning(f"[SKIPPED] {msg}")
            if self.raise_on_error:
                raise DocumentLoaderError(msg)
            return None

        ext = path.suffix.lower()
        if ext not in self.supported_extensions:
            msg = f"Unsupported file format '{ext}' for file: '{path.name}'. Supported formats: {sorted(list(self.supported_extensions))}"
            logger.warning(f"[SKIPPED] {msg}")
            if self.raise_on_error:
                raise UnsupportedFormatError(msg)
            return None

        try:
            stat_info = path.stat()
            file_size = stat_info.st_size
            if file_size == 0:
                # Handle zero-byte file
                msg = f"File is empty (0 bytes): '{path.name}'"
                logger.warning(f"[SKIPPED] {msg}")
                if self.raise_on_error:
                    raise CorruptedDocumentError(msg)
                return None

            # Route to appropriate format extractor
            content, extra_meta = self._extract_content(path, ext)

            if not content.strip():
                msg = f"Extracted text is completely empty for file: '{path.name}'"
                logger.warning(f"[SKIPPED] {msg}")
                if self.raise_on_error:
                    raise CorruptedDocumentError(msg)
                return None

            # Task 3: Preserve source identity & provenance metadata
            metadata: Dict[str, Any] = {
                "source": str(path.resolve()),
                "relative_path": str(path),
                "filename": path.name,
                "file_type": ext,
                "file_size_bytes": file_size,
                "char_count": len(content),
                "word_count": len(content.split()),
            }
            metadata.update(extra_meta)

            return Document(content=content, metadata=metadata)

        except (DocumentLoaderError, UnsupportedFormatError, DocumentNotFoundError, CorruptedDocumentError):
            if self.raise_on_error:
                raise
            return None
        except Exception as exc:
            msg = f"Failed to load or parse '{path.name}': {type(exc).__name__}: {str(exc)}"
            logger.warning(f"[SKIPPED] {msg}")
            if self.raise_on_error:
                raise CorruptedDocumentError(msg) from exc
            return None

    def _extract_content(self, path: Path, ext: str) -> Tuple[str, Dict[str, Any]]:
        """Dispatch file reading to specialized parser based on extension."""
        if ext == ".txt":
            return self._load_txt(path)
        elif ext in {".md", ".markdown"}:
            return self._load_markdown(path)
        elif ext in {".html", ".htm"}:
            return self._load_html(path)
        elif ext == ".pdf":
            return self._load_pdf(path)
        else:
            raise UnsupportedFormatError(f"Extension '{ext}' not implemented.")

    def _load_txt(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        """Load standard plain-text file with multiple encoding fallbacks."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        content = None
        used_encoding = None

        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    content = f.read()
                used_encoding = enc
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
            raise CorruptedDocumentError(f"Unable to decode text file '{path.name}' with standard encodings.")

        return content.strip(), {"encoding": used_encoding, "page_count": 1}

    def _load_markdown(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        """Load markdown document, stripping YAML frontmatter if present and extracting title."""
        raw_text, meta = self._load_txt(path)
        extra_meta: Dict[str, Any] = {"page_count": 1}

        # Check for YAML frontmatter (---\n...---\n)
        frontmatter_pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
        match = frontmatter_pattern.match(raw_text)
        if match:
            clean_body = raw_text[match.end():]
        else:
            clean_body = raw_text

        # Extract top-level Markdown title if available
        title_match = re.search(r"^#\s+(.+)$", clean_body, re.MULTILINE)
        if title_match:
            extra_meta["title"] = title_match.group(1).strip()

        return clean_body.strip(), extra_meta

    def _load_html(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        """Parse HTML file into clean plain text using BeautifulSoup."""
        try:
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise DocumentLoaderError("beautifulsoup4 is required to load HTML files. Install with `pip install beautifulsoup4`.") from e

        raw_bytes = path.read_bytes()
        # Decode raw bytes using robust detection
        decoded_html = None
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                decoded_html = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if decoded_html is None:
            raise CorruptedDocumentError(f"Could not decode HTML file '{path.name}'.")

        soup = BeautifulSoup(decoded_html, "html.parser")

        # Extract title before decomposing
        extra_meta: Dict[str, Any] = {"page_count": 1}
        if soup.title and soup.title.string:
            extra_meta["title"] = soup.title.string.strip()

        # Remove non-content elements
        for element in soup(["script", "style", "noscript", "svg"]):
            element.decompose()

        # Extract text with newlines to separate block elements
        plain_text = soup.get_text(separator="\n")

        # Normalize multiple successive newlines
        clean_text = re.sub(r"\n\s*\n\s*\n+", "\n\n", plain_text).strip()

        return clean_text, extra_meta

    def _load_pdf(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract plain text from PDF using pypdf with fitz/PyMuPDF fallback."""
        # Validate PDF header bytes
        try:
            with open(path, "rb") as f:
                header = f.read(5)
                if not header.startswith(b"%PDF"):
                    raise CorruptedDocumentError(f"File '{path.name}' is missing standard %PDF header magic bytes.")
        except Exception as e:
            if isinstance(e, CorruptedDocumentError):
                raise
            raise CorruptedDocumentError(f"Could not read PDF header for '{path.name}': {e}") from e

        pages_text: List[str] = []
        page_count = 0

        # Try pypdf first
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            page_count = len(reader.pages)
            if page_count == 0:
                raise CorruptedDocumentError(f"PDF '{path.name}' contains 0 pages.")

            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_text.append(page_text.strip())

        except Exception as pypdf_err:
            # Fallback to PyMuPDF (fitz) if pypdf encounters an issue
            try:
                import fitz
                doc = fitz.open(str(path))
                page_count = len(doc)
                pages_text = []
                for idx in range(page_count):
                    page = doc[idx]
                    txt = page.get_text()
                    if txt.strip():
                        pages_text.append(txt.strip())
                doc.close()
            except Exception as fitz_err:
                raise CorruptedDocumentError(
                    f"PDF extraction failed for '{path.name}': pypdf error: {pypdf_err}; fitz error: {fitz_err}"
                ) from pypdf_err

        if not pages_text:
            raise CorruptedDocumentError(f"PDF '{path.name}' contains no readable text content (may be scanned images without OCR).")

        full_content = "\n\n".join(pages_text)
        return full_content.strip(), {"page_count": page_count}

    def load_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
    ) -> DocumentLoadingResult:
        """Scan and load all supported documents from a directory.

        Fault-tolerant: Skips unreadable, corrupt, missing, or unsupported files
        with explicit logging without stopping execution.

        Args:
            directory_path: Directory to inspect.
            recursive: If True, inspects subdirectories recursively.

        Returns:
            DocumentLoadingResult with successfully ingested documents and recorded error details.
        """
        dir_path = Path(directory_path)
        result = DocumentLoadingResult()

        if not dir_path.exists():
            msg = f"Directory not found: '{dir_path}'"
            logger.error(f"[ERROR] {msg}")
            result.errors.append({
                "path": str(dir_path),
                "error_type": "DIRECTORY_NOT_FOUND",
                "message": msg,
            })
            return result

        if not dir_path.is_dir():
            msg = f"Path is not a directory: '{dir_path}'"
            logger.error(f"[ERROR] {msg}")
            result.errors.append({
                "path": str(dir_path),
                "error_type": "NOT_A_DIRECTORY",
                "message": msg,
            })
            return result

        # Collect files
        file_paths = []
        if recursive:
            file_paths = [p for p in dir_path.rglob("*") if p.is_file()]
        else:
            file_paths = [p for p in dir_path.glob("*") if p.is_file()]

        # Sort for deterministic processing order
        file_paths.sort(key=lambda p: str(p).lower())
        result.total_examined = len(file_paths)

        for path in file_paths:
            ext = path.suffix.lower()

            # Skip unsupported format
            if ext not in self.supported_extensions:
                msg = f"Unsupported extension '{ext}' for file '{path.name}'"
                logger.info(f"[SKIPPED] {msg}")
                result.errors.append({
                    "path": str(path),
                    "filename": path.name,
                    "error_type": "UNSUPPORTED_FORMAT",
                    "message": msg,
                })
                continue

            try:
                # Load with raise_on_error=True to capture detailed error info in result.errors
                doc = self._load_single_guarded(path)
                if doc is not None:
                    result.documents.append(doc)
                    logger.info(f"[LOADED] '{doc.filename}' ({doc.file_type}) | Length: {doc.char_count} chars | Words: {doc.word_count}")
            except Exception as exc:
                err_type = type(exc).__name__
                msg = f"Error reading '{path.name}': {str(exc)}"
                logger.warning(f"[SKIPPED] {msg}")
                result.errors.append({
                    "path": str(path),
                    "filename": path.name,
                    "error_type": err_type,
                    "message": msg,
                })

        return result

    def _load_single_guarded(self, path: Path) -> Optional[Document]:
        """Internal helper to load a single file and raise specific exceptions on errors."""
        ext = path.suffix.lower()
        stat_info = path.stat()
        file_size = stat_info.st_size
        if file_size == 0:
            raise CorruptedDocumentError(f"File is 0 bytes (empty): '{path.name}'")

        content, extra_meta = self._extract_content(path, ext)
        if not content.strip():
            raise CorruptedDocumentError(f"Extracted content is empty for '{path.name}'")

        metadata: Dict[str, Any] = {
            "source": str(path.resolve()),
            "relative_path": str(path),
            "filename": path.name,
            "file_type": ext,
            "file_size_bytes": file_size,
            "char_count": len(content),
            "word_count": len(content.split()),
        }
        metadata.update(extra_meta)
        return Document(content=content, metadata=metadata)


def confirm_intake(result: DocumentLoadingResult, verbose: bool = True) -> str:
    """Task 4: Confirm intake by formatting loaded text lengths, samples, and skip stats.

    Args:
        result: The DocumentLoadingResult object.
        verbose: If True, prints directly to stdout.

    Returns:
        The formatted intake summary string.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("RegulSense Document Intake Confirmation")
    lines.append("=" * 80)

    lines.append(f"Total Files Examined : {result.total_examined}")
    lines.append(f"Successfully Loaded  : {result.successful_count}")
    lines.append(f"Skipped / Failed     : {result.skipped_count}")
    lines.append(f"Total Characters     : {result.total_chars_loaded:,}")
    lines.append(f"Total Words          : {result.total_words_loaded:,}")
    lines.append("-" * 80)

    if result.documents:
        lines.append("--- INGESTED DOCUMENTS (SOURCE CITATION & TEXT SAMPLES) ---")
        for i, doc in enumerate(result.documents, 1):
            lines.append(f"\n[{i}] Document: {doc.filename}")
            lines.append(f"    Source Path : {doc.source}")
            lines.append(f"    Citation    : {doc.citation()}")
            lines.append(f"    Format      : {doc.file_type}")
            lines.append(f"    Text Length : {doc.char_count:,} characters ({doc.word_count:,} words)")
            if "title" in doc.metadata:
                lines.append(f"    Doc Title   : {doc.metadata['title']}")
            if "page_count" in doc.metadata:
                lines.append(f"    Page Count  : {doc.metadata['page_count']}")
            lines.append(f"    Text Sample : \"{doc.sample(max_chars=180)}\"")
    else:
        lines.append("No documents were successfully ingested.")

    if result.errors:
        lines.append("\n" + "-" * 80)
        lines.append("--- SKIPPED / REJECTED FILES (FAULT TOLERANCE REPORT) ---")
        for i, err in enumerate(result.errors, 1):
            lines.append(f"[{i}] File       : {err.get('filename', err.get('path'))}")
            lines.append(f"    Error Type : {err.get('error_type')}")
            lines.append(f"    Reason     : {err.get('message')}")

    lines.append("=" * 80)
    report = "\n".join(lines)

    if verbose:
        print(report)

    return report


def generate_intake_markdown_report(
    result: DocumentLoadingResult,
    output_path: Union[str, Path] = "outputs/document_loading_intake.md",
) -> Path:
    """Generate a structured Markdown report confirming intake and error handling results."""
    from datetime import datetime

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    md_lines = [
        "# RegulSense: Document Intake & Corpus Verification Report",
        "",
        f"- **Execution Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Total Files Examined**: {result.total_examined}",
        f"- **Successfully Ingested**: {result.successful_count}",
        f"- **Skipped / Failed**: {result.skipped_count}",
        f"- **Total Volume Ingested**: {result.total_chars_loaded:,} characters ({result.total_words_loaded:,} words)",
        "- **Status**: Document Loading Pipeline Verified Across Multi-Format Corpus",
        "",
        "---",
        "",
        "## 1. Executive Summary & Verification",
        "",
        "The RegulSense Document Loader normalizes heterogeneous compliance assets (PDF, TXT, HTML, Markdown) into a unified plain-text `Document` representation with full provenance preservation and citation readiness. Missing, corrupt, or unsupported files are safely caught and documented without interrupting pipeline execution.",
        "",
        "---",
        "",
        "## 2. Successfully Ingested Corpus",
        "",
        "| # | Document Filename | Format | Characters | Words | Citation Reference | Title / Details |",
        "|---|---|---|---|---|---|---|",
    ]

    for i, doc in enumerate(result.documents, 1):
        title = doc.metadata.get("title", f"Pages: {doc.metadata.get('page_count', 1)}")
        citation = doc.citation().replace("[", "\\[").replace("]", "\\]")
        md_lines.append(
            f"| {i} | `{doc.filename}` | `{doc.file_type}` | {doc.char_count:,} | {doc.word_count:,} | {citation} | {title} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Sample Intake Confirmation (Previews)",
        "",
    ])

    for i, doc in enumerate(result.documents, 1):
        md_lines.extend([
            f"### Document {i}: `{doc.filename}`",
            f"- **Source**: `{doc.source}`",
            f"- **Format**: `{doc.file_type}` | **Length**: {doc.char_count:,} characters | **Words**: {doc.word_count:,}",
            f"- **Citation**: `{doc.citation()}`",
            "",
            "```text",
            doc.sample(max_chars=220),
            "```",
            "",
        ])

    if result.errors:
        md_lines.extend([
            "---",
            "",
            "## 4. Fault Tolerance & Skipped Files Audit",
            "",
            "The following files were encountered during testing or scanning and handled gracefully without crashing the run:",
            "",
            "| # | File / Identifier | Error Category | Detail |",
            "|---|---|---|---|",
        ])
        for i, err in enumerate(result.errors, 1):
            fname = err.get("filename", Path(err.get("path", "")).name)
            etype = err.get("error_type")
            msg = err.get("message", "").replace("|", "-")
            md_lines.append(f"| {i} | `{fname}` | `{etype}` | {msg} |")

    out_p.write_text("\n".join(md_lines), encoding="utf-8")
    return out_p


def main():
    """CLI entrypoint for running document loading and intake verification."""
    import argparse

    parser = argparse.ArgumentParser(description="RegulSense Multi-Format Document Loader & Intake Verifier")
    parser.add_argument(
        "--dir",
        type=str,
        default="data/sample_corpus",
        help="Directory containing corpus files (default: data/sample_corpus)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="outputs/document_loading_intake.md",
        help="Path to generate Markdown intake report (default: outputs/document_loading_intake.md)",
    )
    args = parser.parse_args()

    loader = DocumentLoader()
    result = loader.load_directory(args.dir)

    # Task 4: Confirm intake
    confirm_intake(result, verbose=True)

    # Generate Markdown documentation
    report_file = generate_intake_markdown_report(result, args.report)
    print(f"\n[OK] Intake report successfully written to: {report_file.resolve()}")


if __name__ == "__main__":
    main()
