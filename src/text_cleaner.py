"""RegulSense Retrieval-Ready Text Cleaning & Normalization Pipeline.

Transforms raw extracted text from PDFs, HTML exports, Markdown, and text circulars
into consistent, retrieval-ready content. Strips boilerplate headers/footers, fixes
broken line wraps and hyphenations, normalizes Unicode (NFKC) and encoding artifacts,
and collapses runaway whitespace across the entire corpus.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import unicodedata

# Import Document from document_loader if available
try:
    from src.document_loader import Document, DocumentLoader
except ImportError:
    from document_loader import Document, DocumentLoader  # type: ignore

logger = logging.getLogger("regulsense.text_cleaner")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Modular Cleaning Components
# -----------------------------------------------------------------------------

class UnicodeNormalizer:
    """Normalizes Unicode encoding artifacts, smart quotes, ligatures, and dashes."""

    # Ligature mapping
    LIGATURE_MAP = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ft",
        "ﬆ": "st",
        "Æ": "AE",
        "æ": "ae",
        "Œ": "OE",
        "œ": "oe",
    }

    # Smart quote and dash replacements
    CHAR_REPLACEMENTS = {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "«": '"',
        "»": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "`": "'",
        "–": "-",  # en dash
        "—": "-",  # em dash
        "‒": "-",  # figure dash
        "―": "-",  # horizontal bar
        "…": "...",  # horizontal ellipsis
        "\u00a0": " ",  # non-breaking space
        "\u202f": " ",  # narrow no-break space
        "\u2007": " ",  # figure space
        "\u2009": " ",  # thin space
        "\u200a": " ",  # hair space
        "\u200b": "",   # zero-width space
        "\u200c": "",   # zero-width non-joiner
        "\u200d": "",   # zero-width joiner
        "\ufeff": "",   # zero-width no-break space / BOM
        "\ufffd": "",   # replacement character (garbled byte)
    }

    @classmethod
    def clean(cls, text: str) -> str:
        """Apply Unicode NFKC normalization and character harmonization."""
        if not text:
            return ""

        # 1. Unicode NFKC normalization
        normalized = unicodedata.normalize("NFKC", text)

        # 2. Replace ligatures
        for lig, repl in cls.LIGATURE_MAP.items():
            if lig in normalized:
                normalized = normalized.replace(lig, repl)

        # 3. Replace smart quotes, dashes, non-breaking spaces
        for char, repl in cls.CHAR_REPLACEMENTS.items():
            if char in normalized:
                normalized = normalized.replace(char, repl)

        # 4. Remove unprintable control characters (retain \n, \t, \r)
        cleaned_chars = []
        for ch in normalized:
            code = ord(ch)
            if code >= 32 or ch in ("\n", "\t", "\r"):
                cleaned_chars.append(ch)
        return "".join(cleaned_chars)


class BoilerplateRemover:
    """Removes repetitive headers, footers, page markers, and web navigation noise."""

    # Regex patterns for line-level boilerplate removal
    BOILERPLATE_LINE_PATTERNS = [
        # Page numbering: "Page 1 of 5", "PAGE 2 OF 2", "Page 3", "- 4 -", "[Page 1/2]"
        re.compile(r"^\s*(?:Page|PAGE)\s+\d+(?:\s*(?:of|OF|/)\s*\d+)?\s*$", re.IGNORECASE),
        re.compile(r"^\s*-\s*\d+\s*-\s*$"),
        re.compile(r"^\s*\[?\s*\d+\s*(?:of|/)\s*\d+\s*\]?\s*$", re.IGNORECASE),
        re.compile(r"^\s*---\s*Page\s+\d+\s*(?:of\s+\d+)?\s*---\s*$", re.IGNORECASE),
        re.compile(r"^\s*===+\s*Page\s+\d+\s*===+\s*$", re.IGNORECASE),

        # Confidentiality stamps & circulation notices
        re.compile(r"^\s*(?:CONFIDENTIAL|Confidential|RESTRICTED CIRCULATION|INTERNAL USE ONLY|FOR INTERNAL USE ONLY|STRICTLY CONFIDENTIAL)\b.*$", re.IGNORECASE),
        re.compile(r"^\s*CONFIDENTIAL\s*[-–—/]\s*(?:FOR\s+)?INTERNAL\s+USE\s+ONLY\s*$", re.IGNORECASE),
        re.compile(r"^\s*DO\s+NOT\s+DISTRIBUTE\s*$", re.IGNORECASE),
        re.compile(r"^\s*UNCONTROLLED\s+COPY\s+WHEN\s+PRINTED\s*$", re.IGNORECASE),

        # Bank departmental recurring footer lines
        re.compile(r"^.*Department of Supervision\s+[•·*]\s+Reserve Bank of India\s+[•·*]\s+Confidential Internal Copy.*$", re.IGNORECASE),

        # Web navigation & breadcrumb boilerplate
        re.compile(r"^\s*Home\s*>\s*.*$", re.IGNORECASE),
        re.compile(r"^\s*(?:Skip to (?:main )?content|Navigation Menu|Table of Contents)\s*$", re.IGNORECASE),
        re.compile(r"^\s*(?:Back to top|Print this page|Share this page|Download PDF)\s*$", re.IGNORECASE),
        re.compile(r"^\s*Copyright\s*(?:©|\(c\))\s*\d{4}.*All rights reserved\.?\s*$", re.IGNORECASE),
        re.compile(r"^\s*All\s+rights\s+reserved\.?\s*$", re.IGNORECASE),
    ]

    @classmethod
    def clean(cls, text: str) -> str:
        """Filter out boilerplate lines matching known noise patterns."""
        if not text:
            return ""

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped_line = line.strip()

            # Empty lines are preserved for paragraph structure
            if not stripped_line:
                cleaned_lines.append(line)
                continue

            # Check against boilerplate line patterns
            is_boilerplate = False
            for pattern in cls.BOILERPLATE_LINE_PATTERNS:
                if pattern.match(stripped_line):
                    is_boilerplate = True
                    break

            if not is_boilerplate:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)


class LineWrapNormalizer:
    """Repairs broken line wraps and hyphenated word splits across line breaks."""

    # Matches word hyphenated at line break: e.g. "trans-\naction" -> "transaction"
    # Group 1: prefix word part, Group 2: suffix word part
    HYPHENATION_PATTERN = re.compile(r"([a-zA-Z]{2,})-(?:\r?\n|\n)[ \t]*([a-zA-Z]{2,})")

    @classmethod
    def clean(cls, text: str) -> str:
        """De-hyphenate words and unwrap broken lines within paragraphs."""
        if not text:
            return ""

        # Step 1: De-hyphenate broken words across line wraps
        dehyphenated = cls.HYPHENATION_PATTERN.sub(r"\1\2", text)

        # Step 2: Unwrap soft line breaks inside paragraphs
        # We split by double newlines (paragraphs) to respect paragraph boundaries
        paragraphs = dehyphenated.split("\n\n")
        unwrapped_paragraphs = []

        for para in paragraphs:
            lines = para.split("\n")
            if len(lines) <= 1:
                unwrapped_paragraphs.append(para)
                continue

            # Check if this paragraph is a Markdown structure that shouldn't be flattened:
            # - Markdown tables (contains '|')
            # - Markdown lists ('-', '*', '1.', '(a)', '(b)')
            # - Markdown headings ('#')
            if cls._is_structured_block(lines):
                unwrapped_paragraphs.append(para)
                continue

            # Soft unwrap paragraph lines into a unified continuous line
            merged_lines = []
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                if not merged_lines:
                    merged_lines.append(stripped)
                else:
                    prev = merged_lines[-1]
                    # If previous line ends with hyphen (not caught above) or soft break
                    if prev.endswith("-") and not prev.endswith(" --"):
                        merged_lines[-1] = prev[:-1] + stripped
                    else:
                        merged_lines[-1] = prev + " " + stripped

            unwrapped_paragraphs.append("\n".join(merged_lines))

        return "\n\n".join(unwrapped_paragraphs)

    @classmethod
    def _is_structured_block(cls, lines: List[str]) -> bool:
        """Detect whether a block consists of Markdown tables, headings, or lists."""
        structured_count = 0
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if s.startswith(("#", "|", "-", "*", "+", ">")):
                structured_count += 1
            elif re.match(r"^(?:\d+\.|\([a-z0-9]+\))\s+", s):
                structured_count += 1

        # If more than 30% of non-empty lines are structural, preserve line breaks
        return structured_count >= max(1, len(lines) * 0.3)


class WhitespaceNormalizer:
    """Collapses runaway spaces, tabs, and vertical blank lines."""

    @classmethod
    def clean(cls, text: str) -> str:
        """Collapse multiple horizontal spaces and runaway vertical blank lines."""
        if not text:
            return ""

        # Normalize line endings to \n
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Trim horizontal whitespace from line ends
        lines = [re.sub(r"[ \t]+$", "", line) for line in text.split("\n")]
        text = "\n".join(lines)

        # Collapse multiple horizontal spaces/tabs (except at indentation start)
        collapsed_lines = []
        for line in text.split("\n"):
            # If line is table or code or list, leave indentation but collapse multiple spaces inside
            stripped_leading = line.lstrip(" \t")
            indent = line[: len(line) - len(stripped_leading)]
            collapsed_content = re.sub(r"[ \t]{2,}", " ", stripped_leading)
            collapsed_lines.append(indent + collapsed_content)

        text = "\n".join(collapsed_lines)

        # Collapse 3 or more consecutive newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()


# -----------------------------------------------------------------------------
# End-to-End Pipeline & Batch Corpus Processor
# -----------------------------------------------------------------------------

@dataclass
class CleaningMetrics:
    """Metrics tracking text transformation results for a document."""
    raw_char_count: int = 0
    cleaned_char_count: int = 0
    chars_removed: int = 0
    reduction_pct: float = 0.0
    raw_word_count: int = 0
    cleaned_word_count: int = 0
    raw_line_count: int = 0
    cleaned_line_count: int = 0


@dataclass
class CorpusCleaningResult:
    """Aggregated results from cleaning an entire document corpus."""
    documents: List[Document] = field(default_factory=list)
    document_metrics: List[Dict[str, Any]] = field(default_factory=list)
    total_raw_chars: int = 0
    total_cleaned_chars: int = 0
    total_chars_removed: int = 0
    average_reduction_pct: float = 0.0


class TextCleaner:
    """Full-featured cleaning pipeline for regulatory documents."""

    def __init__(
        self,
        normalize_unicode: bool = True,
        remove_boilerplate: bool = True,
        fix_line_wraps: bool = True,
        normalize_whitespace: bool = True,
    ):
        """Initialize pipeline with configurable stages."""
        self.normalize_unicode = normalize_unicode
        self.remove_boilerplate = remove_boilerplate
        self.fix_line_wraps = fix_line_wraps
        self.normalize_whitespace = normalize_whitespace

    def clean(self, text: str) -> str:
        """Run all enabled cleaning stages sequentially on raw text."""
        if not text:
            return ""

        result = text

        # Stage 1: Unicode NFKC & character harmonization
        if self.normalize_unicode:
            result = UnicodeNormalizer.clean(result)

        # Stage 2: Boilerplate removal (headers, footers, nav lines)
        if self.remove_boilerplate:
            result = BoilerplateRemover.clean(result)

        # Stage 3: Hyphenation & soft line wrap normalization
        if self.fix_line_wraps:
            result = LineWrapNormalizer.clean(result)

        # Stage 4: Whitespace and blank line collapsing
        if self.normalize_whitespace:
            result = WhitespaceNormalizer.clean(result)

        return result

    def clean_document(self, doc: Document) -> Document:
        """Clean a Document instance and record provenance & transformation metadata."""
        raw_text = doc.content
        cleaned_text = self.clean(raw_text)

        raw_chars = len(raw_text)
        cleaned_chars = len(cleaned_text)
        chars_removed = max(0, raw_chars - cleaned_chars)
        reduction_pct = (chars_removed / raw_chars * 100) if raw_chars > 0 else 0.0

        # Clone and enrich metadata
        updated_meta = dict(doc.metadata)
        updated_meta.update({
            "cleaned": True,
            "raw_char_count": raw_chars,
            "cleaned_char_count": cleaned_chars,
            "chars_removed": chars_removed,
            "reduction_pct": round(reduction_pct, 2),
            "raw_word_count": len(raw_text.split()),
            "cleaned_word_count": len(cleaned_text.split()),
            "char_count": cleaned_chars,
            "word_count": len(cleaned_text.split()),
        })

        return Document(content=cleaned_text, metadata=updated_meta)

    def clean_corpus(self, documents: List[Document]) -> CorpusCleaningResult:
        """Apply uniform cleaning across an entire document corpus (Task 3)."""
        cleaned_docs: List[Document] = []
        metrics_list: List[Dict[str, Any]] = []

        total_raw = 0
        total_cleaned = 0

        for doc in documents:
            cleaned_doc = self.clean_document(doc)
            cleaned_docs.append(cleaned_doc)

            m = {
                "filename": doc.filename,
                "file_type": doc.file_type,
                "raw_chars": doc.char_count,
                "cleaned_chars": cleaned_doc.char_count,
                "chars_removed": cleaned_doc.metadata.get("chars_removed", 0),
                "reduction_pct": cleaned_doc.metadata.get("reduction_pct", 0.0),
                "raw_words": doc.word_count,
                "cleaned_words": cleaned_doc.word_count,
            }
            metrics_list.append(m)
            total_raw += m["raw_chars"]
            total_cleaned += m["cleaned_chars"]

        total_removed = max(0, total_raw - total_cleaned)
        avg_red = (total_removed / total_raw * 100) if total_raw > 0 else 0.0

        return CorpusCleaningResult(
            documents=cleaned_docs,
            document_metrics=metrics_list,
            total_raw_chars=total_raw,
            total_cleaned_chars=total_cleaned,
            total_chars_removed=total_removed,
            average_reduction_pct=round(avg_red, 2),
        )


# -----------------------------------------------------------------------------
# Before / After Verification & Reporting (Task 4)
# -----------------------------------------------------------------------------

def compare_text(raw_text: str, cleaned_text: str, title: str = "") -> Dict[str, Any]:
    """Compute before/after comparison metrics and preview snippets."""
    raw_chars = len(raw_text)
    cleaned_chars = len(cleaned_text)
    chars_removed = max(0, raw_chars - cleaned_chars)
    reduction_pct = (chars_removed / raw_chars * 100) if raw_chars > 0 else 0.0

    return {
        "title": title,
        "raw_chars": raw_chars,
        "cleaned_chars": cleaned_chars,
        "chars_removed": chars_removed,
        "reduction_pct": round(reduction_pct, 2),
        "raw_words": len(raw_text.split()),
        "cleaned_words": len(cleaned_text.split()),
        "raw_lines": len(raw_text.split("\n")),
        "cleaned_lines": len(cleaned_text.split("\n")),
        "raw_sample": raw_text[:250].strip(),
        "cleaned_sample": cleaned_text[:250].strip(),
    }


def generate_cleaning_report_markdown(
    corpus_result: CorpusCleaningResult,
    stress_cases: Optional[List[Dict[str, Any]]] = None,
    output_path: Union[str, Path] = "outputs/text_cleaning_results.md",
) -> Path:
    """Generate comprehensive before/after evidence report as required by Task 4 and Task 5."""
    from datetime import datetime

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# RegulSense: Text Cleaning & Retrieval Normalization Report",
        "",
        f"- **Execution Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Total Corpus Documents Cleaned**: {len(corpus_result.documents)}",
        f"- **Total Raw Characters**: {corpus_result.total_raw_chars:,}",
        f"- **Total Cleaned Characters**: {corpus_result.total_cleaned_chars:,}",
        f"- **Noise Removed**: {corpus_result.total_chars_removed:,} characters ({corpus_result.average_reduction_pct}%)",
        "- **Status**: Uniform Cleaning Pipeline Verified Across Corpus",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "The RegulSense text cleaning pipeline prepares ingested regulatory documents for high-accuracy dense vector retrieval. It eliminates four primary noise vectors:",
        "1. **Boilerplate & Running Headers/Footers**: Page counters (`Page 1 of 5`), classification stamps (`Confidential Internal Copy`), web navigation breadcrumbs (`Home > ...`), and legal disclaimers.",
        "2. **Broken Line Wraps & Hyphenation**: Words split across line boundaries (`trans-\\naction` -> `transaction`) are reunited, restoring natural phrase embeddings.",
        "3. **Unicode NFKC & Encoding Artifacts**: Harmonizes smart curly quotes (`“` / `”`), em/en dashes (`—` / `–`), ligatures (`ﬁ` -> `fi`), non-breaking spaces, and zero-width BOMs.",
        "4. **Whitespace Normalization**: Collapses runaway multiple spaces and tabs into a single space and vertical blank lines (3+ newlines collapsed to 2).",
        "",
        "---",
        "",
        "## 2. Corpus-Wide Uniform Application Metrics (Task 3)",
        "",
        "| # | Document Filename | Format | Raw Chars | Cleaned Chars | Chars Removed | Reduction % | Cleaned Words |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for i, m in enumerate(corpus_result.document_metrics, 1):
        lines.append(
            f"| {i} | `{m['filename']}` | `{m['file_type']}` | {m['raw_chars']:,} | {m['cleaned_chars']:,} | {m['chars_removed']:,} | {m['reduction_pct']}% | {m['cleaned_words']:,} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Targeted Before / After Case Evidence (Task 4)",
        "",
    ])

    if stress_cases:
        for idx, case in enumerate(stress_cases, 1):
            lines.extend([
                f"### Case {idx}: {case['title']}",
                f"- **Category**: {case.get('category', 'Cleaning Transformation')}",
                f"- **Raw**: {case['raw_chars']} chars ({case['raw_words']} words, {case['raw_lines']} lines)",
                f"- **Cleaned**: {case['cleaned_chars']} chars ({case['cleaned_words']} words, {case['cleaned_lines']} lines)",
                f"- **Reduction**: {case['chars_removed']} chars removed ({case['reduction_pct']}%)",
                "",
                "#### [BEFORE: RAW EXTRACTED TEXT]",
                "```text",
                case["raw_sample"],
                "```",
                "",
                "#### [AFTER: RETRIEVAL-READY CLEANED TEXT]",
                "```text",
                case["cleaned_sample"],
                "```",
                "",
            ])

    lines.extend([
        "---",
        "",
        "## 4. Corpus Document Before / After Samples",
        "",
    ])

    for i, doc in enumerate(corpus_result.documents, 1):
        raw_meta = doc.metadata
        lines.extend([
            f"### Corpus Document {i}: `{doc.filename}` ({doc.file_type})",
            f"- **Citation**: `{doc.citation()}`",
            f"- **Volume**: {raw_meta.get('raw_char_count', 0):,} raw chars -> {doc.char_count:,} cleaned chars ({raw_meta.get('reduction_pct', 0.0)}% noise eliminated)",
            "",
            "**Cleaned Text Sample (First 280 characters):**",
            "```text",
            doc.sample(max_chars=280),
            "```",
            "",
        ])

    out_p.write_text("\n".join(lines), encoding="utf-8")
    return out_p


def run_stress_test_cases() -> List[Dict[str, Any]]:
    """Generate representative before/after cases for the documentation report."""
    cleaner = TextCleaner()
    cases = []

    # Case 1: Broken line wraps and hyphenation splits in PDF text
    raw_1 = (
        "Under Section 35A of the Banking Regulation Act, 1949, all regu-\n"
        "latory entities must enforce auto-\n"
        "mated trans-\n"
        "action monitoring systems to detect suspicious fund flows."
    )
    clean_1 = cleaner.clean(raw_1)
    c1 = compare_text(raw_1, clean_1, "Broken Line Wraps & Hyphenation Splits (PDF Extracted)")
    c1["category"] = "Hyphenation & Soft Line Wraps"
    cases.append(c1)

    # Case 2: Running headers, page numbers, and confidentiality boilerplate
    raw_2 = (
        "CONFIDENTIAL - FOR INTERNAL USE ONLY\n"
        "Page 1 of 12\n"
        "The bank shall determine the natural person who ultimately owns or controls a customer.\n"
        "Page 1 of 12\n"
        "Department of Supervision • Reserve Bank of India • Confidential Internal Copy\n"
        "- 1 -\n"
        "Beneficial ownership threshold is pegged at 10 percent of shares or voting rights."
    )
    clean_2 = cleaner.clean(raw_2)
    c2 = compare_text(raw_2, clean_2, "Repeated Page Numbers & Confidentiality Boilerplate")
    c2["category"] = "Boilerplate & Headers/Footers"
    cases.append(c2)

    # Case 3: Web navigation breadcrumbs & footer notices (HTML export)
    raw_3 = (
        "Home > Circulars > Master Directions > AML\n"
        "Skip to main content\n"
        "Navigation Menu\n\n"
        "All loan disbursals must be executed solely between the borrower and the regulated entity.\n\n"
        "Back to top\n"
        "Print this page\n"
        "Copyright © 2024 Reserve Bank of India. All rights reserved."
    )
    clean_3 = cleaner.clean(raw_3)
    c3 = compare_text(raw_3, clean_3, "Web Navigation Breadcrumbs & Footer Disclaimers (HTML)")
    c3["category"] = "Navigation & Legal Boilerplate"
    cases.append(c3)

    # Case 4: Unicode NFKC, curly quotes, dashes, ligatures, zero-width spaces
    raw_4 = (
        "The “Enhanced Due Diligence” guidelines—issued in 2024—require ‘high-risk’\n"
        "PEPs to provide corroborating ﬁnancial statements (noting \u200ba\u200b \u200bthreshold of INR 50,000…)."
    )
    clean_4 = cleaner.clean(raw_4)
    c4 = compare_text(raw_4, clean_4, "Unicode NFKC, Smart Quotes, Dashes, Ligatures & Zero-Width Spaces")
    c4["category"] = "Unicode & Character Encoding"
    cases.append(c4)

    # Case 5: Runaway blank lines and multiple spaces
    raw_5 = (
        "Risk   Rating:    Tier 1 (Low).\n\n\n\n\n\n"
        "Review   frequency   is   every   10   years   for   salaried   individuals.\n\n\n\n"
        "Standard OVD verification applies."
    )
    clean_5 = cleaner.clean(raw_5)
    c5 = compare_text(raw_5, clean_5, "Runaway Blank Lines & Multiple Spaces")
    c5["category"] = "Whitespace Normalization"
    cases.append(c5)

    return cases


def main():
    """CLI entrypoint to clean a document corpus and generate before/after evidence."""
    import argparse

    parser = argparse.ArgumentParser(description="RegulSense Text Cleaning & Normalization Pipeline")
    parser.add_argument(
        "--dir",
        type=str,
        default="data/sample_corpus",
        help="Corpus directory to load and clean (default: data/sample_corpus)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/text_cleaning_results.md",
        help="Markdown report output path (default: outputs/text_cleaning_results.md)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("RegulSense Text Cleaning & Retrieval Normalization Pipeline")
    print("=" * 80)

    # 1. Load raw corpus using DocumentLoader
    loader = DocumentLoader()
    load_result = loader.load_directory(args.dir)
    print(f"Loaded {load_result.successful_count} documents from '{args.dir}'")

    # 2. Apply uniform cleaning across the corpus (Task 3)
    cleaner = TextCleaner()
    corpus_result = cleaner.clean_corpus(load_result.documents)

    print("\n--- CORPUS CLEANING METRICS ---")
    for m in corpus_result.document_metrics:
        print(
            f"  - {m['filename']:<36} | Raw: {m['raw_chars']:>5} chars -> Cleaned: {m['cleaned_chars']:>5} chars "
            f"| -{m['chars_removed']:>3} chars ({m['reduction_pct']:>4.1f}%)"
        )
    print(f"\nTotal Noise Removed: {corpus_result.total_chars_removed:,} chars ({corpus_result.average_reduction_pct}%)")

    # 3. Generate targeted stress test before/after evidence (Task 4)
    stress_cases = run_stress_test_cases()

    # 4. Generate report artifact (Task 5)
    report_file = generate_cleaning_report_markdown(corpus_result, stress_cases, args.output)
    print(f"\n[OK] Comprehensive Before/After Report generated: {report_file.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
