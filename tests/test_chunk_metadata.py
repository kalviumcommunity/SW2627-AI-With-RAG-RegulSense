"""Unit tests for RegulSense Chunk Metadata Tagging and Source Tracing.

Verifies:
- Task 1: Source identifier storage (source, filename, document_id) on all chunks.
- Task 2: Additional metadata (section, page_number, chunk_index, char_start, char_end).
- Task 3: Consistent metadata structure across the entire corpus and all chunkers.
- Task 4: Tracing a retrieved chunk back to its source with 100% character verification.
- Task 5: Serialization and export of sample chunks with complete metadata.
"""

import json
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import Document, DocumentLoader
from src.chunker import (
    ChunkMetadata,
    ChunkTracer,
    FixedSizeChunker,
    ParagraphChunker,
    RecursiveStructuralChunker,
    TextChunk,
    build_chunk_metadata,
    extract_document_sections,
    find_chunk_span,
    get_active_section,
    resolve_page_number,
    run_metadata_trace_demonstration,
)


class TestChunkMetadata(unittest.TestCase):
    """Test suite for chunk metadata tagging, schema consistency, and provenance tracing."""

    def setUp(self):
        self.sample_text = (
            "RESERVE BANK OF INDIA\n"
            "CIRCULAR ON DUE DILIGENCE\n\n"
            "1. Preliminary and Statutory Authority\n"
            "In exercise of statutory powers, RBI hereby issues guidelines on customer due diligence.\n\n"
            "2. Customer Due Diligence (CDD) Requirements\n"
            "Regulated entities must undertake client identification and verification procedures before opening accounts.\n"
            "(a) Officially Valid Documents: Passport, Aadhaar, or Voter ID are authorized for KYC.\n"
            "(b) Beneficial Ownership: Natural persons holding at least 10 percent equity must be identified.\n\n"
            "3. Enhanced Due Diligence (EDD) for High-Risk Accounts\n"
            "Accounts for Politically Exposed Persons (PEPs) require written approval from an officer not below the rank of Deputy General Manager.\n\n"
            "4. Record Retention Obligations\n"
            "All transaction records and account opening dossiers must be preserved for not less than five years."
        )
        self.doc_meta = {
            "source": str(PROJECT_ROOT / "data" / "sample_corpus" / "circular_dor_2024_108.txt"),
            "filename": "circular_dor_2024_108.txt",
            "file_type": ".txt",
        }
        self.expected_keys = {
            "source",
            "filename",
            "document_id",
            "file_type",
            "section",
            "page_number",
            "chunk_index",
            "total_chunks",
            "char_start",
            "char_end",
            "char_count",
            "token_count",
            "strategy",
        }

    # -------------------------------------------------------------------------
    # Task 1: Store Source Identifier
    # -------------------------------------------------------------------------

    def test_task1_chunk_stores_source_identifier(self):
        """Task 1: Ensure each chunk stores source document identifiers (source, filename, document_id)."""
        chunker = RecursiveStructuralChunker(target_chunk_size=150, chunk_overlap=30)
        chunks = chunker.split_text(self.sample_text, metadata=self.doc_meta)

        self.assertGreater(len(chunks), 0)
        for ch in chunks:
            self.assertIn("source", ch.metadata)
            self.assertIn("filename", ch.metadata)
            self.assertIn("document_id", ch.metadata)

            self.assertEqual(ch.metadata["filename"], "circular_dor_2024_108.txt")
            self.assertEqual(ch.metadata["document_id"], "circular_dor_2024_108_txt")
            self.assertTrue(ch.metadata["source"].endswith("circular_dor_2024_108.txt"))
            # Chunk ID must incorporate document ID
            self.assertTrue(ch.chunk_id.startswith("circular_dor_2024_108_txt_recursive_"))

    def test_task1_fallback_source_identifier_when_unspecified(self):
        """Task 1: Chunk handles unspecified metadata by defaulting safely."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.split_text(self.sample_text)

        self.assertGreater(len(chunks), 0)
        for ch in chunks:
            self.assertTrue(bool(ch.metadata.get("filename")))
            self.assertTrue(bool(ch.metadata.get("document_id")))
            self.assertIn("doc_fixed_", ch.chunk_id)

    # -------------------------------------------------------------------------
    # Task 2: Additional Metadata (Section, Page, Position)
    # -------------------------------------------------------------------------

    def test_task2_section_detection(self):
        """Task 2: Extract headings and map chunks to active section."""
        sections = extract_document_sections(self.sample_text)
        self.assertGreaterEqual(len(sections), 4)

        titles = [s[1] for s in sections]
        self.assertIn("1. Preliminary and Statutory Authority", titles)
        self.assertIn("2. Customer Due Diligence (CDD) Requirements", titles)
        self.assertIn("3. Enhanced Due Diligence (EDD) for High-Risk Accounts", titles)
        self.assertIn("4. Record Retention Obligations", titles)

        # Verify active section resolution by offset
        cdd_offset = self.sample_text.find("2. Customer Due Diligence")
        edd_offset = self.sample_text.find("3. Enhanced Due Diligence")

        self.assertEqual(get_active_section(sections, cdd_offset + 5), "2. Customer Due Diligence (CDD) Requirements")
        self.assertEqual(get_active_section(sections, edd_offset + 5), "3. Enhanced Due Diligence (EDD) for High-Risk Accounts")

    def test_task2_page_number_resolution(self):
        """Task 2: Resolve 1-based page numbers from page boundaries and defaults."""
        boundaries = [(1, 0, 500), (2, 501, 1000)]
        self.assertEqual(resolve_page_number(100, page_boundaries=boundaries), 1)
        self.assertEqual(resolve_page_number(500, page_boundaries=boundaries), 1)
        self.assertEqual(resolve_page_number(501, page_boundaries=boundaries), 2)
        self.assertEqual(resolve_page_number(800, page_boundaries=boundaries), 2)

        # Single page default
        self.assertEqual(resolve_page_number(250, page_boundaries=None), 1)

    def test_task2_position_and_offsets_attached(self):
        """Task 2: Verify chunk_index, total_chunks, char_start, char_end attached to each chunk."""
        chunker = RecursiveStructuralChunker(target_chunk_size=100, chunk_overlap=20)
        chunks = chunker.split_text(self.sample_text, metadata=self.doc_meta)

        self.assertGreater(len(chunks), 1)
        for idx, ch in enumerate(chunks):
            m = ch.metadata
            self.assertEqual(m["chunk_index"], idx)
            self.assertEqual(m["total_chunks"], len(chunks))
            self.assertGreaterEqual(m["char_start"], 0)
            self.assertGreater(m["char_end"], m["char_start"])
            self.assertEqual(m["char_count"], len(ch.content))
            self.assertGreater(m["token_count"], 0)
            self.assertIsInstance(m["page_number"], int)
            self.assertGreaterEqual(m["page_number"], 1)

    # -------------------------------------------------------------------------
    # Task 3: Consistent Structure Across Corpus & Strategies
    # -------------------------------------------------------------------------

    def test_task3_consistent_schema_across_all_strategies(self):
        """Task 3: Every chunk across FixedSize, Paragraph, and Recursive has identical metadata fields."""
        strategies = [
            FixedSizeChunker(chunk_size=100, chunk_overlap=20),
            ParagraphChunker(max_chunk_size=150),
            RecursiveStructuralChunker(target_chunk_size=120, chunk_overlap=25),
        ]

        for strat in strategies:
            chunks = strat.split_text(self.sample_text, metadata=self.doc_meta)
            self.assertGreater(len(chunks), 0, f"Strategy {strat.strategy_name} yielded 0 chunks")
            for ch in chunks:
                self.assertTrue(
                    self.expected_keys.issubset(ch.metadata.keys()),
                    f"Strategy {strat.strategy_name} missing keys: {self.expected_keys - set(ch.metadata.keys())}",
                )
                # Verify types
                self.assertIsInstance(ch.metadata["source"], str)
                self.assertIsInstance(ch.metadata["filename"], str)
                self.assertIsInstance(ch.metadata["document_id"], str)
                self.assertIsInstance(ch.metadata["file_type"], str)
                self.assertIsInstance(ch.metadata["section"], str)
                self.assertIsInstance(ch.metadata["page_number"], int)
                self.assertIsInstance(ch.metadata["chunk_index"], int)
                self.assertIsInstance(ch.metadata["total_chunks"], int)
                self.assertIsInstance(ch.metadata["char_start"], int)
                self.assertIsInstance(ch.metadata["char_end"], int)
                self.assertIsInstance(ch.metadata["char_count"], int)
                self.assertIsInstance(ch.metadata["token_count"], int)
                self.assertIsInstance(ch.metadata["strategy"], str)

    def test_task3_consistent_structure_across_corpus_formats(self):
        """Task 3: Every chunk across .txt, .pdf, .html, .md maintains uniform metadata structure."""
        loader = DocumentLoader()
        corpus_dir = PROJECT_ROOT / "data" / "sample_corpus"
        if not corpus_dir.exists():
            self.skipTest("Sample corpus directory not found.")

        load_res = loader.load_directory(corpus_dir)
        self.assertGreater(len(load_res.documents), 0)

        chunker = RecursiveStructuralChunker(target_chunk_size=350, chunk_overlap=50)
        for doc in load_res.documents:
            chunks = chunker.split_document(doc)
            self.assertGreater(len(chunks), 0)
            for ch in chunks:
                self.assertTrue(
                    self.expected_keys.issubset(ch.metadata.keys()),
                    f"File {doc.filename} missing metadata keys: {self.expected_keys - set(ch.metadata.keys())}",
                )

    # -------------------------------------------------------------------------
    # Task 4: Trace a Chunk to its Source
    # -------------------------------------------------------------------------

    def test_task4_trace_chunk_exact_slice_match(self):
        """Task 4: Demonstrate that a retrieved chunk traces back to its exact source location."""
        chunker = RecursiveStructuralChunker(target_chunk_size=120, chunk_overlap=20)
        chunks = chunker.split_text(self.sample_text, metadata=self.doc_meta)

        for ch in chunks:
            trace = ChunkTracer.trace(ch, source_text=self.sample_text)
            self.assertTrue(trace.is_verified)
            self.assertEqual(trace.similarity_score, 1.0)
            self.assertGreater(len(trace.matched_slice), 0)
            self.assertIn(">>> [CHUNK CONTENT] <<<", trace.surrounding_context)
            self.assertEqual(trace.source_identifier, "circular_dor_2024_108.txt")

            # Slice from original text must match extracted slice
            original_slice = self.sample_text[trace.char_start : trace.char_end]
            self.assertEqual(original_slice, trace.matched_slice)

    def test_task4_retrieval_simulation_and_trace(self):
        """Task 4: Simulate a compliance retrieval query and trace matching chunk to origin."""
        loader = DocumentLoader()
        corpus_dir = PROJECT_ROOT / "data" / "sample_corpus"
        if not corpus_dir.exists():
            self.skipTest("Sample corpus directory not found.")

        load_res = loader.load_directory(corpus_dir)
        chunker = RecursiveStructuralChunker(target_chunk_size=350, chunk_overlap=50)

        all_chunks = []
        doc_map = {}
        for doc in load_res.documents:
            chunks = chunker.split_document(doc)
            all_chunks.extend(chunks)
            doc_map[doc.filename] = doc.content

        # Compliance query: "Who must approve PEP relationships?"
        matching_chunk = None
        for ch in all_chunks:
            if "Deputy General Manager" in ch.content:
                matching_chunk = ch
                break

        self.assertIsNotNone(matching_chunk)
        src_text = doc_map.get(matching_chunk.metadata["filename"])
        trace = matching_chunk.trace(source_text=src_text)

        self.assertTrue(trace.is_verified)
        self.assertIn(trace.section, [
            "2. Customer Due Diligence (CDD) Requirements",
            "3. Enhanced Due Diligence (EDD) for High-Risk Accounts and PEPs",
            "3. Politically Exposed Persons (PEP) Workflow",
            "Preamble / Document Header",
        ])
        self.assertEqual(trace.page_number, 1)
        self.assertIn("Deputy General Manager", trace.matched_slice)

    # -------------------------------------------------------------------------
    # Task 5: Sample Chunks Export and Serialization
    # -------------------------------------------------------------------------

    def test_task5_chunk_serialization_to_dict(self):
        """Task 5: TextChunk.to_dict produces clean serializable dict with metadata."""
        chunker = RecursiveStructuralChunker(target_chunk_size=100, chunk_overlap=15)
        chunks = chunker.split_text(self.sample_text, metadata=self.doc_meta)

        ch = chunks[0]
        d = ch.to_dict()

        self.assertIsInstance(d, dict)
        self.assertEqual(d["chunk_id"], ch.chunk_id)
        self.assertEqual(d["content"], ch.content)
        self.assertIn("metadata", d)
        self.assertEqual(d["metadata"]["filename"], "circular_dor_2024_108.txt")

        # Must be JSON serializable
        json_str = json.dumps(d)
        self.assertIn("circular_dor_2024_108.txt", json_str)

    def test_task5_exported_sample_chunks_json_validity(self):
        """Task 5: Verify outputs/sample_chunks_with_metadata.json exists and conforms to schema."""
        json_path = PROJECT_ROOT / "outputs" / "sample_chunks_with_metadata.json"
        if not json_path.exists():
            # Run generator
            run_metadata_trace_demonstration()

        self.assertTrue(json_path.exists())
        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        for item in data:
            self.assertIn("chunk_id", item)
            self.assertIn("content", item)
            self.assertIn("metadata", item)
            m = item["metadata"]
            self.assertTrue(self.expected_keys.issubset(m.keys()))


if __name__ == "__main__":
    unittest.main()
