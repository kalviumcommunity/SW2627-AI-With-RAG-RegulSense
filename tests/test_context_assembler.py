"""Unit tests for RegulSense Grounded Context Assembly & Prompt Augmentation Engine.

Validates:
1. Task 1: Formatting retrieved chunks and injecting them into the prompt.
2. Task 2: Strict token budget enforcement reserving room for system instructions,
   user question, and generated answer.
3. Task 3: Inclusion of sequential source markers ([1], [2], document, section, chunk ID).
4. Task 4: Grounding instructions directing the model to answer only from context
   and state when context is insufficient.
5. Task 5: Serialization and report export integrity.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest

from src.context_assembler import (
    AugmentedPrompt,
    ContextAssembler,
    GROUNDED_SYSTEM_INSTRUCTIONS,
    InjectedChunk,
    SourceMarker,
    TokenBudgetLedger,
    TokenBudgetManager,
    TokenBudgetSpec,
    export_augmented_prompt_artifacts,
    generate_augmented_prompt_markdown,
)
from src.vector_db import RetrievedRecord


class TestContextAssembler(unittest.TestCase):
    """Test suite for ContextAssembler, token budgeting, and source markers."""

    def setUp(self):
        """Prepares synthetic regulatory records and assembler instance."""
        self.assembler = ContextAssembler()
        self.mock_records = [
            RetrievedRecord(
                rank=1,
                id="cyber_sec_chunk_001",
                similarity_score=0.8850,
                distance=0.1150,
                document="Any Severity 1 cyber security incident must be reported to the RBI CSITE and CERT-In within 6 hours of detection.",
                metadata={
                    "source_document": "cyber_resilience_framework.pdf",
                    "section": "2. Incident Reporting Timelines",
                    "page_number": 2,
                    "token_count": 25,
                },
            ),
            RetrievedRecord(
                rank=2,
                id="digital_lending_chunk_002",
                similarity_score=0.7420,
                distance=0.2580,
                document="Recovery agents shall not contact borrowers before 08:00 hours or after 19:00 hours under digital lending guidelines.",
                metadata={
                    "source_document": "digital_lending_compliance_note.html",
                    "section": "Code of Conduct for Recovery",
                    "page_number": 1,
                    "token_count": 22,
                },
            ),
        ]

    def test_chunk_injection_and_source_markers(self):
        """Verifies Task 1 & Task 3: Chunks are formatted with sequential [1], [2] source markers."""
        query = "What is the reporting deadline for Severity 1 cyber incidents?"
        prompt = self.assembler.assemble(
            query=query,
            chunks=self.mock_records,
        )

        self.assertIsInstance(prompt, AugmentedPrompt)
        self.assertEqual(len(prompt.injected_chunks), 2)

        # Check marker [1]
        chunk1 = prompt.injected_chunks[0]
        self.assertEqual(chunk1.marker.marker_id, "[1]")
        self.assertEqual(chunk1.marker.source_document, "cyber_resilience_framework.pdf")
        self.assertEqual(chunk1.marker.section, "2. Incident Reporting Timelines")
        self.assertAlmostEqual(chunk1.marker.similarity_score, 0.8850)
        self.assertIn("SOURCE [1]: cyber_resilience_framework.pdf", chunk1.formatted_block)
        self.assertIn("Section: 2. Incident Reporting Timelines", chunk1.formatted_block)

        # Check marker [2]
        chunk2 = prompt.injected_chunks[1]
        self.assertEqual(chunk2.marker.marker_id, "[2]")
        self.assertEqual(chunk2.marker.source_document, "digital_lending_compliance_note.html")
        self.assertIn("SOURCE [2]: digital_lending_compliance_note.html", chunk2.formatted_block)

        # Verify source marker dictionary index
        self.assertIn("[1]", prompt.source_markers)
        self.assertIn("[2]", prompt.source_markers)
        self.assertEqual(prompt.source_markers["[1]"]["source_document"], "cyber_resilience_framework.pdf")

    def test_token_budget_calculation_and_ledger(self):
        """Verifies Task 2: Token budget correctly allocates and audits tokens across all components."""
        spec = TokenBudgetSpec(
            total_budget=1500,
            answer_reserve=400,
            safety_margin=50,
        )

        prompt = self.assembler.assemble(
            query="When must banks report incidents?",
            chunks=self.mock_records,
            budget_spec=spec,
        )

        ledger = prompt.budget_ledger
        self.assertIsInstance(ledger, TokenBudgetLedger)
        self.assertEqual(ledger.total_budget, 1500)
        self.assertEqual(ledger.answer_reserve, 400)
        self.assertEqual(ledger.safety_margin, 50)
        self.assertGreater(ledger.system_tokens, 0)
        self.assertGreater(ledger.query_tokens, 0)
        self.assertGreater(ledger.context_tokens, 0)

        # Check that total allocated equals sum of parts
        expected_total = (
            ledger.system_tokens
            + self.assembler.token_manager.count_tokens(prompt.user_prompt)
            + ledger.answer_reserve
            + ledger.safety_margin
        )
        self.assertEqual(ledger.total_allocated, expected_total)
        self.assertTrue(ledger.is_within_budget)
        self.assertGreaterEqual(ledger.remaining_tokens, 0)
        self.assertEqual(ledger.chunks_included_count, 2)
        self.assertEqual(ledger.chunks_dropped_count, 0)

    def test_token_budget_truncation_enforcement(self):
        """Verifies Task 2: Budget ceiling stops or truncates chunks when available space is small."""
        # Budget calibrated so only 1 chunk fits before ceiling is reached
        tight_spec = TokenBudgetSpec(
            total_budget=600,
            answer_reserve=100,
            safety_margin=20,
        )

        prompt = self.assembler.assemble(
            query="When must banks report incidents?",
            chunks=self.mock_records,
            budget_spec=tight_spec,
        )

        ledger = prompt.budget_ledger
        # Should have dropped chunks or truncated
        self.assertTrue(ledger.was_truncated or ledger.chunks_dropped_count > 0)
        self.assertTrue(ledger.is_within_budget)
        self.assertLessEqual(ledger.total_allocated, 600)

    def test_grounding_instructions_presence(self):
        """Verifies Task 4: Augmented prompt contains strict grounding, citation, and refusal rules."""
        prompt = self.assembler.assemble(
            query="What are the permitted hours for recovery agents?",
            chunks=self.mock_records,
        )

        # Check system prompt constraints
        self.assertIn("STRICT CONTEXT BOUNDARY", prompt.system_prompt)
        self.assertIn("relying SOLELY and EXCLUSIVELY on the regulatory evidence", prompt.system_prompt)
        self.assertIn("MANDATORY SOURCE CITATIONS", prompt.system_prompt)
        self.assertIn("you MUST cite the corresponding source marker (e.g. [1], [2]", prompt.system_prompt)
        self.assertIn("INSUFFICIENT CONTEXT FALLBACK PROTOCOL", prompt.system_prompt)
        self.assertIn("The provided regulatory context does not contain sufficient information to answer this question.", prompt.system_prompt)

        # Check user prompt context boundary
        self.assertIn("--- RETRIEVED REGULATORY CONTEXT ---", prompt.user_prompt)
        self.assertIn("SOURCE [1]:", prompt.user_prompt)
        self.assertIn("--- END REGULATORY CONTEXT ---", prompt.user_prompt)
        self.assertIn("COMPLIANCE INQUIRY:", prompt.user_prompt)

    def test_insufficient_context_empty_chunks_fallback(self):
        """Verifies Task 4: Empty chunk list injects clear fallback and alerts model."""
        prompt = self.assembler.assemble(
            query="What are the Basel III capital adequacy ratios?",
            chunks=[],  # No chunks found
        )

        self.assertEqual(prompt.budget_ledger.chunks_included_count, 0)
        self.assertIn("NO REGULATORY CONTEXT AVAILABLE", prompt.raw_context_block)
        self.assertIn("insufficient", prompt.raw_context_block.lower())

    def test_chat_messages_array_structure(self):
        """Verifies messages payload conforms strictly to OpenAI chat format."""
        prompt = self.assembler.assemble(
            query="Sample inquiry",
            chunks=self.mock_records,
        )

        self.assertEqual(len(prompt.messages), 2)
        self.assertEqual(prompt.messages[0]["role"], "system")
        self.assertEqual(prompt.messages[1]["role"], "user")
        self.assertIn("RegulSense", prompt.messages[0]["content"])
        self.assertIn("Sample inquiry", prompt.messages[1]["content"])

    def test_serialization_and_markdown_export(self):
        """Verifies Task 5: JSON serialization and Markdown artifact generation."""
        prompt = self.assembler.assemble(
            query="What is the 6-hour cyber reporting rule?",
            chunks=self.mock_records,
        )

        data = prompt.to_dict()
        self.assertEqual(data["query"], "What is the 6-hour cyber reporting rule?")
        self.assertIn("[1]", data["source_markers"])
        self.assertEqual(data["budget_ledger"]["chunks_included_count"], 2)

        # Test Markdown formatting
        md_text = generate_augmented_prompt_markdown(
            prompt=prompt,
            llm_completion="Under [1], Severity 1 incidents must be reported within 6 hours.",
        )
        self.assertIn("# RegulSense: Grounded Context Assembly", md_text)
        self.assertIn("Token Budget Allocation & Consumption Ledger", md_text)
        self.assertIn("`[1]`", md_text)
        self.assertIn("`[2]`", md_text)
        self.assertIn("Model Generated Answer", md_text)

        # Test file export
        with tempfile.TemporaryDirectory() as tmp_dir:
            md_path, json_path = export_augmented_prompt_artifacts(
                prompt=prompt,
                llm_completion="Sample completion with [1]",
                output_dir=tmp_dir,
            )
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())

            with open(json_path, "r", encoding="utf-8") as f:
                loaded_json = json.load(f)
                self.assertIn("augmented_prompt", loaded_json)
                self.assertIn("llm_completion", loaded_json)


if __name__ == "__main__":
    unittest.main()
