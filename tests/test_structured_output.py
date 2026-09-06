"""Unit tests for StructuredOutputHandler, resilient parsing, and schema validation."""

import json
from pathlib import Path
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.structured_output_handler import StructuredOutputHandler, ParseResult, ValidationResult


class TestStructuredOutputHandler(unittest.TestCase):
    """Test suite for structured output parsing, error recovery, and validation."""

    def setUp(self):
        self.handler = StructuredOutputHandler()

    def test_direct_clean_json_parsing(self):
        """Verify that a valid JSON object parses directly into a dict."""
        payload = json.dumps({"answer": "Approval required from DGM.", "source": "Section 3(a)"})
        res = self.handler.parse_json_safely(payload)

        self.assertTrue(res.success)
        self.assertEqual(res.parse_strategy_used, "direct_json_loads")
        self.assertFalse(res.repaired)
        self.assertIsInstance(res.data, dict)
        self.assertEqual(res.data["answer"], "Approval required from DGM.")
        self.assertEqual(res.data["source"], "Section 3(a)")

    def test_markdown_fence_extraction(self):
        """Verify extraction from ```json ... ``` markdown code fences."""
        payload = "```json\n{\n  \"answer\": \"Threshold is 10 lakhs.\",\n  \"source\": \"Section 4(a)\"\n}\n```"
        res = self.handler.parse_json_safely(payload)

        self.assertTrue(res.success)
        self.assertEqual(res.parse_strategy_used, "markdown_fence_extraction")
        self.assertTrue(res.repaired)
        self.assertEqual(res.data["answer"], "Threshold is 10 lakhs.")

    def test_conversational_wrapper_extraction(self):
        """Verify extraction when JSON is embedded in conversational intro and outro."""
        payload = (
            "Here is the compliance answer you requested:\n"
            "{\"answer\": \"STR within 7 days.\", \"source\": \"Section 4(c)\"}\n"
            "Hope this helps your audit!"
        )
        res = self.handler.parse_json_safely(payload)

        self.assertTrue(res.success)
        self.assertEqual(res.parse_strategy_used, "regex_bracket_extraction")
        self.assertTrue(res.repaired)
        self.assertEqual(res.data["answer"], "STR within 7 days.")

    def test_syntax_normalization_trailing_comma(self):
        """Verify that trailing commas are cleanly removed before parsing."""
        payload = "{\n  \"answer\": \"5-year retention period.\",\n  \"source\": \"Section 5(a)\",\n}"
        res = self.handler.parse_json_safely(payload)

        self.assertTrue(res.success)
        self.assertEqual(res.parse_strategy_used, "syntax_normalization_repair")
        self.assertTrue(res.repaired)
        self.assertEqual(res.data["answer"], "5-year retention period.")

    def test_syntax_normalization_single_quotes(self):
        """Verify that Python-style single-quoted dictionaries are repaired into valid JSON."""
        payload = "{'answer': 'V-CIP confidence > 95%.', 'source': 'Section 2(c)'}"
        res = self.handler.parse_json_safely(payload)

        self.assertTrue(res.success)
        self.assertEqual(res.parse_strategy_used, "syntax_normalization_repair")
        self.assertTrue(res.repaired)
        self.assertEqual(res.data["answer"], "V-CIP confidence > 95%.")

    def test_catastrophic_malformed_json_handled_gracefully(self):
        """Verify that completely unparseable text returns failure without raising unhandled exceptions."""
        payload = "INTERNAL_SERVER_ERROR: incomplete stream { 'unclosed..."
        res = self.handler.parse_json_safely(payload)

        self.assertFalse(res.success)
        self.assertIsNone(res.data)
        self.assertIsNotNone(res.error_message)
        self.assertIn("JSONDecodeError", res.error_message)

    def test_validation_success_on_complete_schema(self):
        """Verify that complete schema passes validation with status VALID."""
        data = {
            "answer": "Approval required from DGM.",
            "source": "Circular DOR.AML.REC.66",
            "confidence": "HIGH",
        }
        val = self.handler.validate_schema(data)

        self.assertTrue(val.valid)
        self.assertEqual(val.status, "VALID")
        self.assertEqual(len(val.missing_fields), 0)
        self.assertEqual(val.validated_data["confidence"], "HIGH")

    def test_validation_failure_on_missing_required_field(self):
        """Verify that missing required field without alias is rejected with status INVALID."""
        data = {"answer": "Some compliance text."}  # Missing 'source'
        val = self.handler.validate_schema(data)

        self.assertFalse(val.valid)
        self.assertEqual(val.status, "INVALID")
        self.assertIn("source", val.missing_fields)

    def test_validation_alias_recovery(self):
        """Verify that missing required keys are recovered when recognizable aliases exist."""
        data = {
            "response": "Beneficial ownership is 10%.",  # Alias for 'answer'
            "citation": "Section 2(b)",                  # Alias for 'source'
            "certainty": "HIGH",                        # Alias for 'confidence'
        }
        val = self.handler.validate_schema(data)

        self.assertTrue(val.valid)
        self.assertEqual(val.status, "RECOVERED")
        self.assertIn("answer", val.recovered_fields)
        self.assertIn("source", val.recovered_fields)
        self.assertEqual(val.validated_data["answer"], "Beneficial ownership is 10%.")
        self.assertEqual(val.validated_data["source"], "Section 2(b)")

    def test_demonstration_suite_and_report_generation(self):
        """Verify that demonstration suite runs all cases and outputs markdown file."""
        from src.structured_output_handler import run_demonstration_suite

        result = run_demonstration_suite()
        self.assertIn("report_path", result)
        report_file = Path(result["report_path"])
        self.assertTrue(report_file.exists())
        content = report_file.read_text(encoding="utf-8")
        self.assertIn("RegulSense: Structured JSON Generation", content)
        self.assertIn("Case 1: Clean JSON Response", content)
        self.assertIn("Case 4: Malformed Syntax", content)
        self.assertIn("Case 5: Missing Required Field Recovered", content)
        self.assertIn("Case 6: Irreparable Malformed Content", content)


if __name__ == "__main__":
    unittest.main()
