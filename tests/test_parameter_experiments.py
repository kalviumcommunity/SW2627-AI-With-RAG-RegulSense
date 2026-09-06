"""Unit tests for ParameterExperimentRunner and generation hyperparameter evaluation."""

import unittest
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parameter_experiments import ParameterExperimentRunner, ExperimentResult


class TestParameterExperiments(unittest.TestCase):
    """Test suite for generation parameter evaluation."""

    def setUp(self):
        self.runner = ParameterExperimentRunner()
        self.messages = [
            {"role": "system", "content": "You are a banking compliance assistant."},
            {"role": "user", "content": "What are PEP requirements?"},
        ]

    def test_temperature_determinism_at_zero(self):
        """Verify that temperature=0.0 produces identical output across consecutive runs."""
        text_1, finish_1, _ = self.runner.run_completion(
            self.messages,
            temperature=0.0,
            max_tokens=150,
            live=False,
        )
        text_2, finish_2, _ = self.runner.run_completion(
            self.messages,
            temperature=0.0,
            max_tokens=150,
            live=False,
        )

        self.assertEqual(text_1, text_2)
        self.assertEqual(finish_1, "stop")
        self.assertEqual(finish_2, "stop")
        self.assertIn("Deputy General Manager", text_1)

    def test_max_tokens_capping_and_length_finish_reason(self):
        """Verify that tight max_tokens limits length and reports finish_reason='length'."""
        text, finish_reason, _ = self.runner.run_completion(
            self.messages,
            temperature=0.0,
            max_tokens=25,
            live=False,
        )
        tokens = self.runner.count_tokens(text)
        self.assertLessEqual(tokens, 25)
        self.assertEqual(finish_reason, "length")

    def test_stop_sequence_execution(self):
        """Verify that stop sequences halt generation cleanly before subsequent text."""
        text, finish_reason, _ = self.runner.run_completion(
            self.messages,
            temperature=0.0,
            max_tokens=150,
            stop=["\n\n"],
            live=False,
        )
        self.assertEqual(finish_reason, "stop")
        self.assertNotIn("\n\n", text)
        self.assertIn("Deputy General Manager", text)

    def test_top_p_nucleus_restriction(self):
        """Verify top_p restricts sampling vocabulary."""
        text_low_p, finish_low, _ = self.runner.run_completion(
            self.messages,
            temperature=0.8,
            max_tokens=150,
            top_p=0.1,
            live=False,
        )
        text_high_p, finish_high, _ = self.runner.run_completion(
            self.messages,
            temperature=0.8,
            max_tokens=150,
            top_p=0.95,
            live=False,
        )
        self.assertEqual(finish_low, "stop")
        self.assertEqual(finish_high, "stop")
        self.assertNotEqual(text_low_p, text_high_p)

    def test_full_experiment_suite_execution(self):
        """Verify that run_all_experiments compiles all 3 experiment sets and writes output report."""
        results = self.runner.run_all_experiments(live=False)

        self.assertIn("temperature", results)
        self.assertIn("max_tokens", results)
        self.assertIn("stop_and_top_p", results)

        self.assertGreaterEqual(len(results["temperature"]), 3)
        self.assertGreaterEqual(len(results["max_tokens"]), 3)
        self.assertGreaterEqual(len(results["stop_and_top_p"]), 2)

        report_file = PROJECT_ROOT / "outputs" / "parameter_comparison_results.md"
        self.assertTrue(report_file.exists())
        content = report_file.read_text(encoding="utf-8")
        self.assertIn("Recommended Settings for Grounded Banking Compliance Assistant", content)
        self.assertIn("Task 1 — Temperature Variance Analysis", content)
        self.assertIn("Task 2 — Output Capping with `max_tokens`", content)
        self.assertIn("Task 3 — Additional Parameters", content)


if __name__ == "__main__":
    unittest.main()
