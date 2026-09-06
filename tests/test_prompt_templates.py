"""Unit tests for PromptTemplate, ChatPromptTemplate, and cross-feature reuse."""

from pathlib import Path
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompts.templates import (
    PromptTemplate,
    ChatPromptTemplate,
    RAG_COMPLIANCE_USER_TEMPLATE,
    REGUL_SENSE_SYSTEM_TEMPLATE,
    BATCH_AUDIT_USER_TEMPLATE,
)
from src.chat_history_manager import ChatHistoryManager
from src.batch_audit_cli import BatchAuditEvaluator, SAMPLE_AUDIT_BATCH


class TestPromptTemplates(unittest.TestCase):
    """Test suite for template engine and prompt abstraction."""

    def test_placeholder_auto_discovery(self):
        """Verify that named placeholders are automatically extracted from template text."""
        tmpl = PromptTemplate("Hello {name}, your account balance is {balance}.")
        self.assertEqual(tmpl.get_placeholders(), {"name", "balance"})

    def test_successful_rendering(self):
        """Verify dynamic runtime injection into placeholders."""
        tmpl = PromptTemplate("Compliance rule for {entity}: {rule}")
        rendered = tmpl.render(entity="Commercial Banks", rule="Maintain CTR records for 5 years.")
        self.assertEqual(rendered, "Compliance rule for Commercial Banks: Maintain CTR records for 5 years.")

    def test_missing_variable_raises_key_error(self):
        """Verify that rendering without all required placeholders raises a clear KeyError."""
        tmpl = PromptTemplate("Context: {context}\nQuestion: {question}")
        with self.assertRaises(KeyError) as ctx:
            tmpl.render(context="Some circular")
        self.assertIn("question", str(ctx.exception))

    def test_chat_prompt_template_rendering(self):
        """Verify that ChatPromptTemplate returns OpenAI-compatible message dictionaries."""
        chat_tmpl = ChatPromptTemplate(
            system_template="You are an assistant for {bank}.",
            user_template="Explain rule {rule_id}.",
        )
        messages = chat_tmpl.render_messages(bank="Apex Bank", rule_id="Section 4A")

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], {"role": "system", "content": "You are an assistant for Apex Bank."})
        self.assertEqual(messages[1], {"role": "user", "content": "Explain rule Section 4A."})

    def test_shared_rag_template_placeholders(self):
        """Verify that production RAG template contains expected variables."""
        self.assertEqual(RAG_COMPLIANCE_USER_TEMPLATE.get_placeholders(), {"context", "question"})
        rendered = RAG_COMPLIANCE_USER_TEMPLATE.render(
            context="Section 3 text",
            question="What is the approval rank?",
        )
        self.assertIn("--- RETRIEVED REGULATORY CONTEXT ---", rendered)
        self.assertIn("Section 3 text", rendered)
        self.assertIn("Compliance Question: What is the approval rank?", rendered)

    def test_feature_1_chat_history_manager_integration(self):
        """Verify Feature 1 (Chat Path) uses RAG_COMPLIANCE_USER_TEMPLATE."""
        manager = ChatHistoryManager(system_prompt="System message")
        msg = manager.add_user_message(
            query="What are PEP approval rules?",
            context_chunk="DGM approval required.",
        )
        self.assertIn("--- RETRIEVED REGULATORY CONTEXT ---", msg.content)
        self.assertIn("DGM approval required.", msg.content)
        self.assertIn("Compliance Question: What are PEP approval rules?", msg.content)

    def test_feature_2_batch_audit_cli_integration(self):
        """Verify Feature 2 (Batch Audit CLI) uses BATCH_AUDIT_USER_TEMPLATE and RAG_COMPLIANCE_USER_TEMPLATE."""
        evaluator = BatchAuditEvaluator()
        item = SAMPLE_AUDIT_BATCH[0]
        res = evaluator.evaluate_audit_item(item, live=False)

        self.assertEqual(res["audit_id"], item["audit_id"])
        self.assertIn("--- TRANSACTION AUDIT DOSSIER ---", res["rendered_audit_prompt"])
        self.assertIn(item["amount"], res["rendered_audit_prompt"])
        self.assertIn("--- RETRIEVED REGULATORY CONTEXT ---", res["rendered_shared_rag_prompt"])

    def test_batch_pipeline_and_markdown_generation(self):
        """Verify that batch audit pipeline executes and generates documentation report."""
        evaluator = BatchAuditEvaluator()
        result = evaluator.run_batch_pipeline(live=False)

        self.assertIn("report_path", result)
        report_file = Path(result["report_path"])
        self.assertTrue(report_file.exists())
        content = report_file.read_text(encoding="utf-8")
        self.assertIn("RegulSense: Reusable Prompt Templates", content)
        self.assertIn("BATCH_AUDIT_USER_TEMPLATE", content)
        self.assertIn("RAG_COMPLIANCE_USER_TEMPLATE", content)
        self.assertIn("AUDIT-2024-001", content)


if __name__ == "__main__":
    unittest.main()
