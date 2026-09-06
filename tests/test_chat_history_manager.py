"""Unit tests for ChatHistoryManager context window management and token budgeting."""

import unittest
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chat_history_manager import ChatHistoryManager, ChatMessage
from prompts.templates import VARIATION_B_SYSTEM_PROMPT


class TestChatHistoryManager(unittest.TestCase):
    """Test suite for ChatHistoryManager."""

    def setUp(self):
        """Set up test instances before each test."""
        self.system_prompt = VARIATION_B_SYSTEM_PROMPT
        self.manager = ChatHistoryManager(
            system_prompt=self.system_prompt,
            token_budget=600,
            reserve_output_tokens=150,
        )

    def test_initialization_and_system_prompt(self):
        """Verify system prompt is initialized properly as index 0 and marked immutable."""
        self.assertEqual(len(self.manager.messages), 1)
        system_msg = self.manager.messages[0]
        self.assertEqual(system_msg.role, "system")
        self.assertIn("RegulSense", system_msg.content)
        self.assertTrue(system_msg.metadata.get("immutable", False))
        self.assertEqual(self.manager.effective_input_budget, 450)

    def test_token_counting_accuracy(self):
        """Verify token counting reflects content and message overhead."""
        initial_tokens = self.manager.count_total_tokens()
        self.assertGreater(initial_tokens, 0)

        # Add user message
        msg = self.manager.add_user_message("What is the CTR threshold?")
        new_tokens = self.manager.count_total_tokens()
        self.assertGreater(new_tokens, initial_tokens)

        # Confirm message token calculation includes overhead
        msg_tokens = self.manager.count_message_tokens(msg)
        self.assertEqual(new_tokens, initial_tokens + msg_tokens)

    def test_rag_context_augmentation(self):
        """Verify user message properly attaches retrieved regulatory circular chunks."""
        context = "Section 4. All cash transactions exceeding INR 10,00,000 must be reported."
        query = "Explain CTR requirements."
        msg = self.manager.add_user_message(query=query, context_chunk=context)

        self.assertIn("RETRIEVED REGULATORY CONTEXT", msg.content)
        self.assertIn("Compliance Question: Explain CTR requirements.", msg.content)
        self.assertTrue(msg.metadata["has_context"])
        self.assertGreater(msg.metadata["context_tokens"], 0)

    def test_trimming_preserves_system_prompt_and_active_user_message(self):
        """Verify that trimming drops oldest turns while strictly preserving system prompt and active query."""
        # Set a small budget to force immediate trimming
        manager = ChatHistoryManager(
            system_prompt="System: You are an AI assistant.",
            token_budget=150,
            reserve_output_tokens=50,
        )  # Effective budget: 100 tokens

        # Add Turn 1
        manager.add_user_message("First question on KYC policies and customer verification rules?")
        manager.add_assistant_message("Answer 1: Customer identification requires OVD documents.")

        # Add Turn 2
        manager.add_user_message("Second question on Politically Exposed Persons (PEPs) due diligence?")
        manager.add_assistant_message("Answer 2: PEPs require DGM approval and source of funds verification.")

        # Add Turn 3 (Active turn)
        manager.add_user_message("Third question on wire transfer thresholds and reporting?")

        tokens_before = manager.count_total_tokens()
        self.assertGreater(tokens_before, manager.effective_input_budget)

        # Enforce trimming
        result = manager.enforce_token_budget_trimming()
        self.assertEqual(result["action"], "trimmed")
        self.assertGreater(result["pruned_turns"], 0)

        # Verify invariants
        messages = manager.messages
        # 1. System prompt preserved at index 0
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[0].content, "System: You are an AI assistant.")

        # 2. Latest active user query preserved as the last message
        self.assertEqual(messages[-1].role, "user")
        self.assertIn("Third question", messages[-1].content)

        # 3. Tokens are reduced under budget
        self.assertLessEqual(manager.count_total_tokens(), manager.effective_input_budget)

    def test_summarization_strategy(self):
        """Verify that summarization compacts intermediate turns into a summary checkpoint."""
        manager = ChatHistoryManager(
            system_prompt="System: Compliance assistant.",
            token_budget=70,
            reserve_output_tokens=20,
        )  # Effective budget: 50 tokens

        manager.add_user_message("Query 1 on Cash Transaction Reporting and mandatory thresholds?")
        manager.add_assistant_message("Answer 1: All cash transactions exceeding 10 lakhs must be reported to FIU.")
        manager.add_user_message("Query 2 on Suspicious Transaction Reports timeline and rules?")
        manager.add_assistant_message("Answer 2: STR must be submitted within 7 working days.")
        manager.add_user_message("Query 3 on Politically Exposed Persons (PEPs) due diligence?")

        tokens_before = manager.count_total_tokens()
        result = manager.enforce_token_budget_summarization()

        self.assertEqual(result["action"], "summarized")
        # Messages should now be: [system, summary, query 3]
        self.assertEqual(len(manager.messages), 3)
        self.assertEqual(manager.messages[0].role, "system")
        self.assertEqual(manager.messages[1].role, "system")
        self.assertIn("PREVIOUS TURNS SUMMARY", manager.messages[1].content)
        self.assertEqual(manager.messages[2].role, "user")
        self.assertIn("Politically Exposed Persons", manager.messages[2].content)
        self.assertLessEqual(manager.count_total_tokens(), tokens_before)

    def test_api_message_format(self):
        """Verify get_messages_for_api produces clean dictionary payloads."""
        self.manager.add_user_message("Hello")
        self.manager.add_assistant_message("Hi there")
        payload = self.manager.get_messages_for_api()

        self.assertIsInstance(payload, list)
        for msg in payload:
            self.assertIn("role", msg)
            self.assertIn("content", msg)
            self.assertIsInstance(msg["role"], str)
            self.assertIsInstance(msg["content"], str)


if __name__ == "__main__":
    unittest.main()
