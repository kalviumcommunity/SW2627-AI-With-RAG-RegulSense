"""Prompts package for RegulSense."""
from prompts.templates import (
    VARIATION_A_SYSTEM_PROMPT,
    VARIATION_B_SYSTEM_PROMPT,
    TEST_QUERIES,
    format_messages,
    PromptTemplate,
    ChatPromptTemplate,
    RAG_COMPLIANCE_USER_TEMPLATE,
    REGUL_SENSE_SYSTEM_TEMPLATE,
    BATCH_AUDIT_USER_TEMPLATE,
)

__all__ = [
    "VARIATION_A_SYSTEM_PROMPT",
    "VARIATION_B_SYSTEM_PROMPT",
    "TEST_QUERIES",
    "format_messages",
    "PromptTemplate",
    "ChatPromptTemplate",
    "RAG_COMPLIANCE_USER_TEMPLATE",
    "REGUL_SENSE_SYSTEM_TEMPLATE",
    "BATCH_AUDIT_USER_TEMPLATE",
]

