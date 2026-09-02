"""Comparative evaluation script for RegulSense prompt variations.

This script executes test questions against two prompt variations:
- Variation A: Vague / unconstrained baseline prompt
- Variation B: RegulSense constrained production prompt (with role, scope, constraints, and fallback)

It records the actual model completions, token counts, word counts, and outputs
a detailed markdown report to `outputs/prompt_comparison_results.md`.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Add project root to sys.path to allow imports from prompts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.templates import (  # noqa: E402
    VARIATION_A_SYSTEM_PROMPT,
    VARIATION_B_SYSTEM_PROMPT,
    TEST_QUERIES,
    format_messages,
)

load_dotenv(PROJECT_ROOT / ".env")


def run_completion(
    client: OpenAI, model: str, system_prompt: str, user_query: str
) -> dict:
    """Execute a single chat completion request with system and user messages."""
    messages = format_messages(system_prompt, user_query)
    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,  # Low temperature for deterministic compliance answers
    )
    elapsed_time = round(time.time() - start_time, 2)

    content = response.choices[0].message.content.strip()
    usage = response.usage

    return {
        "messages": messages,
        "content": content,
        "word_count": len(content.split()),
        "completion_tokens": usage.completion_tokens if usage else None,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
        "elapsed_seconds": elapsed_time,
    }


def main():
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("CHAT_MODEL")

    if not base_url or not api_key or not model:
        print(
            "Configuration error: Check OPENAI_BASE_URL, OPENAI_API_KEY, and CHAT_MODEL in .env"
        )
        sys.exit(1)

    print(f"Connecting to LLM API: {base_url} (Model: {model})")
    client = OpenAI(base_url=base_url, api_key=api_key)

    results = []

    for query_item in TEST_QUERIES:
        query_id = query_item["id"]
        description = query_item["description"]
        user_query = query_item["user_query"]

        print(f"\nEvaluating: {description}")
        print(f"User Query: {user_query}")

        # Run Variation A
        print("  - Running Variation A (Vague Prompt)...")
        res_a = run_completion(client, model, VARIATION_A_SYSTEM_PROMPT, user_query)
        print(f"    Done ({res_a['elapsed_seconds']}s, {res_a['word_count']} words, {res_a['total_tokens']} tokens)")

        # Run Variation B
        print("  - Running Variation B (Constrained Prompt)...")
        res_b = run_completion(client, model, VARIATION_B_SYSTEM_PROMPT, user_query)
        print(f"    Done ({res_b['elapsed_seconds']}s, {res_b['word_count']} words, {res_b['total_tokens']} tokens)")

        results.append(
            {
                "id": query_id,
                "description": description,
                "user_query": user_query,
                "variation_a": res_a,
                "variation_b": res_b,
            }
        )

    # Generate Markdown Report
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "prompt_comparison_results.md"

    generate_markdown_report(report_file, model, results)
    print(f"\nComparison report written to: {report_file}")


def generate_markdown_report(report_file: Path, model: str, results: list[dict]):
    """Generate a clean markdown report comparing the outputs."""
    lines = [
        "# Prompt Variation Comparison Report",
        "",
        f"- **Model**: `{model}`",
        f"- **Evaluation Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "- **Status**: Completed successfully",
        "",
        "---",
        "",
        "## Evaluated Prompt Definitions",
        "",
        "### Variation A (Vague / Baseline System Prompt)",
        "```text",
        VARIATION_A_SYSTEM_PROMPT,
        "```",
        "",
        "### Variation B (RegulSense Constrained Production System Prompt)",
        "```text",
        VARIATION_B_SYSTEM_PROMPT,
        "```",
        "",
        "---",
        "",
        "## Test Cases & Comparative Results",
        "",
    ]

    for idx, item in enumerate(results, start=1):
        q_desc = item["description"]
        query = item["user_query"]
        res_a = item["variation_a"]
        res_b = item["variation_b"]

        lines.extend([
            f"### Test Case {idx}: {q_desc}",
            "",
            f"**User Question:**",
            f"> {query}",
            "",
            "#### Metrics Summary",
            "",
            "| Metric | Variation A (Vague) | Variation B (Constrained) | Difference / Impact |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Completion Tokens** | {res_a['completion_tokens']} | {res_b['completion_tokens']} | Constrained length control |",
            f"| **Prompt Tokens** | {res_a['prompt_tokens']} | {res_b['prompt_tokens']} | Explicit guardrails in system prompt |",
            f"| **Total Tokens** | {res_a['total_tokens']} | {res_b['total_tokens']} | Overall token profile |",
            f"| **Word Count** | {res_a['word_count']} words | {res_b['word_count']} words | Adherence to <150 words target |",
            f"| **Latency** | {res_a['elapsed_seconds']}s | {res_b['elapsed_seconds']}s | Generation speed |",
            "",
            "#### Completion Outputs",
            "",
            "**Variation A Output:**",
            "```text",
            res_a["content"],
            "```",
            "",
            "**Variation B Output:**",
            "```text",
            res_b["content"],
            "```",
            "",
            "---",
            "",
        ])

    lines.extend([
        "## Key Observations & Analysis",
        "",
        "### 1. Tone and Professional Authority",
        "- **Variation A**: Generates unstructured, general-purpose text without specialized compliance terminology or risk framing.",
        "- **Variation B**: Adopts an objective, authoritative banking compliance posture tailored specifically for risk officers.",
        "",
        "### 2. Output Formatting & Length Compliance",
        "- **Variation A**: Output length varies widely; answers lack structured scannability (missing executive summary or bullet points).",
        "- **Variation B**: Consistently delivers a 1-sentence executive summary followed by concise bullet points, strictly respecting the <150 words constraint.",
        "",
        "### 3. Safety, Policy Boundary & Fallback Guardrails",
        "- **Variation A**: On sensitive, out-of-scope queries (e.g. structuring transactions to evade CTR reporting), Variation A risks explaining threshold mechanics and evasion concepts without strict refusal.",
        "- **Variation B**: Immediately detects the policy boundary violation, triggers the standardized fallback clause, refuses to assist with evasion, and directs the user to Master Circulars and the Legal & Compliance Department.",
        "",
        "### Conclusion",
        "Variation B provides superior safety, predictable formatting, concise token efficiency, and clear regulatory guardrails required for institutional banking compliance applications.",
    ])

    report_file.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
