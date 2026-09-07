# RegulSense: Structured JSON Generation, Resilient Parsing & Validation Report

- **Target Model**: `llama3:latest` (with `response_format={'type': 'json_object'}`)
- **Schema Required Fields**: `answer`, `source`
- **Domain**: Banking Regulatory Compliance (RBI KYC/AML Directives)
- **Execution Timestamp**: 2026-09-07 13:52:10
- **Status**: 100% Graceful Execution Across All Edge Cases

---

## 1. Executive Summary

Downstream banking workflows (audit logging, risk flagging, compliance queues) require reliable, machine-readable 
objects. Unstructured prose introduces parsing ambiguities, while naive `json.loads()` calls crash on minor markdown 
fences, trailing commas, or key discrepancies.

`StructuredOutputHandler` solves this through:
1. **Explicit Schema Prompting**: Constrains output structure to standardized keys (`answer`, `source`, `applicable_rule`, `confidence`, `action_required`).
2. **Multi-Stage Resilient Parser**: Recovers from markdown wrappers, regex bracket boundaries, and syntax errors (trailing commas, single quotes).
3. **Required Field Validation & Soft Recovery**: Verifies critical fields and remaps known LLM aliases (`response` -> `answer`, `citation` -> `source`).
4. **Zero-Crash Failure Handling**: Completely corrupt or truncated text produces structured diagnostic errors rather than unhandled exceptions.

---

## 2. Quantitative Summary of Test Scenarios

| Scenario ID | Test Case | Parse Strategy | Repaired? | Validation Status | Downstream Usability |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `case_1_clean_json_mode` | **Case 1: Clean JSON Response (API Response Mode)** | `direct_json_loads` | No | `VALID` | Ready for Downstream App |
| `case_2_markdown_fenced` | **Case 2: Markdown Code Block Enclosure** | `markdown_fence_extraction` | **Yes** | `VALID` | Ready for Downstream App |
| `case_3_conversational_wrapper` | **Case 3: Conversational Preamble and Postamble** | `regex_bracket_extraction` | **Yes** | `VALID` | Ready for Downstream App |
| `case_4_syntax_malformed_repaired` | **Case 4: Malformed Syntax (Trailing Commas & Single Quotes)** | `syntax_normalization_repair` | **Yes** | `VALID` | Ready for Downstream App |
| `case_5_missing_field_alias_recovered` | **Case 5: Missing Required Field Recovered via Alias Mapping** | `direct_json_loads` | No | `RECOVERED` | Ready for Downstream App |
| `case_6_irreparable_error_handled` | **Case 6: Irreparable Malformed Content Gracefully Handled** | `syntax_repair_failed` | No | `PARSE_FAILED` | Gracefully Rejected (Diagnostic Logged) |

---

## 3. Deep-Dive Case Evaluations

### Scenario 1: Case 1: Clean JSON Response (API Response Mode)
- **Description**: Direct JSON object returned cleanly from the API conforming to schema.
- **Parse Strategy Executed**: `direct_json_loads`
- **Syntax Repair Applied**: `False`
- **Validation Status**: `VALID`

#### Raw Model / Input Text:
```text
{
  "answer": "Accounts classified as high-risk, including PEPs, require approval from an officer not below Deputy General Manager.",
  "source": "RBI Master Direction DOR.AML.REC.66 Section 3(a)",
  "applicable_rule": "Section 3(a), PML Rules 2005",
  "confidence": "HIGH",
  "action_required": "Submit written approval request to DGM before opening account."
}
```

#### Parsed & Validated Python Object (`dict`):
```json
{
  "answer": "Accounts classified as high-risk, including PEPs, require approval from an officer not below Deputy General Manager.",
  "source": "RBI Master Direction DOR.AML.REC.66 Section 3(a)",
  "applicable_rule": "Section 3(a), PML Rules 2005",
  "confidence": "HIGH",
  "action_required": "Submit written approval request to DGM before opening account."
}
```

### Scenario 2: Case 2: Markdown Code Block Enclosure
- **Description**: Model wraps JSON in markdown ```json ... ``` code fence.
- **Parse Strategy Executed**: `markdown_fence_extraction`
- **Syntax Repair Applied**: `True`
- **Validation Status**: `VALID`

#### Raw Model / Input Text:
```text
```json
{
  "answer": "Cash transactions exceeding INR 10,00,000 must be reported monthly to FIU-IND by the 15th day of the succeeding month.",
  "source": "RBI Master Direction Section 4(a)",
  "applicable_rule": "Rule 3, PML (Maintenance of Records) Rules 2005",
  "confidence": "HIGH",
  "action_required": "File Cash Transaction Report (CTR) via FINnet portal."
}
```
```

#### Parsed & Validated Python Object (`dict`):
```json
{
  "answer": "Cash transactions exceeding INR 10,00,000 must be reported monthly to FIU-IND by the 15th day of the succeeding month.",
  "source": "RBI Master Direction Section 4(a)",
  "applicable_rule": "Rule 3, PML (Maintenance of Records) Rules 2005",
  "confidence": "HIGH",
  "action_required": "File Cash Transaction Report (CTR) via FINnet portal."
}
```

### Scenario 3: Case 3: Conversational Preamble and Postamble
- **Description**: Model includes conversational introductory text and polite trailing remarks.
- **Parse Strategy Executed**: `regex_bracket_extraction`
- **Syntax Repair Applied**: `True`
- **Validation Status**: `VALID`

#### Raw Model / Input Text:
```text
Certainly! Here is the regulatory compliance assessment you requested in JSON format:

{
  "answer": "Suspicious Transaction Reports (STR) must be filed within 7 working days of arriving at suspicion, regardless of transaction amount.",
  "source": "RBI Circular Section 4(c)",
  "applicable_rule": "Rule 7(2), PML Rules 2005",
  "confidence": "HIGH",
  "action_required": "Escalate to Principal Officer immediately for FIU-IND STR filing."
}

Please let me know if you need any additional clarification on RBI AML compliance!
```

#### Parsed & Validated Python Object (`dict`):
```json
{
  "answer": "Suspicious Transaction Reports (STR) must be filed within 7 working days of arriving at suspicion, regardless of transaction amount.",
  "source": "RBI Circular Section 4(c)",
  "applicable_rule": "Rule 7(2), PML Rules 2005",
  "confidence": "HIGH",
  "action_required": "Escalate to Principal Officer immediately for FIU-IND STR filing."
}
```

### Scenario 4: Case 4: Malformed Syntax (Trailing Commas & Single Quotes)
- **Description**: Syntax errors that cause standard json.loads() to crash: trailing comma before closing brace and Python single quotes.
- **Parse Strategy Executed**: `syntax_normalization_repair`
- **Syntax Repair Applied**: `True`
- **Validation Status**: `VALID`

#### Raw Model / Input Text:
```text
{
  'answer': 'Transaction records must be retained for at least five years from the transaction date.',
  'source': 'RBI Circular Section 5(a)',
  'applicable_rule': 'Section 12, Banking Regulation Act',
  'confidence': 'HIGH',
  'action_required': 'Ensure audit logs and swift transaction reconstruction archives are maintained.',
}
```

#### Parsed & Validated Python Object (`dict`):
```json
{
  "answer": "Transaction records must be retained for at least five years from the transaction date.",
  "source": "RBI Circular Section 5(a)",
  "applicable_rule": "Section 12, Banking Regulation Act",
  "confidence": "HIGH",
  "action_required": "Ensure audit logs and swift transaction reconstruction archives are maintained."
}
```

### Scenario 5: Case 5: Missing Required Field Recovered via Alias Mapping
- **Description**: Model hallucinated key names ('response' instead of 'answer', 'citation' instead of 'source'). Recovered through aliasing.
- **Parse Strategy Executed**: `direct_json_loads`
- **Syntax Repair Applied**: `False`
- **Validation Status**: `RECOVERED`

#### Raw Model / Input Text:
```text
{
  "response": "Beneficial ownership threshold for corporate entities is set at 10% or more of voting rights or shares.",
  "citation": "RBI Master Direction Section 2(b)",
  "applicable_rule": "Section 2(b) CDD Requirements",
  "confidence": "HIGH",
  "action_required": "Identify natural persons with >=10% ownership during KYC onboarding."
}
```

#### Parsed & Validated Python Object (`dict`):
```json
{
  "applicable_rule": "Section 2(b) CDD Requirements",
  "confidence": "HIGH",
  "action_required": "Identify natural persons with >=10% ownership during KYC onboarding.",
  "answer": "Beneficial ownership threshold for corporate entities is set at 10% or more of voting rights or shares.",
  "source": "RBI Master Direction Section 2(b)"
}
```

> [!NOTE]
> **Soft Recovery Applied**: Missing schema keys were successfully recovered via aliases: `{'answer': 'response', 'source': 'citation'}`.

### Scenario 6: Case 6: Irreparable Malformed Content Gracefully Handled
- **Description**: Catastrophic truncation or completely non-JSON content. Handled without crashing, returning structured diagnostics.
- **Parse Strategy Executed**: `syntax_repair_failed`
- **Syntax Repair Applied**: `False`
- **Validation Status**: `PARSE_FAILED`

#### Raw Model / Input Text:
```text
FATAL_ERROR: Context window connection dropped mid-transmission { 'partial_unclosed_string...
```

#### Graceful Diagnostic Capture (No Crash):
```text
Error: JSONDecodeError: Expecting value at line 1 col 1
```

> [!IMPORTANT]
> The system intercepted the malformed payload, prevented an unhandled exception, and returned a structured failure record for administrative review.

---

## 4. Production Integration Guide

```python
from src.structured_output_handler import StructuredOutputHandler

handler = StructuredOutputHandler()

# End-to-end extraction with guaranteed safety
result = handler.process_query(
    query='What are the PEP onboarding rules under RBI directions?',
    context_chunk=regulatory_text,
)

if result['status'] in ('SUCCESS', 'RECOVERED'):
    data = result['data']
    print(f'Answer: {data["answer"]}')
    print(f'Source: {data["source"]}')
    print(f'Action: {data["action_required"]}')
else:
    print(f'Safe fallback triggered: {result["error"]}')
```