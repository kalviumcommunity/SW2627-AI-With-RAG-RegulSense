# RegulSense: Reusable Prompt Templates & Example Renders Report

- **Engine**: `prompts.templates.PromptTemplate` & `ChatPromptTemplate`
- **Architecture**: Decoupled Prompt Templates (Stored in `prompts/`, Consumed in `src/`)
- **Shared Across Features**: `src/chat_history_manager.py` (Chat) & `src/batch_audit_cli.py` (Batch Audit)
- **Execution Timestamp**: 2026-09-06 16:54:31
- **Status**: Verified Across Multiple Pipeline Features

---

## 1. Executive Summary & Design Architecture

In enterprise software, hardcoding prompt strings inside application logic creates severe maintainability bottlenecks: 
a single compliance phrasing change requires editing scattered string concatenations across multiple files.

`prompts/templates.py` resolves this by introducing:
1. **Template Separation**: Prompts are stored in a standalone configuration module (`prompts/templates.py`) completely isolated from application logic.
2. **Named Placeholders**: Named placeholders (`{context}`, `{question}`, `{audit_id}`) declare explicit variable dependencies.
3. **Missing Variable Enforcement**: `render()` raises immediate `KeyError` if any mandatory placeholder is missing at runtime.
4. **Cross-Feature Reuse**: The exact same `RAG_COMPLIANCE_USER_TEMPLATE` is reused by both the **Interactive Chat Session** (`chat_history_manager.py`) and the **Batch Audit CLI** (`batch_audit_cli.py`).

---

## 2. Shared Template Specifications

### Template 1: `RAG_COMPLIANCE_USER_TEMPLATE` (General RAG Query)
- **Consumers**: `src/chat_history_manager.py`, `src/batch_audit_cli.py`
- **Input Variables**: `context`, `question`
```text
--- RETRIEVED REGULATORY CONTEXT ---
{context}
------------------------------------

Compliance Question: {question}
```

### Template 2: `BATCH_AUDIT_USER_TEMPLATE` (Structured Audit Dossier)
- **Consumers**: `src/batch_audit_cli.py`
- **Input Variables**: `audit_id`, `transaction_type`, `amount`, `context`, `question`
```text
--- TRANSACTION AUDIT DOSSIER ---
Audit Item ID: {audit_id}
Transaction Type: {transaction_type}
Amount: {amount}
Relevant Circular: {context}
---------------------------------

Compliance Officer Inquiry: {question}
Determine if this transaction requires statutory reporting and cite specific rule provisions.
```

### Template 3: `REGUL_SENSE_SYSTEM_TEMPLATE` (Parameterized System Prompt)
- **Consumers**: Chat Sessions, Batch Evaluator
- **Input Variables**: `assistant_name`, `bank_entity`
```text
You are {assistant_name}, an AI Regulatory Compliance Specialist for {bank_entity}. Your duty is to assist risk officers and internal audit staff with regulatory compliance questions.

SCOPE OF ASSISTANCE:
- You ONLY provide general informational summaries of banking regulations, AML/KYC standards, internal risk controls, and supervisory reporting standards.
- You MUST NOT provide formal legal advice, guarantee regulatory approvals, speculate on non-public bank policies, or assist in circumventing compliance controls.

RESPONSE CONSTRAINTS:
1. Tone: Maintain a strictly professional, neutral, and risk-conscious tone.
2. Structure: Begin with a direct 1-sentence answer, followed by 2 to 3 concise bullet points highlighting key compliance obligations or caveats.
3. Length: Keep your entire response under 150 words. Be direct and avoid conversational filler.

FALLBACK PROTOCOL:
If a query is outside banking compliance, requests personal investment/tax evasion advice, or requires circulars/documents not provided, decline to answer and respond with:
"I cannot advise on this matter as it falls outside verified regulatory compliance guidelines. Please refer to the relevant Master Circular or escalate to the Bank's Compliance & Legal Department."
```

---

## 3. Runtime Variable Injection & Example Renders (Task 2, 3, & 5)

Below are example renders showing templates dynamically filled at runtime with live regulatory data.

### Example Render 1: AUDIT-2024-001 (Cash Deposit)

#### Feature A: Rendered via `BATCH_AUDIT_USER_TEMPLATE` (Batch Path):
```text
--- TRANSACTION AUDIT DOSSIER ---
Audit Item ID: AUDIT-2024-001
Transaction Type: Cash Deposit
Amount: INR 14,50,000
Relevant Circular: Section 4(a). Cash Transaction Reports (CTRs):
All cash transactions of the value of more than rupees ten lakhs or its equivalent in foreign currency must be reported monthly to FIU-IND by the 15th day of the succeeding month.
---------------------------------

Compliance Officer Inquiry: Does this deposit breach mandatory CTR thresholds, and what is the FIU-IND filing deadline?
Determine if this transaction requires statutory reporting and cite specific rule provisions.
```

#### Feature B: Rendered via `RAG_COMPLIANCE_USER_TEMPLATE` (Shared Chat Path):
```text
--- RETRIEVED REGULATORY CONTEXT ---
Section 4(a). Cash Transaction Reports (CTRs):
All cash transactions of the value of more than rupees ten lakhs or its equivalent in foreign currency must be reported monthly to FIU-IND by the 15th day of the succeeding month.
------------------------------------

Compliance Question: Does this deposit breach mandatory CTR thresholds, and what is the FIU-IND filing deadline?
```

#### Compliance Audit Model Assessment:
```text
NON-COMPLIANCE RISK DETECTED: The cash deposit of INR 14,50,000 exceeds the statutory INR 10,00,000 CTR threshold.
* Statutory Obligation: A Cash Transaction Report (CTR) must be submitted to FIU-IND.
* Deadline: Must be filed no later than the 15th day of the succeeding month under Rule 3 of PML Rules, 2005.
```

---

### Example Render 2: AUDIT-2024-002 (Account Onboarding)

#### Feature A: Rendered via `BATCH_AUDIT_USER_TEMPLATE` (Batch Path):
```text
--- TRANSACTION AUDIT DOSSIER ---
Audit Item ID: AUDIT-2024-002
Transaction Type: Account Onboarding
Amount: Initial Wire: USD 250,000
Relevant Circular: Section 3(a). Approval from Senior Management:
Establishing relationships with PEPs, their family members, or close associates requires written approval from an officer not below the rank of Deputy General Manager (DGM).
---------------------------------

Compliance Officer Inquiry: Was AGM-level branch approval legally sufficient to activate this PEP account under RBI guidelines?
Determine if this transaction requires statutory reporting and cite specific rule provisions.
```

#### Feature B: Rendered via `RAG_COMPLIANCE_USER_TEMPLATE` (Shared Chat Path):
```text
--- RETRIEVED REGULATORY CONTEXT ---
Section 3(a). Approval from Senior Management:
Establishing relationships with PEPs, their family members, or close associates requires written approval from an officer not below the rank of Deputy General Manager (DGM).
------------------------------------

Compliance Question: Was AGM-level branch approval legally sufficient to activate this PEP account under RBI guidelines?
```

#### Compliance Audit Model Assessment:
```text
CRITICAL COMPLIANCE BREACH: Approval from an Assistant General Manager (AGM) is legally invalid for onboarding a PEP.
* Statutory Mandate: Establishing a banking relationship with a PEP requires written approval from an officer not below Deputy General Manager (DGM) rank.
* Remedial Action: Freeze debit transactions and immediately escalate dossier to DGM for formal ratification.
```

---

### Example Render 3: AUDIT-2024-003 (Structured Wire Transfers)

#### Feature A: Rendered via `BATCH_AUDIT_USER_TEMPLATE` (Batch Path):
```text
--- TRANSACTION AUDIT DOSSIER ---
Audit Item ID: AUDIT-2024-003
Transaction Type: Structured Wire Transfers
Amount: 4 x INR 9,50,000 within 48 hours
Relevant Circular: Section 4(c). Suspicious Transaction Reports (STRs):
If any transaction gives rise to reasonable suspicion of illicit funds, evasion, or terrorism financing, an STR shall be furnished to FIU-IND within seven working days of arriving at such a conclusion. STR obligations apply regardless of transaction amount or whether a CTR has been submitted.
---------------------------------

Compliance Officer Inquiry: Does structuring transactions below INR 10 lakhs exempt the bank from filing an STR?
Determine if this transaction requires statutory reporting and cite specific rule provisions.
```

#### Feature B: Rendered via `RAG_COMPLIANCE_USER_TEMPLATE` (Shared Chat Path):
```text
--- RETRIEVED REGULATORY CONTEXT ---
Section 4(c). Suspicious Transaction Reports (STRs):
If any transaction gives rise to reasonable suspicion of illicit funds, evasion, or terrorism financing, an STR shall be furnished to FIU-IND within seven working days of arriving at such a conclusion. STR obligations apply regardless of transaction amount or whether a CTR has been submitted.
------------------------------------

Compliance Question: Does structuring transactions below INR 10 lakhs exempt the bank from filing an STR?
```

#### Compliance Audit Model Assessment:
```text
STRUCTURING ALERT: Subdividing transactions into amounts below INR 10 lakhs is an overt indicator of evasion (smurfing).
* Obligation: An STR must be submitted to FIU-IND within 7 working days under Section 4(c).
* Caveat: Sub-threshold amounts do not exempt the entity from STR obligations.
```

---

## 4. Multi-Feature Code Integration Evidence (Task 3 & 4)

### Feature 1: Interactive Chat Path (`src/chat_history_manager.py`)
```python
from prompts.templates import RAG_COMPLIANCE_USER_TEMPLATE

def add_user_message(self, query: str, context_chunk: Optional[str] = None):
    if context_chunk:
        # Dynamically renders template from prompts/ module
        formatted_content = RAG_COMPLIANCE_USER_TEMPLATE.render(
            context=context_chunk.strip(),
            question=query.strip(),
        )
    ...
```

### Feature 2: Batch Audit CLI (`src/batch_audit_cli.py`)
```python
from prompts.templates import RAG_COMPLIANCE_USER_TEMPLATE, BATCH_AUDIT_USER_TEMPLATE

def evaluate_audit_item(self, item: dict):
    # Reuses the exact same template definition without modifying business logic
    rendered_prompt = BATCH_AUDIT_USER_TEMPLATE.render(
        audit_id=item['audit_id'],
        transaction_type=item['transaction_type'],
        amount=item['amount'],
        context=item['circular_section'],
        question=item['question'],
    )
```