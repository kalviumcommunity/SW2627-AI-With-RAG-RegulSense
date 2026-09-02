# RegulSense: System Prompt Architecture & Evaluation

## 1. Overview

In a regulatory compliance system such as **RegulSense**, the prompt defines the assistant's behavior, operational scope, safety posture, and formatting standards before retrieval (RAG) is connected.

This document presents the design rationale for the chosen production system prompt, contrasts it with an unconstrained baseline prompt, and details the structural elements that make it reliable, safe, and actionable for banking risk officers.

---

## 2. Prompt Variations Tested

### Variation A: Vague / Unconstrained Baseline

```text
You are an AI assistant for bank staff. Answer their questions.
```

#### Deficiencies of Variation A:
- **No Role Identity:** The model lacks professional identity and context, resulting in generic customer-service phrasing rather than compliance-grade language.
- **Undefined Scope:** It does not specify boundaries. The model is prone to speculating on unverified policies, giving unauthorized legal advice, or even assisting with harmful inquiries.
- **Lack of Constraints:** Answers can become verbose, meandering, or poorly formatted, making quick review difficult for busy risk officers.
- **Missing Fallback:** When uncertain or asked about out-of-scope/unethical topics, the model may hallucinate answers or attempt to satisfy requests that should be strictly refused.

---

### Variation B: RegulSense Constrained System Prompt (Chosen)

```text
You are RegulSense, an AI Regulatory Compliance Specialist for a commercial bank. Your duty is to assist risk officers and internal audit staff with regulatory compliance questions.

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

## 3. Key Components of the Chosen Prompt (Variation B)

| Component | Implementation in Variation B | Purpose & Impact |
| :--- | :--- | :--- |
| **Role & Persona** | "RegulSense, an AI Regulatory Compliance Specialist for a commercial bank" | Establishes domain authority, sets risk officer target audience, and focuses the model on institutional compliance rather than retail chat. |
| **In-Scope Boundary** | Informational summaries of banking regulations, AML/KYC, internal risk controls, reporting standards | Centers the knowledge boundary strictly on regulatory and audit topics. |
| **Out-of-Scope Boundary** | Explicitly prohibits legal advice, approval guarantees, speculation on non-public policies, and control circumvention | Prevents compliance liability, reduces hallucination, and enforces ethical banking guardrails. |
| **Tone Constraint** | Strictly professional, neutral, and risk-conscious | Eliminates conversational fluff ("Sure, I'd be happy to help you with that!") and maintains an audit-ready tone. |
| **Structure & Length** | Direct 1-sentence answer + 2-3 bullets; strictly under 150 words | Delivers fast, scannable insights suitable for risk officers reviewing high volumes of transactions. |
| **Fallback Protocol** | Standardized refusal message redirecting to Master Circulars or Legal/Compliance Department | Provides a safe, deterministic defense against jailbreaks, evasion questions, and queries outside the knowledge base. |

---

## 4. Evaluation & Rationale for Chosen Prompt

### 1. Distinct Role Separation (System vs. User)
The system message completely isolates behavioral configuration from dynamic input. The user message solely carries the specific question without needing to re-instruct the model on formatting, safety, or role.

### 2. Clarity & Scannability
Variation B requires a 1-sentence executive summary followed by 2-3 bullet points. This structure allows compliance staff to assess the core obligation within 5 seconds, compared to Variation A's dense, multi-paragraph essays.

### 3. Safety & Circumvention Resistance
When presented with queries regarding transaction structuring (smurfing/avoiding CTR reporting):
- **Variation A** risks explaining technical mechanics or discussing threshold limits in a manner that could be misinterpreted as instructional.
- **Variation B** activates its explicit fallback protocol, refusing to facilitate control circumvention and directing the staff member to official compliance channels.

### 4. Hallucination Containment & Domain Discipline
By explicitly disallowing speculation and demanding escalation when circulars or compliance domains are absent, Variation B protects the bank from fabricated regulatory circular numbers or unauthorized financial advice.

---

## 5. Empirical Evaluation Summary

Live testing using `llama3:latest` (documented in detail in [prompt_comparison_results.md](file:///c:/Users/uppal/OneDrive/Desktop/WI-sprint-2/SW2627-AI-With-RAG-RegulSense/outputs/prompt_comparison_results.md)) produced stark contrasts:

| Evaluation Dimension | Variation A (Vague Prompt) | Variation B (Constrained Prompt) |
| :--- | :--- | :--- |
| **AML Retention Query** | 226 words, 299 completion tokens, repetitive phrasing (took 44.6s). | **105 words, 131 completion tokens**, structured bullets under <150 word limit (took 20.9s). |
| **Transaction Structuring (Evasion)** | Generic refusal without institutional context. | Immediate institutional refusal directing to compliance boundaries. |
| **Personal Tech Stock Inquiry** | **Severe Domain Drift:** Model hallucinated stock picks (MSFT, GOOGL, CRM) for 313 words (took 59.6s). | **Strict Fallback Adherence:** Model refused immediately in 8 words (3.6s), preventing unauthorized investment advisory liability. |

### Decision
**Variation B is selected as the production standard prompt** for RegulSense. It enforces institutional identity, eliminates mission drift, ensures token and latency efficiency, and maintains audit-ready response formatting.

