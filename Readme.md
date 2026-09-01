# RegulSense RAG Assistant

## Problem Statement

A large bank maintains compliance circulars, internal audit reports, and regulatory updates, but risk officers cannot quickly confirm which current rule governs a transaction without reading through conflicting historical documents.

---

## Solution

RegulSense uses a Retrieval-Augmented Generation (RAG) pipeline to search through regulatory and compliance documents and provide context-aware answers.

The system is designed to:

- Retrieve relevant regulatory documents
- Identify currently active rules
- Detect outdated or superseded regulations
- Compare conflicting documents
- Prioritize the latest applicable regulation
- Provide explainable answers with source references

---

## Development Environment Setup

This repository contains the foundational workspace for RegulSense, a Retrieval-Augmented Generation (RAG) assistant designed to help risk and compliance officers identify the currently applicable rule from regulatory circulars, internal audit reports, and compliance documents.

## Project Structure

```text
rag-app-starter/
│
├── data/              # Local source documents
│   └── .gitkeep
│
├── src/               # Application source code
│   ├── __init__.py
│   └── main.py
│
├── prompts/           # Prompt templates
│   └── .gitkeep
│
├── outputs/           # Generated application outputs
│   └── .gitkeep
│
├── .env.example       # Environment variable template
├── .gitignore         # Files excluded from Git
├── requirements.txt   # Python dependencies
└── README.md
```

## Prerequisites

* Python 3.10 or later
* pip
* Git

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd rag-app-starter
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

#### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
.venv\Scripts\activate
```

#### macOS/Linux

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Copy `.env.example` to `.env`.

#### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

#### macOS/Linux

```bash
cp .env.example .env
```

Update the `.env` file with the required API configuration:

```env
OPENAI_BASE_URL=
OPENAI_API_KEY=
CHAT_MODEL=
EMBEDDING_MODEL=
```

Never commit the `.env` file because it may contain sensitive API credentials.

### 6. Run the application

```bash
python src/main.py
```

Expected output:

```text
RegulSense RAG Environment Setup Successful!
Environment variables loaded successfully.
```

## Dependencies

* OpenAI-compatible API client
* ChromaDB for vector storage
* python-dotenv for environment variable management
* FastAPI for future backend development
* Uvicorn ASGI server

## Security

The `.gitignore` file excludes:

* `.venv/`
* `node_modules/`
* `.env`
* Local data files
* Generated outputs
* Python cache files

No API keys or sensitive environment files are committed to the repository.

## Clean Run Verification

The project setup was verified using a clean environment with the following workflow:

1. Create a new virtual environment.
2. Install dependencies from `requirements.txt`.
3. Copy `.env.example` to `.env`.
4. Configure required environment variables.
5. Run `python src/main.py`.

The application started successfully, confirming that the workspace setup is reproducible on a fresh machine.
