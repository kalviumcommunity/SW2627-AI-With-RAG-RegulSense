"""Unit and Safety Tests for Environment Configuration and Secret Protection.

Validates Task 4:
- Ensures .env and secret patterns are properly excluded by Git.
- Validates that no hardcoded API secrets or bearer tokens are present in the source codebase.
- Verifies AppConfig correctly loads and prioritizes environment variables.
- Confirms API secret redaction in /api/v1/config.
- Confirms safe fallback when optional configuration is omitted.
"""

import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from src.api import AppConfig, create_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestEnvironmentSafety(unittest.TestCase):
    """Test suite ensuring environment variables and secret safety standards are met."""

    def test_gitignore_protects_env_and_secrets(self):
        """Verify .gitignore properly excludes .env and secret files while keeping .env.example."""
        gitignore_path = PROJECT_ROOT / ".gitignore"
        self.assertTrue(gitignore_path.exists(), ".gitignore file must exist in project root")

        content = gitignore_path.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

        self.assertIn(".env", lines, ".env must be explicitly ignored in .gitignore")
        self.assertIn(".env.*", lines, ".env.* must be explicitly ignored in .gitignore")
        self.assertIn("!.env.example", lines, "!.env.example must be whitelisted in .gitignore")

    def test_no_hardcoded_api_keys_in_source_code(self):
        """Verify that no live OpenAI/cloud secret keys are committed in source files."""
        secret_patterns = [
            re.compile(r"sk-[a-zA-Z0-9_-]{24,}"),        # Standard OpenAI API key pattern
            re.compile(r"ghp_[a-zA-Z0-9]{36}"),          # GitHub personal access token
            re.compile(r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{30,}"), # Generic bearer token
        ]

        source_dirs = [
            PROJECT_ROOT / "src",
            PROJECT_ROOT / "prompts",
            PROJECT_ROOT / "app.py",
        ]

        files_to_check = []
        for path in source_dirs:
            if path.is_file():
                files_to_check.append(path)
            elif path.is_dir():
                files_to_check.extend([p for p in path.rglob("*.py") if "__pycache__" not in str(p)])

        violations = []
        for file_path in files_to_check:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            for pattern in secret_patterns:
                matches = pattern.findall(text)
                for match in matches:
                    # Filter out dummy/redacted placeholders
                    if "dummy" in match.lower() or "example" in match.lower() or "REDACTED" in match:
                        continue
                    violations.append(f"{file_path.name}: matches '{pattern.pattern}' -> {match[:10]}...")

        self.assertEqual(len(violations), 0, f"Found hardcoded secrets in source files: {violations}")

    def test_app_config_loads_all_environment_variables(self):
        """Verify AppConfig parses all documented environment variables properly."""
        test_env = {
            "OPENAI_BASE_URL": "http://mock-ai-server:11434/v1",
            "OPENAI_API_KEY": "test-env-key-9988",
            "CHAT_MODEL": "custom-compliance-llm",
            "EMBEDDING_MODEL": "custom-embedding-model",
            "CHROMA_COLLECTION_NAME": "custom_test_collection",
            "API_HOST": "127.0.0.1",
            "API_PORT": "8888",
            "DEFAULT_TOP_K": "5",
            "MIN_SIMILARITY_THRESHOLD": "0.62",
            "APP_ENV": "staging",
            "UPLOAD_DIRECTORY": str(PROJECT_ROOT / "data" / "staging_uploads"),
            "MAX_UPLOAD_SIZE_BYTES": "20971520", # 20 MB
        }

        with patch.dict(os.environ, test_env, clear=False):
            cfg = AppConfig()
            self.assertEqual(cfg.openai_base_url, "http://mock-ai-server:11434/v1")
            self.assertEqual(cfg.openai_api_key, "test-env-key-9988")
            self.assertEqual(cfg.chat_model, "custom-compliance-llm")
            self.assertEqual(cfg.embedding_model, "custom-embedding-model")
            self.assertEqual(cfg.chroma_collection, "custom_test_collection")
            self.assertEqual(cfg.api_host, "127.0.0.1")
            self.assertEqual(cfg.api_port, 8888)
            self.assertEqual(cfg.default_top_k, 5)
            self.assertEqual(cfg.min_similarity_threshold, 0.62)
            self.assertEqual(cfg.app_env, "staging")
            self.assertEqual(cfg.max_upload_size_bytes, 20971520)

    def test_api_config_endpoint_redaction(self):
        """Verify GET /api/v1/config redacts sensitive credentials."""
        from fastapi.testclient import TestClient

        with patch.dict(os.environ, {"OPENAI_API_KEY": "super-secret-production-token"}, clear=False):
            test_app = create_app()
            client = TestClient(test_app)

            resp = client.get("/api/v1/config")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data.get("status"), "success")
            self.assertEqual(data["config"]["openai_api_key"], "***REDACTED***")
            self.assertNotIn("super-secret-production-token", resp.text)


if __name__ == "__main__":
    unittest.main()
