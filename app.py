"""Root entrypoint for RegulSense Streamlit Chat Interface.

Usage:
    streamlit run app.py
"""

from pathlib import Path
import runpy
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Execute main chat application
chat_app_path = PROJECT_ROOT / "src" / "chat_app.py"
runpy.run_path(str(chat_app_path), run_name="__main__")
