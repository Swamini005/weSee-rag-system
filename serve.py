"""One-command entry point: installs dependencies on first run, checks config,
builds the index if needed, then serves the API on port 8000.

Usage:  python serve.py
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REQUIRED_MODULES = ["fastapi", "uvicorn", "sentence_transformers", "faiss", "groq", "dotenv", "numpy"]


def bootstrap():
    missing = [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        print(f"First run: installing dependencies ({', '.join(missing)}) ...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r",
            str(Path(__file__).parent / "requirements.txt"),
        ])


def check_config():
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        sys.exit(
            "GROQ_API_KEY is not set.\n"
            "Copy .env.example to .env and paste your key (free at https://console.groq.com)."
        )


if __name__ == "__main__":
    bootstrap()
    check_config()
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)