"""Entry point aplikasi.

Menjalankan GUI CustomTkinter (ASR real-time + TTS).

Jika model belum ada, jalankan dulu:
- scripts/record_dataset.py
- train/train_mlp.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Friendly error if user runs with system Python instead of the project's venv
    try:
        from gui.app_gui import run
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or "<unknown>"
        in_venv = hasattr(sys, "base_prefix") and sys.prefix != sys.base_prefix

        print("\n[ERROR] Python dependency is missing:")
        print(f"- Missing module: {missing}")
        print(f"- Python executable: {sys.executable}")
        print(f"- Running inside venv: {in_venv}\n")

        print("Fix (Windows):")
        print("1) Create venv (once):  python -m venv .venv")
        print("2) Activate venv:")
        print("   - PowerShell:  .\\.venv\\Scripts\\Activate.ps1")
        print("   - CMD:         .\\.venv\\Scripts\\activate.bat")
        print("3) Install deps:        pip install -r requirements.txt")
        print("4) Run app:             python app.py\n")

        print("Alternative (no activation):")
        print("- .\\.venv\\Scripts\\python.exe app.py\n")
        raise SystemExit(1) from e

    run()


if __name__ == "__main__":
    main()
