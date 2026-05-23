from __future__ import annotations

import sys
from pathlib import Path

import customtkinter as ctk

# Allow running from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.asr_panel import ASRPanel
from gui.tts_panel import TTSPanel


class MainApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("PTU 2026 — ASR (MFCC+MLP) + TTS (ID)")
        self.geometry("1100x650")
        self.minsize(1000, 600)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Shared state for ASR→TTS integration
        self.last_asr_text: str = ""

        self.asr_panel = ASRPanel(self, on_text_update=self._on_asr_text)
        self.asr_panel.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)

        self.tts_panel = TTSPanel(self, get_last_asr_text=lambda: self.last_asr_text)
        self.tts_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=16)

    def _on_asr_text(self, text: str) -> None:
        # `text` berasal dari ASRPanel.
        # Setelah perubahan fitur, `text` biasanya sudah berupa info_text (bukan hanya label).
        self.last_asr_text = text

        # Requirement: otomatis kirim info_text ke textbox TTS.
        # Ini menggantikan alur lama yang mengharuskan klik tombol "Use ASR".
        try:
            self.tts_panel.set_text_from_asr(text)
        except Exception:
            # Best-effort: jika GUI state belum siap, jangan crash aplikasi.
            pass


def run() -> None:
    app = MainApp()
    app.mainloop()
