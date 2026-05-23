from __future__ import annotations

from pathlib import Path
from typing import Callable

import customtkinter as ctk
from tkinter import filedialog

from utils.config import ProjectPaths
from utils.tts import TTSGender, TTSSettings, TTSSpeed, save_mp3_gtts, save_wav_pyttsx3, speak_async


class TTSPanel(ctk.CTkFrame):
    def __init__(self, master, get_last_asr_text: Callable[[], str]):  # noqa: ANN001
        super().__init__(master)

        self._get_last_asr_text = get_last_asr_text
        root = Path(__file__).resolve().parents[1]
        self._paths = ProjectPaths(root=root)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        title = ctk.CTkLabel(self, text="TTS — Bahasa Indonesia", font=ctk.CTkFont(size=18, weight="bold"))
        title.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 8))

        self.status_var = ctk.StringVar(value="Ready")
        self.speed_var = ctk.StringVar(value=TTSSpeed.normal.value)
        self.gender_var = ctk.StringVar(value=TTSGender.male.value)

        status_frame = ctk.CTkFrame(self)
        status_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=8)
        status_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(status_frame, text="Status:").grid(row=0, column=0, padx=12, pady=10, sticky="w")
        ctk.CTkLabel(status_frame, textvariable=self.status_var).grid(row=0, column=1, padx=12, pady=10, sticky="w")

        self.textbox = ctk.CTkTextbox(self, height=140)
        self.textbox.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 8))
        self.textbox.insert("1.0", "Halo, ini demo Text-to-Speech Bahasa Indonesia.")

        options_frame = ctk.CTkFrame(self)
        options_frame.grid(row=3, column=0, sticky="ew", padx=16, pady=8)
        options_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(options_frame, text="Speed:").grid(row=0, column=0, padx=12, pady=12, sticky="w")
        self.speed_menu = ctk.CTkOptionMenu(
            options_frame,
            values=[TTSSpeed.slow.value, TTSSpeed.normal.value, TTSSpeed.fast.value],
            variable=self.speed_var,
        )
        self.speed_menu.grid(row=0, column=1, padx=12, pady=12, sticky="ew")

        ctk.CTkLabel(options_frame, text="Gender:").grid(row=1, column=0, padx=12, pady=(0, 12), sticky="w")
        self.gender_menu = ctk.CTkOptionMenu(
            options_frame,
            values=[TTSGender.male.value, TTSGender.female.value],
            variable=self.gender_var,
        )
        self.gender_menu.grid(row=1, column=1, padx=12, pady=(0, 12), sticky="ew")

        btns_frame = ctk.CTkFrame(self)
        btns_frame.grid(row=4, column=0, sticky="ew", padx=16, pady=8)
        btns_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.btn_use_asr = ctk.CTkButton(btns_frame, text="Use ASR", command=self._use_asr)
        self.btn_use_asr.grid(row=0, column=0, padx=12, pady=12, sticky="ew")

        self.btn_play = ctk.CTkButton(btns_frame, text="Play", command=self._play)
        self.btn_play.grid(row=0, column=1, padx=12, pady=12, sticky="ew")

        self.btn_save = ctk.CTkButton(btns_frame, text="Save Audio", command=self._save)
        self.btn_save.grid(row=0, column=2, padx=12, pady=12, sticky="ew")

        hint = ctk.CTkLabel(
            self,
            text=(
                "Save: WAV (offline via pyttsx3) atau MP3 (gTTS butuh internet).\n"
                "Default save: outputs/tts_output.wav"
            ),
            justify="left",
            anchor="w",
        )
        hint.grid(row=5, column=0, sticky="nw", padx=16, pady=(8, 16))

    # === API untuk integrasi ASR → TTS textbox ===
    # Dipanggil dari MainApp saat ASR menghasilkan teks terbaru.
    # Ini membuat demo UAS lebih "otomatis": user cukup bicara, lalu textbox TTS terisi.
    def set_text_from_asr(self, text: str) -> None:
        """Isi textbox TTS dari hasil ASR (dipanggil programmatically)."""

        def update() -> None:
            self.textbox.delete("1.0", "end")
            self.textbox.insert("1.0", (text or "").strip())
            self.status_var.set("Auto-filled from ASR")

        # Aman dipanggil baik dari thread maupun main thread.
        self.after(0, update)

    def _settings(self) -> TTSSettings:
        return TTSSettings(speed=TTSSpeed(self.speed_var.get()), gender=TTSGender(self.gender_var.get()))

    def _get_text(self) -> str:
        return self.textbox.get("1.0", "end").strip()

    def _use_asr(self) -> None:
        text = self._get_last_asr_text().strip()
        if not text:
            self.status_var.set("Belum ada hasil ASR")
            return
        self.textbox.delete("1.0", "end")
        self.textbox.insert("1.0", text)
        self.status_var.set("Copied from ASR")

    def _play(self) -> None:
        text = self._get_text()
        if not text:
            self.status_var.set("Teks kosong")
            return
        self.status_var.set("Playing...")
        speak_async(text, self._settings(), on_done=lambda: self.after(0, lambda: self.status_var.set("Ready")))

    def _save(self) -> None:
        text = self._get_text()
        if not text:
            self.status_var.set("Teks kosong")
            return

        default_name = "tts_output.wav"
        out_path_str = filedialog.asksaveasfilename(
            title="Save TTS Audio",
            initialdir=str(self._paths.outputs_dir),
            initialfile=default_name,
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav"), ("MP3 audio", "*.mp3")],
        )

        if not out_path_str:
            self.status_var.set("Save dibatalkan")
            return

        out_path = Path(out_path_str)
        self.status_var.set("Saving...")

        try:
            if out_path.suffix.lower() == ".mp3":
                save_mp3_gtts(text, out_path, self._settings())
            else:
                save_wav_pyttsx3(text, out_path, self._settings())
        except Exception as e:  # noqa: BLE001
            self.status_var.set(f"Save gagal: {e}")
            return

        self.status_var.set(f"Saved: {out_path.name}")
