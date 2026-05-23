from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

import customtkinter as ctk
import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from utils.asr_infer import (
    ASRBundle,
    ASRResult,
    RealtimeASR,
    SILENCE_LABEL,
    UNKNOWN_LABEL,
    load_asr_bundle,
    predict_one,
    top_k_predictions,
)
from utils.audio import record_fixed_duration
from utils.config import DEFAULT_AUDIO, ProjectPaths
from utils.country_info import (
    format_country_info_speech,
    format_country_info_text,
    format_country_info_tts_text,
    get_country_info,
)
from utils.tts import TTSSettings, TTSSpeed, speak_async


class ASRPanel(ctk.CTkFrame):
    def __init__(self, master, on_text_update: Callable[[str], None]):  # noqa: ANN001
        super().__init__(master)

        self._on_text_update = on_text_update

        root = Path(__file__).resolve().parents[1]
        self._paths = ProjectPaths(root=root)

        self._bundle: ASRBundle | None = None
        self._engine: RealtimeASR | None = None

        # Demo safety & auto speak
        self._safe_mode = False
        self._last_spoken_label: str = ""
        self._last_spoken_time = 0.0
        self._speak_cooldown_sec = 2.5

        self.grid_rowconfigure(6, weight=1)
        self.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(self, text="ASR — Pengenalan Negara ASEAN", font=ctk.CTkFont(size=18, weight="bold"))
        title.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 8))

        self.status_var = ctk.StringVar(value="Idle")
        self.mode_var = ctk.StringVar(value="Mode: Realtime")
        self.pred_var = ctk.StringVar(value="-")
        self.conf_var = ctk.StringVar(value="0%")

        self.auto_tts_var = ctk.BooleanVar(value=True)
        self.threshold_var = ctk.DoubleVar(value=0.85)
        self.threshold_lbl_var = ctk.StringVar(value="Auto TTS threshold: 85%")

        # Info negara (ditampilkan & dibacakan setelah prediksi ASR)
        # Disimpan sebagai teks multi-line agar mudah dibaca saat presentasi.
        self.country_info_var = ctk.StringVar(value="-")

        # Status
        status_frame = ctk.CTkFrame(self)
        status_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=8)
        status_frame.grid_columnconfigure(1, weight=1)
        status_frame.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(status_frame, text="Status:").grid(row=0, column=0, padx=12, pady=10, sticky="w")
        self.status_lbl = ctk.CTkLabel(status_frame, textvariable=self.status_var)
        self.status_lbl.grid(row=0, column=1, padx=12, pady=10, sticky="w")

        self.progress = ctk.CTkProgressBar(status_frame)
        self.progress.grid(row=0, column=2, padx=12, pady=10, sticky="ew")
        self.progress.set(0)
        self.progress.stop()

        self.mode_lbl = ctk.CTkLabel(status_frame, textvariable=self.mode_var)
        self.mode_lbl.grid(row=1, column=0, columnspan=3, padx=12, pady=(0, 10), sticky="w")

        # Controls
        control_frame = ctk.CTkFrame(self)
        control_frame.grid(row=2, column=0, sticky="ew", padx=16, pady=8)
        control_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.btn_start = ctk.CTkButton(control_frame, text="Start Realtime", command=self._start)
        self.btn_start.grid(row=0, column=0, padx=12, pady=12, sticky="ew")

        self.btn_stop = ctk.CTkButton(control_frame, text="Stop", command=self._stop, state="disabled")
        self.btn_stop.grid(row=0, column=1, padx=12, pady=12, sticky="ew")

        self.btn_once = ctk.CTkButton(control_frame, text="Record Once", command=self._record_once)
        self.btn_once.grid(row=0, column=2, padx=12, pady=12, sticky="ew")

        # Result + Auto TTS
        result_frame = ctk.CTkFrame(self)
        result_frame.grid(row=3, column=0, sticky="ew", padx=16, pady=8)
        result_frame.grid_columnconfigure(1, weight=1)
        result_frame.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(result_frame, text="Prediksi:").grid(row=0, column=0, padx=12, pady=(10, 4), sticky="w")
        pred_lbl = ctk.CTkLabel(result_frame, textvariable=self.pred_var, font=ctk.CTkFont(size=20, weight="bold"))
        pred_lbl.grid(row=0, column=1, padx=12, pady=(10, 4), sticky="w")

        self.auto_switch = ctk.CTkSwitch(result_frame, text="Auto TTS", variable=self.auto_tts_var)
        self.auto_switch.grid(row=0, column=2, padx=12, pady=(10, 4), sticky="e")

        ctk.CTkLabel(result_frame, text="Confidence:").grid(row=1, column=0, padx=12, pady=(4, 10), sticky="w")
        conf_lbl = ctk.CTkLabel(result_frame, textvariable=self.conf_var)
        conf_lbl.grid(row=1, column=1, padx=12, pady=(4, 10), sticky="w")

        self.threshold_lbl = ctk.CTkLabel(result_frame, textvariable=self.threshold_lbl_var)
        self.threshold_lbl.grid(row=1, column=2, padx=12, pady=(4, 0), sticky="e")

        self.threshold_slider = ctk.CTkSlider(
            result_frame,
            from_=0.50,
            to=0.95,
            number_of_steps=45,
            variable=self.threshold_var,
            command=self._on_threshold_change,
        )
        self.threshold_slider.grid(row=2, column=0, columnspan=3, padx=12, pady=(0, 12), sticky="ew")

        # Top-3
        topk_frame = ctk.CTkFrame(self)
        topk_frame.grid(row=4, column=0, sticky="ew", padx=16, pady=8)
        topk_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(topk_frame, text="Top-3 Predictions", anchor="w").grid(
            row=0, column=0, padx=12, pady=(12, 4), sticky="ew"
        )
        self.topk_box = ctk.CTkTextbox(topk_frame, height=90)
        self.topk_box.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="ew")
        self.topk_box.insert("1.0", "-")

        # Info negara
        # Requirement: Setelah ASR memprediksi negara, tampilkan info (lokasi/ibu kota/mata uang)
        # di GUI. TTS akan membacakan info yang sama jika confidence >= threshold.
        info_frame = ctk.CTkFrame(self)
        info_frame.grid(row=5, column=0, sticky="ew", padx=16, pady=8)
        info_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(info_frame, text="Info Negara", anchor="w").grid(
            row=0, column=0, padx=12, pady=(12, 4), sticky="ew"
        )
        self.info_box = ctk.CTkTextbox(info_frame, height=110)
        self.info_box.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="ew")
        self.info_box.insert("1.0", "-")

        # Plot
        plot_frame = ctk.CTkFrame(self)
        plot_frame.grid(row=6, column=0, sticky="nsew", padx=16, pady=(8, 16))
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(plot_frame, text="Waveform & MFCC", anchor="w").grid(
            row=0, column=0, padx=12, pady=(12, 4), sticky="ew"
        )

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        self._draw_empty_plot()

        # load model lazily
        self.after(50, self._load_model)

    def _on_threshold_change(self, v: float) -> None:
        pct = int(round(float(v) * 100))
        self.threshold_lbl_var.set(f"Auto TTS threshold: {pct}%")

    def _set_mode(self, mode: str) -> None:
        self.mode_var.set(f"Mode: {mode}")

    def _load_model(self) -> None:
        model_path = self._paths.models_dir / "asr_mlp.joblib"
        meta_path = self._paths.models_dir / "asr_meta.json"
        if not model_path.exists() or not meta_path.exists():
            self.status_var.set("Model tidak ditemukan. Jalankan: train/train_mlp.py")
            self.btn_start.configure(state="disabled")
            self.btn_once.configure(state="disabled")
            return

        try:
            self._bundle = load_asr_bundle(model_path, meta_path)
            self.status_var.set("Ready")
        except Exception as e:  # noqa: BLE001
            self.status_var.set(f"Gagal load model: {e}")
            self.btn_start.configure(state="disabled")
            self.btn_once.configure(state="disabled")

    def _start(self) -> None:
        if self._bundle is None:
            self.status_var.set("Model belum siap")
            return
        if self._engine is not None and self._engine.is_running:
            return

        self._safe_mode = False
        self._set_mode("Realtime")
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_once.configure(state="disabled")
        self.status_var.set("Starting mic...")
        self.progress.start()

        def start_engine() -> None:
            try:
                engine = RealtimeASR(bundle=self._bundle, cfg=DEFAULT_AUDIO)
                engine.on_status = lambda s: self._safe_set_status(s)
                engine.on_result = lambda r: self._safe_on_result(r)
                self._engine = engine
                engine.start()
            except Exception as e:  # noqa: BLE001
                self._safe_mode = True
                self._safe_set_status(f"Streaming gagal: {e} (Safe Mode: gunakan Record Once)")
                self.after(0, lambda: self.btn_start.configure(state="disabled"))
                self.after(0, lambda: self.btn_stop.configure(state="disabled"))
                self.after(0, lambda: self.btn_once.configure(state="normal"))
                self.after(0, lambda: self.progress.stop())

        threading.Thread(target=start_engine, daemon=True).start()

    def _stop(self) -> None:
        if self._engine is not None:
            self._engine.stop()
        self.btn_start.configure(state="normal" if not self._safe_mode else "disabled")
        self.btn_stop.configure(state="disabled")
        self.btn_once.configure(state="normal")
        self.progress.stop()
        self._set_mode("Single-shot")

    def _record_once(self) -> None:
        if self._bundle is None:
            self.status_var.set("Model belum siap")
            return

        # stop streaming to avoid mic conflicts
        if self._engine is not None and self._engine.is_running:
            self._stop()

        self._set_mode("Single-shot")
        self.btn_once.configure(state="disabled")
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="disabled")
        self.progress.start()

        duration = float(self._bundle.meta.duration_sec)
        sr = int(self._bundle.meta.sample_rate)

        def work() -> None:
            try:
                self._safe_set_status(f"Recording {duration:.1f}s...")
                rec = record_fixed_duration(duration_sec=duration, sample_rate=sr, channels=1)

                self._safe_set_status("Extracting MFCC...")
                res = predict_one(rec.audio, sample_rate=rec.sample_rate, bundle=self._bundle, cfg=DEFAULT_AUDIO)

                self._safe_set_status("Predicting...")
                self._safe_on_result(res)
                self._safe_set_status("Ready")
            except Exception as e:  # noqa: BLE001
                self._safe_set_status(f"Record Once error: {e}")
            finally:
                self.after(0, lambda: self.progress.stop())
                self.after(0, lambda: self.btn_once.configure(state="normal"))
                if not self._safe_mode:
                    self.after(0, lambda: self.btn_start.configure(state="normal"))

        threading.Thread(target=work, daemon=True).start()

    def _safe_set_status(self, s: str) -> None:
        self.after(0, lambda: self.status_var.set(s))

    def _safe_on_result(self, r: ASRResult) -> None:
        def update() -> None:
            self.pred_var.set(r.label)
            self.conf_var.set(f"{r.confidence * 100:.1f}%")

            # === Integrasi ASR → TTS textbox ===
            # Requirement:
            # - Dulu: hanya mengirim predicted_label ke textbox TTS.
            # - Sekarang: kirim info lengkap negara (info_text).
            # Catatan:
            # - Untuk menjaga demo tetap jelas, hanya kirim info jika label ada di COUNTRY_INFO.
            # - Jika tidak ada info, fallback ke label saja.
            if get_country_info(r.label) is not None:
                info_text = format_country_info_tts_text(r.label)
            else:
                info_text = r.label
            self._on_text_update(info_text)

            self._draw_plots(r.audio, r.mfcc)
            self._update_topk(r)
            self._update_country_info(r)
            self._maybe_auto_speak(r)

        self.after(0, update)

    def _update_country_info(self, r: ASRResult) -> None:
        """Update kotak info negara di GUI.

        Catatan:
        - Info tetap ditampilkan meskipun confidence rendah.
        - Tetapi TTS tetap mengikuti threshold (lihat _maybe_auto_speak).
        """

        if r.label in {"-", "", SILENCE_LABEL, UNKNOWN_LABEL}:
            text = "-"
        else:
            text = format_country_info_text(r.label)

        # Pastikan textbox selalu di-refresh dengan cara yang stabil.
        self.info_box.delete("1.0", "end")
        self.info_box.insert("1.0", text)

    def _update_topk(self, r: ASRResult) -> None:
        if self._bundle is None:
            return
        if r.label == SILENCE_LABEL:
            self.topk_box.delete("1.0", "end")
            self.topk_box.insert("1.0", "-")
            return
        top3 = top_k_predictions(self._bundle, r.proba, k=3)
        lines = [f"{i+1}. {lab} ({conf*100:.1f}%)" for i, (lab, conf) in enumerate(top3)]
        self.topk_box.delete("1.0", "end")
        self.topk_box.insert("1.0", "\n".join(lines))

    def _maybe_auto_speak(self, r: ASRResult) -> None:
        if not bool(self.auto_tts_var.get()):
            return
        thr = float(self.threshold_var.get())
        if r.confidence < thr:
            return
        if r.label == "-" or not r.label.strip():
            return
        if r.label in {SILENCE_LABEL, UNKNOWN_LABEL}:
            return

        # Requirement: jika prediksi bukan negara yang ada di COUNTRY_INFO,
        # jangan bacakan info (fallback ini mencegah demo menjadi membingungkan).
        if get_country_info(r.label) is None:
            return

        now = time.time()
        if now - self._last_spoken_time < self._speak_cooldown_sec:
            return
        if r.label == self._last_spoken_label:
            return

        self._last_spoken_label = r.label
        self._last_spoken_time = now

        # Speak in background (keep GUI responsive)
        self.status_var.set("Speaking...")
        speak_async(
            # Dibacakan: kalimat ringkas berisi lokasi + ibu kota + mata uang
            format_country_info_speech(r.label),
            TTSSettings(speed=TTSSpeed.normal),
            on_done=lambda: self.after(
                0,
                lambda: self.status_var.set(
                    "Listening..." if (self._engine is not None and self._engine.is_running) else "Ready"
                ),
            ),
        )

    def _draw_empty_plot(self) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_title("Waveform & MFCC")
        ax.text(0.5, 0.5, "Belum ada audio", ha="center", va="center")
        self.canvas.draw_idle()

    def _draw_plots(self, audio: np.ndarray, mfcc: np.ndarray) -> None:
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 2])

        # Downsample waveform for speed
        wav = np.asarray(audio, dtype=np.float32)
        if wav.size > 2500:
            idx = np.linspace(0, wav.size - 1, 2500).astype(int)
            wav = wav[idx]

        ax1 = self.fig.add_subplot(gs[0, 0])
        ax1.set_title("Waveform")
        ax1.plot(wav, linewidth=0.8)
        ax1.set_xlim(0, max(1, wav.size - 1))
        ax1.set_ylabel("Amp")
        ax1.grid(True, alpha=0.2)

        ax2 = self.fig.add_subplot(gs[1, 0])
        ax2.set_title("MFCC")
        im = ax2.imshow(mfcc, aspect="auto", origin="lower")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Koef")
        self.fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        self.canvas.draw_idle()
