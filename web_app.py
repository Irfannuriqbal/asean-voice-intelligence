"""web_app.py — Web version (Gradio) untuk demo UAS.

Fokus:
- UI dashboard modern (dark / neon cyan / glassmorphism)
- Realtime microphone panel (push-to-talk recording via komponen Audio)
- CNN inference: MFCC → pad/crop → reshape (n_frames, n_mfcc) → predict
- Country info panel (lokasi, ibu kota, mata uang, bahasa resmi)
- Text-to-Speech panel (gTTS jika ada internet, fallback pyttsx3 offline)
- MFCC heatmap visualization

Catatan penting:
- File ini TIDAK mengubah app.py desktop yang sudah ada.
- Model yang dipakai sesuai request: models/asr_cnn.h5

Cara menjalankan:
  1) aktifkan environment yang punya TensorFlow (disarankan `cnn-env` / Python 3.11)
      - PowerShell: ./cnn-env/Scripts/Activate.ps1
  2) install dependencies web
     - pip install -r requirements_web.txt
  3) run
     - python web_app.py
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib
import numpy as np

# Matplotlib untuk server (tanpa GUI)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils.audio import preprocess_for_asr
from utils.config import AudioConfig
from utils.country_info import format_country_info_speech, get_country_info
from utils.features import extract_mfcc
from utils.tts import TTSGender, TTSSettings, TTSSpeed, save_mp3_gtts, save_wav_pyttsx3


# === Paths ===
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "asr_cnn.h5"
TRAIN_REPORT_PATH = ROOT / "outputs" / "train_report_cnn.json"
META_PATH = ROOT / "models" / "asr_meta.json"  # fallback jika report tidak ada

ASSETS_DIR = ROOT / "assets"
WEB_OUT_DIR = ROOT / "outputs" / "web_demo"
WEB_TTS_DIR = WEB_OUT_DIR / "tts"


# === UI small constants ===
TITLE = "ASEAN Voice Intelligence"
SUBTITLE = "CNN-based ASEAN Speech Recognition & Country Analytics"


# === Official language mapping (untuk panel web) ===
# Dibuat eksplisit agar mudah dijelaskan saat presentasi.
OFFICIAL_LANGUAGE_ID: dict[str, str] = {
    "Indonesia": "Bahasa Indonesia",
    "Malaysia": "Bahasa Melayu",
    "Singapura": "Bahasa Inggris (utama), juga Melayu / Mandarin / Tamil",
    "Thailand": "Bahasa Thai",
    "Vietnam": "Bahasa Vietnam",
    "Laos": "Bahasa Lao",
    "Myanmar": "Bahasa Burma",
    "Filipina": "Filipino dan Bahasa Inggris",
    "Brunei": "Bahasa Melayu",
    "Kamboja": "Bahasa Khmer",
}


@dataclass(frozen=True)
class CnnAssets:
    model: Any
    labels: list[str]
    cfg: AudioConfig
    n_frames: int
    n_mfcc: int


_MODEL_LOCK = threading.Lock()
_ASSETS: CnnAssets | None = None


def _load_audio_config() -> AudioConfig:
    """Ambil AudioConfig dari train report atau meta."""
    if TRAIN_REPORT_PATH.exists():
        data = json.loads(TRAIN_REPORT_PATH.read_text(encoding="utf-8"))
        cfg = data.get("audio_config") or {}
        return AudioConfig(**cfg)

    if META_PATH.exists():
        data = json.loads(META_PATH.read_text(encoding="utf-8"))
        cfg = ((data.get("audio") or {}).get("config")) or {}
        # asr_meta.json di project ini punya struktur berbeda; fallback ke default jika gagal
        try:
            return AudioConfig(**cfg)
        except Exception:
            return AudioConfig()

    return AudioConfig()


def _load_labels_and_shape() -> tuple[list[str], int, int]:
    """Ambil label order & input shape CNN.

    Sumber utama: outputs/train_report_cnn.json
    """
    if not TRAIN_REPORT_PATH.exists():
        # Fallback aman (urutan sesuai dataset folder di repo)
        labels = [
            "Brunei",
            "Filipina",
            "Indonesia",
            "Kamboja",
            "Laos",
            "Malaysia",
            "Myanmar",
            "Singapura",
            "Thailand",
            "Vietnam",
        ]
        return labels, 121, 20

    data = json.loads(TRAIN_REPORT_PATH.read_text(encoding="utf-8"))
    labels = list(map(str, data.get("labels") or []))
    shape = data.get("cnn", {}).get("input_shape") or [121, 20]
    n_frames = int(shape[0])
    n_mfcc = int(shape[1])
    if not labels:
        raise RuntimeError("train_report_cnn.json ada, tapi labels kosong.")
    return labels, n_frames, n_mfcc


def load_assets() -> CnnAssets:
    """Load model CNN + metadata sekali (cache global)."""
    global _ASSETS
    if _ASSETS is not None:
        return _ASSETS

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model CNN tidak ditemukan: {MODEL_PATH}")

    labels, n_frames, n_mfcc = _load_labels_and_shape()
    cfg = _load_audio_config()

    # TensorFlow/Keras load
    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        raise RuntimeError(
            "Gagal load models/asr_cnn.h5. Pastikan TensorFlow + h5py terinstall.\n"
            "Saran: pip install tensorflow h5py"
        ) from e

    _ASSETS = CnnAssets(model=model, labels=labels, cfg=cfg, n_frames=n_frames, n_mfcc=n_mfcc)
    return _ASSETS


def _ensure_audio_tuple(audio: Any) -> tuple[int, np.ndarray] | None:
    """Normalisasi input Audio dari Gradio.

    Gradio bisa memberi:
    - None
    - (sample_rate, np.ndarray)
    - dict-like (tergantung versi)
    """
    if audio is None:
        return None

    # Most common: (sr, data)
    if isinstance(audio, (tuple, list)) and len(audio) == 2:
        sr = int(audio[0])
        data = np.asarray(audio[1], dtype=np.float32)
        return sr, data

    # Some versions may pass dict
    if isinstance(audio, dict) and "sample_rate" in audio and "data" in audio:
        sr = int(audio["sample_rate"])
        data = np.asarray(audio["data"], dtype=np.float32)
        return sr, data

    return None


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def _pad_or_crop_frames(mfcc: np.ndarray, n_frames_target: int) -> np.ndarray:
    """Pad/crop MFCC di axis frame (axis=1)."""
    m = np.asarray(mfcc, dtype=np.float32)
    if m.shape[1] < n_frames_target:
        pad = n_frames_target - m.shape[1]
        m = np.pad(m, pad_width=((0, 0), (0, pad)), mode="constant", constant_values=0.0)
    elif m.shape[1] > n_frames_target:
        m = m[:, :n_frames_target]
    return m


def _mfcc_heatmap_image(mfcc: np.ndarray) -> np.ndarray:
    """Render MFCC heatmap ke image (numpy RGB) untuk Gradio."""
    plt.close("all")
    fig = plt.figure(figsize=(8, 3.2), dpi=150)
    ax = fig.add_subplot(111)

    # Styling gelap agar match dashboard
    fig.patch.set_facecolor("#070B14")
    ax.set_facecolor("#070B14")

    im = ax.imshow(mfcc, aspect="auto", origin="lower", cmap="magma")
    ax.set_title("MFCC Heatmap", color="white", fontsize=12, pad=10)
    ax.set_xlabel("Time Frames", color="#cbd5e1")
    ax.set_ylabel("MFCC Coeff", color="#cbd5e1")
    ax.tick_params(colors="#94a3b8", labelsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")

    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=False)
    buf.seek(0)

    # Convert to numpy RGB
    import PIL.Image

    img = PIL.Image.open(buf).convert("RGB")
    return np.array(img)


def _render_prediction_md(label: str, conf: float, topk: list[tuple[str, float]]) -> str:
    conf_pct = int(round(conf * 100))

    topk_items = "".join(
        [
            (
                "<div class='topk-item'>"
                f"<div class='topk-rank'>#{i + 1}</div>"
                f"<div class='topk-label'>{lab}</div>"
                f"<div class='topk-score'>{p * 100:.1f}%</div>"
                "</div>"
            )
            for i, (lab, p) in enumerate(topk)
        ]
    )

    return (
        f"<div class='prediction-big'>{label}</div>"
        f"<div class='confidence-row'>"
        f"  <div class='confidence-big'><span class='conf-chip'>Confidence</span><b>{conf_pct}%</b></div>"
        f"  <div class='confidence-spark'></div>"
        f"</div>"
        f"<div class='conf-bar' style='--conf:{conf_pct}%;'><div></div></div>"
        f"<div class='topk-title'>Top Predictions</div>"
        f"<div class='topk-list'>{topk_items}</div>"
    )


def _render_country_md(label: str) -> str:
    info = get_country_info(label)
    if not info:
        return "<div style='color: rgba(255,255,255,0.72)'>-</div>"

    lang = OFFICIAL_LANGUAGE_ID.get(label, "-")

    return (
        "<div class='country-card'>"
        f"  <div class='country-title'>{label}</div>"
        "  <div class='country-grid'>"
        f"    <div class='kv'><div class='k'>🌏 Lokasi</div><div class='v'>{info['lokasi']}</div></div>"
        f"    <div class='kv'><div class='k'>🏛️ Ibu kota</div><div class='v'>{info['ibu_kota']}</div></div>"
        f"    <div class='kv'><div class='k'>💰 Mata uang</div><div class='v'>{info['mata_uang']}</div></div>"
        f"    <div class='kv'><div class='k'>🗣️ Bahasa resmi</div><div class='v'>{lang}</div></div>"
        "  </div>"
        "</div>"
    )


def _status_pill(text: str, pulse: bool = False) -> str:
    dot = "<span class='pulse'></span>" if pulse else "<span class='ai-dot'></span>"
    return f"<div class='status-pill'>{dot}<span>{text}</span></div>"


def predict_cnn(
    audio: Any,
    tts_speed: str,
    tts_gender: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, str, str, np.ndarray | None, str]:
    """Pipeline utama: Audio → preprocess → MFCC → CNN → UI outputs.

    Return:
      - status_html
      - prediction_html
      - country_html
      - mfcc_image
      - tts_text (untuk textbox)
    """

    parsed = _ensure_audio_tuple(audio)
    if parsed is None:
        return (
            _status_pill("Idle — klik mic untuk rekam", pulse=False),
            "<div style='color: rgba(255,255,255,0.72)'>-</div>",
            "<div style='color: rgba(255,255,255,0.72)'>-</div>",
            None,
            "",
        )

    sr_in, wav_in = parsed

    # Flatten ke mono
    wav = np.asarray(wav_in, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    if _rms(wav) < 0.0025:
        return (
            _status_pill("Suara terlalu pelan / hening", pulse=False),
            "<div class='prediction-big'>Suara hening</div>",
            "<div style='color: rgba(255,255,255,0.72)'>-</div>",
            None,
            "",
        )

    assets = load_assets()

    progress(0.15, desc="Preprocessing audio")
    x = preprocess_for_asr(
        wav,
        sample_rate=int(sr_in),
        target_sr=int(assets.cfg.sample_rate),
        target_duration_sec=float(assets.cfg.duration_sec),
        do_trim=True,
    )

    progress(0.40, desc="Extracting MFCC")
    mfcc = extract_mfcc(x, sample_rate=int(assets.cfg.sample_rate), cfg=assets.cfg)  # (n_mfcc, n_frames)
    mfcc_fixed = _pad_or_crop_frames(mfcc, assets.n_frames)

    # CNN input: (n_frames, n_mfcc)
    x_cnn = mfcc_fixed.T.astype(np.float32)
    x_cnn = np.expand_dims(x_cnn, axis=0)

    progress(0.65, desc="Running CNN inference")
    with _MODEL_LOCK:
        y = assets.model.predict(x_cnn, verbose=0)

    y = np.asarray(y).reshape(-1)
    if y.size != len(assets.labels):
        raise RuntimeError(f"Output model size {y.size} != labels {len(assets.labels)}")

    idx = int(np.argmax(y))
    conf = float(y[idx])
    label = assets.labels[idx]

    # Top-3
    topk_idxs = np.argsort(y)[::-1][:3]
    topk = [(assets.labels[int(i)], float(y[int(i)])) for i in topk_idxs]

    progress(0.85, desc="Rendering dashboard")
    pred_html = _render_prediction_md(label, conf, topk)
    country_html = _render_country_md(label)
    mfcc_img = _mfcc_heatmap_image(mfcc_fixed)

    # Text untuk TTS
    tts_text = format_country_info_speech(label) if get_country_info(label) else label

    status_html = _status_pill("AI Ready — prediksi selesai", pulse=False)
    progress(1.0, desc="Done")
    return status_html, pred_html, country_html, mfcc_img, tts_text


def generate_tts_audio(
    text: str,
    tts_speed: str = "normal",
    tts_gender: str = "male",
) -> str:
    """Generate file audio untuk diputar di web.

    Urutan:
    1) gTTS (lang=id) → MP3 (kualitas bagus, butuh internet)
    2) pyttsx3 → WAV (offline fallback)
    """

    text = (text or "").strip()
    if not text:
        raise ValueError("TTS text kosong")

    WEB_TTS_DIR.mkdir(parents=True, exist_ok=True)

    # Filename unik supaya tidak ketimpa
    ts = int(time.time() * 1000)

    speed = TTSSpeed(tts_speed) if tts_speed in {"slow", "normal", "fast"} else TTSSpeed.normal
    gender = TTSGender.female if str(tts_gender).lower().startswith("f") else TTSGender.male
    settings = TTSSettings(speed=speed, gender=gender)

    out_mp3 = WEB_TTS_DIR / f"tts_{ts}.mp3"
    try:
        save_mp3_gtts(text, out_mp3, settings=settings)
        return str(out_mp3)
    except Exception:
        out_wav = WEB_TTS_DIR / f"tts_{ts}.wav"
        save_wav_pyttsx3(text, out_wav, settings=settings)
        return str(out_wav)


def tts_button_handler(tts_text: str, tts_speed: str, tts_gender: str) -> tuple[str, str]:
    """Handler tombol Play Voice.

    Return:
      - status_html
      - audio_path (untuk gr.Audio output)
    """

    status = _status_pill("Generating voice...", pulse=True)
    audio_path = generate_tts_audio(tts_text, tts_speed=tts_speed, tts_gender=tts_gender)
    status = _status_pill("Voice ready — klik Play", pulse=False)
    return status, audio_path


def _placeholder_html(message: str) -> str:
    return f"<div class='placeholder-block'>{message}</div>"


def handle_audio_change(
    audio: Any,
    auto_analyze: bool,
    tts_speed: str,
    tts_gender: str,
) -> tuple[str, str, str, np.ndarray | None, str]:
    if audio is None:
        return (
            _status_pill("Idle — klik mic untuk rekam", pulse=False),
            _placeholder_html("Rekam suara, lalu klik Analyze atau aktifkan Auto Analyze."),
            _placeholder_html("Informasi negara akan ditampilkan setelah prediksi."),
            None,
            "",
        )

    if not auto_analyze:
        return (
            _status_pill("Siap untuk Analyze manual", pulse=False),
            _placeholder_html("Tekan Analyze untuk menjalankan prediksi."),
            _placeholder_html("Informasi negara akan ditampilkan setelah prediksi."),
            None,
            "",
        )

    return predict_cnn(audio, tts_speed, tts_gender)


def build_ui() -> gr.Blocks:
    css_path = ROOT / "style.css"
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

    # Gradio 6: CSS disarankan dipass lewat launch(css=...)
    with gr.Blocks(title=TITLE) as demo:
        # === HEADER ===
        gr.HTML(
            f"""
            <div class='ai-header'>
                <div class='ai-header-top'>
                    <div class='ai-branding'>
                        <div class='ai-icon-stack' aria-hidden='true'>
                            <div class='ai-icon'>ASEAN</div>
                        </div>
                        <div class='brand-copy'>
                            <div class='ai-headline'>ASEAN Speech Intelligence</div>
                            <div class='ai-title'>{TITLE}</div>
                            <div class='ai-subtitle'>{SUBTITLE}</div>
                        </div>
                    </div>
                    <button id='theme_toggle' class='theme-switcher' type='button' onclick="document.documentElement.classList.toggle('light-mode');">
                        Switch Light Mode
                    </button>
                </div>
                <div class='ai-badge'><span class='ai-dot'></span><span>ASEAN region • Country insight • MFCC analytics • TTS</span></div>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            # === LEFT COLUMN ===
            with gr.Column(scale=5, min_width=360):
                with gr.Group(elem_classes=["glass-card", "neon-outline"]):
                    gr.Markdown("### Realtime Microphone", elem_classes=["card-title"])
                    gr.Markdown(
                        "Push-to-talk: klik tombol mic → bicara 1 kata → stop → sistem akan auto predict jika Auto Analyze aktif. Atau gunakan **Analyze** secara manual.",
                        elem_classes=["card-subtitle"],
                    )

                    audio_in = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Microphone Input (waveform)",
                        elem_id="mic_audio",
                    )

                    with gr.Row(elem_classes=["action-row"]):
                        btn_analyze = gr.Button("Analyze", variant="primary", elem_id="btn_analyze")
                        auto_analyze = gr.Checkbox(value=True, label="Auto Analyze after record", elem_id="auto_analyze")

                    status_html = gr.HTML(
                        _status_pill("Idle — klik mic untuk rekam", pulse=False),
                        elem_id="status_pill",
                    )

                with gr.Group(elem_classes=["glass-card", "neon-outline"]):
                    gr.Markdown("### MFCC Visualization", elem_classes=["card-title"])
                    gr.Markdown("Heatmap MFCC (analytics view)", elem_classes=["card-subtitle"])
                    mfcc_img = gr.Image(type="numpy", label="MFCC Heatmap", elem_id="mfcc_img")

            # === RIGHT COLUMN ===
            with gr.Column(scale=6, min_width=420):
                with gr.Group(elem_classes=["glass-card", "neon-outline"]):
                    gr.Markdown("### AI Prediction", elem_classes=["card-title"])
                    prediction_html = gr.HTML(
                        "<div style='color: rgba(255,255,255,0.72)'>-</div>",
                        elem_id="prediction_panel",
                    )

                with gr.Group(elem_classes=["glass-card", "neon-outline"]):
                    gr.Markdown("### Country Information", elem_classes=["card-title"])
                    country_html = gr.HTML(
                        "<div style='color: rgba(255,255,255,0.72)'>-</div>",
                        elem_id="country_panel",
                    )

                with gr.Group(elem_classes=["glass-card", "neon-outline"]):
                    gr.Markdown("### Text-to-Speech", elem_classes=["card-title"])
                    gr.Markdown(
                        "Teks otomatis terisi dari hasil prediksi. Klik **Play Voice** untuk membuat audio.",
                        elem_classes=["card-subtitle"],
                    )

                    with gr.Row():
                        tts_speed = gr.Dropdown(
                            choices=["slow", "normal", "fast"],
                            value="normal",
                            label="Speed",
                        )
                        tts_gender = gr.Dropdown(
                            choices=["male", "female"],
                            value="male",
                            label="Voice",
                        )

                    tts_text = gr.Textbox(
                        label="TTS Text",
                        placeholder="(akan terisi otomatis setelah prediksi)",
                        lines=3,
                        elem_id="tts_text",
                    )

                    with gr.Row():
                        btn_tts = gr.Button("Play Voice", variant="primary", elem_id="btn_tts")
                        tts_audio = gr.Audio(label="Generated Voice", autoplay=False, elem_id="tts_audio")

        # === Events ===
        btn_analyze.click(
            fn=predict_cnn,
            inputs=[audio_in, tts_speed, tts_gender],
            outputs=[status_html, prediction_html, country_html, mfcc_img, tts_text],
        )

        audio_in.change(
            fn=handle_audio_change,
            inputs=[audio_in, auto_analyze, tts_speed, tts_gender],
            outputs=[status_html, prediction_html, country_html, mfcc_img, tts_text],
        )

        btn_tts.click(
            fn=tts_button_handler,
            inputs=[tts_text, tts_speed, tts_gender],
            outputs=[status_html, tts_audio],
        )

        # Small footer
        gr.Markdown(
            "<div style='margin-top:10px; color: rgba(255,255,255,0.55); font-size: 12px;'>"
            "Demo UAS • CNN (Conv1D) • MFCC analytics • Bahasa Indonesia"
            "</div>",
        )

    # Simpan css string sebagai attribute agar bisa dipakai di main()
    demo._custom_css = css  # type: ignore[attr-defined]
    return demo


def main() -> None:
    # Eager load supaya error dependency ketahuan di awal
    load_assets()

    demo = build_ui()
    demo.queue(default_concurrency_limit=2)
    css = getattr(demo, "_custom_css", "")
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, css=css)


if __name__ == "__main__":
    main()
