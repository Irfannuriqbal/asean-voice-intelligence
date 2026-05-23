from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


# pyttsx3 (SAPI5) tidak stabil jika dipanggil paralel dari banyak thread.
# Untuk demo UAS, kita serialisasi aksesnya supaya tidak muncul error:
# "RuntimeError: run loop already started".
_PYTTSX3_LOCK = threading.Lock()


class TTSSpeed(str, Enum):
    slow = "slow"
    normal = "normal"
    fast = "fast"


class TTSGender(str, Enum):
    male = "Male"
    female = "Female"


@dataclass(frozen=True)
class TTSSettings:
    speed: TTSSpeed = TTSSpeed.normal
    gender: TTSGender = TTSGender.male


def _select_voice_id(voices: list, gender: TTSGender) -> str | None:
    """Pick a voice id based on gender heuristics (Windows SAPI5).

    Returns None if no reasonable match.
    """
    if not voices:
        return None

    def score(v) -> int:  # noqa: ANN001
        name = f"{getattr(v, 'name', '')} {getattr(v, 'id', '')}".lower()
        s = 0
        if gender == TTSGender.female:
            if "female" in name:
                s += 5
            if "zira" in name or "susan" in name or "hazel" in name:
                s += 4
        else:
            if "male" in name:
                s += 5
            if "david" in name or "mark" in name or "james" in name:
                s += 4
        # prefer microsoft voices slightly
        if "microsoft" in name:
            s += 1
        return s

    ranked = sorted(voices, key=score, reverse=True)
    best = ranked[0]
    return getattr(best, "id", None)


def _rate_for_speed(speed: TTSSpeed) -> int:
    # Nilai rate SAPI5 bervariasi antar device; ini cukup untuk demo
    if speed == TTSSpeed.slow:
        return 140
    if speed == TTSSpeed.fast:
        return 220
    return 180


def speak_pyttsx3(text: str, settings: TTSSettings) -> None:
    """TTS offline via pyttsx3 (Windows SAPI5).

    Catatan:
    - Bahasa/aksen tergantung voice yang tersedia di Windows.
    - Cocok untuk demo presentasi karena tidak butuh internet.
    """
    # Kunci global agar TTS tidak overlap.
    with _PYTTSX3_LOCK:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", _rate_for_speed(settings.speed))
        try:
            voices = list(engine.getProperty("voices") or [])
            vid = _select_voice_id(voices, settings.gender)
            if vid:
                engine.setProperty("voice", vid)
        except Exception:
            # Voice selection is best-effort; fallback to default voice
            pass
        engine.say(text)
        engine.runAndWait()


def save_wav_pyttsx3(text: str, out_path: Path, settings: TTSSettings) -> None:
    """Simpan output TTS ke file (umumnya .wav) memakai pyttsx3."""
    # Serialisasi juga untuk mode save (pakai pyttsx3 engine yang sama backend-nya)
    with _PYTTSX3_LOCK:
        import pyttsx3

        out_path.parent.mkdir(parents=True, exist_ok=True)
        engine = pyttsx3.init()
        engine.setProperty("rate", _rate_for_speed(settings.speed))
        try:
            voices = list(engine.getProperty("voices") or [])
            vid = _select_voice_id(voices, settings.gender)
            if vid:
                engine.setProperty("voice", vid)
        except Exception:
            pass
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()


def save_mp3_gtts(text: str, out_path: Path, settings: TTSSettings) -> None:
    """Simpan output TTS ke MP3 memakai gTTS (butuh internet).

    gTTS mendukung bahasa Indonesia lewat lang='id'.
    Speed:
    - slow: didukung (slow=True)
    - normal/fast: gTTS hanya punya slow/normal → fast akan diperlakukan normal
    """
    from gtts import gTTS

    out_path.parent.mkdir(parents=True, exist_ok=True)

    slow = settings.speed == TTSSpeed.slow
    tts = gTTS(text=text, lang="id", slow=slow)
    tts.save(str(out_path))


def speak_async(text: str, settings: TTSSettings, on_done: Callable[[], None] | None = None) -> None:
    """Jalankan speak di thread agar GUI tetap responsif."""

    def run() -> None:
        try:
            speak_pyttsx3(text, settings)
        finally:
            if on_done is not None:
                try:
                    on_done()
                except Exception:
                    pass

    threading.Thread(target=run, daemon=True).start()
