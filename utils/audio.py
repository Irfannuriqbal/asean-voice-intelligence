from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class RecordingResult:
    audio: np.ndarray  # shape: (n_samples,)
    sample_rate: int
    duration_sec: float


def record_fixed_duration(duration_sec: float, sample_rate: int, channels: int = 1) -> RecordingResult:
    """Rekam audio durasi tetap. Paling stabil untuk dataset kata."""
    import sounddevice as sd

    n_samples = int(duration_sec * sample_rate)
    audio = sd.rec(frames=n_samples, samplerate=sample_rate, channels=channels, dtype="float32")
    sd.wait()
    audio = np.squeeze(audio)
    return RecordingResult(audio=audio, sample_rate=sample_rate, duration_sec=duration_sec)


def record_until_enter(sample_rate: int, channels: int = 1) -> RecordingResult:
    """Rekam sampai user menekan ENTER.

    Cocok untuk demo, tetapi untuk dataset biasanya lebih konsisten pakai durasi tetap.
    """

    import sounddevice as sd

    q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ANN001
        if status:
            # status bisa berisi underrun/overflow; biarkan saja untuk sekarang
            pass
        q.put(indata.copy())

    print("Recording... tekan ENTER untuk stop")
    start = time.time()
    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype="float32", callback=callback):
        input()  # blocking sampai ENTER

    chunks: list[np.ndarray] = []
    while not q.empty():
        chunks.append(q.get())

    if not chunks:
        audio = np.zeros((0,), dtype=np.float32)
    else:
        audio = np.concatenate(chunks, axis=0)
        audio = np.squeeze(audio)

    duration = max(0.0, time.time() - start)
    return RecordingResult(audio=audio, sample_rate=sample_rate, duration_sec=duration)


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


def load_wav_mono(path: Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if target_sr is not None and sr != target_sr:
        # librosa resample cukup reliable, tapi biar utils tetap ringan:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def normalize_audio(audio: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < eps:
        return audio
    return audio / peak


def pad_or_truncate(audio: np.ndarray, sample_rate: int, duration_sec: float) -> np.ndarray:
    target_len = int(duration_sec * sample_rate)
    if audio.size == target_len:
        return audio
    if audio.size > target_len:
        return audio[:target_len]
    # pad di akhir
    pad_len = target_len - audio.size
    return np.pad(audio, (0, pad_len), mode="constant")


def trim_silence(audio: np.ndarray, top_db: int = 25) -> np.ndarray:
    """Trim bagian hening di awal/akhir (berbasis amplitude)."""
    if audio.size == 0:
        return audio
    import librosa

    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed.astype(np.float32)


def preprocess_for_asr(
    audio: np.ndarray,
    sample_rate: int,
    target_sr: int,
    target_duration_sec: float,
    do_trim: bool = True,
) -> np.ndarray:
    """Pipeline preprocessing minimal untuk ASR kata terbatas."""
    audio = np.asarray(audio, dtype=np.float32)

    if sample_rate != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    if do_trim:
        audio = trim_silence(audio)

    audio = normalize_audio(audio)
    audio = pad_or_truncate(audio, sample_rate=sample_rate, duration_sec=target_duration_sec)
    return audio


def list_wav_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.wav"):
        if p.is_file():
            yield p
