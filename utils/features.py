from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from utils.config import AudioConfig


def extract_mfcc(audio: np.ndarray, sample_rate: int, cfg: AudioConfig) -> np.ndarray:
    """Ekstraksi MFCC.

    Return:
        mfcc: shape (n_mfcc, n_frames)
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    # Normalisasi per koefisien (opsional tapi membantu stabilitas)
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
    return mfcc.astype(np.float32)


def mfcc_to_feature_vector(mfcc: np.ndarray) -> np.ndarray:
    """Ubah MFCC 2D menjadi 1D untuk model MLP.

    Teknik sederhana dan mudah dijelaskan:
    - mean + std per koefisien MFCC
    - tambah delta & delta-delta (mean + std) untuk menangkap dinamika ucapan
    """
    mfcc = np.asarray(mfcc, dtype=np.float32)

    def stats(m: np.ndarray) -> np.ndarray:
        mean = np.mean(m, axis=1)
        std = np.std(m, axis=1)
        return np.concatenate([mean, std], axis=0)

    # Base MFCC stats
    base = stats(mfcc)

    # Temporal dynamics (still MFCC-based)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    feat = np.concatenate([base, stats(d1), stats(d2)], axis=0)
    return feat.astype(np.float32)


def plot_mfcc(mfcc: np.ndarray, out_path: Path, title: str = "MFCC") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 3))
    librosa.display.specshow(mfcc, x_axis="time")
    plt.colorbar(format="%+2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_feature_config(cfg: AudioConfig, out_path: Path) -> None:
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
