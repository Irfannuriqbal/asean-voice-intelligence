from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.audio import list_wav_files, load_wav_mono, preprocess_for_asr
from utils.config import AudioConfig
from utils.features import extract_mfcc, mfcc_to_feature_vector


@dataclass(frozen=True)
class DatasetItem:
    path: Path
    label: str


def scan_dataset(dataset_dir: Path) -> list[DatasetItem]:
    """Scan dataset dengan asumsi folder per label: dataset/<label>/*.wav"""
    items: list[DatasetItem] = []
    for wav_path in list_wav_files(dataset_dir):
        if wav_path.parent == dataset_dir:
            # file wav langsung di root dataset → skip
            continue
        label = wav_path.parent.name
        items.append(DatasetItem(path=wav_path, label=label))
    return sorted(items, key=lambda x: (x.label, x.path.name))


def build_feature_matrix(
    items: list[DatasetItem],
    cfg: AudioConfig,
    target_sr: int,
    target_duration_sec: float,
) -> tuple[np.ndarray, list[str]]:
    """Load+preprocess audio → MFCC → feature vector.

    Output:
        X: shape (n_samples, 2*n_mfcc)
        y: list label string
    """
    X_list: list[np.ndarray] = []
    y_list: list[str] = []

    for it in items:
        audio, sr = load_wav_mono(it.path, target_sr=None)
        audio = preprocess_for_asr(
            audio,
            sample_rate=sr,
            target_sr=target_sr,
            target_duration_sec=target_duration_sec,
            do_trim=True,
        )
        mfcc = extract_mfcc(audio, sample_rate=target_sr, cfg=cfg)
        feat = mfcc_to_feature_vector(mfcc)
        X_list.append(feat)
        y_list.append(it.label)

    X = np.stack(X_list, axis=0).astype(np.float32)
    return X, y_list
