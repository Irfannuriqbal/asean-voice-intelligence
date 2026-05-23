from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib

from utils.config import AudioConfig


@dataclass(frozen=True)
class ASRMeta:
    labels: list[str]
    audio_config: AudioConfig
    sample_rate: int
    duration_sec: float


def save_model(model, model_path: Path) -> None:  # noqa: ANN001
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: Path):  # noqa: ANN001
    return joblib.load(model_path)


def save_meta(meta: ASRMeta, meta_path: Path) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": meta.labels,
        "sample_rate": meta.sample_rate,
        "duration_sec": meta.duration_sec,
        "audio_config": asdict(meta.audio_config),
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_meta(meta_path: Path) -> ASRMeta:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg = AudioConfig(**payload["audio_config"])
    return ASRMeta(
        labels=list(payload["labels"]),
        audio_config=cfg,
        sample_rate=int(payload["sample_rate"]),
        duration_sec=float(payload["duration_sec"]),
    )
