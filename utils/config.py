from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def recordings_dir(self) -> Path:
        return self.root / "recordings"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"


ASEAN_COUNTRY_LABELS_ID = [
    "Indonesia",
    "Malaysia",
    "Singapura",
    "Thailand",
    "Vietnam",
    "Laos",
    "Myanmar",
    "Filipina",
    "Brunei",
    "Kamboja",
]


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    duration_sec: float = 1.2
    n_mfcc: int = 20
    n_fft: int = 512
    hop_length: int = 160  # ~10ms @16kHz
    win_length: int = 400  # ~25ms @16kHz
    fmin: int = 50
    fmax: int = 8000


DEFAULT_AUDIO = AudioConfig()
