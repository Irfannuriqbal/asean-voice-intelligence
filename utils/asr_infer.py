from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd

from utils.audio import preprocess_for_asr
from utils.config import AudioConfig
from utils.features import extract_mfcc, mfcc_to_feature_vector
from utils.model_io import ASRMeta, load_meta, load_model


SILENCE_LABEL = "Suara hening"
UNKNOWN_LABEL = "Tidak dikenali"

# Tolerant thresholds for laptop microphones
MIN_RMS_SILENT = 0.003
MIN_RMS_SOFT = 0.005


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def _silence_result(bundle: ASRBundle, cfg: AudioConfig, audio: np.ndarray) -> ASRResult:
    # Use stable placeholder arrays so GUI plotting doesn't crash.
    classes = getattr(bundle.label_encoder, "classes_", None)
    try:
        n_classes = int(len(classes)) if classes is not None else 1
    except Exception:  # noqa: BLE001
        n_classes = 1
    n_classes = max(1, n_classes)
    proba = np.zeros((n_classes,), dtype=np.float32)
    mfcc = np.zeros((int(cfg.n_mfcc), 1), dtype=np.float32)
    wav = np.asarray(audio, dtype=np.float32).reshape(-1)
    return ASRResult(label=SILENCE_LABEL, confidence=0.0, proba=proba, mfcc=mfcc, audio=wav)


@dataclass(frozen=True)
class ASRBundle:
    pipeline: object
    label_encoder: object
    meta: ASRMeta


@dataclass(frozen=True)
class ASRResult:
    label: str
    confidence: float  # 0..1
    proba: np.ndarray  # shape (n_classes,)
    mfcc: np.ndarray  # shape (n_mfcc, n_frames)
    audio: np.ndarray  # waveform (preprocessed), shape (n_samples,)


def top_k_predictions(bundle: ASRBundle, proba: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    """Ambil top-k prediksi (label, confidence) dari probabilitas."""
    if proba.ndim != 1:
        proba = proba.reshape(-1)
    k = max(1, min(int(k), int(proba.size)))
    idxs = np.argsort(proba)[::-1][:k]
    labels = bundle.label_encoder.inverse_transform(idxs)
    return [(str(labels[i]), float(proba[int(idxs[i])])) for i in range(len(idxs))]


def load_asr_bundle(model_path: Path, meta_path: Path) -> ASRBundle:
    obj = load_model(model_path)
    if not isinstance(obj, dict) or "pipeline" not in obj or "label_encoder" not in obj:
        raise ValueError("Format model tidak sesuai. Jalankan train/train_mlp.py untuk membuat model.")

    meta = load_meta(meta_path)
    return ASRBundle(pipeline=obj["pipeline"], label_encoder=obj["label_encoder"], meta=meta)


def predict_one(
    audio: np.ndarray,
    sample_rate: int,
    bundle: ASRBundle,
    cfg: AudioConfig,
    *,
    do_trim: bool = True,
) -> ASRResult:
    """Prediksi satu potong audio (1 kata)."""
    # Gate using RAW audio RMS (more reliable than post-normalization)
    if _rms(audio) < MIN_RMS_SILENT:
        return _silence_result(bundle, cfg, audio)

    x = preprocess_for_asr(
        audio,
        sample_rate=sample_rate,
        target_sr=bundle.meta.sample_rate,
        target_duration_sec=bundle.meta.duration_sec,
        do_trim=do_trim,
    )
    # Secondary gate after preprocess
    if _rms(x) < MIN_RMS_SILENT:
        return _silence_result(bundle, cfg, x)

    mfcc = extract_mfcc(x, sample_rate=bundle.meta.sample_rate, cfg=cfg)
    feat = mfcc_to_feature_vector(mfcc)

    # pipeline sklearn mendukung predict_proba
    proba = bundle.pipeline.predict_proba(feat.reshape(1, -1))[0]
    idx = int(np.argmax(proba))
    label = str(bundle.label_encoder.inverse_transform([idx])[0])
    conf = float(proba[idx])
    return ASRResult(
        label=label,
        confidence=conf,
        proba=np.asarray(proba, dtype=np.float32),
        mfcc=mfcc,
        audio=x.astype(np.float32),
    )


class RealtimeASR:
    """Engine inferensi real-time dari mikrofon.

    Mekanisme:
    - Ambil audio stream dari mic
    - Simpan ke ring buffer
    - Tiap interval tertentu, ambil window terakhir (durasi target)
    - Ekstraksi MFCC dan predict_proba
    - Smoothing probabilitas agar stabil
    """

    def __init__(
        self,
        bundle: ASRBundle,
        cfg: AudioConfig,
        update_interval_sec: float = 0.35,
        smoothing_windows: int = 5,
        device: int | None = None,
    ) -> None:
        self._bundle = bundle
        self._cfg = cfg
        self._device = device

        self._sr = bundle.meta.sample_rate
        self._target_len = int(bundle.meta.sample_rate * bundle.meta.duration_sec)
        self._update_interval_sec = float(update_interval_sec)

        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: sd.InputStream | None = None

        self._buffer = np.zeros((0,), dtype=np.float32)
        self._proba_hist: deque[np.ndarray] = deque(maxlen=int(smoothing_windows))

        self._last_energy_status = 0.0
        self._last_silence_emit = 0.0

        self.on_result: Callable[[ASRResult], None] | None = None
        self.on_status: Callable[[str], None] | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return

        self._stop.clear()
        self._buffer = np.zeros((0,), dtype=np.float32)
        self._proba_hist.clear()

        def callback(indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ANN001
            if status:
                # underrun/overflow → tetap lanjut
                pass
            # indata shape (frames, channels)
            chunk = np.mean(indata, axis=1).astype(np.float32)
            self._q.put(chunk)

        try:
            self._stream = sd.InputStream(
                samplerate=self._sr,
                channels=1,
                dtype="float32",
                device=self._device,
                callback=callback,
                blocksize=0,
            )
            self._stream.start()
        except Exception as e:  # noqa: BLE001
            self._stream = None
            if self.on_status:
                self.on_status(f"Mic error: {e}")
            raise

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        if self.on_status:
            self.on_status("Listening...")

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
        if self.on_status:
            self.on_status("Stopped")

    def _worker(self) -> None:
        last_emit = 0.0
        while not self._stop.is_set():
            try:
                chunk = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            # Append ke buffer, simpan yang terakhir saja supaya buffer tidak membesar
            self._buffer = np.concatenate([self._buffer, chunk], axis=0)
            if self._buffer.size > self._target_len * 3:
                self._buffer = self._buffer[-self._target_len * 3 :]

            now = time.time()
            if now - last_emit < self._update_interval_sec:
                continue
            last_emit = now

            if self._buffer.size < self._target_len:
                continue

            window = self._buffer[-self._target_len :]

            # Energy feedback: if too quiet, show "Suara hening" and skip inference.
            w_rms = _rms(window)
            now2 = time.time()
            if w_rms < MIN_RMS_SILENT:
                # Clear history so next speech isn't biased by old proba
                self._proba_hist.clear()
                if self.on_status and (now2 - self._last_energy_status) > 0.8:
                    self._last_energy_status = now2
                    self.on_status(f"Listening... ({SILENCE_LABEL})")
                if self.on_result and (now2 - self._last_silence_emit) > 0.8:
                    # Emit a result occasionally so label updates in GUI
                    self._last_silence_emit = now2
                    self.on_result(_silence_result(self._bundle, self._cfg, window))
                continue
            if w_rms < MIN_RMS_SOFT:
                if self.on_status and (now2 - self._last_energy_status) > 0.8:
                    self._last_energy_status = now2
                    self.on_status("Listening... (suara kecil)")

            try:
                res = predict_one(window, sample_rate=self._sr, bundle=self._bundle, cfg=self._cfg, do_trim=False)
            except Exception as e:  # noqa: BLE001
                if self.on_status:
                    self.on_status(f"Infer error: {e}")
                continue

            # Smoothing probabilitas agar lebih stabil
            self._proba_hist.append(res.proba)
            proba_smoothed = np.mean(np.stack(list(self._proba_hist), axis=0), axis=0)
            idx = int(np.argmax(proba_smoothed))
            label = str(self._bundle.label_encoder.inverse_transform([idx])[0])
            conf = float(proba_smoothed[idx])
            res2 = ASRResult(
                label=label,
                confidence=conf,
                proba=proba_smoothed.astype(np.float32),
                mfcc=res.mfcc,
                audio=res.audio,
            )

            if self.on_result:
                self.on_result(res2)

        # cleanup
        self.stop()
