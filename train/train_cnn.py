from __future__ import annotations

"""Training CNN (Conv1D) untuk pembanding metode ASR.

Tujuan skrip ini:
- Menambah metode CNN TANPA merusak pipeline MLP yang sudah ada.
- Menggunakan dataset yang sama (dataset negara ASEAN 10 kelas) dan MFCC yang sama.
- Menyimpan artifact CNN ke file terpisah (suffix _cnn) agar mudah dibandingkan.

Catatan penting:
- CNN dilatih OFFLINE (eksperimen/penelitian), bukan untuk realtime inference di app.
- TensorFlow sering bergantung pada versi Python tertentu. Jika import TensorFlow gagal,
  ikuti instruksi yang tercetak di terminal.

Output yang dihasilkan:
- models/asr_cnn.h5
- outputs/classification_report_cnn.txt
- outputs/confusion_matrix_cnn.png
- outputs/train_report_cnn.json
- outputs/cnn_training_plot.png
"""

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Allow running this file directly: `python train/train_cnn.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.audio import load_wav_mono, preprocess_for_asr
from utils.config import DEFAULT_AUDIO, ProjectPaths
from utils.dataset import scan_dataset
from utils.features import extract_mfcc


def _build_mfcc_tensor(
    dataset_items,
    *,
    target_sr: int,
    target_duration_sec: float,
) -> tuple[np.ndarray, list[str]]:
    """Load dataset → preprocess → MFCC → tensor untuk CNN.

    CNN input (Conv1D):
    - Kita treat "time" sebagai n_frames
    - dan "features" sebagai n_mfcc

    Jadi shape per-sample: (n_frames, n_mfcc)
    Output X: (n_samples, n_frames, n_mfcc)

    Perhatian:
    - n_frames bisa berbeda 1-2 frame tergantung rounding.
      Karena itu, kita pakai padding/truncation agar semua sample punya panjang sama.
    """

    mfcc_list: list[np.ndarray] = []
    y_list: list[str] = []

    # Pertama, ekstrak semua MFCC
    for it in dataset_items:
        audio, sr = load_wav_mono(it.path, target_sr=None)
        audio = preprocess_for_asr(
            audio,
            sample_rate=sr,
            target_sr=target_sr,
            target_duration_sec=target_duration_sec,
            do_trim=True,
        )
        mfcc = extract_mfcc(audio, sample_rate=target_sr, cfg=DEFAULT_AUDIO)  # (n_mfcc, n_frames)
        mfcc_list.append(mfcc)
        y_list.append(it.label)

    # Tentukan n_frames target (median biar robust terhadap outlier)
    frame_counts = np.array([m.shape[1] for m in mfcc_list], dtype=int)
    n_frames_target = int(np.median(frame_counts))
    n_mfcc = int(mfcc_list[0].shape[0]) if mfcc_list else int(DEFAULT_AUDIO.n_mfcc)

    X_list: list[np.ndarray] = []
    for mfcc in mfcc_list:
        m = np.asarray(mfcc, dtype=np.float32)
        # Pad/crop di axis frame (axis=1)
        if m.shape[1] < n_frames_target:
            pad = n_frames_target - m.shape[1]
            m = np.pad(m, pad_width=((0, 0), (0, pad)), mode="constant", constant_values=0.0)
        elif m.shape[1] > n_frames_target:
            m = m[:, :n_frames_target]

        # Transpose untuk Conv1D: (n_frames, n_mfcc)
        X_list.append(m.T)

    X = np.stack(X_list, axis=0).astype(np.float32)

    # Safety check
    if X.ndim != 3 or X.shape[1] != n_frames_target or X.shape[2] != n_mfcc:
        raise RuntimeError(f"Bad X shape: {X.shape}")

    return X, y_list


def _plot_training(history: dict[str, list[float]], out_path: Path) -> None:
    """Simpan plot training accuracy/loss (jika tersedia) untuk laporan."""

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Accuracy
    ax = axes[0]
    if "accuracy" in history:
        ax.plot(history.get("accuracy", []), label="train")
    if "val_accuracy" in history:
        ax.plot(history.get("val_accuracy", []), label="val")
    ax.set_title("CNN Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()

    # Loss
    ax = axes[1]
    if "loss" in history:
        ax.plot(history.get("loss", []), label="train")
    if "val_loss" in history:
        ax.plot(history.get("val_loss", []), label="val")
    ax.set_title("CNN Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    # TensorFlow/Keras import di dalam main agar errornya lebih friendly.
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except Exception as e:  # noqa: BLE001
        print("\n[ERROR] TensorFlow/Keras belum siap di environment ini.")
        print(f"- Detail: {e}\n")
        print("Solusi umum (Windows):")
        print("1) Gunakan Python yang didukung TensorFlow (seringnya 3.10/3.11/3.12 tergantung versi TF)")
        print("2) Buat venv baru, lalu install:  pip install tensorflow")
        print("3) Jalankan lagi:             python train/train_cnn.py\n")
        raise SystemExit(1)

    root = ROOT
    paths = ProjectPaths(root=root)

    dataset_items = scan_dataset(paths.dataset_dir)
    if not dataset_items:
        raise SystemExit(
            "Dataset kosong. Rekam dulu dengan: python scripts/record_dataset.py --label Indonesia --count 10"
        )

    # Build MFCC tensor untuk CNN
    X, y = _build_mfcc_tensor(
        dataset_items,
        target_sr=DEFAULT_AUDIO.sample_rate,
        target_duration_sec=DEFAULT_AUDIO.duration_sec,
    )

    # Encode label string → integer
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split dataset (mirip train_mlp.py, tapi pakai train_test_split sesuai requirement)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc,
    )

    n_frames = int(X.shape[1])
    n_mfcc = int(X.shape[2])
    n_classes = int(len(le.classes_))

    # === Model CNN sederhana (Conv1D) ===
    # Input: (time_steps=n_frames, features=n_mfcc)
    model = keras.Sequential(
        [
            layers.Input(shape=(n_frames, n_mfcc)),
            layers.Conv1D(filters=32, kernel_size=5, activation="relu", padding="same"),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Training settings
    epochs = 35
    batch_size = 32

    # EarlyStopping untuk mencegah overfitting dan mempercepat eksperimen
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)
    ]

    print("\n=== CNN Training ===")
    print(f"Samples: {X.shape[0]} | Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Input shape: (frames={n_frames}, mfcc={n_mfcc}) | Classes: {n_classes}")

    hist = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    # === Evaluation ===
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        digits=4,
        zero_division=0,
        output_dict=False,
    )

    print("\n=== CNN Evaluation ===")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(report_text)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(n_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix CNN (acc={acc:.3f})")
    plt.tight_layout()

    out_cm = paths.outputs_dir / "confusion_matrix_cnn.png"
    out_cm.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_cm, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix: {out_cm}")

    # Save classification report
    out_report_txt = paths.outputs_dir / "classification_report_cnn.txt"
    out_report_json = paths.outputs_dir / "classification_report_cnn.json"
    out_report_txt.write_text(report_text, encoding="utf-8")
    out_report_json.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    print(f"Saved classification report: {out_report_txt}")

    # Save model
    out_model = paths.models_dir / "asr_cnn.h5"
    out_model.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save(out_model)
    except Exception as e:  # noqa: BLE001
        print("\n[ERROR] Gagal menyimpan model ke format .h5")
        print(f"- Detail: {e}")
        print("- Coba install dependency HDF5:  pip install h5py")
        print("- Atau gunakan environment/versi TensorFlow yang sesuai.\n")
        raise
    print(f"Saved CNN model: {out_model}")

    # Save training plot
    out_plot = paths.outputs_dir / "cnn_training_plot.png"
    _plot_training(hist.history, out_plot)
    print(f"Saved training plot: {out_plot}")

    # Save train report JSON
    macro_avg = report_dict.get("macro avg", {}) if isinstance(report_dict, dict) else {}
    weighted_avg = report_dict.get("weighted avg", {}) if isinstance(report_dict, dict) else {}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "CNN (Conv1D)",
        "accuracy": float(acc),
        "precision_macro": float(macro_avg.get("precision", 0.0)),
        "recall_macro": float(macro_avg.get("recall", 0.0)),
        "f1_macro": float(macro_avg.get("f1-score", 0.0)),
        "precision_weighted": float(weighted_avg.get("precision", 0.0)),
        "recall_weighted": float(weighted_avg.get("recall", 0.0)),
        "f1_weighted": float(weighted_avg.get("f1-score", 0.0)),
        "n_samples": int(X.shape[0]),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_classes": int(n_classes),
        "labels": list(map(str, le.classes_)),
        "audio_config": asdict(DEFAULT_AUDIO),
        "cnn": {
            "epochs_requested": int(epochs),
            "batch_size": int(batch_size),
            "optimizer": "Adam(lr=1e-3)",
            "loss": "sparse_categorical_crossentropy",
            "input_shape": [int(n_frames), int(n_mfcc)],
            "layers": [
                "Conv1D(32,k=5)",
                "MaxPool(2)",
                "Conv1D(64,k=3)",
                "MaxPool(2)",
                "Flatten",
                "Dense(128)",
                f"Dense({n_classes})",
            ],
        },
        "history": {k: [float(vv) for vv in v] for k, v in (hist.history or {}).items()},
        "artifacts": {
            "model_h5": str(out_model.as_posix()),
            "confusion_matrix_png": str(out_cm.as_posix()),
            "classification_report_txt": str(out_report_txt.as_posix()),
            "classification_report_json": str(out_report_json.as_posix()),
            "training_plot_png": str(out_plot.as_posix()),
        },
    }

    out_train_report = paths.outputs_dir / "train_report_cnn.json"
    out_train_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved train report: {out_train_report}")


if __name__ == "__main__":
    main()
