from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running this file directly: `python train/train_mlp.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.config import DEFAULT_AUDIO, ProjectPaths
from utils.dataset import build_feature_matrix, scan_dataset
from utils.model_io import ASRMeta, save_meta, save_model


def main() -> None:
    root = ROOT
    paths = ProjectPaths(root=root)

    dataset_items = scan_dataset(paths.dataset_dir)
    if not dataset_items:
        raise SystemExit(
            "Dataset kosong. Rekam dulu dengan: python scripts/record_dataset.py --label Indonesia --count 10"
        )

    X, y = build_feature_matrix(
        dataset_items,
        cfg=DEFAULT_AUDIO,
        target_sr=DEFAULT_AUDIO.sample_rate,
        target_duration_sec=DEFAULT_AUDIO.duration_sec,
    )

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    (train_idx, test_idx) = next(splitter.split(X, y_enc))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=32,
                    learning_rate_init=1e-3,
                    max_iter=400,
                    random_state=42,
                    early_stopping=True,
                    n_iter_no_change=15,
                    verbose=True,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Full classification report (dict + text) for documentation
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

    print("\n=== Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(report_text)

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix (acc={acc:.3f})")
    plt.tight_layout()
    out_cm = paths.outputs_dir / "confusion_matrix.png"
    out_cm.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_cm, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix: {out_cm}")

    # Tambahan untuk skema artifact perbandingan penelitian (MLP vs CNN)
    # Simpan versi kedua dengan suffix _mlp, tanpa menghapus file lama.
    out_cm_mlp = paths.outputs_dir / "confusion_matrix_mlp.png"
    try:
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax2, xticks_rotation=45, cmap="Blues", colorbar=False)
        plt.title(f"Confusion Matrix MLP (acc={acc:.3f})")
        plt.tight_layout()
        plt.savefig(out_cm_mlp, dpi=150)
        plt.close(fig2)
    except Exception:  # noqa: BLE001
        pass

    # Confusion matrix analysis: top confusions (off-diagonal)
    supports = np.bincount(y_test, minlength=len(le.classes_)).astype(int)
    confusions: list[dict[str, object]] = []
    for true_i in range(cm.shape[0]):
        for pred_i in range(cm.shape[1]):
            if true_i == pred_i:
                continue
            c = int(cm[true_i, pred_i])
            if c <= 0:
                continue
            support_true = int(supports[true_i]) if true_i < supports.size else 0
            rate = (c / support_true) if support_true > 0 else 0.0
            confusions.append(
                {
                    "true": str(le.classes_[true_i]),
                    "pred": str(le.classes_[pred_i]),
                    "count": c,
                    "rate_within_true": float(rate),
                }
            )
    confusions.sort(key=lambda d: (d["count"], d["rate_within_true"]), reverse=True)
    top_confusions = confusions[:10]

    # Save report artifacts
    out_report_txt = paths.outputs_dir / "classification_report.txt"
    out_report_json = paths.outputs_dir / "classification_report.json"
    out_report_txt.write_text(report_text, encoding="utf-8")
    out_report_json.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    print(f"Saved classification report: {out_report_txt}")

    out_report_txt_mlp = paths.outputs_dir / "classification_report_mlp.txt"
    out_report_json_mlp = paths.outputs_dir / "classification_report_mlp.json"
    out_report_txt_mlp.write_text(report_text, encoding="utf-8")
    out_report_json_mlp.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    # Save model + metadata
    model_path = paths.models_dir / "asr_mlp.joblib"
    meta_path = paths.models_dir / "asr_meta.json"

    meta = ASRMeta(
        labels=list(le.classes_),
        audio_config=DEFAULT_AUDIO,
        sample_rate=DEFAULT_AUDIO.sample_rate,
        duration_sec=DEFAULT_AUDIO.duration_sec,
    )

    save_model({"pipeline": model, "label_encoder": le}, model_path)
    save_meta(meta, meta_path)

    # Save quick report json for presentation
    macro_avg = report_dict.get("macro avg", {}) if isinstance(report_dict, dict) else {}
    weighted_avg = report_dict.get("weighted avg", {}) if isinstance(report_dict, dict) else {}
    # Keep accuracy_score as source of truth; still mirror classification_report accuracy if present.
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
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
        "n_classes": int(len(le.classes_)),
        "labels": list(le.classes_),
        "model": {
            "type": "Pipeline(StandardScaler + MLPClassifier)",
            "mlp_hidden_layer_sizes": [128, 64],
            "mlp_activation": "relu",
            "mlp_alpha": 1e-4,
            "mlp_batch_size": 32,
            "mlp_learning_rate_init": 1e-3,
            "mlp_max_iter": 400,
            "mlp_early_stopping": True,
            "random_state": 42,
        },
        "artifacts": {
            "confusion_matrix_png": str(out_cm.as_posix()),
            "classification_report_txt": str(out_report_txt.as_posix()),
            "classification_report_json": str(out_report_json.as_posix()),
        },
        "confusion_analysis_top": top_confusions,
    }
    (paths.outputs_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Simpan juga versi _mlp.
    (paths.outputs_dir / "train_report_mlp.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model: {model_path}")
    print(f"Saved meta:  {meta_path}")


if __name__ == "__main__":
    main()
