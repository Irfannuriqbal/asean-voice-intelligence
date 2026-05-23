from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.asr_infer import RealtimeASR, load_asr_bundle
from utils.config import DEFAULT_AUDIO, ProjectPaths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realtime ASR CLI (mic → prediksi + confidence)")
    p.add_argument("--model", default="models/asr_mlp.joblib")
    p.add_argument("--meta", default="models/asr_meta.json")
    p.add_argument("--interval", type=float, default=0.35, help="Update interval (detik)")
    p.add_argument("--smooth", type=int, default=5, help="Jumlah window smoothing")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    paths = ProjectPaths(root=ROOT)
    model_path = (ROOT / args.model).resolve()
    meta_path = (ROOT / args.meta).resolve()

    if not model_path.exists() or not meta_path.exists():
        raise SystemExit("Model belum ada. Jalankan: .\\.venv\\Scripts\\python.exe train\\train_mlp.py")

    bundle = load_asr_bundle(model_path, meta_path)
    engine = RealtimeASR(bundle=bundle, cfg=DEFAULT_AUDIO, update_interval_sec=args.interval, smoothing_windows=args.smooth)

    def on_status(s: str) -> None:
        print(f"[status] {s}")

    def on_result(r) -> None:  # noqa: ANN001
        print(f"{r.label:12s} | conf={r.confidence * 100:5.1f}%")

    engine.on_status = on_status
    engine.on_result = on_result

    engine.start()
    print("Running... tekan Ctrl+C untuk stop")

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
