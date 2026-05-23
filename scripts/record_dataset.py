from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Allow running this file directly: `python scripts/record_dataset.py ...`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.audio import record_fixed_duration, save_wav


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rekam dataset kata (1 kata) ke dataset/<label>/*.wav")
    p.add_argument("--label", required=True, help="Nama label/folder, misal: Indonesia")
    p.add_argument("--count", type=int, default=10, help="Jumlah rekaman")
    p.add_argument("--duration", type=float, default=1.2, help="Durasi per rekaman (detik)")
    p.add_argument("--sr", type=int, default=16000, help="Sample rate")
    p.add_argument("--out", default="dataset", help="Folder output (default: dataset)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = ROOT

    dataset_root = (root / args.out).resolve()
    label_dir = dataset_root / args.label
    label_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output folder: {label_dir}")
    print("Tips: ucapkan 1 kata jelas, jarak mic konsisten.")

    for i in range(1, args.count + 1):
        input(f"[{i}/{args.count}] Tekan ENTER untuk mulai rekam...")
        rec = record_fixed_duration(duration_sec=args.duration, sample_rate=args.sr, channels=1)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{args.label}_{ts}_{i:03d}.wav"
        out_path = label_dir / fname
        save_wav(out_path, rec.audio, rec.sample_rate)

        print(f"Saved: {out_path}")

    print("Selesai. Lanjutkan rekam label lain.")


if __name__ == "__main__":
    main()
