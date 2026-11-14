"""Bistun paper-defect synthetic dataset generator.

Generates a labelled synthetic dataset of grayscale paper-web images for the
six defect classes in ``backend.app.core.paper_render.CLASSES``:

    clean, stain, hole, non_uniform, scratch, inclusion

The same rendering primitives power the live process simulator, so the
classifier trained here sees the same data distribution the dashboard streams
in real time -- a key TRL-5 fidelity argument.

Outputs (under data/synthetic/):
  images/<class>/<idx>.jpg            -- JPEG q88, 256x256 grayscale
  labels.csv                          -- path,label,severity_sum,n_defects,seed
  stats.json                          -- per-class counts + totals
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# allow running both as `python scripts/generate_synthetic_data.py` (repo root)
# and from elsewhere -- import the backend package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.paper_render import CLASSES, LABELS, render_sample  # noqa: E402

OUT_DIR = ROOT / "data" / "synthetic"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=200,
                    help="images per class (default 200 -> 1200 total)")
    ap.add_argument("--size", type=int, default=256, help="image side (px)")
    ap.add_argument("--quality", type=int, default=88, help="JPEG quality")
    ap.add_argument("--seed", type=int, default=20261114, help="master RNG seed")
    args = ap.parse_args()

    master = np.random.default_rng(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "images").mkdir(exist_ok=True)

    rows: list[dict] = []
    counts: dict[str, int] = {}
    for cls in CLASSES:
        cls_dir = OUT_DIR / "images" / cls
        cls_dir.mkdir(exist_ok=True)
        counts[cls] = 0
        for i in range(args.per_class):
            seed = int(master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed)
            img, recs = render_sample(cls, h=args.size, w=args.size, rng=rng)
            rel = f"images/{cls}/{i:04d}.jpg"
            cv2.imwrite(str(OUT_DIR / rel), img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            sev_sum = round(sum(r.severity for r in recs), 4)
            rows.append({
                "path": rel, "label": cls,
                "severity_sum": sev_sum, "n_defects": len(recs), "seed": seed,
            })
            counts[cls] += 1
        print(f"[ok] {cls:12s} -> {counts[cls]} images")

    # labels.csv
    with (OUT_DIR / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "severity_sum", "n_defects", "seed"])
        w.writeheader()
        w.writerows(rows)

    stats = {
        "per_class": counts,
        "total": sum(counts.values()),
        "size_px": args.size,
        "jpeg_quality": args.quality,
        "seed": args.seed,
        "classes": CLASSES,
        "labels_en_fa": {k: v for k, v in LABELS.items()},
        "source": "synthetic (procedural paper-web renderer, see backend/app/core/paper_render.py)",
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {stats['total']} images, labels.csv + stats.json under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
