"""Train the defect classifier on synthetic (+ optional real) features.

Pipeline:
  1. Iterate data/synthetic/images/*/*.jpg (label = parent dir).
  2. Optionally iterate data/real/images/*.jpg using the NEU->paper mapping in
     data/real/manifest.csv (lower weight, used as a held-out sanity set by
     default -- pass --include-real-train to fold it into training).
  3. Extract the 25-dim feature vector with PaperDefectDetector for each image.
  4. Stratified train/test split, train DefectClassifier (GradientBoosting).
  5. Evaluate, save models/classifier.joblib + meta + metrics.json.

Usage:
  python scripts/train_classifier.py
  python scripts/train_classifier.py --per-class-limit 150 --include-real-train
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.classifier import DefectClassifier  # noqa: E402
from backend.app.core.detector import FEATURE_NAMES, PaperDefectDetector  # noqa: E402
from backend.app.core.paper_render import CLASSES  # noqa: E402

DATA_SYN = ROOT / "data" / "synthetic"
DATA_REAL = ROOT / "data" / "real"
MODEL_PATH = ROOT / "models" / "classifier.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def collect_synthetic(per_class_limit: int) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    base = DATA_SYN / "images"
    if not base.exists():
        return out
    for cls in CLASSES:
        d = base / cls
        if not d.exists():
            continue
        files = sorted(d.glob("*.jpg"))
        if per_class_limit:
            files = files[:per_class_limit]
        for f in files:
            out.append((f, cls))
    return out


def collect_real(include_train: bool) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    man = DATA_REAL / "manifest.csv"
    if not man.exists():
        return out
    for row in _read_csv(man):
        paper_cls = row.get("paper_class")
        if paper_cls not in CLASSES:
            continue
        p = ROOT / row["path"]
        if p.exists():
            out.append((p, paper_cls))
    return out


def extract_features(samples, detector: PaperDefectDetector) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, y, paths = [], [], []
    for path, label in samples:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        res = detector.detect(img)
        X.append(res.features)
        y.append(label)
        paths.append(str(path))
    if not X:
        raise SystemExit("[error] no samples -- generate synthetic data first")
    return np.array(X, dtype=np.float32), np.array(y), paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class-limit", type=int, default=0, help="cap synthetic per class (0 = all)")
    ap.add_argument("--include-real-train", action="store_true",
                    help="fold real NEU-DET images into training (default: eval only)")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    detector = PaperDefectDetector()

    syn = collect_synthetic(args.per_class_limit)
    real = collect_real(args.include_real_train)
    print(f"[info] synthetic samples: {len(syn)}   real samples: {len(real)}")

    train_samples = list(syn)
    if args.include_real_train:
        train_samples += real
        real_eval = []          # real is in train -> not a separate eval set
    else:
        real_eval = real

    X, y, paths = extract_features(train_samples, detector)
    print(f"[info] features extracted: X={X.shape}  classes={sorted(set(y))}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y)

    clf = DefectClassifier().fit(Xtr, ytr)
    train_metrics = clf.evaluate(Xtr, ytr)
    test_metrics = clf.evaluate(Xte, yte)
    print(f"[ok] train acc={train_metrics['accuracy']:.3f}  test acc={test_metrics['accuracy']:.3f}")
    print(test_metrics["report"])

    real_metrics: dict = {}
    if real_eval:
        Xr, yr, _ = extract_features(real_eval, detector)
        real_metrics = clf.evaluate(Xr, yr)
        print(f"[ok] REAL (held-out) acc={real_metrics['accuracy']:.3f}  n={real_metrics['n']}")

    clf.save(MODEL_PATH, meta={
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_size": args.test_size,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "real_accuracy": real_metrics.get("accuracy"),
        "real_n": real_metrics.get("n"),
    })

    metrics_out = {
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_report": test_metrics["report"],
        "real_accuracy": real_metrics.get("accuracy"),
        "real_report": real_metrics.get("report"),
        "feature_names": FEATURE_NAMES,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "classes": CLASSES,
    }
    METRICS_PATH.write_text(json.dumps(metrics_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] model -> {MODEL_PATH}\n         metrics -> {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
