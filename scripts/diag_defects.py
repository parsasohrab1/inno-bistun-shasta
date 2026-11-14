"""Diagnostic: defect-kind breakdown on clean vs defective samples."""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from backend.app.core.classifier import DefectClassifier
from backend.app.core.detector import PaperDefectDetector
from backend.app.core.paper_render import CLASSES, render_sample

det = PaperDefectDetector()
det.attach_classifier(DefectClassifier.load(ROOT / "models" / "classifier.joblib"))

rng = np.random.default_rng(3)
for label in CLASSES:
    counts = Counter()
    qsum = 0.0
    for _ in range(10):
        img, _ = render_sample(label, 256, 256, rng=rng)
        r = det.detect(img)
        for d in r.defects:
            counts[d.kind] += 1
        qsum += r.quality_score
    print(f"{label:12s} avg_q={qsum/10:.3f}  defect-kinds(total over 10): {dict(counts)}")
