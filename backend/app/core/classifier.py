"""Trained defect classifier wrapping a scikit-learn model.

The detector (``detector.py``) extracts a 25-dim feature vector per frame; this
module trains a GradientBoosting classifier on those features and exposes a
tiny, dependency-light inference API used by the detector and the backend.

Artifacts:
  * models/classifier.joblib        -- trained pipeline (StandardScaler + clf)
  * models/classifier_meta.json     -- classes, feature names, metrics
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .detector import FEATURE_NAMES
from .paper_render import CLASSES


class DefectClassifier:
    """Load/save/predict wrapper around a sklearn Pipeline."""

    def __init__(self, pipeline: Pipeline | None = None, classes: list[str] | None = None) -> None:
        self.pipeline = pipeline
        self.classes = classes or list(CLASSES)
        # sklearn exposes classes_ after fit; keep ours in canonical order.
        self._classes_order = None

    @property
    def classes_(self) -> list[str]:
        if self.pipeline is not None and hasattr(self.pipeline, "classes_"):
            return list(self.pipeline.classes_)
        return list(self.classes)

    # -- training ------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DefectClassifier":
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.08,
                subsample=0.9, random_state=42)),
        ])
        self.pipeline.fit(X, y)
        self._classes_order = list(self.pipeline.classes_)
        return self

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.pipeline.predict(X)
        acc = float(accuracy_score(y, y_pred))
        # Pin the label set so classes absent from y still appear (support=0)
        # instead of raising a target_names size mismatch.
        labels = list(self.classes_)
        rep = classification_report(
            y, y_pred, labels=labels, target_names=labels, zero_division=0)
        return {"accuracy": acc, "report": rep, "n": int(len(y))}

    # -- inference -----------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X.reshape(1, -1) if X.ndim == 1 else X)

    def predict_one(self, feat: np.ndarray) -> tuple[str, float]:
        """Return (class_label, max_probability) for one feature vector."""
        proba = self.predict_proba(feat)[0]
        idx = int(np.argmax(proba))
        return self.classes_[idx], float(proba[idx])

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path, meta: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        meta_out = {
            "classes": self.classes_,
            "feature_names": FEATURE_NAMES,
            "feature_dim": len(FEATURE_NAMES),
            "model_type": "GradientBoostingClassifier",
            **(meta or {}),
        }
        (path.with_suffix(".meta.json")).write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DefectClassifier":
        path = Path(path)
        pipeline = joblib.load(path)
        meta_path = path.with_suffix(".meta.json")
        classes = list(CLASSES)
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                classes = meta.get("classes", classes)
            except Exception:
                pass
        return cls(pipeline=pipeline, classes=classes)
