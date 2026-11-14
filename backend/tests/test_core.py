"""Unit tests for the core ML + control modules."""
import numpy as np
import pytest

from backend.app.core.apc import APCController, ProcessState
from backend.app.core.classifier import DefectClassifier
from backend.app.core.detector import FEATURE_DIM, PaperDefectDetector
from backend.app.core.paper_render import CLASSES, render_sample
from backend.app.core.simulator import PaperMachineSimulator


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def detector():
    d = PaperDefectDetector()
    try:
        from backend.app.config import MODEL_PATH
        if MODEL_PATH.exists():
            d.attach_classifier(DefectClassifier.load(MODEL_PATH))
    except Exception:
        pass
    return d


# -- renderer ---------------------------------------------------------------
def test_render_sample_shape(rng):
    img, recs = render_sample("stain", 256, 256, rng=rng)
    assert img.shape == (256, 256)
    assert img.dtype == np.uint8
    assert len(recs) >= 1


def test_render_sample_all_classes(rng):
    for cls in CLASSES:
        img, _ = render_sample(cls, 128, 128, rng=rng)
        assert img.shape == (128, 128)


# -- detector ---------------------------------------------------------------
def test_detector_clean_is_clean(detector, rng):
    img, _ = render_sample("clean", 256, 256, rng=rng)
    res = detector.detect(img)
    assert res.quality_score >= 0.9
    assert len(res.defects) <= 2
    assert res.features.shape == (FEATURE_DIM,)


def test_detector_finds_stain(detector, rng):
    img, _ = render_sample("stain", 256, 256, rng=rng)
    res = detector.detect(img)
    kinds = {d.kind for d in res.defects}
    assert "stain" in kinds or res.pred_class == "stain"
    assert res.quality_score < 0.95


def test_detector_finds_hole(detector, rng):
    img, _ = render_sample("hole", 256, 256, rng=rng)
    res = detector.detect(img)
    assert any(d.kind in {"hole", "inclusion"} for d in res.defects) or res.pred_class == "hole"


def test_detector_feature_dim(detector, rng):
    img, _ = render_sample("scratch", 256, 256, rng=rng)
    res = detector.detect(img)
    assert res.features.shape == (FEATURE_DIM,)
    assert set(res.feature_dict.keys()) >= {"mean", "std", "defect_area_ratio"}


# -- APC --------------------------------------------------------------------
def test_apc_out_of_spec_reduces_defects():
    apc = APCController()
    st = ProcessState(speed=820, steam_pressure=3.6, chem_dosage=1.6,
                      consistency=4.5, defect_rate=0.12, quality_score=0.7)
    act = apc.step(st)
    assert act.defect_rate_reduction > 0
    assert act.tonnes_saved_per_h > 0
    assert act.predicted_defect_rate <= 0.031 or not act.in_spec


def test_apc_in_spec_pushes_throughput():
    # At nominal setpoints quality is slack (dr=k0=2% < 3% target). The
    # economic MPC maximises throughput, so it pushes speed up until quality
    # binds at the target -- not "no change".
    apc = APCController()
    st = ProcessState(speed=700, steam_pressure=4.5, chem_dosage=2.0,
                      consistency=4.5, defect_rate=0.02, quality_score=0.95)
    act = apc.step(st)
    assert act.in_spec
    assert act.predicted_defect_rate <= 0.031
    assert act.deltas["speed"] >= -1.0  # does not reduce throughput when in spec


def test_apc_savings_nonnegative():
    apc = APCController()
    st = ProcessState(speed=820, steam_pressure=3.6, chem_dosage=1.6,
                      consistency=4.5, defect_rate=0.12, quality_score=0.7)
    act = apc.step(st)
    assert act.energy_saved_gj_per_h >= 0
    assert act.co2_saved_kg_per_h >= 0


# -- simulator --------------------------------------------------------------
def test_simulator_tick_structure():
    sim = PaperMachineSimulator(seed=1)
    ev = sim.tick()
    for key in ("t", "frame_jpeg_b64", "overlay_jpeg_b64", "detection", "state", "apc", "cumulative"):
        assert key in ev
    assert ev["label_truth"] in CLASSES
    assert len(ev["frame_jpeg_b64"]) > 100
    assert ev["cumulative"]["frames"] == 1


def test_simulator_run_accumulates():
    sim = PaperMachineSimulator(seed=2)
    for _ in range(15):
        sim.tick()
    assert sim.cum.frames == 15
    assert sim.cum.co2_saved_kg >= 0
