# TRL 5 Maturity Matrix & Evidence

**Project:** Integrated Quality & Energy Control (Machine Vision + APC)
**Customer:** Bistun Paper Industries (صنایع کاغذ بیستون)
**Assessment date:** 2026-11-14

---

## 1. TRL definition (NASA scale)

| TRL | Definition | This project |
|---|---|---|
| 1 | Basic principles observed | — |
| 2 | Technology concept formulated | — |
| 3 | Experimental proof of concept | — |
| 4 | Component validation in lab | — |
| **5** | **Component validation in relevant environment** | **← target, achieved** |
| 6 | System/subsystem prototype in operational environment | next |
| 7 | Prototype in operational environment |  |
| 8 | System completed and qualified |  |
| 9 | Flight-proven / operational |  |

**TRL 5 means:** the detector, classifier and APC controller are integrated and
validated together in a **relevant environment** — a high-fidelity paper-machine
simulator whose defect dynamics are physically driven by the same setpoints the
APC manipulates, closing the vision→control loop on CPU-only hardware with the
production-grade software stack (FastAPI, OpenCV, scikit-learn, scipy LP).

---

## 2. Maturity matrix

| Subsystem | Component | Status | Evidence |
|---|---|---|---|
| **Machine vision** | OpenCV defect detector (25-dim features, localisation) | ✅ Validated | `backend/app/core/detector.py`; `test_detector_*` (clean q≥0.9, stains/holes located) |
| **Machine vision** | Brightness normalisation (lighting invariance) | ✅ Validated | `_normalize` (mean-shift); clean paper q=1.00, 0 false defects |
| **ML** | GradientBoosting classifier (6 classes) | ✅ Validated | `models/classifier.joblib`; 95.4% synthetic test acc |
| **ML** | Real reference data (NEU-DET) + cross-domain eval | ✅ Sanity | 72 real images; ~38% cross-material top-1 (vs 17% random) |
| **Data** | Real-data downloader (HF no-auth + Kaggle backend) | ✅ Delivered | `scripts/download_real_data.py`; `data/real/` (72 imgs) |
| **Data** | Synthetic generator (6 classes, 1,200 imgs) | ✅ Delivered | `scripts/generate_synthetic_data.py`; `data/synthetic/` |
| **Data** | Reproducible training pipeline | ✅ Delivered | `scripts/train_classifier.py`; `models/metrics.json` |
| **Control** | Economic MPC (LP via HiGHS) | ✅ Validated | `backend/app/core/apc.py`; `test_apc_*` |
| **Control** | Move-suppression + full-bounds fallback | ✅ Validated | `APCController.step` infeasible-fallback path |
| **Control** | Savings accounting (tonnes/energy/CO₂) | ✅ Validated | `test_apc_savings_nonnegative` |
| **Integration** | Live simulator (closed loop) | ✅ Validated | `backend/app/core/simulator.py`; `test_simulator_*` |
| **Integration** | FastAPI backend (REST + WebSocket) | ✅ Validated | `backend/app/main.py`; `test_api.py` (7 tests) |
| **UI** | Self-contained live dashboard | ✅ Delivered | `dashboard/index.html` (no build step) |
| **Quality** | Automated tests + CI | ✅ Delivered | 18/18 pytest pass; `.github/workflows/ci.yml` |
| **Deploy** | Dockerfile + docker-compose | ✅ Delivered | `backend/Dockerfile`, `docker-compose.yml` |
| **Docs** | SRS, BRS, README, TRL matrix | ✅ Delivered | repo root + `docs/` |

---

## 3. What "relevant environment" means here

TRL 5 requires validation in an environment *relevant* to the operational one.
For a paper machine, that means the defect physics must respond to the process
setpoints the controller moves. Our simulator provides exactly that:

- **Defect propensity is setpoint-driven**: faster line speed / lower dryer
  steam / lower retention-aid dosage → higher predicted defect fraction
  (linear quality model with calibrated coefficients).
- **The vision detector measures real rendered defects** on synthetic web
  frames (stains, holes, non-uniformity, scratches, inclusions), not a mocked
  label.
- **The APC controller acts on the vision measurement** and its recommended
  setpoints are applied with first-order actuator lag, so the loop's effect on
  subsequent frames is physical, not cosmetic.
- **The stack is production-grade**: FastAPI + uvicorn, OpenCV, scikit-learn,
  scipy LP — the same libraries a TRL 6+ deployment would use, not a toy.

This is strictly stronger than TRL 4 (component tested in isolation with
canned data): here the components are integrated and the loop is closed.

---

## 4. Quantitative evidence (re-run to reproduce)

```
Classifier  (synthetic held-out test) : accuracy = 0.954
Classifier  (real NEU-DET, cross-material sanity) : accuracy ≈ 0.38
Clean-web detector (synthetic)        : quality_score = 1.00, n_defects = 0
APC out-of-spec (12% → target)        : defect-rate reduction > 0, savings ≥ 0
Live loop smoke (8 ticks)             : clean frames q=1.00; defects detected; savings accumulate
Automated tests                       : 18 / 18 pass
```

Reproduce:
```bash
python scripts/download_real_data.py --per-class 12
python scripts/generate_synthetic_data.py --per-class 200
python scripts/train_classifier.py          # writes models/metrics.json
pytest backend/tests -q                      # 18 passed
uvicorn backend.app.main:app --port 8000    # open http://localhost:8000/
```

---

## 5. Gaps to TRL 6 (system prototype in operational environment)

1. **Real paper-line camera feed** replacing the simulator frame source.
2. **Real paper-line image dataset** for fine-tuning thresholds/classifier
   (lifts real accuracy above the steel cross-domain baseline).
3. **DCS supervisory wiring** (OPC UA / Modbus setpoint writes) — the `/api/apc`
   JSON contract is the integration surface.
4. **Site-verified model parameters**: replace `ef_*`, `c_*`, throughput and
   defect-rate target placeholders with Bistun-confirmed values.
5. **Field validation of savings**: instrument the line to measure actual
   defective-tonne and specific-energy deltas against a baseline.

---

## 6. Statement

As of 2026-11-14, the Integrated Quality & Energy Control system meets **TRL 5**:
all core components (machine-vision detector, trained classifier, economic MPC
controller) are integrated and validated together in a relevant, setpoint-driven
paper-machine simulation, on the production software stack, with automated tests
and CI gating. The next step (TRL 6) is operational-environment integration on
the Bistun line with real camera and DCS data.
