# Software Requirements Specification (SRS)

**Project:** Integrated Quality & Energy Control based on Machine Vision and APC
**Customer:** Bistun Paper Industries (صنایع کاغذ بیستون)
**Repository:** `inno-bistun-shasta`
**Document standard:** IEEE 830-1998 structure
**Maturity target:** TRL 5 (component validation in a relevant environment)
**Version:** 1.0 — 2026-11-14

---

## 1. Introduction

### 1.1 Purpose
This SRS specifies the functional, interface, performance and quality
requirements for a software system that combines **machine-vision surface-defect
detection** with **Advanced Process Control (APC)** on a paper-machine web. The
system is delivered at **Technology Readiness Level 5** — the integrated
detector, classifier and controller are validated together in a relevant
(high-fidelity simulated paper-line) environment.

The core thesis of the product: **every defective tonne that is not produced is
energy and emissions that are not spent.** By detecting surface defects
(stains, holes, non-uniformity, scratches, inclusions) in real time and
re-optimising process setpoints (line speed, dryer steam, retention-aid dosage)
through an MPC controller, the system prevents the production of off-spec paper
that would otherwise be re-pulped or discarded, directly cutting specific energy
(GJ/t) and CO₂ (kg/t).

### 1.2 Scope
The system covers, end-to-end:

1. **Machine vision** — a classical computer-vision detector that localises and
   classifies surface defects on the paper web from camera frames, producing a
   25-dimensional feature vector, per-defect bounding boxes, a quality score
   (0–1) and a defect-area ratio.
2. **Classification** — a trained scikit-learn (Gradient Boosting) classifier on
   the extracted features, assigning one of six defect classes.
3. **Advanced Process Control** — an economic linear MPC that, given the
   measured defect rate and quality from vision, computes energy-optimal
   setpoints subject to a quality (defect-rate) specification, and quantifies
   the resulting energy, CO₂ and defective-tonne savings.
4. **Live process simulator** — a calibrated paper-machine simulator that
   physically links setpoints to defect propensity, closing the vision→APC loop
   and providing the TRL-5 relevant environment.
5. **Backend API** — a FastAPI service exposing synchronous inference and APC
   endpoints plus a WebSocket live stream.
6. **Live monitoring dashboard** — a self-contained web UI streaming the live
   web camera, defect overlay, APC state and cumulative savings.

Out of scope for TRL 5: physical camera/PLC integration on the real Bistun line,
operator DCS wiring, and field tuning on real paper images (these are TRL 6–7
activities, flagged in §6 and the TRL maturity matrix).

### 1.3 Definitions and acronyms
| Term | Meaning |
|---|---|
| APC | Advanced Process Control — multivariable supervisory control above the base DCS |
| MPC | Model Predictive Control — optimisation-based control over a receding horizon |
| MV | Machine Vision — automated visual inspection |
| TRL | Technology Readiness Level (1–9, NASA scale) |
| Defect class | One of: clean, stain, hole, non_uniform, scratch, inclusion |
| Quality score | Scalar 0–1 (1 = defect-free web), computed from localised defect area + severity |
| Defect-area ratio | Fraction of web pixels covered by localised defects |
| Specific energy | Energy per tonne of paper (GJ/t) |
| `dr` | Predicted/measured defect fraction (0–1) |
| LP | Linear Program (solved with HiGHS via `scipy.optimize.linprog`) |

### 1.4 References
- IEEE 830-1998, *Recommended Practice for Software Requirements Specifications*.
- Song & Yan, "A noise robust method based on completed local binary patterns
  for hot-rolled steel strip surface defects," *Ironmaking & Steelmaking* (NEU
  surface-defect dataset, used as real reference data — see §3.4).
- Bistun Paper Industries internal process context (kraft/printing-paper line).

### 1.5 Overview
§2 describes the product in context. §3 gives the specific functional and
non-functional requirements. §4 covers interfaces. §5 states verification. §6
lists TRL-6+ evolution and assumptions.

---

## 2. Overall description

### 2.1 Product perspective
The system is a standalone supervisory application that sits above the paper
machine's existing DCS/SCADA. It ingests a camera stream of the moving web,
runs vision inference, and emits recommended setpoint changes (and savings
metrics) to the operator and, in a future TRL 6+ step, to the DCS in
supervisory setpoint mode. At TRL 5 the camera feed is produced by a
high-fidelity simulator that stands in for the real line.

```
  +-----------+    frame    +----------------+   features   +----------------+
  | Camera /  |----------->| Machine-Vision |------------->|   Classifier   |
  | Simulator |            |   Detector     |              |  (GradientBoost)|
  +-----------+            +----------------+              +----------------+
                                                                   | class, q, dr
                                                                   v
                                   +----------------+   setpoints  +----------------+
                                   |  Operator /    |<------------|   APC / MPC    |
                                   |  Dashboard     |  savings    |   Controller   |
                                   +----------------+             +----------------+
```

### 2.2 Product functions
1. **Acquire** a grayscale web frame (256×256 typical).
2. **Detect & localise** defects with bounding boxes, severity and kind.
3. **Classify** the dominant defect class with a confidence.
4. **Score** the web quality (0–1) and defect-area ratio.
5. **Control**: solve the economic MPC for energy-optimal setpoints that hold
   the predicted defect rate ≤ target; report deltas, predicted energy/CO₂, and
   avoided defective tonnes / energy / CO₂.
6. **Stream** all of the above live to the dashboard over WebSocket.
7. **Expose** synchronous REST endpoints for batch inference and standalone APC.

### 2.3 User characteristics
- **Process / quality engineer**: uses the dashboard to monitor live quality and
  APC actions, and the REST API to batch-evaluate offline image sets.
- **Automation engineer**: integrates the `/api/detect` and `/api/apc`
  endpoints with the DCS in TRL 6+.
- **Plant management**: reads the cumulative savings KPIs and the BRS ROI.

### 2.4 Constraints
- CPU-only deployment (no GPU required); deterministic, auditable CV pipeline.
- No deep-learning dependency in the control path (MPC is linear, solved with
  LP — appropriate for QA sign-off).
- Single-board/edge-box deployable; the dashboard is a single self-contained
  HTML file (no build step, no external runtime dependencies).

### 2.5 Assumptions and dependencies
- The web camera provides a top-lit, grayscale view of the moving web at
  ≥2 fps, 256×256 or larger.
- Stock-consistency (a disturbance) is available from the stock-prep DCS tag.
- The defect-rate target (default 3%) is settable by the engineer.

---

## 3. Specific requirements

### 3.1 Functional requirements

#### FR-1 — Defect detection and localisation
The detector SHALL ingest a grayscale or BGR frame and return:
- a list of localised defects, each with `kind ∈ {stain, hole, non_uniform,
  scratch, inclusion}`, bounding box `(x, y, w, h)`, severity (0–1) and area;
- a 25-dimensional feature vector (schema fixed in `FEATURE_NAMES`);
- a `quality_score` (0–1, 1 = defect-free) and `defect_area_ratio` (0–1).

On a defect-free synthetic web, `quality_score ≥ 0.90` and `n_defects ≤ 2`
(verified — see §5).

#### FR-2 — Defect classification
A trained classifier SHALL map the 25-dim feature vector to one of six classes
(`clean, stain, hole, non_uniform, scratch, inclusion`) with a confidence
(max softmax-equivalent probability). On a held-out synthetic test set the
classifier SHALL achieve ≥ 90% top-1 accuracy (measured: **95.4%**).

#### FR-3 — Cross-domain reference (real data)
The detector SHALL also run on real surface-defect reference images (NEU-DET
steel-strip, mapped to the paper taxonomy). This is a *cross-material sanity*
check, not a production accuracy target: the classifier trained only on
synthetic paper achieves **~38% top-1** on steel (vs 17% random for 6 classes),
demonstrating partial structural transfer; full class accuracy on real paper
requires paper-line data collection (TRL 6+).

#### FR-4 — Advanced Process Control (economic MPC)
Given a `ProcessState` (speed, steam_pressure, chem_dosage, consistency,
defect_rate, quality_score), the controller SHALL solve an LP that:
- minimises `c_steam·steam + c_chem·chem − v_thr·speed` (controllable energy
  minus throughput value);
- subject to the linear quality constraint `dr(x) ≤ dr_target`;
- subject to actuator bounds and (soft) per-step move limits, with a
  full-bounds fallback when the move-limited LP is infeasible.

It SHALL return recommended setpoints, deltas, predicted defect rate / energy
(GJ/t) / CO₂ (kg/t), and the headline savings: **defective tonnes avoided/h,
energy avoided/h, CO₂ avoided/h** (always ≥ 0, derived from the defect-rate
reduction × throughput).

When the current defect rate exceeds the target, the controller SHALL reduce
the predicted defect rate to ≤ target (or report `in_spec=false` if actuator
limits prevent it).

#### FR-5 — Live process simulator
A simulator SHALL render web frames whose defect content is driven by the
current setpoints (faster speed / lower steam → higher defect propensity),
run the detector and APC on each frame, apply the recommended setpoints with a
first-order actuator lag, and accumulate cumulative savings. One `tick()`
SHALL return a JSON-serialisable event containing the frame (JPEG base64),
overlay, detection, state, APC action and cumulative stats.

#### FR-6 — Backend API
The backend SHALL expose:
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | service + model-load status |
| GET | `/api/classes` | defect classes with EN/FA labels |
| GET | `/api/metrics` | trained-model metrics |
| POST | `/api/detect` | base64 image → detection (+ optional overlay) |
| POST | `/api/detect/file` | multipart image → detection |
| POST | `/api/apc` | process state → control action + savings |
| GET | `/api/engine/state` | live simulator snapshot |
| POST | `/api/engine/control` | pause / resume / reset / seed |
| WS | `/ws/live` | live paper-machine stream |

#### FR-7 — Live dashboard
The dashboard SHALL connect to `/ws/live`, render the overlay frame, show the
predicted class/confidence/quality, the APC setpoints + deltas + rationale, and
cumulative savings (CO₂, energy, tonnes avoided). It SHALL provide
pause/resume/reset/new-run controls and time-series charts (quality, defect
rate, CO₂-saved rate) plus a defect-event feed and class distribution.

#### FR-8 — Data assets
The repository SHALL include:
- a **real-data downloader** (`scripts/download_real_data.py`) supporting a
  no-auth Hugging Face mirror (default) and a Kaggle backend (with credentials),
  writing `data/real/images/` + `manifest.csv` + `SOURCE.md`;
- a **synthetic-data generator** (`scripts/generate_synthetic_data.py`) producing
  ≥ 1200 labelled images across the six classes with `labels.csv` + `stats.json`;
- a **training script** (`scripts/train_classifier.py`) producing
  `models/classifier.joblib` + `models/metrics.json`.

### 3.2 Non-functional requirements
| ID | Requirement |
|---|---|
| NFR-1 Performance | Detector inference ≤ 30 ms / 256² frame on a single CPU core; MPC LP ≤ 5 ms. |
| NFR-2 Determinism | The CV pipeline is parameter-deterministic; the same frame yields the same result. |
| NFR-3 Availability | The live engine auto-restarts its loop on exception and stops when no clients are connected. |
| NFR-4 Portability | Runs on Linux x86_64 with Python 3.10+; no GPU. Dockerfile provided. |
| NFR-5 Maintainability | ≥ 80% of public modules covered by pytest; CI gates on tests. |
| NFR-6 Auditability | Control path is a linear LP (no neural net); model + metrics are versioned artifacts. |
| NFR-7 Security | No credentials in the repo; the GitHub PAT is never logged; Kaggle creds read from env only. |
| NFR-8 Observability | Every live event is self-describing JSON; `/api/engine/state` exposes a snapshot. |

### 3.3 Performance targets (TRL 5 relevant environment)
- Live stream cadence: 3 fps (configurable).
- Simulated process time: 1 s per tick.
- Classifier test accuracy: ≥ 90% (measured 95.4%).
- Clean-web quality score: ≥ 0.90 (measured 1.00 on synthetic clean).

---

## 4. Interfaces

### 4.1 External interfaces
- **Camera / image input**: JPEG/PNG, grayscale or BGR, base64 or multipart.
- **Operator UI**: HTML5 + WebSocket (self-contained `dashboard/index.html`).
- **Future DCS**: supervisory setpoint writes (TRL 6+) via OPC UA / Modbus —
  not implemented at TRL 5 but the `/api/apc` JSON contract is the integration
  surface.

### 4.2 Internal interfaces (Python)
- `PaperDefectDetector.detect(img) -> DetectionResult`
- `DefectClassifier.predict_one(feat) -> (class, confidence)`
- `APCController.step(ProcessState) -> ControlAction`
- `PaperMachineSimulator.tick() -> dict`
- `LiveEngine` (singleton): `add_client`, `remove_client`, `control`

---

## 5. Verification (TRL 5 evidence)

| Requirement | Verification | Result |
|---|---|---|
| FR-1 clean-web quality | pytest `test_detector_clean_is_clean` | q ≥ 0.9, n_def ≤ 2 ✅ |
| FR-1 defect localisation | pytest `test_detector_finds_stain/hole` | defects located ✅ |
| FR-2 classifier accuracy | `models/metrics.json` | 95.4% test acc ✅ |
| FR-3 cross-domain real | held-out NEU-DET eval | ~38% top-1 (sanity) ✅ |
| FR-4 APC reduces defects | pytest `test_apc_out_of_spec_reduces_defects` | dr reduction > 0 ✅ |
| FR-4 APC savings ≥ 0 | pytest `test_apc_savings_nonnegative` | energy/CO₂/h ≥ 0 ✅ |
| FR-5 simulator structure | pytest `test_simulator_tick_structure` | all keys present ✅ |
| FR-6 API | pytest `test_api.py` (7 tests) | all endpoints 200 ✅ |
| NFR-1 performance | smoke timing | ~5 ms detect, ~1 ms LP ✅ |
| NFR-5 tests | `pytest backend/tests` | 18/18 pass ✅ |

Full evidence and the maturity matrix: `docs/TRL5_Maturity_Matrix.md`.

---

## 6. TRL 6+ evolution and assumptions
1. Collect real paper-line images from the Bistun camera; fine-tune thresholds
   and retrain (lifts real-data accuracy above the cross-domain steel baseline).
2. Wire `/api/apc` outputs to the DCS in supervisory setpoint mode (OPC UA).
3. Add a camera-acquisition worker replacing the simulator frame source.
4. Extend the MPC to a receding-horizon formulation with measured moisture and
   a throughput/economic objective calibrated to Bistun's actual energy tariffs.
5. Add alarm / batch-report generation and historical KPI storage.

### 6.1 Assumptions requiring user confirmation (placeholders)
- **Energy tariffs and emission factors** for the Bistun site are currently
  illustrative defaults (Iran grid gas-heavy). Replace `ef_*` and `c_*` in
  `backend/app/core/apc.py` DEFAULTS with site-verified values before any
  production saving claim.
- **Throughput** (default 18 t/h) and **defect-rate target** (default 3%) are
  placeholders pending confirmation from the Bistun process team.
