# Business Requirements Specification (BRS)

**Project:** Integrated Quality & Energy Control based on Machine Vision and APC
**Customer:** Bistun Paper Industries (صنایع کاغذ بیستون)
**Document type:** Business case + requirements
**Version:** 1.0 — 2026-11-14

---

## 1. Business opportunity

Bistun Paper Industries operates a continuous paper web on which surface defects
— stains (لکه), holes (سوراخ), formation non-uniformity (نایکنواختی), scratches
(خراش) and inclusions (آخال) — cause off-spec tonnes that are re-pulped or
discarded. Today, defect detection is largely manual or post-hoc: by the time a
defect is seen, many metres of defective web have already consumed pulp stock,
steam for drying, electrical drive power and chemical additives — only to be
thrown away. Simultaneously, the drying section (the dominant energy consumer on
a paper machine) is frequently run with a safety margin ("over-drying") to hedge
against moisture-related defects, wasting steam.

This project delivers a single integrated system that (a) **detects surface
defects in real time with machine vision**, and (b) **re-optimises the process
setpoints with an Advanced Process Controller (MPC)** so that the line produces
fewer defective tonnes at minimum specific energy. The unifying business logic:

> **Every defective tonne that is not produced is energy, chemical and emissions
> that are not spent.** Preventing the defect is cheaper than re-pulping it.

## 2. Business objectives (measurable)

| ID | Objective | Target (TRL 5 → TRL 7) |
|---|---|---|
| BO-1 | Reduce the off-spec (defective) tonnage rate | from a 6% uncontrolled baseline to ≤ 3% under APC |
| BO-2 | Cut specific energy (GJ/t) | eliminate over-drying; ~1.0 GJ/h saved at steady state |
| BO-3 | Cut CO₂ emissions (kg/t) | ~46 kg/h avoided at steady state |
| BO-4 | Provide real-time operator visibility | live dashboard + WebSocket stream (delivered) |
| BO-5 | Reach TRL 5 | integrated validation in a relevant (simulated) environment (delivered) |

## 3. Stakeholders

- **Plant management** — owns the ROI / payback decision.
- **Process & quality engineers** — daily users of the dashboard; define the
  defect-rate target and alarm thresholds.
- **Automation / DCS engineers** — integrate `/api/apc` to the DCS (TRL 6+).
- **Sustainability / EHS** — reports the CO₂ savings.
- **IT / OT security** — approves the edge deployment and network path.

## 4. Quantified savings (computed from the APC model)

The figures below are computed directly from the calibrated economic MPC model
in `backend/app/core/apc.py` (specific energy **1.93 GJ/t**, CO₂ **84.7 kg/t** at
the nominal operating point; throughput **18 t/h**, **8,000 operating h/y**).
They are **illustrative pending site verification** of energy tariffs, emission
factors and throughput (see §7 placeholders) but the method is auditable and
reproducible.

### 4.1 Scenario A — steady-state (conservative)
Uncontrolled defect rate 6% → APC holds 3% target.

| Metric | Per hour | Per year (8,000 h) |
|---|---:|---:|
| Defective tonnes avoided | 0.54 t | **4,320 t** |
| Energy avoided | 1.04 GJ | **8,338 GJ ≈ 30 MWh** |
| CO₂ avoided | 45.8 kg | **366 t** |

### 4.2 Scenario B — off-spec correction (line drift)
Line drifts to 12% defective → APC corrects to 3%.

| Metric | Per hour | Per year (8,000 h) |
|---|---:|---:|
| Defective tonnes avoided | 1.62 t | **12,960 t** |
| Energy avoided | 3.02 GJ | **24,157 GJ ≈ 87 MWh** |
| CO₂ avoided | 131.6 kg | **1,053 t** |

## 5. Economic value & ROI (Scenario A, conservative)

Using illustrative unit values — paper **$600/t**, natural gas **$8/GJ**, CO₂
avoided value **$5/t**:

| Value stream | Annual |
|---|---:|
| Paper tonnage saved (4,320 t × $600) | **$2,592,000** |
| Energy saved (8,338 GJ × $8) | **$66,700** |
| CO₂ value (366 t × $5) | **$1,830** |
| **Total avoided value / year** | **≈ $2.66 M** |

The dominant value is **avoided defective paper** (~97% of the total), which is
the whole point of the product: the energy and emissions embedded in those
tonnes are saved as a consequence. Energy and CO₂ savings, while smaller in
dollar terms, are the headline sustainability KPIs and are themselves
measurable and reportable.

### 5.1 Indicative payback
A TRL 6–7 deployment on one Bistun line (edge box + camera + integration) is
budgeted in the low five-figure USD range. Against ≈ $2.6 M/y of avoided value,
**payback is well under one month** of operation at the Scenario A rate. Even at
a tenth of Scenario A (a 0.3 percentage-point defect-rate reduction), payback is
inside a year.

## 6. Business requirements

| ID | Requirement | Maps to |
|---|---|---|
| BR-1 | Detect and localise the five defect families on the live web. | SRS FR-1 |
| BR-2 | Classify the dominant defect and report a quality score. | SRS FR-2 |
| BR-3 | Automatically re-optimise speed/steam/chemical to hold defect rate ≤ target at minimum energy. | SRS FR-4 |
| BR-4 | Report avoided tonnes, energy and CO₂ in real time and cumulatively. | SRS FR-4, FR-7 |
| BR-5 | Give operators a live monitoring dashboard (bilingual EN/FA labels). | SRS FR-7 |
| BR-6 | Provide REST + WebSocket APIs for integration and a live stream. | SRS FR-6 |
| BR-7 | Ship with real reference data + synthetic data + reproducible training. | SRS FR-8 |
| BR-8 | Be deployable on CPU-only edge hardware (no GPU, no cloud dependency). | SRS NFR-1/4 |
| BR-9 | Reach TRL 5 with automated tests and CI gating. | SRS §5, NFR-5 |

## 7. Assumptions, placeholders & risks

- **PLACEHOLDER — energy tariffs & emission factors:** the `ef_*` and `c_*`
  constants in `apc.py` are illustrative (Iran grid, gas-heavy). They must be
  replaced with Bistun-site-verified values before any external saving claim.
- **PLACEHOLDER — throughput (18 t/h) and defect-rate target (3%):** pending
  confirmation from the Bistun process team.
- **Cross-domain accuracy:** the classifier is trained on synthetic paper; real
  NEU-DET steel images are used only as a structural sanity check (~38% top-1).
  Real paper-line image collection is a TRL 6 activity.
- **Risk — model drift:** lighting and stock changes shift the camera grey
  level; the detector applies mean normalisation for lighting invariance, but a
  periodic re-threshold/retrain cadence must be defined at deployment.
- **Risk — actuator authority:** APC is supervisory; final DCS authority and
  safety interlocks remain with the existing control system.

## 8. Success criteria (TRL 5)
1. The closed loop runs live: vision detects defects → APC retunes setpoints →
   defect rate falls → savings accumulate on the dashboard. ✅ (verified)
2. Classifier ≥ 90% on synthetic test set (measured 95.4%). ✅
3. Clean-web frames read quality ≥ 0.90 (measured 1.00). ✅
4. APC savings non-negative and monotonically cumulative. ✅
5. 18/18 automated tests pass in CI. ✅

## 9. Out of scope (TRL 6+)
Physical camera + PLC integration; DCS supervisory wiring (OPC UA); real
paper-line data collection and fine-tuning; multi-line deployment; alarm and
historian integration; formal safety review.
