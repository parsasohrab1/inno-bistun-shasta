# inno-bistun-shasta

**Bistun Paper Industries (صنایع کاغذ بیستون) — Integrated Quality & Energy Control
based on Machine Vision and APC.**

A TRL-5 system that combines **real-time machine-vision surface-defect
detection** with an **economic MPC Advanced Process Controller** on a paper
machine, so the line produces fewer defective tonnes at minimum specific energy.
The core idea:

> Every defective tonne that is not produced is energy, chemical and emissions
> that are not spent. Vision detects defects → APC re-optimises setpoints →
> defects fall → energy and CO₂ are saved.

- **Vision**: classical OpenCV detector (25-dim features, per-defect
  localisation, 6 classes) — fast, deterministic, CPU-only, no GPU.
- **ML**: Gradient Boosting classifier on the features (**95.4%** synthetic
  test accuracy).
- **Control**: economic linear MPC (LP via HiGHS) minimising controllable energy
  minus throughput value, subject to a defect-rate quality spec.
- **Real data**: NEU-DET surface-defect images (no-auth mirror) + Kaggle-ready
  downloader.
- **Synthetic data**: 1,200 labelled paper-web images, same renderer as the live
  simulator.
- **Backend**: FastAPI (REST + WebSocket live stream).
- **Dashboard**: self-contained HTML live monitoring (charts, defect feed, APC
  state, cumulative savings).
- **CI/CD**: GitHub Actions (lint + test on push).

---

## Repository structure

```
inno-bistun-shasta/
├── README.md                      # this file
├── SRS.md                         # Software Requirements Specification (IEEE 830)
├── BRS.md                         # Business Requirements + ROI / savings
├── docs/
│   └── TRL5_Maturity_Matrix.md    # TRL 5 evidence & maturity matrix
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app, serves dashboard, lifespan
│   │   ├── config.py              # paths + tunables
│   │   ├── api/{routes.py,ws.py}  # REST + WebSocket endpoints
│   │   └── core/
│   │       ├── paper_render.py    # paper texture + defect rendering primitives
│   │       ├── detector.py        # OpenCV defect detector + 25-dim features
│   │       ├── classifier.py      # scikit-learn classifier wrapper
│   │       ├── apc.py             # economic MPC (LP) controller
│   │       ├── simulator.py       # closed-loop paper-machine simulator
│   │       └── live.py            # WebSocket live engine (singleton)
│   ├── tests/                     # pytest (18 tests)
│   ├── requirements.txt
│   └── Dockerfile
├── dashboard/
│   └── index.html                 # self-contained live monitoring UI
├── scripts/
│   ├── download_real_data.py      # real NEU-DET (HF no-auth + Kaggle backend)
│   ├── generate_synthetic_data.py # synthetic paper-defect dataset
│   ├── train_classifier.py        # train + evaluate + save model
│   ├── smoke_test.py              # end-to-end pipeline smoke
│   └── diag_defects.py            # detector diagnostic
├── data/
│   ├── real/                      # 72 real NEU-DET images + manifest + SOURCE.md
│   └── synthetic/                 # 1,200 synthetic images + labels.csv + stats.json
├── models/                        # classifier.joblib + metrics.json
├── .github/workflows/             # CI
├── docker-compose.yml
└── conftest.py
```

---

## Quick start

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

# 2. (re)build the data + model  -- already committed, re-run to reproduce
python scripts/download_real_data.py --per-class 12
python scripts/generate_synthetic_data.py --per-class 200
python scripts/train_classifier.py          # -> models/classifier.joblib

# 3. run the backend + live dashboard
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000/   (live monitoring dashboard)
```

The dashboard auto-connects to the WebSocket live stream (`/ws/live`) and shows
the web camera with defect overlay, predicted class, quality score, the APC
setpoints/deltas/rationale, time-series charts and cumulative savings (CO₂,
energy, defective tonnes avoided). Buttons: Pause / Resume / Reset / New Run.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | service + model-load status |
| GET | `/api/classes` | defect classes (EN + FA) |
| GET | `/api/metrics` | trained-model metrics (accuracy, reports) |
| POST | `/api/detect` | `{image_b64, return_overlay}` → detection result |
| POST | `/api/detect/file` | multipart image → detection result |
| POST | `/api/apc` | `{speed, steam_pressure, chem_dosage, consistency, defect_rate, quality_score}` → control action + savings |
| GET | `/api/engine/state` | live simulator snapshot |
| POST | `/api/engine/control?action=reset` | pause / resume / reset / seed |
| WS | `/ws/live` | live paper-machine stream (frame + detection + APC + cumulative) |

Example:
```bash
curl -X POST http://localhost:8000/api/apc \
  -H 'Content-Type: application/json' \
  -d '{"speed":820,"steam_pressure":3.6,"chem_dosage":1.6,"consistency":4.5,"defect_rate":0.10,"quality_score":0.7}'
```

---

## Defect taxonomy

| Class | EN | FA | NEU-DET analog (real reference) |
|---|---|---|---|
| `clean` | Clean | سالم | — |
| `stain` | Stain | لکه | patches |
| `hole` | Hole | سوراخ | pitted_surface |
| `non_uniform` | Non-uniformity | نایکنواختی | crazing, rolled-in_scale |
| `scratch` | Scratch | خراش | scratches |
| `inclusion` | Inclusion | آخال | inclusion |

---

## How the loop works

1. The **simulator** renders a paper-web frame whose defect content is driven by
   the current setpoints (faster speed / lower steam → higher defect
   propensity) using the shared `paper_render` primitives.
2. The **detector** runs on the frame: brightness-normalises it, finds dark
   blobs (stains), sharp punctures (holes/inclusions), elongated components
   (scratches) and grid non-uniformity, extracts the 25-dim feature vector,
   scores quality (0–1) and (if the classifier is loaded) predicts the class.
3. The **APC controller** takes the measured defect rate + quality, solves the
   economic LP (minimise controllable energy − throughput value, subject to the
   defect-rate spec and actuator bounds), and returns new setpoints + savings.
4. The simulator applies the setpoints with first-order actuator lag and
   accumulates savings. Every connected dashboard receives the frame + overlay +
   detection + APC + cumulative stats over WebSocket.

---

## Results (TRL 5)

| Metric | Value |
|---|---|
| Classifier — synthetic held-out test accuracy | **95.4%** |
| Classifier — real NEU-DET (cross-material sanity) | ~38% (vs 17% random) |
| Clean-web quality score (synthetic) | 1.00 (0 false defects) |
| Detector inference time (256², CPU) | ~5 ms |
| APC LP solve time | ~1 ms |
| Automated tests | 18 / 18 pass |

**Indicative annual savings (conservative Scenario A, 6%→3% defect rate,
8,000 h/y):** ~4,320 defective tonnes avoided, ~8,338 GJ (≈30 MWh) energy
avoided, ~366 t CO₂ avoided; ≈ $2.66 M avoided value/year. See `BRS.md` for the
full derivation and placeholders to verify for the Bistun site.

> ⚠️ Energy tariffs, emission factors, throughput and the defect-rate target are
> **illustrative placeholders** pending confirmation from the Bistun process
> team. Replace the `ef_*` / `c_*` constants in `backend/app/core/apc.py` before
> any external saving claim.

---

## Docker

```bash
docker compose up --build      # backend on :8000, dashboard at /
# or
docker build -f backend/Dockerfile -t bistun-inno-shasta .
docker run -p 8000:8000 bistun-inno-shasta
```

---

## Testing & CI

```bash
pytest backend/tests -q          # 18 passed
```

GitHub Actions (`.github/workflows/ci.yml`) runs lint + tests on every push and
PR; `cd.yml` builds and publishes the Docker image on tagged releases.

---

## License & data provenance

- Code: project-internal (Bistun Paper Industries).
- Real reference data: NEU Surface Defect Database (NEU-DET), via a public
  no-auth Hugging Face mirror. Upstream NEU-DET declares **no explicit
  license** — treat as research/non-commercial. Verify before any commercial
  use. See `data/real/SOURCE.md`.
- Synthetic data: generated by this project's renderer (no third-party license).
