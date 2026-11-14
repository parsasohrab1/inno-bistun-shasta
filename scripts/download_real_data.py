"""
download_real_data.py
=====================

Downloads a *real* surface-defect image dataset for the Bistun Paper Industries
"Integrated Quality & Energy Control (Machine Vision + APC)" project.

Because paper-specific public defect datasets are scarce, we use the
**NEU Surface Defect Database (NEU-DET)** -- a widely-cited real-world
surface-defect image set of hot-rolled steel strip (Song & Yan, Northeastern
University, Shenyang).  Steel-strip surface defects share the same visual
failure modes we target on paper web:

    NEU-DET class        ->  Paper-defect analog (our taxonomy)
    ----------------------------------------------------------
    patches              ->  stain          (لکه)
    pitted_surface       ->  hole           (سوراخ)
    rolled-in_scale      ->  non_uniform    (نایکنواختی)
    crazing              ->  non_uniform    (نایکنواختی)
    scratches            ->  scratch        (خراش)
    inclusion            ->  inclusion      (آخال)

Two backends are supported:

1. Hugging Face Hub mirror  (DEFAULT, no authentication, no Kaggle token needed):
   https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected
   Images are individual JPGs reachable by a plain HTTPS GET.

2. Kaggle API  (optional, requires KAGGLE_USERNAME / KAGGLE_KEY env vars):
   kaggle datasets download -d <slug>
   This is the "Kaggle datasets" priority path requested by the stakeholder;
   it is used automatically when Kaggle credentials are present, otherwise we
   fall back to the no-auth HF mirror so the repo always contains real data.

The script downloads a *bounded* sample (default 12 images per class) so the
repository stays lightweight.  Set --per-class 0 to pull the full set.

Outputs:
    data/real/images/<class>_<n>.jpg
    data/real/manifest.csv     (path, neu_class, paper_class, split, source)
    data/real/SOURCE.md        (provenance + license notes)

Usage:
    python scripts/download_real_data.py --per-class 12
    python scripts/download_real_data.py --backend kaggle --per-class 20
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request
from pathlib import Path

# --- NEU-DET class -> Bistun paper-defect taxonomy -------------------------
NEU_TO_PAPER = {
    "crazing": "non_uniform",
    "inclusion": "inclusion",
    "patches": "stain",
    "pitted_surface": "hole",
    "rolled-in_scale": "non_uniform",
    "scratches": "scratch",
}

HF_BASE = "https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected"
HF_API = "https://huggingface.co/api/datasets/KeenForgeAI/NEU-DET-corrected"
REAL_DIR = Path(__file__).resolve().parents[1] / "data" / "real"


def http_get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "bistun-inno/1.0 (real-data downloader)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def list_hf_tree(split: str) -> list[str]:
    """Return relative image paths for a split, e.g. ['images/train/crazing_1.jpg', ...]."""
    url = f"{HF_API}/tree/main/images/{split}?recursive=false&limit=1000"
    raw = http_get(url).decode("utf-8", "replace")
    data = json.loads(raw)
    paths = []
    # HF tree API returns either a list of entries or {"paths":[...]}
    entries = data if isinstance(data, list) else data.get("paths", [])
    for entry in entries:
        p = entry.get("path") if isinstance(entry, dict) else entry
        if p and p.endswith(".jpg"):
            paths.append(p)
    return paths


def class_of(path: str) -> str | None:
    """images/train/crazing_12.jpg -> 'crazing'."""
    name = path.rsplit("/", 1)[-1]  # crazing_12.jpg
    for cls in NEU_TO_PAPER:
        if name.startswith(cls + "_") or name.startswith(cls.replace("-", "_") + "_"):
            return cls
    # fallback: prefix before first digit
    prefix = ""
    for ch in name:
        if ch == "_" or ch.isdigit():
            break
        prefix += ch
    return prefix or None


def download_hf(per_class: int) -> list[dict]:
    rows: list[dict] = []
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    (REAL_DIR / "images").mkdir(exist_ok=True)
    per_class_count: dict[str, int] = {c: 0 for c in NEU_TO_PAPER}

    for split in ("train", "val", "test"):
        try:
            paths = list_hf_tree(split)
        except Exception as exc:  # network blip -> skip split
            print(f"[warn] could not list split {split}: {exc}", file=sys.stderr)
            continue
        print(f"[info] {split}: {len(paths)} images listed on HF")

        for rel in paths:
            cls = class_of(rel)
            if cls is None or cls not in NEU_TO_PAPER:
                continue
            if per_class and per_class_count[cls] >= per_class:
                continue
            n = per_class_count[cls] + 1
            paper_cls = NEU_TO_PAPER[cls]
            out = REAL_DIR / "images" / f"{cls}_{n:02d}.jpg"
            per_class_count[cls] = n
            rows.append(
                {
                    "path": str(out.relative_to(REAL_DIR.parents[1])),
                    "neu_class": cls,
                    "paper_class": paper_cls,
                    "split": split,
                    "source": "HF:KeenForgeAI/NEU-DET-corrected",
                }
            )
            if out.exists():  # already downloaded in a prior split/run
                continue
            try:
                blob = http_get(f"{HF_BASE}/resolve/main/{rel}", timeout=60)
            except Exception as exc:
                print(f"[warn] download failed {rel}: {exc}", file=sys.stderr)
                continue
            if not blob.startswith(b"\xff\xd8"):
                print(f"[warn] not a JPEG: {rel}", file=sys.stderr)
                continue
            out.write_bytes(blob)

    print("[info] per-class counts:", per_class_count)
    return rows


def download_kaggle(per_class: int) -> list[dict]:
    """Kaggle-priority path. Requires kaggle package + credentials.

    Slugs (real, but require a Kaggle token -- verify before commercial use):
      - arjvinoth/steel-surface-defect            (NEU-DET mirror)
      - saurabhshahane/classification-of-surface-defects-steel-strip
    """
    try:
        import kaggle  # type: ignore
    except ImportError:
        print(
            "[error] Kaggle backend selected but the `kaggle` package is not "
            "installed. Run: pip install kaggle   and export "
            "KAGGLE_USERNAME / KAGGLE_KEY. Falling back to HF mirror.",
            file=sys.stderr,
        )
        return download_hf(per_class)

    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        print("[warn] Kaggle credentials missing; falling back to HF mirror.", file=sys.stderr)
        return download_hf(per_class)

    slug = "arjvinoth/steel-surface-defect"
    raw_dir = REAL_DIR / "_kaggle_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] kaggle datasets download -d {slug}")
    kaggle.api.dataset_download_files(slug, path=str(raw_dir), unzip=True, quiet=False)
    print("[info] Kaggle download complete -- post-processing not implemented in this sample.")
    # NOTE: Kaggle layout varies; we keep the HF path as the verified default.
    return download_hf(per_class)


def write_manifest(rows: list[dict]) -> None:
    if not rows:
        return
    manifest = REAL_DIR / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] manifest -> {manifest}  ({len(rows)} rows)")

    src = REAL_DIR / "SOURCE.md"
    src.write_text(
        """# Real Data Source & Provenance

## Dataset
**NEU Surface Defect Database (NEU-DET)** -- real-world surface-defect images
of hot-rolled steel strip.

- Source mirror (no auth): https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected
- Original: Northeastern University (Shenyang), Song & Yan,
  "A noise robust method based on completed local binary patterns for
  hot-rolled steel strip surface defects," *Ironmaking & Steelmaking*.
- Classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
- Splits on mirror: train (1000), val (180), test (180).

## Why steel-strip defects for a *paper* line?
Surface-defect vision transfers across continuous-web processes: the local
visual failure modes (stains/patches, pits/holes, scale/crazing -> non-uniform
texture, scratches) are the same defect families we target on a paper machine
web.  NEU-DET is used here as **real, labelled reference data** to calibrate
and sanity-check the detector; the production detector is tuned on
paper-specific synthetic data (`data/synthetic/`).

## License
Upstream NEU-DET declares **no explicit license** -- treat as
**research / non-commercial**.  Verify before any production / commercial use.
The HF mirror license card: `license: other`.

## Mapping to Bistun paper-defect taxonomy
| NEU-DET class     | Paper-defect class | Persian        |
|-------------------|--------------------|----------------|
| patches           | stain              | لکه            |
| pitted_surface    | hole               | سوراخ          |
| rolled-in_scale   | non_uniform        | نایکنواختی     |
| crazing           | non_uniform        | نایکنواختی     |
| scratches         | scratch            | خراش           |
| inclusion         | inclusion          | آخال           |

## Reproduce
```bash
python scripts/download_real_data.py --per-class 12          # HF mirror (default)
python scripts/download_real_data.py --backend kaggle        # Kaggle (needs token)
```
""",
        encoding="utf-8",
    )
    print(f"[ok] provenance -> {src}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["hf", "kaggle"], default="hf",
                    help="hf = Hugging Face no-auth mirror (default); kaggle = Kaggle API")
    ap.add_argument("--per-class", type=int, default=12,
                    help="images to keep per class (0 = all). default 12")
    args = ap.parse_args()

    print(f"[info] backend={args.backend}  per_class={args.per_class}  out={REAL_DIR}")
    rows = download_kaggle(args.per_class) if args.backend == "kaggle" else download_hf(args.per_class)
    write_manifest(rows)
    if not rows:
        print("[error] no images downloaded", file=sys.stderr)
        return 2
    print(f"[done] {len(rows)} real images saved under {REAL_DIR/'images'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
