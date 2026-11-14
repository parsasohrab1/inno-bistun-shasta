# Real Data Source & Provenance

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
