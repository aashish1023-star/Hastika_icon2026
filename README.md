# HASTIKA @ ICON-2026

Code and saved artifacts for the HASTIKA @ ICON-2026 shared task on Kannada-English code-mixed hate speech and target identification.

This repository is intended for inspection and verification. It contains the training notebook, saved validation/test probabilities, prediction files, and result figures. It does **not** contain the raw dataset, the paper source, PDFs, bibliography files, or manuscript notes.

## Tasks

| Task | Problem | Labels |
|---|---|---|
| A | Binary hate speech detection | `Hate`, `Non-hate` |
| B | Fine-grained target identification | `Gender`, `Political`, `Religion`, `Geo-political`, `Violence`, `Others` |

## Data

The raw HASTIKA data is **not redistributed** here. Use the official starting kit:

```text
https://github.com/shankarb14/Hastika-ICON2026
```

The notebook clones that repository to `/tmp/Hastika-ICON2026` and reads the six CSV files from its `data/` directory.

| Task | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| A | 6,446 | 806 | 806 | 8,058 |
| B | 3,159 | 395 | 396 | 3,950 |

Task B is the hate-only subset and is split independently of Task A.

## Models

- `ai4bharat/IndicBERTv2-MLM-only`
- `l3cube-pune/kannada-bert`
- `google/muril-base-cased`

Each model is fine-tuned independently for each task. No ensemble is used.

## Training setup

| Setting | Value |
|---|---|
| Max epochs | 6 |
| Early-stopping patience | 3 |
| Early-stopping metric | Validation macro-F1 |
| Batch size | 32 |
| Learning rate | `3e-5` |
| Max sequence length | 128 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Seed | 42 |
| Task B class weighting | Square-root inverse frequency |

Early stopping was enabled for all six runs but never triggered. Selected epochs were `5, 5, 4` for Task A and `6, 6, 6` for Task B.

## Validation results

These are **validation** results, not official test or leaderboard scores. Test gold labels are not public.

### Task A

| Model | Macro-F1 | Weighted-F1 | Accuracy | Best epoch |
|---|---:|---:|---:|---:|
| IndicBERTv2 | 0.7832 | 0.7836 | 0.7854 | 5 |
| Kannada-BERT | 0.8052 | 0.8052 | 0.8052 | 5 |
| MuRIL | **0.8212** | **0.8213** | **0.8213** | 4 |

### Task B

| Model | Macro-F1 | Weighted-F1 | Accuracy | Best epoch |
|---|---:|---:|---:|---:|
| IndicBERTv2 | **0.6182** | **0.6923** | **0.6987** | 6 |
| Kannada-BERT | 0.4495 | 0.5676 | 0.6481 | 6 |
| MuRIL | 0.3764 | 0.5344 | 0.6228 | 6 |

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── verify_results.py
├── notebookde561d9cfd.ipynb
└── hastika_all_predictions/
    ├── results_summary.csv
    ├── figures/
    ├── ai4bharat__IndicBERTv2-MLM-only/
    ├── google__muril-base-cased/
    └── l3cube-pune__kannada-bert/
```

Each model directory contains:

```text
taskA_binary/
  predictions.csv
  val_proba.npy
  test_proba.npy
  taskA_binary_predictions.zip

taskB_finegrained/
  predictions.csv
  val_proba.npy
  test_proba.npy
  taskB_finegrained_predictions.zip
```

The `paramanandaAI_*` files are exact duplicates of the corresponding main prediction files, not alternative or ensemble submissions.

## Reproducing the notebook

1. Install the Python dependencies:

```bash
pip install -r requirements.txt
```

2. Provide the official data. The notebook expects the starting-kit repository at `/tmp/Hastika-ICON2026` and its data under `/tmp/Hastika-ICON2026/data`. Adjust `REPO_DIR` if running outside Kaggle/Linux.

3. Set a Hugging Face token if needed for model downloads. The notebook is configured to leave `HF_TOKEN` empty and fall back to the Kaggle secret named `HF_TOKEN`.

4. Run `notebookde561d9cfd.ipynb`. On Kaggle, outputs are written to `/kaggle/working/outputs`; elsewhere they default to `./outputs`.

## Verifying saved results

`verify_results.py` recomputes the validation metrics from `val_proba.npy` and compares them with `results_summary.csv`. It also checks prediction headers, row counts, ID uniqueness, probability normalisation, and `argmax(test_proba)` against `predictions.csv`.

Download the official labelled validation files first, then run:

```bash
python verify_results.py --data-dir /path/to/Hastika-ICON2026/data
```

Expected checks include:

- Task A: 806 validation rows, 806 test rows, two probability columns
- Task B: 395 validation rows, 396 test rows, six probability columns
- Prediction files have header `id,label` and unique IDs
- Validation macro-F1, weighted-F1, and accuracy match to four decimals

## Important notes

- Only validation metrics are reported here. No official test or leaderboard score is claimed.
- Exact bitwise reproducibility is not guaranteed: the runs use one seed and do not pin package versions.
- The raw dataset and pretrained model weights are not included and remain subject to their original licenses.
- No license has been declared for this repository. Contact the repository owner before reuse.
