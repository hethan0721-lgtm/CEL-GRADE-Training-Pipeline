# Synthetic Tabular Demo Data (M0_CLIN)

**Every record in this directory is fictitious.** Nothing here was sampled,
copied, perturbed, de-identified, or otherwise derived from any real
patient record. All values are drawn from a random-number generator defined
in [`generate_synthetic_data.py`](generate_synthetic_data.py) in this same
directory.

This data exists solely to let `text_pipeline/src/main.py` run end-to-end
(train / internal-eval / external-eval) without any real clinical data, so
the code's interface, column expectations, and output format can be
exercised and tested. **It does not represent any real patient population,
clinical distribution, or medical relationship**, and must not be used to
draw any clinical or scientific conclusion.

## Files

- `synthetic_train.csv` — 150 fictitious rows, used as the training file (the pipeline performs its own internal 80/20 split on this file).
- `synthetic_external.csv` — 50 fictitious rows, used as the external validation file.
- `generate_synthetic_data.py` — the generator script; re-run it to regenerate both files deterministically (a fixed seed local to this script, independent of the model's training seed).

CSV (not `.xlsx`) is used here because the repository's root `.gitignore`
excludes `*.xlsx` but allow-lists `examples/**/*.csv`. `text_pipeline`'s
`DataLoader` supports `.csv`, `.xlsx`, and `.xls` equally — CSV is simply
what the public examples ship as.

## Columns

Column names and types mirror what `text_pipeline.src.config.Config`
actually requires as input (see `text_pipeline/README.md` for the full
field reference). The public example files contain **only** an anonymous
identifier, the 5 canonical model features, and the target column — no
extra/legacy fields:

| Column | Type | Notes |
|---|---|---|
| `sample_id` | text | Anonymous synthetic identifier (`SYN-TR-####` / `SYN-EX-####`); not a real patient ID, not a model input, silently dropped by the ingestion whitelist. |
| `脱位程度` | categorical | Used feature. |
| `矫正视力` | numeric | Used feature. |
| `矫正球镜度数(D)` | numeric | Used feature. |
| `矫正柱镜度数(D)` | numeric | Used feature. |
| `IOLMaster-Cyl(D)` | numeric | Used feature. |
| `是否需要手术` | text (`手术` / `不手术`) | Target column. |

Historical, never-used columns (`是否配合`, `年龄`) are intentionally **not**
included here. `DataLoader`'s tolerance for such legacy/extra input columns
(they are silently dropped and can never reach the model) is exercised by a
synthetic, in-test-only DataFrame in
`tests/test_text_pipeline_input_validation.py`, not by these checked-in
example files.

No name, ID-card number, medical record number, phone number, address, or
province column is present. A small fraction of cells are intentionally
left blank, and a small fraction of numeric cells intentionally include the
raw-text artifacts (`+`, `-`, full-width `−`) that real optometry exports
sometimes contain, so the demo data actually exercises the pipeline's
cleaning/imputation code paths rather than arriving pre-cleaned. Some
`脱位程度` cells are also intentionally left blank to exercise the
historical categorical-missing-value behaviour described in
`text_pipeline/README.md` (missing categorical values become an explicit
`"nan"` category, not a most-frequent-imputed value).

## Regenerating

```bash
python examples/synthetic_tabular_data/generate_synthetic_data.py
```
