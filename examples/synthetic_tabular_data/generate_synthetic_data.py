#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates fully synthetic demo data for the M0_CLIN text/tabular pipeline
(text_pipeline/). Every value in the output files is drawn from a fictitious
random-number generator defined in this script -- nothing is sampled,
copied, or derived from any real patient record.

Column names and dtypes mirror ``text_pipeline.src.config.Config`` so the
files can be fed directly to ``text_pipeline/src/main.py`` (train,
internal-eval, external-eval). Only the columns the canonical model actually
uses are emitted: an anonymous ``sample_id``, the 5 canonical features
(``脱位程度`` + 4 numeric columns -- see ``Config.FEATURE_COLUMNS``), and the
target column. The historical, never-used ``是否配合``/``年龄`` columns are
deliberately NOT included in the public example data; DataLoader's
tolerance for such extra/legacy columns (they are silently dropped and
never reach the model) is covered separately by a synthetic, in-test-only
DataFrame in tests/test_text_pipeline_input_validation.py -- it does not
need to be demonstrated via checked-in example files.

Outputs (CSV -- .xlsx is not used for the public examples because the
repository's root .gitignore excludes *.xlsx but allow-lists
examples/**/*.csv):
  synthetic_train.csv     -- used as the training file (80/20 internal
                              split happens inside the pipeline)
  synthetic_external.csv  -- used as the external validation file

This script writes no PHI and contains no absolute machine paths; it is
safe to re-run to regenerate the example files at any time.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Seed for reproducibility of the *example data only*. This is unrelated to
# and independent from Config.RANDOM_STATE (the model-training seed).
EXAMPLE_DATA_SEED = 12345

DISLOCATION_LEVELS = ['轻度', '中度', '重度']  # mild / moderate / severe (fictitious)

OUTPUT_DIR = Path(__file__).resolve().parent


def _make_numeric_with_artifacts(rng, values, messy_fraction=0.04, missing_fraction=0.05):
    """
    Turn a float array into an ``object`` array that occasionally contains
    the same kind of raw-text artifacts real optometry exports have (a
    trailing '+' / '-' sign, or a full-width minus), plus some genuinely
    missing cells -- so the demo data actually exercises
    ``DataLoader.clean_numeric_columns`` / ``SimpleImputer`` instead of
    arriving pre-cleaned.
    """
    values = np.round(values.astype(float), 2).astype(object)
    n = len(values)

    n_missing = max(1, int(n * missing_fraction))
    missing_idx = rng.choice(n, size=n_missing, replace=False)
    for i in missing_idx:
        values[i] = np.nan

    remaining = [i for i in range(n) if i not in set(missing_idx)]
    n_messy = max(1, int(n * messy_fraction))
    messy_idx = rng.choice(remaining, size=min(n_messy, len(remaining)), replace=False)
    for i in messy_idx:
        v = values[i]
        suffix = rng.choice(['+', '-', ''])
        values[i] = f"{v:.2f}{suffix}".replace('-', '−', 1) if suffix == '-' else f"{v:.2f}{suffix}"

    return values


def _make_categorical_with_missing(rng, values, missing_fraction=0.04):
    values = values.astype(object)
    n = len(values)
    n_missing = max(1, int(n * missing_fraction))
    missing_idx = rng.choice(n, size=n_missing, replace=False)
    for i in missing_idx:
        values[i] = np.nan
    return values


def _generate_frame(rng, n_rows, id_prefix, positive_rate_target):
    dislocation = rng.choice(DISLOCATION_LEVELS, size=n_rows, p=[0.5, 0.3, 0.2])

    corrected_vision = np.clip(rng.uniform(0.05, 1.0, size=n_rows), 0.02, 1.2)
    spherical_power = np.clip(rng.normal(-4.0, 4.0, size=n_rows), -20.0, 4.0)
    cylindrical_power = np.clip(rng.normal(-1.5, 1.2, size=n_rows), -6.0, 0.0)
    iolmaster_cyl = np.clip(rng.uniform(0.0, 4.0, size=n_rows), 0.0, 6.0)

    # Purely illustrative synthetic relationship (NOT a medical claim) so the
    # demo data produces a non-degenerate mix of both classes: more severe
    # dislocation + worse (lower) corrected vision nudges the fictitious
    # surgery probability upward.
    severity_score = np.select(
        [dislocation == '轻度', dislocation == '中度', dislocation == '重度'],
        [0.0, 0.35, 0.7]
    )
    vision_score = 1.0 - corrected_vision  # worse vision -> higher score
    logits = -1.2 + 2.2 * severity_score + 1.1 * vision_score + rng.normal(0, 0.4, size=n_rows)
    prob = 1.0 / (1.0 + np.exp(-logits))
    # Rescale so the realized positive rate lands close to the target, purely
    # so smoke tests reliably see both classes; still fully synthetic.
    prob = prob * (positive_rate_target / prob.mean())
    prob = np.clip(prob, 0.02, 0.98)
    surgery = rng.binomial(1, prob)
    target = np.where(surgery == 1, '手术', '不手术')

    df = pd.DataFrame({
        'sample_id': [f"{id_prefix}-{i+1:04d}" for i in range(n_rows)],
        '脱位程度': _make_categorical_with_missing(rng, dislocation),
        '矫正视力': _make_numeric_with_artifacts(rng, corrected_vision),
        '矫正球镜度数(D)': _make_numeric_with_artifacts(rng, spherical_power),
        '矫正柱镜度数(D)': _make_numeric_with_artifacts(rng, cylindrical_power),
        'IOLMaster-Cyl(D)': _make_numeric_with_artifacts(rng, iolmaster_cyl),
        '是否需要手术': target,
    })
    return df


def main():
    rng = np.random.default_rng(EXAMPLE_DATA_SEED)

    train_df = _generate_frame(rng, n_rows=150, id_prefix='SYN-TR', positive_rate_target=0.40)
    external_df = _generate_frame(rng, n_rows=50, id_prefix='SYN-EX', positive_rate_target=0.40)

    train_path = OUTPUT_DIR / 'synthetic_train.csv'
    external_path = OUTPUT_DIR / 'synthetic_external.csv'

    # utf-8-sig so the Chinese column names open correctly in Excel on
    # Windows too; text_pipeline's DataLoader reads CSVs with the same
    # encoding, so this round-trips exactly either way.
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    external_df.to_csv(external_path, index=False, encoding='utf-8-sig')

    print(f"Wrote {len(train_df)} synthetic training rows to {train_path}")
    print(f"Wrote {len(external_df)} synthetic external-validation rows to {external_path}")
    print("\nTraining target distribution:")
    print(train_df['是否需要手术'].value_counts())
    print("\nExternal target distribution:")
    print(external_df['是否需要手术'].value_counts())


if __name__ == '__main__':
    main()
