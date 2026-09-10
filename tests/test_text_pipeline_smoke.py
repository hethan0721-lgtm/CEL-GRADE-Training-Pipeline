"""
End-to-end smoke test for M0_CLIN (text_pipeline), synthetic data only.

Exercises the full CLI pipeline (train -> internal-eval -> external-eval)
against the bundled fully-synthetic example files (150 / 50 rows), writing
all artifacts to a pytest tmp_path so nothing touches the real repository.
This is deliberately the ONLY test that runs actual model training, and it
only ever does so on synthetic data -- no real patient data is read or
could be read, since the only input paths used are the synthetic example
files checked into examples/synthetic_tabular_data/.
"""

from pathlib import Path

from text_pipeline.src.main import build_run_config, run_external_eval, run_internal_eval, run_train

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / 'examples' / 'synthetic_tabular_data'
TRAIN_FILE = EXAMPLES_DIR / 'synthetic_train.csv'
EXTERNAL_FILE = EXAMPLES_DIR / 'synthetic_external.csv'

METRICS = ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')


def test_full_pipeline_smoke_on_synthetic_data(tmp_path):
    config, output_dir = build_run_config(
        input_data=str(TRAIN_FILE),
        external_data=str(EXTERNAL_FILE),
        output_dir=str(tmp_path / 'run'),
        seed=42,
    )

    model, trainer, internal_results, external_results = run_train(config, output_dir, verbose=False)

    # Artifacts saved where the CLI promises them.
    assert (output_dir / 'models' / 'feature_engineer.pkl').exists()
    assert (output_dir / 'models' / config.BEST_MODEL_NAME).exists()
    assert (output_dir / 'eval_internal' / 'classification_report.txt').exists()
    assert (output_dir / 'eval_external' / 'classification_report.txt').exists()

    for metric in METRICS:
        assert 0.0 <= internal_results[metric] <= 1.0
        assert 0.0 <= external_results[metric] <= 1.0

    # internal-eval reconstructs the exact same held-out partition (same
    # file, same seed) and must score identically without retraining.
    internal_eval_results = run_internal_eval(config, output_dir, verbose=False)
    for metric in METRICS:
        assert internal_eval_results[metric] == internal_results[metric]

    # external-eval on the same external file, loading saved artifacts only
    # (no fitting), must also reproduce the training-time external metrics.
    ext_only_config, _ = build_run_config(
        input_data=str(EXTERNAL_FILE),
        output_dir=str(tmp_path / 'run'),
        seed=42,
    )
    external_eval_results = run_external_eval(ext_only_config, output_dir, verbose=False)
    for metric in METRICS:
        assert external_eval_results[metric] == external_results[metric]
