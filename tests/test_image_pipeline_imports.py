"""
Module import test for M0_IMG (image_pipeline).

Verifies every module imports cleanly under its real package path
(``image_pipeline.src.*``). Importing must not create the outputs/
directory, read any dataset, download ImageNet weights, start training,
or load a checkpoint -- it must be a pure definition-time import.
"""

import importlib

MODULES = [
    'image_pipeline',
    'image_pipeline.src',
    'image_pipeline.src.config',
    'image_pipeline.src.data_utils',
    'image_pipeline.src.models',
    'image_pipeline.src.train',
    'image_pipeline.src.evaluate',
    'image_pipeline.src.main',
]


def test_all_modules_import_cleanly():
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        assert module is not None


def test_importing_main_does_not_create_outputs_dir():
    main_module = importlib.import_module('image_pipeline.src.main')
    default_outputs_dir = main_module.IMAGE_PIPELINE_DIR / 'outputs'
    assert not default_outputs_dir.exists(), (
        "Importing image_pipeline.src.main must not create the default "
        "outputs/ directory as a side effect of import"
    )


def test_importing_main_does_not_create_models_or_results_dirs():
    main_module = importlib.import_module('image_pipeline.src.main')
    outputs_dir = main_module.IMAGE_PIPELINE_DIR / 'outputs'
    assert not (outputs_dir / 'models').exists()
    assert not (outputs_dir / 'results').exists()
