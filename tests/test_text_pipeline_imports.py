"""
Module import test for M0_CLIN (text_pipeline).

Verifies every module imports cleanly under its real package path
(``text_pipeline.src.*``) now that the code lives under
``text_pipeline/src`` -- this is a regression test for the
``from text_classification...`` import bug that broke every module after
the repository move.
"""

import importlib

MODULES = [
    'text_pipeline',
    'text_pipeline.src',
    'text_pipeline.src.config',
    'text_pipeline.src.data_loader',
    'text_pipeline.src.feature_engineering',
    'text_pipeline.src.models',
    'text_pipeline.src.train',
    'text_pipeline.src.evaluate',
    'text_pipeline.src.main',
]


def test_all_modules_import_cleanly():
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        assert module is not None


def test_package_exports_are_reachable():
    import text_pipeline.src as pkg

    assert pkg.__all__, "text_pipeline.src.__all__ should not be empty"
    for name in pkg.__all__:
        assert hasattr(pkg, name), f"text_pipeline.src is missing exported symbol: {name}"


def test_main_module_has_no_leftover_text_classification_references():
    import text_pipeline.src.main as main_module

    source_path = main_module.__file__
    with open(source_path, encoding='utf-8') as f:
        source = f.read()
    assert 'text_classification' not in source, (
        "main.py still references the old 'text_classification' package path"
    )
