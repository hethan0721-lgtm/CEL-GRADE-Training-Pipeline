"""
Shared pytest configuration for the CEL-GRADE-Training-Pipeline test suite.

Ensures the repository root is importable as a package root (so
``import text_pipeline.src...`` works) regardless of the directory pytest
is invoked from, without requiring any repo-root packaging files.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
