"""
Whitelist-only compatibility loader for the frozen M0_CLIN (7-feature)
XGBoost release weights.

Why this file exists
---------------------
The authoritative, already-frozen training run's pickles
(``pretrained/best_xgboost_model.pkl``, ``pretrained/feature_engineer.pkl``,
published alongside this repository) were produced inside the private
training project, where the embedded ``Config`` object is an instance of
``text_classification_7feature.code.config.Config``. This repository ships
the equivalent class at ``text_pipeline.src.config.Config`` instead. A plain
``pickle.load()`` therefore fails with ``ModuleNotFoundError`` when loading
those specific frozen files, because the old module path does not exist
here.

This loader does NOT vendor, copy, or reconstruct any code from the private
training project. It only teaches the unpickler a single, explicit mapping
from that one old class identity to this repo's already-published,
structurally equivalent ``Config`` class, and only resolves the small,
fixed set of third-party library classes the frozen artifacts actually
need (``xgboost``, ``sklearn``, ``numpy``, ``builtins``, ``collections``).
Any other module/class reference -- old or new, known or unknown -- raises
``UnpicklingError`` immediately. Nothing is silently guessed or broadly
mapped.

``SurgeryClassifier.load()`` / ``FeatureEngineer.load()`` (see ``models.py``
/ ``feature_engineering.py``) try a plain ``pickle.load()`` first and only
fall back to this whitelist-only unpickler on ``ModuleNotFoundError``. That
keeps the normal train -> save -> load round trip for freshly trained
models completely unchanged (it never touches this module at all); this
compatibility path only ever activates for the specific frozen legacy
files distributed with this release.

Verified inventory (see the release-preparation audit that produced this
mapping): the ONLY project-specific class embedded in either frozen pickle
is ``text_classification_7feature.code.config.Config``. No legacy
``FeatureEngineer``, ``SurgeryClassifier``, ``DataLoader``, or any other
private-project class is embedded in these two files.
"""

from __future__ import annotations

import pickle
from typing import Any

# ---------------------------------------------------------------------------
# The single, explicit legacy -> public mapping. Add an entry here ONLY after
# (a) confirming via a safe pickle scan that the old class is actually
# embedded, and (b) confirming this repo's replacement is structurally
# compatible (same or superset of attributes actually used at runtime).
# ---------------------------------------------------------------------------
_LEGACY_CLASS_MAP: dict[tuple[str, str], str] = {
    ("text_classification_7feature.code.config", "Config"):
        "text_pipeline.src.config",
}

# Module prefixes that are legitimate third-party / stdlib dependencies the
# frozen artifacts actually reference, always resolved via the normal,
# unmodified pickle machinery. This is an allowlist, not a denylist:
# anything not covered here and not in _LEGACY_CLASS_MAP raises rather than
# being silently allowed.
_SAFE_THIRD_PARTY_PREFIXES = (
    "xgboost",
    "sklearn",
    "numpy",
    "builtins",
    "collections",
)


class UnknownLegacyReferenceError(pickle.UnpicklingError):
    """Raised when the pickle references a module/class this loader has not
    been explicitly told is safe. Fail loudly rather than guess."""


class CompatibilityUnpickler(pickle.Unpickler):
    """A ``pickle.Unpickler`` that only special-cases the whitelisted legacy
    ``Config`` class; everything else is resolved normally or rejected."""

    def find_class(self, module: str, name: str) -> Any:
        key = (module, name)
        if key in _LEGACY_CLASS_MAP:
            target_module = _LEGACY_CLASS_MAP[key]
            imported = __import__(target_module, fromlist=[name])
            return getattr(imported, name)

        if any(module == p or module.startswith(p + ".") for p in _SAFE_THIRD_PARTY_PREFIXES):
            return super().find_class(module, name)

        raise UnknownLegacyReferenceError(
            f"Refusing to unpickle unmapped/unknown reference "
            f"'{module}.{name}'. This loader only whitelists the legacy "
            f"Config class and a fixed set of third-party library prefixes "
            f"({', '.join(_SAFE_THIRD_PARTY_PREFIXES)}). If this is a "
            f"genuinely new, verified-safe reference, add it explicitly to "
            f"_LEGACY_CLASS_MAP or _SAFE_THIRD_PARTY_PREFIXES after manual "
            f"review -- do not broaden this check generically."
        )


def load(path: str) -> Any:
    """Load a pickle file using the whitelist-only compatibility unpickler.

    Args:
        path: Path to a trusted, already-hash-verified pickle file (the
            frozen model or feature-engineer artifact). This function
            performs no hash verification itself -- callers are expected to
            have already checked SHA256 against ``pretrained/SHA256SUMS.txt``
            before calling this.

    Returns:
        The unpickled object.
    """
    with open(path, "rb") as f:
        return CompatibilityUnpickler(f).load()
