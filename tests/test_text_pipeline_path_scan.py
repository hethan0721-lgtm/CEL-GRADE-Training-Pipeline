"""
Absolute-path scan test for the M0_CLIN public release.

Recursively scans the directories this pass is allowed to touch
(text_pipeline/, examples/synthetic_tabular_data/, tests/, docs/) for
hardcoded absolute filesystem paths -- Windows drive-letter paths
(``C:\\...``, ``F:/...``), Unix home directories, and specifically the
forbidden ``F:\\eye`` / ``E:\\eye`` paths -- so no personal or
institution-specific machine path can ship in the public repository.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ['text_pipeline', 'examples/synthetic_tabular_data', 'tests', 'docs']
TEXT_EXTENSIONS = {'.py', '.md', '.txt', '.cfg', '.ini', '.json', '.toml', '.yaml', '.yml'}

# Directories whose *contents* are expected to vary at runtime (model
# artifacts, generated plots/reports) and are not part of the reviewed
# source tree.
EXCLUDED_DIR_NAMES = {'outputs', '__pycache__', '.pytest_cache'}

THIS_FILE = Path(__file__).resolve()

WINDOWS_DRIVE_BACKSLASH = re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:\\')
WINDOWS_DRIVE_FORWARDSLASH = re.compile(r'(?<![A-Za-z0-9:])[A-Za-z]:/(?!/)')
UNIX_HOME_DIR = re.compile(r'/(?:home|Users|root|mnt)/[\w.\-]+')

FORBIDDEN_SUBSTRINGS = ['F:\\eye', 'E:\\eye', 'F:/eye', 'E:/eye', 'C:\\Users']


def _iter_text_files():
    for rel_dir in SCAN_DIRS:
        base = REPO_ROOT / rel_dir
        if not base.exists():
            continue
        for path in base.rglob('*'):
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            if EXCLUDED_DIR_NAMES & set(p.name for p in path.parents):
                continue
            yield path


def test_no_hardcoded_absolute_filesystem_paths():
    offenders = []
    for path in _iter_text_files():
        if path == THIS_FILE:
            continue  # this file legitimately contains the patterns as literals/regex
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue

        for pattern in (WINDOWS_DRIVE_BACKSLASH, WINDOWS_DRIVE_FORWARDSLASH, UNIX_HOME_DIR):
            for match in pattern.finditer(text):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {match.group(0)!r}")

        for needle in FORBIDDEN_SUBSTRINGS:
            if needle in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: contains forbidden path {needle!r}")

    assert not offenders, "Found hardcoded absolute filesystem paths:\n" + "\n".join(offenders)
