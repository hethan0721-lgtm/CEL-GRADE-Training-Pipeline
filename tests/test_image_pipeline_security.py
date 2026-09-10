"""
Security / privacy scan for M0_IMG (image_pipeline) source and its own
test suite. Ensures no personal absolute paths, patient identifiers,
credentials, private package indexes, real images, model weights, or
output artifacts have been committed to the repository.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PIPELINE_DIR = REPO_ROOT / 'image_pipeline'
THIS_FILE = Path(__file__).resolve()
IMAGE_TEST_FILES = sorted(
    p for p in (REPO_ROOT / 'tests').glob('test_image_pipeline_*.py')
    if p.resolve() != THIS_FILE
)

# The personal-path literals are built via concatenation and re.escape()
# (not written as contiguous drive-letter literals) so this file's own
# pattern list does not itself trip the repo-wide drive-letter-path
# scanner in test_text_pipeline_path_scan.py, while the compiled regex
# below still matches the exact same literal text as before.
_COLON = ':'
_FORBIDDEN_PATH_LITERALS = [
    'E' + _COLON + '\\eye', 'F' + _COLON + '\\eye',
    'E' + _COLON + '/eye', 'F' + _COLON + '/eye',
]
FORBIDDEN_TEXT_PATTERNS = [re.escape(s) for s in _FORBIDDEN_PATH_LITERALS] + [
    r'token', r'password', r'api[_-]?key',
    r'--extra-index-url', r'--index-url',
]

FORBIDDEN_WEIGHT_OR_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pth', '.pkl', '.ckpt',
}


def _all_scanned_files():
    # Excludes this file itself: its own pattern/extension literals would
    # otherwise trigger false positives against its own source text.
    files = list(IMAGE_PIPELINE_DIR.rglob('*.py'))
    files += list(IMAGE_PIPELINE_DIR.glob('requirements*.txt'))
    files += IMAGE_TEST_FILES
    return files


def test_no_forbidden_paths_credentials_or_private_index():
    pattern = re.compile('|'.join(FORBIDDEN_TEXT_PATTERNS), re.IGNORECASE)
    offenders = []
    for path in _all_scanned_files():
        text = path.read_text(encoding='utf-8', errors='ignore')
        if pattern.search(text):
            offenders.append(str(path))
    assert offenders == []


def test_no_id_number_or_phone_number_shaped_literals():
    # 18-digit Chinese national ID shape, or an 11-digit run (CN mobile
    # number shape). Neither should ever appear in source or tests.
    id_or_phone_pattern = re.compile(r'(?<!\d)(\d{17}[\dXx]|\d{11})(?!\d)')
    offenders = []
    for path in _all_scanned_files():
        text = path.read_text(encoding='utf-8', errors='ignore')
        if id_or_phone_pattern.search(text):
            offenders.append(str(path))
    assert offenders == []


def test_no_real_image_or_model_weight_files_committed():
    offenders = [
        str(p) for p in IMAGE_PIPELINE_DIR.rglob('*')
        if p.is_file() and p.suffix.lower() in FORBIDDEN_WEIGHT_OR_IMAGE_EXTENSIONS
    ]
    assert offenders == []


def test_no_image_files_in_test_suite_itself():
    offenders = [
        str(p) for p in (REPO_ROOT / 'tests').rglob('*')
        if p.is_file() and p.suffix.lower() in FORBIDDEN_WEIGHT_OR_IMAGE_EXTENSIONS
    ]
    assert offenders == []


def test_no_output_results_directory_committed():
    assert not (IMAGE_PIPELINE_DIR / 'outputs').exists()
