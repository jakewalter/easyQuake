import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

import pytest

# Ensure the repository root (parent of tests/) is on sys.path so tests can
# import the local `easyQuake` package without needing an editable install.
_tests_dir = Path(__file__).parent.resolve()
_repo_root = _tests_dir.parent
if (_repo_root / 'easyQuake').is_dir():
    sys.path.insert(0, str(_repo_root))


# Default pickers exercised by the consolidated test harness. The
# 'Seisbench' entry will be skipped automatically if no model is found.
DEFAULT_PICKERS = ['GPD', 'EQTransformer', 'PhaseNet', 'STALTA', 'Seisbench']


@pytest.fixture(scope='session')
def seisbench_model():
    """Return a path to a Seisbench model if available, otherwise None.

    Priority: SEISBENCH_MODEL_PATH env var, then repo local
    easyQuake/seisbench/models/ directory (common extensions).
    """
    env = os.environ.get('SEISBENCH_MODEL_PATH')
    if env:
        return env if Path(env).exists() else None

    # Auto-discover model in repo relative to the tests/ directory
    tests_dir = Path(__file__).parent.resolve()
    repo_root = tests_dir.parent
    default_models_dir = repo_root / 'easyQuake' / 'seisbench' / 'models'
    if not default_models_dir.exists():
        return None

    candidates = list(default_models_dir.glob('*.pth')) + list(default_models_dir.glob('*.pt')) + list(default_models_dir.glob('*.ckpt'))
    if not candidates:
        return None

    # Prefer files with 'best' in the name when available
    bests = [c for c in candidates if 'best' in c.name.lower()]
    selected = bests[0] if bests else candidates[0]
    return str(selected)


@pytest.fixture(params=DEFAULT_PICKERS)
def machine_picker(request, seisbench_model):
    """Parametrized fixture that yields each picker name.

    If the Seisbench model is not present, the Seisbench param is skipped.
    """
    picker = request.param
    if picker == 'Seisbench' and not seisbench_model:
        pytest.skip('Seisbench model not found; skipping Seisbench tests')
    return picker


@pytest.fixture(scope='session')
def project_folder():
    """Create a temporary test project with the sample MSEED files and
    yield its path. The folder is removed after the test session.
    """
    tests_dir = Path(__file__).parent.resolve()
    test_files = ['O2.WILZ.EHE.mseed', 'O2.WILZ.EHN.mseed', 'O2.WILZ.EHZ.mseed']

    missing = [f for f in test_files if not (tests_dir / f).exists()]
    if missing:
        pytest.skip(f"Missing test data files in {tests_dir}: {missing}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_project = tests_dir / f'test_project_{timestamp}'
    date_str = '20240101'
    date_dir = temp_project / date_str
    date_dir.mkdir(parents=True, exist_ok=True)

    for f in test_files:
        shutil.copy2(tests_dir / f, date_dir / f)

    yield temp_project

    # cleanup
    try:
        shutil.rmtree(temp_project)
    except Exception:
        pass


@pytest.fixture
def test_date():
    """Return the fixed test date string used by the test harness."""
    return '20240101'
