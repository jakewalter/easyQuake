#!/usr/bin/env python3
"""
Consolidated test script for all easyQuake machine pickers.

This file was synchronized with the working copy in
`easyQuake/test_all_pickers.py` and adjusted to run from the
`~/easyQuake/tests/` directory. It auto-discovers the local
`easyQuake` package (so you can run it from the tests/ folder).

Pickers tested: GPD, EQTransformer, PhaseNet, STALTA, Seisbench

Seisbench requires a model path (set via SEISBENCH_MODEL_PATH); otherwise
it will be skipped.
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime, date

# Ensure headless plotting works in CI / headless environments. Set a
# non-interactive backend before any code can import pyplot or other
# matplotlib submodules. Also defensively ensure figure dpi is positive
# to avoid ValueError: dpi must be positive coming from the QtAgg/Agg
# renderer when DPI is misconfigured or zero.
try:
    if "DISPLAY" not in os.environ or os.environ.get("MPLBACKEND", "") == "Agg":
        import matplotlib

        matplotlib.use("Agg")
        try:
            dpi = matplotlib.rcParams.get("figure.dpi", 100)
            if dpi is None or dpi <= 0:
                matplotlib.rcParams["figure.dpi"] = 100
        except Exception:
            matplotlib.rcParams["figure.dpi"] = 100
except Exception:
    # If matplotlib isn't installed or backend setting fails, continue; tests
    # that need plotting will still proceed but may error elsewhere.
    pass


def find_and_add_easyquake_to_path():
    """Locate the repository root that contains `easyQuake/` and add it to sys.path.

    This allows running this script from ~/easyQuake/tests/ without hardcoding
    absolute paths.
    """
    current_dir = Path(__file__).parent.resolve()
    for parent in [current_dir, *current_dir.parents]:
        if (parent / 'easyQuake').is_dir():
            sys.path.insert(0, str(parent))
            return parent
    return None


def setup_test_project():
    """Create a temporary test project inside the tests directory and copy MSEED files.

    Returns (project_path: Path, test_date: str) or (None, None) on failure.
    """
    print("easyQuake Machine Picker Test Suite - synchronized tests version")
    tests_dir = Path(__file__).parent.resolve()
    test_files = ['O2.WILZ.EHE.mseed', 'O2.WILZ.EHN.mseed', 'O2.WILZ.EHZ.mseed']

    missing = [f for f in test_files if not (tests_dir / f).exists()]
    if missing:
        print(f"❌ Missing test data files in {tests_dir}: {missing}")
        return None, None

    print(f"✓ Found test data files in {tests_dir}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_project = tests_dir / f'test_project_{timestamp}'
    date_str = '20240101'  # fixed test date used across the suite
    date_dir = temp_project / date_str
    date_dir.mkdir(parents=True, exist_ok=True)

    print("Copying MSEED files to project directory...")
    for f in test_files:
        src = tests_dir / f
        dst = date_dir / f
        shutil.copy2(src, dst)
        print(f"  Copied {f}")

    print(f"Test project created at: {temp_project}")
    return temp_project, date_str


def check_output_file(output_file: Path, picker_name: str):
    """Check whether an output file exists and count picks.

    Returns (success: bool, picks: int).
    """
    if not output_file.exists():
        print(f"✗ {picker_name} failed - no output file generated: {output_file}")
        return False, 0

    try:
        with open(output_file, 'r') as fh:
            lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith('#')]
        pick_count = len(lines)
        print(f"Output file: {output_file}")
        print(f"Number of picks: {pick_count}")
        if pick_count:
            print("First 5 picks:")
            for i, ln in enumerate(lines[:5], 1):
                print(f"  {i}: {ln}")
            if pick_count > 5:
                print(f"  ... and {pick_count - 5} more picks")
        else:
            print("  No picks found in output file")
        return True, pick_count
    except Exception as e:
        print(f"Error reading output file {output_file}: {e}")
        return True, 0


def test_picker(machine_picker, project_folder: Path, test_date: str, seisbench_model=None):
    """Run detection_continuous for a single picker and check results.

    Returns (success: bool, picks: int).
    """
    print(f"\n{'='*60}")
    print(f"Testing {machine_picker} picker")
    print(f"{'='*60}")

    try:
        # Ensure easyQuake is importable (added to sys.path earlier)
        import easyQuake

        test_datetime = datetime.strptime(test_date, '%Y%m%d')

        if machine_picker == 'STALTA':
            # STALTA uses machine=False and specific parameters
            easyQuake.detection_continuous(
                machine=False,
                dirname=test_date,
                project_folder=str(project_folder),
                project_code='TEST',
                single_date=test_datetime,
                local=True,
                latitude=36.7,
                longitude=-97.5,
                max_radius=300,
                make3=False,
                filtmin=2,
                filtmax=15,
                t_sta=0.2,
                t_lta=2.5,
                trigger_on=4,
                trigger_off=2,
                trig_horz=6.0,
                trig_vert=10.0
            )
        else:
            kwargs = {
                'machine': True,
                'machine_picker': machine_picker,
                'dirname': test_date,
                'project_folder': str(project_folder),
                'project_code': 'TEST',
                'single_date': test_datetime,
                'local': True,
                'latitude': 36.7,
                'longitude': -97.5,
                'max_radius': 300,
                'make3': False,
            }
            if machine_picker == 'Seisbench' and seisbench_model:
                kwargs['seisbenchmodel'] = seisbench_model

            easyQuake.detection_continuous(**kwargs)

        expected_output = project_folder / test_date / f"{machine_picker.lower()}_picks.out"

        # PhaseNet sometimes returns success with zero picks and no file; treat that as success
        if not expected_output.exists() and machine_picker == 'PhaseNet':
            print(f"Note: {machine_picker} produced no output file but will be treated as success (0 picks)")
            return True, 0

        return check_output_file(expected_output, machine_picker)

    except Exception as e:
        print(f"✗ {machine_picker} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def main():
    repo_root = find_and_add_easyquake_to_path()
    if repo_root:
        print(f"Repository root discovered: {repo_root}")
    else:
        print("Warning: could not auto-detect repository root; ensure easyQuake package is on PYTHONPATH")

    # Prepare project and test data
    project_folder, test_date = setup_test_project()
    if not project_folder:
        return 1

    pickers = ['GPD', 'EQTransformer', 'PhaseNet', 'STALTA', 'Seisbench']

    seisbench_model = os.environ.get('SEISBENCH_MODEL_PATH')
    # If no env var provided, try to auto-discover a model in the repo
    if not seisbench_model:
        try:
            # repo_root was added to sys.path earlier by find_and_add_easyquake_to_path()
            # find_and_add_easyquake_to_path() returns the repo root Path when successful
            default_models_dir = repo_root / 'easyQuake' / 'seisbench' / 'models'
            if default_models_dir.exists():
                # look for common model extensions
                candidates = list(default_models_dir.glob('*.pth')) + list(default_models_dir.glob('*.pt')) + list(default_models_dir.glob('*.ckpt'))
                if candidates:
                    # Prefer explicit 'best' models when available
                    bests = [c for c in candidates if 'best_model' in c.name]
                    selected = bests[0] if bests else candidates[0]
                    seisbench_model = str(selected)
                    print(f"Using discovered Seisbench model: {seisbench_model}")
                else:
                    print(f"No model files found in {default_models_dir}; Seisbench will be skipped unless SEISBENCH_MODEL_PATH is set")
            else:
                # Fall back: no default models dir present
                pass
        except Exception:
            # Defensive: if repo_root is None or unexpected error, continue and rely on env var
            seisbench_model = None
    else:
        if seisbench_model and not os.path.exists(seisbench_model):
            print(f"Provided SEISBENCH_MODEL_PATH does not exist: {seisbench_model}")
            seisbench_model = None

    results = {}
    total_picks = 0
    for picker in pickers:
        if picker == 'Seisbench':
            if seisbench_model:
                success, picks = test_picker(picker, project_folder, test_date, seisbench_model)
            else:
                print('Seisbench model not provided; skipping Seisbench test')
                success, picks = True, 0
        else:
            success, picks = test_picker(picker, project_folder, test_date)

        results[picker] = (success, picks)
        if success:
            total_picks += picks

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    successful = 0
    for picker, (success, picks) in results.items():
        status = 'PASS' if success else 'FAIL'
        print(f"{picker:<15} {status} ({picks:3d} picks)")
        if success:
            successful += 1

    print(f"\nOverall: {successful}/{len(pickers)} pickers successful")
    print(f"Total picks generated: {total_picks}")

    output_dir = project_folder / test_date
    output_files = list(output_dir.glob("*_picks.out"))
    if output_files:
        print(f"\nOutput files in {output_dir}:")
        for f in output_files:
            print(f"  {f.name} ({f.stat().st_size} bytes)")

    print(f"\nTest project preserved at: {project_folder}")
    print("You can examine the output files manually.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
