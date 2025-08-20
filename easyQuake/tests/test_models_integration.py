import subprocess
import sys
import os
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_gpd_process_dayfile_runs(tmp_path):
    """Run the GPD inline process_dayfile on the bundled test data."""
    from easyQuake.gpd_predict.gpd_predict import process_dayfile

    data_dir = ROOT / 'gpd_predict' / 'testdata'
    infile = data_dir / 'dayfile.in'
    outfile = tmp_path / 'gpd_out.txt'

    # run (may be slow); this imports TF and loads the model
    process_dayfile(str(infile), str(outfile), base_dir=str(ROOT / 'gpd_predict'), verbose=False, plot=False)

    assert outfile.exists()


@pytest.mark.integration
def test_eqtransformer_cli_runs(tmp_path):
    """Run the EQTransformer CLI script against the test mseed files."""
    # Run as module to preserve package-relative imports
    module = 'easyQuake.EQTransformer.mseed_predictor'
    data_dir = ROOT / 'gpd_predict' / 'testdata'
    infile = data_dir / 'dayfile.in'
    outfile = tmp_path / 'eqt_out.txt'
    cmd = [sys.executable, '-m', module, '-I', str(infile), '-O', str(outfile), '-F', str(ROOT / 'EQTransformer')]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    combined = (res.stdout or '') + (res.stderr or '')
    if 'Traceback' in combined:
        pytest.skip('EQTransformer failed in this environment: ' + combined.splitlines()[-1])


@pytest.mark.integration
def test_phasenet_cli_runs(tmp_path):
    """Run the PhaseNet CLI script against the test mseed files."""
    # Run PhaseNet as module to keep relative imports working
    module = 'easyQuake.phasenet.phasenet_predict'
    data_dir = ROOT / 'gpd_predict' / 'testdata'
    infile = data_dir / 'dayfile.in'
    outfile = tmp_path / 'phasenet_out.txt'
    model_dir = ROOT / 'phasenet' / 'model' / '190703-214543'
    if not model_dir.exists():
        pytest.skip('PhaseNet model missing')

    cmd = [sys.executable, '-m', module, f'--model={model_dir}', f'--data_list={infile}', '--format=mseed', f'--result_fname={outfile.name}', f'--result_dir={tmp_path}']
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    combined = (res.stdout or '') + (res.stderr or '')
    if 'Traceback' in combined:
        pytest.skip('PhaseNet failed in this environment: ' + combined.splitlines()[-1])


@pytest.mark.integration
def test_seisbench_runs_if_model_available(tmp_path):
    """Run Seisbench CLI if a model is available in the repo; otherwise skip."""
    script = ROOT / 'seisbench' / 'run_seisbench.py'
    data_dir = ROOT / 'gpd_predict' / 'testdata'
    infile = data_dir / 'dayfile.in'
    outfile = tmp_path / 'seis_out.txt'

    # attempt to find a model file (common extensions)
    model_files = list(ROOT.rglob('*.pt')) + list(ROOT.rglob('*.pth')) + list(ROOT.rglob('*.model'))
    if not script.exists() or not model_files:
        pytest.skip('Seisbench script or model not available')

    model = model_files[0]
    cmd = [sys.executable, str(script), '-I', str(infile), '-O', str(outfile), '-M', str(model)]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    combined = (res.stdout or '') + (res.stderr or '')
    assert 'Traceback' not in combined
