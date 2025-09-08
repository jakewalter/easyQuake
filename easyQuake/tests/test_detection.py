import os
import sys
import datetime
import subprocess
import pathlib

import pytest

# Prefer the repo root on sys.path so imports resolve to local package
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from easyQuake.easyQuake import detection_continuous


def _make_project(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    day = proj / "day1"
    day.mkdir()
    # write a simple dayfile that won't be actually read by the mocked runners
    dayfile = day / "dayfile.in"
    dayfile.write_text("/tmp/A.mseed /tmp/B.mseed /tmp/C.mseed\n")
    return proj, "day1"


def test_gpd_inline_called(monkeypatch, tmp_path):
    proj, dirname = _make_project(tmp_path)

    # Ensure make_dayfile returns our prepared dayfile path
    monkeypatch.setattr("easyQuake.easyQuake.make_dayfile", lambda d, m: str(proj / dirname / "dayfile.in"))

    called = {}

    def fake_process_dayfile(infile, outfile, base_dir=None, verbose=False, plot=False):
        called['infile'] = infile
        called['outfile'] = outfile
        with open(outfile, 'w') as f:
            f.write('gpd-pick')

    import easyQuake.gpd_predict.gpd_predict as gpd_mod
    monkeypatch.setattr(gpd_mod, 'process_dayfile', fake_process_dayfile)

    detection_continuous(dirname=dirname, project_folder=str(proj), project_code='PC', local=True, machine=True, machine_picker='GPD', single_date=datetime.datetime(2020,1,1))

    outpath = proj / dirname / 'gpd_picks.out'
    assert outpath.exists()
    assert called['outfile'] == str(outpath)


def test_eqtransformer_fallback_to_cli(monkeypatch, tmp_path):
    proj, dirname = _make_project(tmp_path)
    monkeypatch.setattr("easyQuake.easyQuake.make_dayfile", lambda d, m: str(proj / dirname / "dayfile.in"))

    # cause the inline call to raise TypeError so the code falls back to CLI
    def eqt_main():
        raise TypeError('bad signature')

    import easyQuake.EQTransformer.mseed_predictor as eqt_mod
    monkeypatch.setattr(eqt_mod, 'main', eqt_main)

    calls = []

    def fake_system(cmd):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(os, 'system', fake_system)

    detection_continuous(dirname=dirname, project_folder=str(proj), project_code='PC', local=True, machine=True, machine_picker='EQTransformer', single_date=datetime.datetime(2020,1,1))

    assert any('mseed_predictor' in c for c in calls)


def test_phasenet_cli_invoked(monkeypatch, tmp_path):
    proj, dirname = _make_project(tmp_path)
    monkeypatch.setattr("easyQuake.easyQuake.make_dayfile", lambda d, m: str(proj / dirname / "dayfile.in"))

    # simulate successful CLI run
    def fake_run(cmd, shell, capture_output, text):
        class R:
            returncode = 0
            stdout = 'ok'
            stderr = ''

        # record the command for assertion
        fake_run.cmd = cmd
        return R()

    monkeypatch.setattr(subprocess, 'run', fake_run)

    detection_continuous(dirname=dirname, project_folder=str(proj), project_code='PC', local=True, machine=True, machine_picker='PhaseNet', single_date=datetime.datetime(2020,1,1))

    assert hasattr(fake_run, 'cmd')
    assert 'phasenet_predict.py' in fake_run.cmd or 'phasenet_predict' in fake_run.cmd


def test_seisbench_skips_without_model(tmp_path):
    proj, dirname = _make_project(tmp_path)
    # No monkeypatch needed: just ensure it runs without a model and doesn't raise
    detection_continuous(dirname=dirname, project_folder=str(proj), project_code='PC', local=True, machine=True, machine_picker='Seisbench', single_date=datetime.datetime(2020,1,1))


def test_stalta_branch_calls_queue(monkeypatch, tmp_path):
    proj, dirname = _make_project(tmp_path)
    monkeypatch.setattr("easyQuake.easyQuake.make_dayfile", lambda d, m: str(proj / dirname / "dayfile.in"))

    called = {}

    def fake_queue(infile, outfile, dirname2, *args, **kwargs):
        called['infile'] = infile
        called['outfile'] = outfile

    monkeypatch.setattr("easyQuake.easyQuake.queue_sta_lta", fake_queue)

    # call with machine=False which triggers STALTA path
    detection_continuous(dirname=dirname, project_folder=str(proj), project_code='PC', local=True, machine=False, single_date=datetime.datetime(2020,1,1))

    assert 'outfile' in called
