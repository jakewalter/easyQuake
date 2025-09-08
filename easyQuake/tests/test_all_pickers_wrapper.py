import subprocess
import sys
from pathlib import Path


def test_run_test_all_pickers_script(tmp_path):
    """Run the legacy test_all_pickers.py script and ensure it completes successfully."""
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / 'test_all_pickers.py'
    assert script.exists(), f"Script not found: {script}"

    # Run the script in a fresh python subprocess and capture output for debugging
    res = subprocess.run([sys.executable, str(script)], cwd=str(repo_root), capture_output=True, text=True)
    print('stdout:\n', res.stdout)
    print('stderr:\n', res.stderr)
    assert res.returncode == 0, f"test_all_pickers.py exited with {res.returncode}\nSTDERR:\n{res.stderr}"
