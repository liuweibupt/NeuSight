import subprocess
import sys
from pathlib import Path


def test_import_neusight_succeeds_on_cpu_only_torch():
    repo = Path(__file__).resolve().parents[1]
    code = "import neusight; print('ok')"
    proc = subprocess.run(
        [sys.executable, '-c', code],
        cwd=repo,
        env={**__import__('os').environ, 'PYTHONPATH': str(repo)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert 'ok' in proc.stdout
