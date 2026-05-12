import subprocess
import sys


def test_r_oracle_parity_smoke():
    proc = subprocess.run(
        [sys.executable, "tools/r_oracle_parity.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode == 77:
        import pytest

        pytest.skip(proc.stderr.strip())
    assert proc.returncode == 0, proc.stderr
