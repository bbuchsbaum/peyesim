"""Run R eyesim as a live oracle from pytest.

Parity test modules build one R script per module, run it once, and compare
the JSON it prints against peyesim::

    from tests.r_oracle import requires_r, run_r

    R_CODE = '''
    out <- list(value = similarity(d1, d2, method = "pearson"))
    '''

    @pytest.fixture(scope="module")
    def r():
        return run_r(R_CODE)

The script runs after ``pkgload::load_all()`` of the R source tree, which is
``$EYESIM_SOURCE_DIR`` when set and ``<repo>/eyesim`` otherwise. It must assign
a list to ``out``; that list is serialised with ``jsonlite::toJSON`` (data frames
by column, ``NA`` as ``null``, 15 significant digits). Tests are skipped when
``Rscript``, ``pkgload``/``jsonlite`` or the R source tree are unavailable.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_PRELUDE = r"""
suppressPackageStartupMessages(pkgload::load_all(Sys.getenv("EYESIM_SOURCE_DIR"), quiet = TRUE))
suppressPackageStartupMessages(library(jsonlite))
"""

_EPILOGUE = r"""
cat("<<<JSON\n")
cat(toJSON(out, dataframe = "columns", auto_unbox = TRUE, digits = I(15),
           na = "null", null = "null", force = TRUE))
"""


def eyesim_source_dir() -> Path:
    env = os.environ.get("EYESIM_SOURCE_DIR")
    return Path(env) if env else REPO_ROOT / "eyesim"


@functools.lru_cache(maxsize=1)
def r_unavailable_reason() -> str | None:
    if shutil.which("Rscript") is None:
        return "Rscript is not available"
    src = eyesim_source_dir()
    if not (src / "DESCRIPTION").is_file():
        return f"R eyesim source not found at {src} (set EYESIM_SOURCE_DIR)"
    probe = subprocess.run(
        ["Rscript", "-e", "cat(requireNamespace('pkgload', quietly=TRUE) && "
         "requireNamespace('jsonlite', quietly=TRUE))"],
        text=True, capture_output=True, check=False,
    )
    if probe.returncode != 0 or not probe.stdout.strip().endswith("TRUE"):
        return "R oracle requires the R packages 'pkgload' and 'jsonlite'"
    return None


class _RequiresR:
    """``@requires_r`` marker that probes R lazily, at the first decorated test."""

    def __call__(self, obj):
        reason = r_unavailable_reason()
        return pytest.mark.skipif(reason is not None, reason=reason or "")(obj)


requires_r = _RequiresR()


def run_r(code: str) -> dict:
    """Run ``code`` against R eyesim and return the parsed ``out`` list."""
    reason = r_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    env = dict(os.environ, EYESIM_SOURCE_DIR=str(eyesim_source_dir()))
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as handle:
        handle.write(_PRELUDE + code + _EPILOGUE)
        script = Path(handle.name)
    try:
        proc = subprocess.run(["Rscript", str(script)], text=True, capture_output=True,
                              check=False, env=env)
    finally:
        script.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"R oracle failed:\n{proc.stderr}")
    return json.loads(proc.stdout.split("<<<JSON\n", 1)[1])


def as_float(value) -> np.ndarray:
    """Convert R JSON (with ``null`` for ``NA``) to a float array with NaN."""
    if value is None:
        return np.array(np.nan)
    if isinstance(value, list):
        return np.array([np.nan if v is None else v for v in _flatten(value)], dtype=float).reshape(
            np.shape(_nan_fill(value))
        )
    return np.array(value, dtype=float)


def _flatten(value):
    for v in value:
        if isinstance(v, list):
            yield from _flatten(v)
        else:
            yield v


def _nan_fill(value):
    if isinstance(value, list):
        return [_nan_fill(v) for v in value]
    return np.nan if value is None else value
