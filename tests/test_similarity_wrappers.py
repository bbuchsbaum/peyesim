"""Regression tests for similarity wrapper bugs.

Ports the R bug fixes from R/similarity.R:
1. run_similarity_analysis() rejecting FixationGroup/Scanpath inputs
2. fixation_similarity(method="overlap") returning dict instead of scalar
3. scanpath_similarity(method="multimatch") not expanding columns
4. Extra args (time_samples, screensize) not forwarded
"""

import numpy as np
import pandas as pd
import pytest

from peyesim import (
    fixation_group,
    scanpath,
    eye_table,
    add_scanpath,
)
from peyesim.similarity import (
    similarity,
    fixation_similarity,
    scanpath_similarity,
)


def _make_eyetab():
    """Build a small eye_table with 2 images x 2 phases."""
    rng = np.random.default_rng(42)
    rows = []
    for phase in ["enc", "ret"]:
        for img in ["img1", "img2"]:
            nfix = 10
            rows.append(pd.DataFrame({
                "x": rng.uniform(0, 500, nfix),
                "y": rng.uniform(0, 500, nfix),
                "onset": np.arange(1, nfix + 1, dtype=float),
                "duration": rng.uniform(100, 300, nfix),
                "image": img,
                "phase": phase,
            }))
    df = pd.concat(rows, ignore_index=True)
    return eye_table("x", "y", "duration", "onset",
                     groupvar=["phase", "image"], data=df)


# ---- Bug 1+2: fixation_similarity with overlap returns scalar ----

def test_fixation_similarity_overlap_returns_scalar():
    eyetab = _make_eyetab()
    enc = eyetab[eyetab["phase"] == "enc"].reset_index(drop=True)
    ret = eyetab[eyetab["phase"] == "ret"].reset_index(drop=True)

    result = fixation_similarity(enc, ret, match_on="image", method="overlap")
    assert "eye_sim" in result.columns
    assert result["eye_sim"].dtype == float
    assert len(result) == 2


# ---- Bug 1: fixation_similarity with sinkhorn ----

def test_fixation_similarity_sinkhorn():
    eyetab = _make_eyetab()
    enc = eyetab[eyetab["phase"] == "enc"].reset_index(drop=True)
    ret = eyetab[eyetab["phase"] == "ret"].reset_index(drop=True)

    result = fixation_similarity(enc, ret, match_on="image", method="sinkhorn")
    assert "eye_sim" in result.columns
    assert all(0 <= v <= 1 for v in result["eye_sim"])


# ---- Bug 3: scanpath_similarity expands multimatch columns ----

def test_scanpath_similarity_multimatch_expands_columns():
    eyetab = _make_eyetab()
    enc = add_scanpath(eyetab[eyetab["phase"] == "enc"].reset_index(drop=True))
    ret = add_scanpath(eyetab[eyetab["phase"] == "ret"].reset_index(drop=True))

    result = scanpath_similarity(enc, ret, match_on="image",
                                 method="multimatch", screensize=(500, 500))

    expected_cols = ["mm_vector", "mm_direction", "mm_length",
                     "mm_position", "mm_duration", "mm_position_emd"]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"
        assert result[col].notna().all(), f"NaN in {col}"

    # Should NOT have a single eye_sim column
    assert "eye_sim" not in result.columns


def test_scanpath_similarity_multimatch_with_permutations():
    eyetab = _make_eyetab()
    enc = add_scanpath(eyetab[eyetab["phase"] == "enc"].reset_index(drop=True))
    ret = add_scanpath(eyetab[eyetab["phase"] == "ret"].reset_index(drop=True))

    result = scanpath_similarity(enc, ret, match_on="image",
                                 method="multimatch", screensize=(500, 500),
                                 permutations=1)

    # Should have _perm and _diff columns for each metric
    for metric in ["mm_vector", "mm_direction", "mm_length",
                   "mm_position", "mm_duration", "mm_position_emd"]:
        assert f"{metric}_perm" in result.columns, f"Missing {metric}_perm"
        assert f"{metric}_diff" in result.columns, f"Missing {metric}_diff"


# ---- Bug 4: kwargs forwarded (screensize, time_samples) ----

# ---- Short scanpath edge case: mm_position_emd in NA return ----

def test_multimatch_short_scanpath_returns_all_six_keys():
    rng = np.random.default_rng(99)
    fg_short = fixation_group([0, 100], [0, 100], [100, 100], [1.0, 2.0])
    fg_normal = fixation_group([0, 50, 100], [0, 50, 100], [100, 100, 100],
                               [1.0, 2.0, 3.0])
    from peyesim.multimatch import multi_match
    result = multi_match(scanpath(fg_short), scanpath(fg_normal),
                         screensize=(500, 500))
    expected = ["mm_vector", "mm_direction", "mm_length",
                "mm_position", "mm_duration", "mm_position_emd"]
    for k in expected:
        assert k in result, f"Missing key: {k}"
    assert len(result) == 6


def test_scanpath_similarity_short_scanpath_with_permutations():
    """Short scanpaths should produce all-NaN columns without crashing."""
    rows = []
    for phase in ["enc", "ret"]:
        for img in ["img1", "img2"]:
            rows.append(pd.DataFrame({
                "x": [100.0, 200.0], "y": [100.0, 200.0],
                "onset": [1.0, 2.0], "duration": [100.0, 100.0],
                "image": img, "phase": phase,
            }))
    df = pd.concat(rows, ignore_index=True)
    eyetab = eye_table("x", "y", "duration", "onset",
                       groupvar=["phase", "image"], data=df)
    enc = add_scanpath(eyetab[eyetab["phase"] == "enc"].reset_index(drop=True))
    ret = add_scanpath(eyetab[eyetab["phase"] == "ret"].reset_index(drop=True))

    result = scanpath_similarity(enc, ret, match_on="image",
                                 method="multimatch", screensize=(500, 500),
                                 permutations=1)
    mm_cols = [c for c in result.columns if c.startswith("mm_")]
    assert len(mm_cols) == 18  # 6 metrics x 3 (raw, perm, diff)


# ---- Bug 4: kwargs forwarded (screensize, time_samples) ----

def test_similarity_forwards_screensize():
    rng = np.random.default_rng(7)
    fg1 = fixation_group(rng.uniform(0, 500, 10), rng.uniform(0, 500, 10),
                         rng.uniform(100, 300, 10), np.arange(1, 11, dtype=float))
    fg2 = fixation_group(rng.uniform(0, 500, 10), rng.uniform(0, 500, 10),
                         rng.uniform(100, 300, 10), np.arange(1, 11, dtype=float))

    sp1, sp2 = scanpath(fg1), scanpath(fg2)
    result = similarity(sp1, sp2, method="multimatch", screensize=(1000, 1000))
    assert isinstance(result, dict)
    assert "mm_vector" in result


def test_similarity_forwards_time_samples():
    rng = np.random.default_rng(7)
    fg1 = fixation_group(rng.uniform(0, 500, 10), rng.uniform(0, 500, 10),
                         rng.uniform(100, 300, 10),
                         np.cumsum(rng.uniform(10, 50, 10)))
    fg2 = fixation_group(rng.uniform(0, 500, 10), rng.uniform(0, 500, 10),
                         rng.uniform(100, 300, 10),
                         np.cumsum(rng.uniform(10, 50, 10)))

    result = similarity(fg1, fg2, method="overlap",
                        time_samples=np.arange(0, 200, 10))
    assert isinstance(result, float)


# ---- Ported from R: identical fixation groups -> overlap similarity of 1.0 ----

def test_fixation_similarity_overlap_identical_returns_one():
    """Identical fixation groups should produce high overlap similarity."""
    rng = np.random.default_rng(55)
    rows = []
    for trial in ["t1", "t2", "t3"]:
        nfix = 8
        # Use contiguous fixations that cover time range densely
        durations = np.full(nfix, 100.0)
        onsets = np.arange(nfix, dtype=float) * 100  # 0, 100, 200, ...
        rows.append(pd.DataFrame({
            "x": rng.uniform(0, 500, nfix),
            "y": rng.uniform(0, 500, nfix),
            "onset": onsets,
            "duration": durations,
            "trial": trial,
        }))
    df = pd.concat(rows, ignore_index=True)
    tab = eye_table("x", "y", "duration", "onset", groupvar="trial", data=df)

    # Sample only within the fixation time window so all samples are covered
    time_samples = np.arange(0, 800, 20)
    result = fixation_similarity(tab, tab, match_on="trial", method="overlap",
                                 time_samples=time_samples)
    assert "eye_sim" in result.columns
    # Identical data compared to itself must yield perfect overlap
    assert all(result["eye_sim"] == 1.0)
