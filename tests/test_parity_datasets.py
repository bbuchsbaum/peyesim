"""The bundled wynn_study / wynn_test data hold exactly R eyesim's data.

R loads the datasets through ``pkgload::load_all()`` and reports, per dataset,
the column names and classes, the grouping, every non-fixation column, the
number of fixations per row, and an md5 digest of every fixation written as
``row,index,x,y,duration,onset,group_index`` with ``%.17g``. Python writes the
same text from ``peyesim.load_dataset()``, so equal digests mean the
coordinates agree bit for bit, in the same order.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from peyesim import FixationGroup, load_dataset
from tests.r_oracle import requires_r, run_r

R_CODE = r"""
canon <- function(d) {
  lines <- unlist(lapply(seq_along(d$fixgroup), function(k) {
    g <- d$fixgroup[[k]]
    if (!nrow(g)) return(character(0))
    sprintf("%d,%s,%s,%s,%s,%s,%s", k, sprintf("%.17g", as.numeric(g$index)), sprintf("%.17g", g$x),
            sprintf("%.17g", g$y), sprintf("%.17g", as.numeric(g$duration)),
            sprintf("%.17g", as.numeric(g$onset)), sprintf("%.17g", as.numeric(g$group_index)))
  }))
  f <- tempfile()
  writeLines(lines, f)
  unname(tools::md5sum(f))
}
info <- function(d) {
  meta <- as.data.frame(dplyr::ungroup(d))[, setdiff(names(d), "fixgroup"), drop = FALSE]
  list(names = names(d), classes = lapply(d, function(v) class(v)[1]),
       group_vars = dplyr::group_vars(d),
       meta = lapply(meta, as.character),
       fg_classes = unique(vapply(d$fixgroup, function(g) paste(class(g), collapse = "/"), "")),
       fg_names = unique(vapply(d$fixgroup, function(g) paste(names(g), collapse = ","), "")),
       nfix = vapply(d$fixgroup, nrow, integer(1)),
       md5 = canon(d))
}
out <- list(wynn_study = info(wynn_study), wynn_test = info(wynn_test))
"""


@pytest.fixture(scope="module")
def r():
    return run_r(R_CODE)


def _canon_md5(frame: pd.DataFrame) -> str:
    lines = []
    for k, fg in enumerate(frame["fixgroup"], start=1):
        f = fg.to_pandas(copy=False)
        cols = [f[c].to_numpy(dtype=float) for c in ("index", "x", "y", "duration", "onset", "group_index")]
        for vals in zip(*cols):
            lines.append(f"{k}," + ",".join("%.17g" % v for v in vals))
    return hashlib.md5(("\n".join(lines) + "\n").encode()).hexdigest()


def _as_r_character(series: pd.Series) -> list[str]:
    out = []
    for v in series:
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            out.append(str(int(v)))
        else:
            out.append(str(v))
    return out


@requires_r
@pytest.mark.parametrize("name", ["wynn_study", "wynn_test"])
def test_bundled_dataset_matches_r(r, name):
    expected = r[name]
    data = load_dataset(name)
    assert list(data.columns) == expected["names"]
    assert len(data) == len(expected["nfix"])
    # R groups both datasets by ImageNumber and Subject; peyesim returns a plain frame.
    assert expected["group_vars"] == ["ImageNumber", "Subject"]
    for col, values in expected["meta"].items():
        assert _as_r_character(data[col]) == values, col
    assert expected["fg_classes"] == "fixation_group/tbl_df/tbl/data.frame"
    assert expected["fg_names"] == "index,x,y,duration,onset,group_index"
    assert all(isinstance(fg, FixationGroup) for fg in data["fixgroup"])
    assert [len(fg) for fg in data["fixgroup"]] == expected["nfix"]
    assert _canon_md5(data) == expected["md5"]


@requires_r
def test_wynn_study_pools_the_four_study_viewings(r):
    # NEWS: a wynn_study row pools a participant's four study viewings, so onsets
    # restart within a row. Check that the bundled copy has the pooled layout.
    data = load_dataset("wynn_study")
    restarts = [int(np.sum(np.diff(fg["onset"].to_numpy()) < 0)) for fg in data["fixgroup"]]
    assert max(restarts) == 3 and np.mean(restarts) > 2.5
    assert len(data) == 12240 and len(set(data["ImageNumberS"])) == 12240
