#!/usr/bin/env python3
"""Regenerate peyesim's bundled datasets from R eyesim, with exact doubles.

peyesim ships ``src/peyesim/data/<name>.json.gz`` because ``pyreadr`` cannot
read the nested fixation-group list columns of eyesim's ``.rda`` files. This
script loads the R package with ``pkgload::load_all()`` (from
``$EYESIM_SOURCE_DIR``, default ``./eyesim``), writes every fixation with
``%.17g`` so no digits are lost, and rebuilds the JSON companions. The output
is deterministic (gzip mtime 0), so re-running it on unchanged data gives
byte-identical files; ``tests/test_parity_datasets.py`` checks the result
against R.

Usage: python tools/regenerate_datasets.py [wynn_study wynn_test]
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "src" / "peyesim" / "data"
DEFAULT = ["wynn_study", "wynn_test"]

R_CODE = r"""
args <- commandArgs(TRUE)
outdir <- args[1]
suppressMessages(pkgload::load_all(Sys.getenv("EYESIM_SOURCE_DIR", "eyesim"), quiet = TRUE))
for (n in args[-1]) {
  d <- get(n)
  meta <- as.data.frame(dplyr::ungroup(d)[, setdiff(names(d), "fixgroup")])
  fac <- vapply(meta, is.factor, logical(1))
  meta[fac] <- lapply(meta[fac], as.character)
  jsonlite::write_json(list(columns = names(d), meta = meta), file.path(outdir, paste0(n, ".meta.json")),
                       dataframe = "columns", digits = NA, na = "null")
  rows <- do.call(rbind, lapply(seq_along(d$fixgroup), function(k) {
    g <- d$fixgroup[[k]]
    if (is.null(g) || !nrow(g)) return(NULL)
    data.frame(row = k, index = g$index,
               x = sprintf("%.17g", g$x), y = sprintf("%.17g", g$y),
               duration = sprintf("%.17g", g$duration), onset = sprintf("%.17g", g$onset),
               group_index = sprintf("%.17g", g$group_index))
  }))
  utils::write.csv(rows, file.path(outdir, paste0(n, ".fix.csv")), row.names = FALSE)
}
"""


def _whole(text: str) -> int:
    value = float(text)
    if not value.is_integer():
        raise ValueError(f"expected an integer value, got {text}")
    return int(value)


def regenerate(names: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "dump.R"
        script.write_text(R_CODE)
        subprocess.run(["Rscript", str(script), tmp, *names], check=True)
        for name in names:
            spec = json.loads((Path(tmp) / f"{name}.meta.json").read_text())
            meta = spec["meta"]
            columns = [c for c in spec["columns"] if c != "fixgroup"]
            n_rows = len(meta[columns[0]]) if columns else 0
            fixations: dict[int, list[dict]] = {}
            with open(Path(tmp) / f"{name}.fix.csv", newline="") as handle:
                for rec in csv.DictReader(handle):
                    fixations.setdefault(int(rec["row"]), []).append({
                        "index": _whole(rec["index"]),
                        "x": float(rec["x"]),
                        "y": float(rec["y"]),
                        "duration": _whole(rec["duration"]),
                        "onset": _whole(rec["onset"]),
                        "group_index": float(rec["group_index"]),
                    })
            rows = []
            for i in range(n_rows):
                row = {col: meta[col][i] for col in spec["columns"] if col != "fixgroup"}
                row["fixgroup"] = fixations.get(i + 1, [])
                # keep R's column order, with fixgroup where R has it
                rows.append({col: row[col] for col in spec["columns"]})
            buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as gz:
                gz.write(json.dumps(rows, separators=(",", ":")).encode("utf-8"))
            (DATA / f"{name}.json.gz").write_bytes(buffer.getvalue())
            print(f"{name}: {n_rows} rows, {sum(len(v) for v in fixations.values())} fixations")


if __name__ == "__main__":
    regenerate(sys.argv[1:] or DEFAULT)
