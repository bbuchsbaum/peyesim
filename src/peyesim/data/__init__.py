"""Packaged example datasets from the R eyesim package."""

from __future__ import annotations

import gzip
import json
from importlib import resources
from pathlib import Path

import pandas as pd

from peyesim.fixations import FixationGroup


DATASETS = ("wynn_study", "wynn_test", "wynn_study_image")


def available_datasets() -> tuple[str, ...]:
    """Return the names of datasets bundled with ``peyesim``."""
    return DATASETS


def data_path(name: str) -> Path:
    """Return a filesystem path for a bundled ``.rda`` dataset."""
    if name not in DATASETS:
        valid = ", ".join(DATASETS)
        raise ValueError(f"Unknown dataset {name!r}. Valid datasets are: {valid}.")
    return Path(resources.files(__name__).joinpath(f"{name}.rda"))


def _json_data_path(name: str) -> Path:
    if name not in DATASETS:
        valid = ", ".join(DATASETS)
        raise ValueError(f"Unknown dataset {name!r}. Valid datasets are: {valid}.")
    return Path(resources.files(__name__).joinpath(f"{name}.json.gz"))


def _fixation_group_from_records(records) -> FixationGroup | None:
    if records is None:
        return None
    return FixationGroup(pd.DataFrame.from_records(records))


def _load_json_dataset(name: str):
    path = _json_data_path(name)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = json.load(handle)
    frame = pd.DataFrame.from_records(rows)
    if "fixgroup" in frame.columns:
        frame.loc[:, "fixgroup"] = frame["fixgroup"].map(_fixation_group_from_records)
    if name == "wynn_study_image":
        frame.attrs["origin"] = (400, 300)
    return frame


def load_dataset(name: str):
    """Load a bundled example dataset as a pandas ``DataFrame``.

    The R source data contain nested fixation-group list columns that
    ``pyreadr`` cannot currently parse, so peyesim ships Python-readable
    companion files for the bundled datasets. The original ``.rda`` files
    remain available through :func:`data_path` for users who need them.
    """
    json_path = _json_data_path(name)
    if json_path.exists():
        return _load_json_dataset(name)

    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError(
            "load_dataset() requires the optional dependency 'pyreadr'. "
            "Install it with `pip install peyesim[data]`."
        ) from exc

    result = pyreadr.read_r(str(data_path(name)))
    if name in result:
        return result[name]
    if len(result) == 1:
        return next(iter(result.values()))
    return result
