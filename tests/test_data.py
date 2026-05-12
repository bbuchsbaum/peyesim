import pytest

import pandas as pd

from peyesim import FixationGroup, available_datasets, data_path, load_dataset


def test_packaged_r_datasets_are_available():
    assert available_datasets() == ("wynn_study", "wynn_test", "wynn_study_image")
    for name in available_datasets():
        path = data_path(name)
        assert path.name == f"{name}.rda"
        assert path.exists()
        assert path.stat().st_size > 0


def test_data_path_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        data_path("missing")


def test_load_dataset_uses_bundled_python_copy_when_pyreadr_missing(monkeypatch):
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "pyreadr":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    loaded = load_dataset("wynn_test")
    assert isinstance(loaded, pd.DataFrame)
    assert isinstance(loaded["fixgroup"].iloc[0], FixationGroup)


def test_load_dataset_with_pyreadr_when_available():
    loaded = load_dataset("wynn_test")
    assert isinstance(loaded, pd.DataFrame)
    assert loaded.shape[0] == 12240
    assert isinstance(loaded["fixgroup"].iloc[0], FixationGroup)


def test_load_dataset_preserves_eye_table_origin_metadata():
    loaded = load_dataset("wynn_study_image")
    assert loaded.attrs["origin"] == (400, 300)
