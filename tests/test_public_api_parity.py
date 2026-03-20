import numpy as np
import pandas as pd
import pytest

from peyesim import (
    EyeTable,
    as_eye_table,
    center,
    coords,
    density_matrix,
    fixation_group,
    gen_density,
    get_density,
    normalize,
    rep_fixations,
    rescale,
    sample_fixations,
    template_sample,
)


def test_fixation_wrapper_exports_dispatch_to_fixation_group_methods():
    fg = fixation_group(
        x=[10, 20],
        y=[20, 40],
        onset=[0, 100],
        duration=[100, 200],
    )

    np.testing.assert_allclose(coords(fg), np.array([[10, 20], [20, 40]]))

    centered = center(fg, origin=(10, 20))
    np.testing.assert_allclose(centered["x"], [0, 10])
    np.testing.assert_allclose(centered["y"], [0, 20])

    normalized = normalize(fg, xbounds=(0, 20), ybounds=(0, 40))
    np.testing.assert_allclose(normalized["x"], [0.5, 1.0])
    np.testing.assert_allclose(normalized["y"], [0.5, 1.0])

    scaled = rescale(fg, sx=2.0, sy=0.5)
    np.testing.assert_allclose(scaled["x"], [20, 40])
    np.testing.assert_allclose(scaled["y"], [10, 20])

    replicated = rep_fixations(fg, resolution=1)
    assert len(replicated) == 300


def test_sample_fixations_fast_path_matches_r_semantics_before_first_onset():
    fg = fixation_group(
        x=[5, 15],
        y=[10, 30],
        onset=[100, 200],
        duration=[100, 100],
    )

    sampled = sample_fixations(fg, time=np.array([50, 100, 150]), fast=True)

    assert np.isnan(sampled["x"].iloc[0])
    assert np.isnan(sampled["y"].iloc[0])
    assert sampled["x"].iloc[1] == pytest.approx(5)
    assert sampled["x"].iloc[2] == pytest.approx(5)


def test_concat_fixation_groups_rejects_invalid_inputs():
    fg = fixation_group(x=[1], y=[2], onset=[0], duration=[100])

    from peyesim import concat_fixation_groups

    with pytest.raises(TypeError, match="must be fixation_group"):
        concat_fixation_groups(fg, pd.DataFrame({"x": [1]}))


def test_as_eye_table_reapplies_class_and_preserves_origin():
    raw = pd.DataFrame(
        {
            "trial": ["A"],
            "fixgroup": [fixation_group(x=[1], y=[2], onset=[0], duration=[100])],
        }
    )
    raw.attrs["origin"] = (320.0, 240.0)

    tab = as_eye_table(raw)

    assert isinstance(tab, EyeTable)
    assert tab.origin == (320.0, 240.0)


def test_get_density_and_density_matrix_extract_raw_matrix():
    z = np.array([[1.0, 2.0], [3.0, 4.0]])
    dens = gen_density(np.array([0.0, 1.0]), np.array([0.0, 1.0]), z)

    np.testing.assert_allclose(get_density(dens), z)
    np.testing.assert_allclose(density_matrix(dens), z)


def test_template_sample_adds_sampled_density_column():
    z = np.array([[0.1, 0.2], [0.3, 0.9]])
    dens = gen_density(np.array([0.0, 1.0]), np.array([0.0, 1.0]), z)
    fg = fixation_group(x=[1.0], y=[1.0], onset=[0.0], duration=[100.0])
    source = pd.DataFrame({"density": [dens], "fixgroup": [fg]})

    sampled = template_sample(source, template="density", outcol="sample_out")

    assert "sample_out" in sampled.columns
    assert sampled["sample_out"].iloc[0]["z"].iloc[0] == pytest.approx(0.9)
