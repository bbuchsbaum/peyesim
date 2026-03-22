"""Tests for the entropy module."""

import numpy as np
import pandas as pd
import pytest

from peyesim.entropy import entropy_from_mass, fixation_entropy
from peyesim.density import EyeDensity, EyeDensityMultiscale, eye_density
from peyesim.fixations import FixationGroup


def _make_fg(x, y, onset=None, duration=None):
    """Helper to build a FixationGroup from coordinate arrays."""
    n = len(x)
    if onset is None:
        onset = np.arange(n, dtype=float) * 200
    if duration is None:
        duration = np.full(n, 150.0)
    df = pd.DataFrame({"x": x, "y": y, "onset": onset, "duration": duration})
    return FixationGroup(df)


# ── entropy_from_mass ──────────────────────────────────────────────────


class TestEntropyFromMass:
    def test_uniform_normalized_equals_one(self):
        mass = np.ones(100)
        assert entropy_from_mass(mass, normalize=True) == pytest.approx(1.0)

    def test_concentrated_less_than_one(self):
        mass = np.zeros(100)
        mass[0] = 1.0
        assert entropy_from_mass(mass, normalize=True) == pytest.approx(0.0)

    def test_unnormalized_uniform(self):
        n = 64
        mass = np.ones(n)
        # Unnormalized entropy in nats == ln(n)
        assert entropy_from_mass(mass, normalize=False) == pytest.approx(np.log(n))

    def test_base2(self):
        mass = np.ones(8)
        assert entropy_from_mass(mass, normalize=False, base=2.0) == pytest.approx(3.0)

    def test_empty_returns_nan(self):
        assert np.isnan(entropy_from_mass(np.array([])))

    def test_all_zero_returns_nan(self):
        assert np.isnan(entropy_from_mass(np.zeros(5)))

    def test_non_finite_ignored(self):
        mass = np.array([1.0, 1.0, np.nan, np.inf, 1.0])
        # Three equal positive values -> normalized entropy = 1.0
        assert entropy_from_mass(mass, normalize=True) == pytest.approx(1.0)


# ── fixation_entropy on EyeDensity ─────────────────────────────────────


class TestFixationEntropyDensity:
    def test_uniform_density_normalized(self):
        """A perfectly uniform density map should have normalized entropy ~ 1.0."""
        n = 50
        x = np.linspace(0, 100, n)
        y = np.linspace(0, 100, n)
        z = np.ones((n, n)) / (n * n)
        ed = EyeDensity(x=x, y=y, z=z, sigma=10.0)
        ent = fixation_entropy(ed, normalize=True)
        assert ent == pytest.approx(1.0, abs=1e-6)

    def test_concentrated_density(self):
        """A density concentrated in one cell should have low entropy."""
        n = 50
        x = np.linspace(0, 100, n)
        y = np.linspace(0, 100, n)
        z = np.zeros((n, n))
        z[25, 25] = 1.0
        ed = EyeDensity(x=x, y=y, z=z, sigma=10.0)
        ent = fixation_entropy(ed, normalize=True)
        assert ent == pytest.approx(0.0)


# ── fixation_entropy on EyeDensityMultiscale ───────────────────────────


class TestFixationEntropyMultiscale:
    def _make_multiscale(self):
        n = 30
        x = np.linspace(0, 100, n)
        y = np.linspace(0, 100, n)
        z_uniform = np.ones((n, n)) / (n * n)
        z_peaked = np.zeros((n, n))
        z_peaked[15, 15] = 1.0
        ed1 = EyeDensity(x=x, y=y, z=z_uniform, sigma=10.0)
        ed2 = EyeDensity(x=x, y=y, z=z_peaked, sigma=30.0)
        return EyeDensityMultiscale(scales=[ed1, ed2])

    def test_mean_aggregation(self):
        edm = self._make_multiscale()
        ent = fixation_entropy(edm, normalize=True)
        # Mean of 1.0 and 0.0
        assert ent == pytest.approx(0.5, abs=1e-6)

    def test_none_aggregation(self):
        edm = self._make_multiscale()
        per_scale = fixation_entropy(edm, normalize=True, aggregate="none")
        assert isinstance(per_scale, list)
        assert len(per_scale) == 2
        assert per_scale[0] == pytest.approx(1.0, abs=1e-6)
        assert per_scale[1] == pytest.approx(0.0, abs=1e-6)


# ── fixation_entropy on FixationGroup ──────────────────────────────────


class TestFixationEntropyFixationGroup:
    def test_density_method_runs(self):
        rng = np.random.default_rng(42)
        fg = _make_fg(rng.uniform(100, 900, 50), rng.uniform(100, 700, 50))
        ent = fixation_entropy(fg, method="density")
        assert 0.0 < ent <= 1.0

    def test_grid_method_runs(self):
        rng = np.random.default_rng(42)
        fg = _make_fg(rng.uniform(100, 900, 50), rng.uniform(100, 700, 50))
        ent = fixation_entropy(fg, method="grid")
        assert 0.0 < ent <= 1.0

    def test_grid_concentrated(self):
        """Fixations all at the same point should yield low entropy."""
        fg = _make_fg(np.full(10, 500.0), np.full(10, 500.0))
        ent = fixation_entropy(fg, method="grid",
                               xbounds=(0, 1000), ybounds=(0, 1000))
        assert ent < 0.2

    def test_empty_fixation_group_returns_nan(self):
        fg = _make_fg(np.array([]), np.array([]))
        assert np.isnan(fixation_entropy(fg, method="density"))
        assert np.isnan(fixation_entropy(fg, method="grid"))

    def test_single_fixation_density_returns_nan(self):
        fg = _make_fg(np.array([500.0]), np.array([500.0]))
        assert np.isnan(fixation_entropy(fg, method="density"))

    def test_invalid_method_raises(self):
        fg = _make_fg(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        with pytest.raises(ValueError, match="Unknown method"):
            fixation_entropy(fg, method="invalid")

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            fixation_entropy("not_a_valid_input")
