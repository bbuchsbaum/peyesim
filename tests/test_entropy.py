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


# ── Ported from R test_fixation_entropy.R ─────────────────────────────


class TestFixationEntropyScalingInvariance:
    def test_fixation_entropy_scaling_invariance(self):
        """Entropy should be invariant to density scaling (multiplying z by a constant)."""
        z1 = np.array([[1, 2], [3, 4]], dtype=float)
        z2 = z1 * 10
        ed1 = EyeDensity(x=np.array([0, 1.0]), y=np.array([0, 1.0]), z=z1, sigma=1.0)
        ed2 = EyeDensity(x=np.array([0, 1.0]), y=np.array([0, 1.0]), z=z2, sigma=1.0)
        assert abs(fixation_entropy(ed1) - fixation_entropy(ed2)) < 1e-12


class TestGridEntropyAnalyticProbabilities:
    def test_grid_entropy_analytic_probabilities(self):
        """Grid entropy with 2 occupied cells of equal count should be log(2)/log(4) normalized."""
        fg = FixationGroup(pd.DataFrame({
            "x": [0.1, 0.2, 0.8, 0.85],
            "y": [0.1, 0.2, 0.8, 0.85],
            "onset": [0.0, 100.0, 200.0, 300.0],
            "duration": [100.0, 100.0, 100.0, 100.0],
        }))
        # 2x2 grid, 2 occupied cells with 2 fixations each -> p=[0.5, 0.5]
        # unnormalized entropy base 2: -2*(0.5*log2(0.5)) = 1.0
        ent = fixation_entropy(fg, method="grid", grid_size=(2, 2),
                               xbounds=(0, 1), ybounds=(0, 1),
                               normalize=False, base=2)
        assert abs(ent - 1.0) < 1e-10
        # normalized: 1.0 / log2(4) = 0.5
        ent_norm = fixation_entropy(fg, method="grid", grid_size=(2, 2),
                                    xbounds=(0, 1), ybounds=(0, 1),
                                    normalize=True, base=2)
        assert abs(ent_norm - 0.5) < 1e-10


class TestGridEntropyOrderInvariance:
    def test_grid_entropy_order_invariance(self):
        """Grid entropy should not depend on fixation order."""
        fg = FixationGroup(pd.DataFrame({
            "x": [0.1, 0.2, 0.8, 0.85, 0.55],
            "y": [0.1, 0.2, 0.8, 0.85, 0.45],
            "onset": [0.0, 100.0, 200.0, 300.0, 400.0],
            "duration": [100.0] * 5,
        }))
        fg_shuf = FixationGroup(pd.DataFrame({
            "x": [0.55, 0.8, 0.1, 0.85, 0.2],
            "y": [0.45, 0.8, 0.1, 0.85, 0.2],
            "onset": [0.0, 100.0, 200.0, 300.0, 400.0],
            "duration": [100.0] * 5,
        }))
        ent1 = fixation_entropy(fg, method="grid", grid_size=(3, 3),
                                xbounds=(0, 1), ybounds=(0, 1))
        ent2 = fixation_entropy(fg_shuf, method="grid", grid_size=(3, 3),
                                xbounds=(0, 1), ybounds=(0, 1))
        assert abs(ent1 - ent2) < 1e-10


class TestFixationEntropyNormalizedRange:
    def test_fixation_entropy_normalized_in_unit_range(self):
        """Normalized entropy should be in [0, 1]."""
        rng = np.random.default_rng(123)
        fg = FixationGroup(pd.DataFrame({
            "x": rng.uniform(0, 1, 25),
            "y": rng.uniform(0, 1, 25),
            "onset": np.arange(0, 2500, 100, dtype=float),
            "duration": np.full(25, 100.0),
        }))
        ent_grid = fixation_entropy(fg, method="grid", grid_size=(5, 5),
                                    xbounds=(0, 1), ybounds=(0, 1))
        ent_dens = fixation_entropy(fg, method="density", sigma=0.1,
                                    xbounds=(0, 1), ybounds=(0, 1),
                                    outdim=(25, 25))
        assert 0 <= ent_grid <= 1
        assert 0 <= ent_dens <= 1


class TestFixationGroupDensityEntropyMatchesEyeDensity:
    def test_fixation_group_density_entropy_matches_eye_density(self):
        """Computing entropy via fixation_group(method='density') should match direct EyeDensity entropy."""
        fg = FixationGroup(pd.DataFrame({
            "x": [0.2, 0.25, 0.75, 0.8],
            "y": [0.2, 0.25, 0.75, 0.8],
            "onset": [0.0, 100.0, 200.0, 300.0],
            "duration": [100.0] * 4,
        }))
        dens = eye_density(fg, sigma=0.08, xbounds=(0, 1), ybounds=(0, 1),
                           outdim=(30, 30))
        ent_fg = fixation_entropy(fg, method="density", sigma=0.08,
                                  xbounds=(0, 1), ybounds=(0, 1),
                                  outdim=(30, 30))
        ent_dens = fixation_entropy(dens)
        assert abs(ent_fg - ent_dens) < 1e-12
