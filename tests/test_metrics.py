import numpy as np
import pytest

from dsp import metrics


def test_si_sdr_identity_is_large():
    x = np.random.default_rng(0).standard_normal(4000)
    assert metrics.si_sdr(x, x) > 100


def test_si_sdr_invariant_to_scale():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4000)
    estimate = x + 0.1 * rng.standard_normal(4000)
    assert metrics.si_sdr(x, 3.0 * estimate) == pytest.approx(metrics.si_sdr(x, estimate), abs=1e-6)


def test_snr_db_known_case():
    x = np.ones(1000)
    noise = 0.1 * np.random.default_rng(0).standard_normal(1000)
    measured = metrics.snr_db(x, x + noise)
    assert 18 < measured < 22


def test_erle_negative_when_output_is_quieter():
    d = np.random.default_rng(0).standard_normal(2000)
    assert metrics.erle_db(d, 0.1 * d) > 15


def test_segmental_snr_finite():
    x = np.random.default_rng(0).standard_normal(8000)
    value = metrics.segmental_snr(x, x + 0.1 * x)
    assert np.isfinite(value)
