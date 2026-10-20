import numpy as np

from dsp.lms import lms, nlms, rls
from dsp.metrics import erle_db


def system_identification_setup(n=8000, order=16, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    h = rng.standard_normal(order) * 0.3
    d = np.convolve(x, h)[:n]
    d += 1e-3 * rng.standard_normal(n)
    return x, d, h


def test_lms_reduces_error():
    x, d, h = system_identification_setup()
    out = lms(x, d, mu=0.01, order=len(h))
    assert erle_db(d, out["output"]) > 10.0
    assert out["mse"][-1] < out["mse"][0]


def test_nlms_beats_lms_on_same_problem():
    x, d, h = system_identification_setup()
    out_lms = lms(x, d, mu=0.01, order=len(h))
    out_nlms = nlms(x, d, mu=0.5, order=len(h))
    assert erle_db(d, out_nlms["output"]) > erle_db(d, out_lms["output"])


def test_rls_converges_faster_than_lms():
    x, d, h = system_identification_setup()
    out_lms = lms(x, d, mu=0.01, order=len(h))
    out_rls = rls(x, d, lam=0.99, order=len(h))

    early = slice(2, 12)
    assert np.mean(out_rls["mse"][early]) < np.mean(out_lms["mse"][early])
    assert erle_db(d, out_rls["output"]) > 20.0


def test_nlms_is_stable_at_full_step_size():
    x, d, h = system_identification_setup()
    out = nlms(x, d, mu=1.0, order=len(h))
    assert np.all(np.isfinite(out["output"]))
    assert erle_db(d, out["output"]) > 10.0
