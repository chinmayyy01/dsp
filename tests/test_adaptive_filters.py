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


def test_freeze_mask_holds_weights():
    x, d, h = system_identification_setup(n=2000, order=8)
    freeze = np.zeros(len(x), dtype=bool)
    out = lms(x, d, mu=0.01, order=8, freeze=np.ones(len(x), dtype=bool))
    np.testing.assert_allclose(out["output"], d, atol=1e-12)

    split = len(x) // 2
    freeze[:split] = False
    freeze[split:] = True
    out = lms(x, d, mu=0.01, order=8, freeze=freeze)
    assert np.all(np.isfinite(out["weights"]))
    assert out["weights"].shape == (8,)


def test_filters_handle_signals_shorter_than_a_block():
    x = np.random.default_rng(0).standard_normal(100)
    d = x + 0.1 * np.random.default_rng(1).standard_normal(100)
    for fn in (lms, nlms, rls):
        out = fn(x, d, order=8)
        assert len(out["mse"]) == 1
        assert np.all(np.isfinite(out["output"]))
