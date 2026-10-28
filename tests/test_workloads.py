import numpy as np
import pytest

from autoscaler.env.workloads import SHAPES, make_workload


@pytest.mark.parametrize("shape", SHAPES)
def test_same_seed_same_workload(shape):
    a = make_workload(shape, 1440, np.random.default_rng(3))
    b = make_workload(shape, 1440, np.random.default_rng(3))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("shape", SHAPES)
def test_different_seed_different_workload(shape):
    a = make_workload(shape, 1440, np.random.default_rng(3))
    b = make_workload(shape, 1440, np.random.default_rng(4))
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("steps", [1440, 360])
def test_values_in_range(shape, steps):
    w = make_workload(shape, steps, np.random.default_rng(1))
    assert w.shape == (steps,)
    assert w.min() >= 30.0 and w.max() <= 460.0


def test_daily_ramp_burst_landmarks():
    w = make_workload("daily_ramp_burst", 1440, np.random.default_rng(0))
    # burst at step ~1050 is the highest region of the day
    assert abs(int(np.argmax(np.convolve(w, np.ones(15) / 15, "same"))) - 1050) < 40


def test_step_jump():
    w = make_workload("step", 1440, np.random.default_rng(0))
    assert w[:500].mean() < 150 and w[700:].mean() > 300


def test_unknown_shape():
    with pytest.raises(ValueError):
        make_workload("nope", 10, np.random.default_rng(0))
