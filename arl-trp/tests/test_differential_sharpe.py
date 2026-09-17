import pytest

from arl_trp.rewards.differential_sharpe import (
    DifferentialSharpeReward,
)


def test_initial_state():
    estimator = DifferentialSharpeReward()

    assert estimator.count == 0
    assert estimator.mean == 0.0
    assert estimator.variance == 0.0


def test_first_observation_returns_zero():
    estimator = DifferentialSharpeReward(
        min_observations=3
    )

    result = estimator.update(10.0)

    assert result == 0.0
    assert estimator.count == 1


def test_burn_in_returns_zero():
    estimator = DifferentialSharpeReward(
        min_observations=3
    )

    assert estimator.update(1.0) == 0.0
    assert estimator.update(2.0) == 0.0


def test_signal_activates_after_minimum_observations():
    estimator = DifferentialSharpeReward(
        min_observations=3
    )

    estimator.update(1.0)
    estimator.update(2.0)

    result = estimator.update(3.0)

    assert result != 0.0


def test_statistics_are_updated():
    estimator = DifferentialSharpeReward(
        min_observations=3
    )

    estimator.update(1.0)
    estimator.update(2.0)
    estimator.update(3.0)

    assert estimator.count == 3
    assert estimator.mean == pytest.approx(2.0)
    assert estimator.variance > 0.0


def test_variance_floor_prevents_division_problem():
    estimator = DifferentialSharpeReward(
        min_observations=2,
        variance_floor=1e-8,
    )

    estimator.update(1.0)

    result = estimator.update(1.0)

    assert result == 0.0


def test_reset_clears_state():
    estimator = DifferentialSharpeReward(
        min_observations=2
    )

    estimator.update(1.0)
    estimator.update(2.0)

    estimator.reset()

    assert estimator.count == 0
    assert estimator.mean == 0.0
    assert estimator.variance == 0.0
