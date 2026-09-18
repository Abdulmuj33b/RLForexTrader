import pytest

from arl_trp.rewards.combined import CombinedReward
from arl_trp.rewards.differential_sharpe import (
    DifferentialSharpeReward,
)


def test_zero_beta_equals_economic_reward():
    reward = CombinedReward(
        beta=0.0,
        differential_sharpe=DifferentialSharpeReward(
            min_observations=3
        ),
    )

    result = reward.calculate(
        equity_before=100.0,
        equity_after=101.0,
    )

    assert result == pytest.approx(1.0)


def test_burn_in_uses_economic_reward_only():
    reward = CombinedReward(
        beta=1.0,
        differential_sharpe=DifferentialSharpeReward(
            min_observations=3
        ),
    )

    result = reward.calculate(
        equity_before=100.0,
        equity_after=101.0,
    )

    assert result == pytest.approx(1.0)


def test_combined_reward_includes_differential_signal():
    differential = DifferentialSharpeReward(
        min_observations=2
    )

    reward = CombinedReward(
        beta=1.0,
        differential_sharpe=differential,
    )

    first = reward.calculate(100.0, 101.0)
    second = reward.calculate(101.0, 103.0)

    assert first == pytest.approx(1.0)
    assert second != pytest.approx(2.0)


def test_beta_zero_removes_differential_component():
    reward = CombinedReward(
        beta=0.0,
        differential_sharpe=DifferentialSharpeReward(
            min_observations=2
        ),
    )

    reward.calculate(100.0, 101.0)

    result = reward.calculate(
        101.0,
        103.0,
    )

    assert result == pytest.approx(2.0)


def test_reset_resets_differential_sharpe():
    reward = CombinedReward(
        beta=0.5,
        differential_sharpe=DifferentialSharpeReward(
            min_observations=2
        ),
    )

    reward.calculate(100.0, 101.0)
    reward.calculate(101.0, 102.0)

    reward.reset()

    assert reward.differential_sharpe.count == 0
    assert reward.differential_sharpe.mean == 0.0
    assert reward.differential_sharpe.variance == 0.0


def test_negative_beta_is_rejected():
    with pytest.raises(ValueError):
        CombinedReward(beta=-1.0)
