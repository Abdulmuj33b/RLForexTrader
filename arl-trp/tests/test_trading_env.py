import numpy as np
from datetime import datetime, timedelta

import pytest

from arl_trp.environment.observation import ObservationNormalizer
from arl_trp.domain.market import MarketSnapshot
from arl_trp.domain.order import Action
from arl_trp.environment.trading_env import TradingEnvironment
from arl_trp.execution.simulator import (
    ExecutionConfig,
    ExecutionSimulator,
)


def make_market_data():
    """
    Create a small deterministic market sequence.

    Bid/ask spread = 1.0 throughout.
    """

    prices = [
        (100.0, 101.0),
        (102.0, 103.0),
        (104.0, 105.0),
        (103.0, 104.0),
    ]

    data = []

    for i, (bid, ask) in enumerate(prices):
        data.append(
            MarketSnapshot(
                timestamp=datetime(2026, 1, 1)
                + timedelta(minutes=i),
                open=bid,
                high=ask,
                low=bid - 1.0,
                close=bid + 0.5,
                volume=1000.0,
                bid=bid,
                ask=ask,
            )
        )

    return data


def make_long_market_data(length=502):
    """
    Create a deterministic market sequence long enough
    to cross the 500-observation reward burn-in boundary.
    """

    data = []

    for i in range(length):
        bid = 100.0 + i
        ask = bid + 1.0

        data.append(
            MarketSnapshot(
                timestamp=datetime(2026, 1, 1)
                + timedelta(minutes=i),
                open=bid,
                high=ask,
                low=bid - 1.0,
                close=bid + 0.5,
                volume=1000.0,
                bid=bid,
                ask=ask,
            )
        )

    return data


def make_test_normalizer():
    """
    Create a deterministic fitted normalizer for tests.

    This is test-only normalization data.

    Production normalization statistics must come from the
    training-fold pipeline and must never be fitted inside
    TradingEnvironment.
    """

    observations = np.array(
        [
            [
                0.00,
                0.01,
                -0.01,
                0.00,
                0.01,
                100.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            [
                0.00,
                0.01,
                -0.01,
                0.01,
                0.01,
                110.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            [
                0.00,
                0.01,
                -0.01,
                0.02,
                0.01,
                120.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
        ],
        dtype=np.float32,
    )

    return ObservationNormalizer().fit(observations)


def make_environment(
    initial_balance=10_000.0,
    commission=0.0,
    slippage=0.0,
):
    execution = ExecutionSimulator(
        ExecutionConfig(
            slippage=slippage,
            commission_per_unit=commission,
        )
    )

    env = TradingEnvironment(
        market_data=make_market_data(),
        initial_balance=initial_balance,
        execution=execution,
        quantity=1.0,
        reward_beta=0.0,
        observation_window=1,
    )

    env.set_observation_normalizer(make_test_normalizer())

    return env


def test_environment_reset_resets_reward_estimator():
    execution = ExecutionSimulator(
        ExecutionConfig()
    )

    env = TradingEnvironment(
        market_data=make_long_market_data(),
        initial_balance=10_000.0,
        execution=execution,
        quantity=1.0,
    )

    env.set_observation_normalizer(make_test_normalizer())

    env.reset()

    for _ in range(10):
        env.step(Action.HOLD)

    assert (
        env.reward.differential_sharpe.count
        == 10
    )

    env.reset()

    assert (
        env.reward.differential_sharpe.count
        == 0
    )

    assert (
        env.reward.differential_sharpe.mean
        == 0.0
    )

    assert (
        env.reward.differential_sharpe.variance
        == 0.0
    )


def test_reset_starts_flat():
    env = make_environment()

    observation, _ = env.reset()

    assert env.index == 0
    assert env.position.is_flat
    assert env.accounting.balance == 10_000.0
    assert env.accounting.peak_equity == 10_000.0
    assert observation.shape == (10,)


def test_long_opens_at_ask():
    env = make_environment()

    env.reset()

    _, _, _, _, info = env.step(Action.LONG)

    transition = info["transition"]

    assert transition.fill_price == 101.0
    assert transition.position_before == 0.0
    assert transition.position_after == 1.0


def test_short_opens_at_bid():
    env = make_environment()

    env.reset()

    _, _, _, _, info = env.step(Action.SHORT)

    transition = info["transition"]

    assert transition.fill_price == 100.0
    assert transition.position_before == 0.0
    assert transition.position_after == -1.0


def test_hold_maintains_long_position():
    env = make_environment()

    env.reset()

    env.step(Action.LONG)

    _, _, _, _, info = env.step(Action.HOLD)

    transition = info["transition"]

    assert transition.position_before == 1.0
    assert transition.position_after == 1.0


def test_hold_maintains_short_position():
    env = make_environment()

    env.reset()

    env.step(Action.SHORT)

    _, _, _, _, info = env.step(Action.HOLD)

    transition = info["transition"]

    assert transition.position_before == -1.0
    assert transition.position_after == -1.0


def test_close_long_realizes_pnl():
    env = make_environment()

    env.reset()

    env.step(Action.LONG)

    # LONG entered at ask = 101.
    # Next market has bid = 102.
    env.step(Action.HOLD)

    _, reward, _, _, info = env.step(Action.CLOSE)

    transition = info["transition"]

    assert transition.position_before == 1.0
    assert transition.position_after == 0.0
    assert transition.fill_price == 104.0
    assert transition.gross_pnl == 3.0
    assert transition.transaction_cost == 0.0
    assert transition.net_pnl == 3.0
    assert reward == 0.0


def test_close_short_realizes_pnl():
    env = make_environment()

    env.reset()

    env.step(Action.SHORT)

    # SHORT entered at bid = 100.
    # Close occurs on the third market bar:
    # ask = 105.
    env.step(Action.HOLD)

    _, reward, _, _, info = env.step(Action.CLOSE)

    transition = info["transition"]

    assert transition.position_before == -1.0
    assert transition.position_after == 0.0
    assert transition.fill_price == 105.0
    assert transition.gross_pnl == -5.0
    assert transition.transaction_cost == 0.0
    assert transition.net_pnl == -5.0


def test_opposite_action_closes_long_instead_of_reversing():
    env = make_environment()

    env.reset()

    env.step(Action.LONG)

    _, _, _, _, info = env.step(Action.SHORT)

    transition = info["transition"]

    assert transition.position_before == 1.0
    assert transition.position_after == 0.0


def test_opposite_action_closes_short_instead_of_reversing():
    env = make_environment()

    env.reset()

    env.step(Action.SHORT)

    _, _, _, _, info = env.step(Action.LONG)

    transition = info["transition"]

    assert transition.position_before == -1.0
    assert transition.position_after == 0.0


def test_commission_reduces_equity_when_opening():
    env = make_environment(
        commission=2.0
    )

    env.reset()

    _, _, _, _, info = env.step(Action.LONG)

    transition = info["transition"]

    assert transition.transaction_cost == 2.0
    assert transition.net_pnl == -2.0
    assert env.accounting.balance == 9_998.0


def test_spread_creates_immediate_entry_loss():
    env = make_environment()

    env.reset()

    env.step(Action.LONG)

    # Entry = ask 101.
    # Next bid = 102, so after price movement the
    # unrealized P/L is +1.
    assert env.accounting.unrealized_pnl(
        env.position,
        env.market_data[1],
    ) == 1.0


def test_reward_is_equity_change():
    env = make_environment()

    env.reset()

    _, reward, _, _, info = env.step(
        Action.LONG
    )

    transition = info["transition"]

    assert reward == pytest.approx(
        transition.equity_after
        - transition.equity_before
    )


def test_trade_transition_contains_audit_information():
    env = make_environment()

    env.reset()

    _, _, _, _, info = env.step(Action.LONG)

    transition = info["transition"]

    assert transition.timestamp == (
        env.market_data[0].timestamp
    )

    assert transition.action == Action.LONG
    assert transition.position_before == 0.0
    assert transition.position_after == 1.0
    assert transition.fill_price == 101.0


def test_environment_terminates_at_dataset_end():
    env = make_environment()

    env.reset()

    terminated = False

    while not terminated:
        _, _, terminated, _, _ = env.step(
            Action.HOLD
        )

    assert terminated
    assert env.index == len(env.market_data) - 1


def test_reset_restores_environment_state():
    env = make_environment()

    env.reset()

    env.step(Action.LONG)

    env.reset()

    assert env.index == 0
    assert env.position.is_flat
    assert env.accounting.balance == 10_000.0
    assert env.accounting.peak_equity == 10_000.0


def test_environment_is_deterministic():
    env1 = make_environment()
    env2 = make_environment()

    env1.reset()
    env2.reset()

    actions = [
        Action.LONG,
        Action.HOLD,
        Action.CLOSE,
    ]

    results1 = []
    results2 = []

    for action in actions:
        obs1, reward1, term1, trunc1, info1 = (
            env1.step(action)
        )

        obs2, reward2, term2, trunc2, info2 = (
            env2.step(action)
        )

        results1.append(
            (
                obs1,
                reward1,
                term1,
                trunc1,
                info1["transition"],
            )
        )

        results2.append(
            (
                obs2,
                reward2,
                term2,
                trunc2,
                info2["transition"],
            )
        )

    for result1, result2 in zip(
        results1,
        results2,
    ):
        obs1, reward1, term1, trunc1, transition1 = result1
        obs2, reward2, term2, trunc2, transition2 = result2

        assert np.array_equal(obs1, obs2)
        assert reward1 == reward2
        assert term1 == term2
        assert trunc1 == trunc2
        assert transition1 == transition2


def test_reward_burn_in_activates_after_500_observations():
    execution = ExecutionSimulator(
        ExecutionConfig()
    )

    env = TradingEnvironment(
        market_data=make_long_market_data(),
        initial_balance=10_000.0,
        execution=execution,
        quantity=1.0,
    )

    env.set_observation_normalizer(make_test_normalizer())

    env.reset()

    # Open a long position.
    env.step(Action.LONG)

    # The first reward observation has already been consumed.
    assert env.reward.differential_sharpe.count == 1

    # Consume observations 2 through 499.
    for _ in range(498):
        env.step(Action.HOLD)

    assert (
        env.reward.differential_sharpe.count
        == 499
    )

    # Observation 500 is where the differential-Sharpe
    # component becomes active.
    _, reward, _, _, info = env.step(
        Action.HOLD
    )

    transition = info["transition"]

    economic_reward = (
        transition.equity_after
        - transition.equity_before
    )

    assert (
        env.reward.differential_sharpe.count
        == 500
    )

    assert reward != pytest.approx(
        economic_reward
    )