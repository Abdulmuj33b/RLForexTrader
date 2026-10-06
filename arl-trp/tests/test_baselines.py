from datetime import datetime, timedelta

import numpy as np
import pytest

from arl_trp.domain.market import MarketSnapshot
from arl_trp.domain.order import Action
from arl_trp.evaluation.baselines import (
    BaselineEvaluator,
    BuyAndHoldBaseline,
    ConstrainedRandomBaseline,
    FlatBaseline,
    MomentumBaseline,
)
from arl_trp.execution.simulator import (
    ExecutionConfig,
    ExecutionSimulator,
)
from arl_trp.environment.observation import ObservationNormalizer
from arl_trp.environment.trading_env import TradingEnvironment


def make_market_data(length: int = 100):
    data = []

    for index in range(length):
        bid = 100.0 + index
        ask = bid + 1.0

        data.append(
            MarketSnapshot(
                timestamp=datetime(2026, 1, 1)
                + timedelta(minutes=index),
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


def make_environment():
    observations = np.array(
        [
            [
                0.0,
                0.01,
                -0.01,
                0.0,
                0.01,
                100.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            [
                0.0,
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
                0.0,
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

    normalizer = ObservationNormalizer().fit(
        observations
    )

    environment = TradingEnvironment(
        market_data=make_market_data(),
        initial_balance=10_000.0,
        execution=ExecutionSimulator(
            ExecutionConfig()
        ),
        quantity=1.0,
    )

    environment.set_observation_normalizer(
        normalizer
    )

    return environment


def test_flat_baseline_always_holds():
    policy = FlatBaseline()

    for _ in range(10):
        action, _ = policy.predict(None)

        assert action == Action.HOLD


def test_buy_and_hold_enters_once_then_holds():
    policy = BuyAndHoldBaseline()

    first_action, _ = policy.predict(None)
    second_action, _ = policy.predict(None)
    third_action, _ = policy.predict(None)

    assert first_action == Action.LONG
    assert second_action == Action.HOLD
    assert third_action == Action.HOLD


def test_buy_and_hold_reset_restarts_policy():
    policy = BuyAndHoldBaseline()

    policy.predict(None)
    policy.reset()

    action, _ = policy.predict(None)

    assert action == Action.LONG


def test_momentum_uses_positive_return_for_long():
    environment = make_environment()
    policy = MomentumBaseline(lookback_bars=5)

    environment.reset()

    environment.index = 10

    action = policy.action(environment)

    assert action == Action.LONG


def test_momentum_holds_during_warmup():
    environment = make_environment()
    policy = MomentumBaseline(lookback_bars=20)

    environment.reset()

    environment.index = 10

    action = policy.action(environment)

    assert action == Action.HOLD


def test_momentum_rejects_invalid_lookback():
    with pytest.raises(ValueError):
        MomentumBaseline(lookback_bars=0)


def test_random_probabilities_must_sum_to_one():
    with pytest.raises(ValueError):
        ConstrainedRandomBaseline(
            hold_probability=0.5,
            long_probability=0.5,
            short_probability=0.5,
            close_probability=0.5,
        )


def test_random_baseline_is_reproducible():
    policy_a = ConstrainedRandomBaseline(seed=42)
    policy_b = ConstrainedRandomBaseline(seed=42)

    actions_a = [
        policy_a.predict(None)[0]
        for _ in range(50)
    ]

    actions_b = [
        policy_b.predict(None)[0]
        for _ in range(50)
    ]

    assert actions_a == actions_b


def test_random_baseline_changes_with_different_seed():
    policy_a = ConstrainedRandomBaseline(seed=42)
    policy_b = ConstrainedRandomBaseline(seed=43)

    actions_a = [
        policy_a.predict(None)[0]
        for _ in range(50)
    ]

    actions_b = [
        policy_b.predict(None)[0]
        for _ in range(50)
    ]

    assert actions_a != actions_b


def test_baseline_evaluator_uses_same_environment_accounting():
    environment = make_environment()
    evaluator = BaselineEvaluator()
    policy = FlatBaseline()

    result = evaluator.evaluate(
        policy=policy,
        environment=environment,
    )

    assert result.steps == 36
    assert result.equity_curve.ndim == 1
    assert result.equity_curve.size == 37
    assert result.trade_pnls.size == 0
    assert np.all(
        result.actions == int(Action.HOLD)
    )


def test_buy_and_hold_baseline_enters_through_environment():
    environment = make_environment()
    evaluator = BaselineEvaluator()
    policy = BuyAndHoldBaseline()

    result = evaluator.evaluate(
        policy=policy,
        environment=environment,
    )

    assert result.steps == 36
    assert result.actions[0] == int(Action.LONG)

    # The position remains open because the baseline never
    # explicitly closes it.
    assert environment.position.is_long


def test_baseline_evaluator_can_run_random_policy():
    environment = make_environment()
    evaluator = BaselineEvaluator()

    policy = ConstrainedRandomBaseline(
        seed=42,
    )

    result = evaluator.evaluate(
        policy=policy,
        environment=environment,
    )

    assert result.steps == 36
    assert result.equity_curve.size == 37
    assert result.actions.size == 36