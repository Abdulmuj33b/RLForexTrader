import numpy as np
import pytest
from datetime import datetime, timedelta

from arl_trp.domain.market import MarketSnapshot
from arl_trp.domain.order import Action
from arl_trp.execution.simulator import ExecutionConfig, ExecutionSimulator
from arl_trp.environment.trading_env import TradingEnvironment
from arl_trp.environment.observation import ObservationNormalizer


def make_market_data(count: int = 70):
    base_time = datetime(2026, 1, 1)

    return [
        MarketSnapshot(
            timestamp=base_time + timedelta(minutes=5 * i),
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.0 + i,
            volume=1000.0 + i * 10,
            bid=99.9 + i,
            ask=100.1 + i,
        )
        for i in range(count)
    ]


def make_normalizer():
    observations = np.array(
        [
            [
                0.0,
                0.01,
                -0.01,
                0.001,
                0.002,
                1000.0 + i * 100,
                0.0,
                0.0,
                1.0,
                0.0,
            ]
            for i in range(100)
        ],
        dtype=np.float32,
    )

    return ObservationNormalizer().fit(observations)


def make_environment(market_data=None):
    if market_data is None:
        market_data = make_market_data()

    env = TradingEnvironment(
        market_data=market_data,
        initial_balance=10_000.0,
        execution=ExecutionSimulator(
            ExecutionConfig(
                slippage=0.0,
                commission_per_unit=0.0,
            )
        ),
        quantity=1.0,
        reward_beta=0.0,
        observation_window=64,
    )

    env.set_observation_normalizer(make_normalizer())

    return env

def test_environment_accepts_observation_normalizer():
    normalizer = make_normalizer()

    env = TradingEnvironment(
        market_data=make_market_data(),
        initial_balance=10_000.0,
        execution=ExecutionSimulator(
            ExecutionConfig()
        ),
        quantity=1.0,
        reward_beta=0.0,
    )

    env.set_observation_normalizer(normalizer)

    assert env.observation_normalizer is normalizer


def test_reset_starts_at_first_valid_decision_bar():
    env = make_environment()

    observation = env.reset()

    assert env.index == 63
    assert observation.shape == (64, 10)


def test_reset_initializes_full_observation_buffer():
    env = make_environment()

    observation = env.reset()

    assert len(env.observation_buffer) == 64
    assert env.observation_buffer.is_ready is True
    assert observation.shape == (64, 10)


def test_reset_observation_contains_exactly_first_64_bars():
    env = make_environment()

    observation = env.reset()

    expected = np.array(
        [
            env.observation_buffer.get()[i]
            for i in range(64)
        ],
        dtype=np.float32,
    )

    assert np.array_equal(observation, expected)


def test_step_advances_one_market_bar():
    env = make_environment()

    env.reset()

    assert env.index == 63

    observation, _, terminated, _, _ = env.step(
        Action.HOLD
    )

    assert env.index == 64
    assert terminated is False
    assert observation.shape == (64, 10)


def test_observation_window_remains_64_bars_after_step():
    env = make_environment()

    env.reset()

    observation, _, _, _, _ = env.step(
        Action.HOLD
    )

    assert observation.shape == (64, 10)
    assert len(env.observation_buffer) == 64
    assert env.observation_buffer.is_ready is True


def test_step_observation_discards_oldest_bar():
    env = make_environment()

    env.reset()

    initial_window = env.observation_buffer.get()

    observation, _, _, _, _ = env.step(
        Action.HOLD
    )

    updated_window = env.observation_buffer.get()

    assert np.array_equal(
        updated_window[:-1],
        initial_window[1:],
    )

    assert np.array_equal(
        observation,
        updated_window,
    )


def test_observation_does_not_use_future_bars():
    market_data = make_market_data()

    env = make_environment(market_data)

    observation = env.reset()

    assert env.index == 63
    assert len(observation) == 64

    # The observation must end at the current decision bar.
    # No information from bar 64+ may be present.
    assert env.observation_buffer.get().shape == (64, 10)


def test_dataset_shorter_than_observation_window_is_rejected():
    env = make_environment(
        make_market_data(count=63)
    )

    with pytest.raises(ValueError):
        env.reset()


def test_reset_rebuilds_same_initial_observation():
    env = make_environment()

    first_observation = env.reset()

    env.step(Action.HOLD)
    env.step(Action.HOLD)

    second_observation = env.reset()

    assert env.index == 63
    assert np.array_equal(
        first_observation,
        second_observation,
    )


def test_reset_clears_and_rebuilds_observation_buffer():
    env = make_environment()

    env.reset()

    env.step(Action.HOLD)
    env.step(Action.HOLD)

    assert len(env.observation_buffer) == 64

    env.reset()

    assert len(env.observation_buffer) == 64
    assert env.observation_buffer.is_ready is True


def test_environment_requires_fitted_normalizer():
    env = TradingEnvironment(
        market_data=make_market_data(),
        initial_balance=10_000.0,
        execution=ExecutionSimulator(
            ExecutionConfig()
        ),
        quantity=1.0,
        reward_beta=0.0,
    )

    with pytest.raises(RuntimeError):
        env.reset()


def test_normalizer_statistics_are_not_modified_by_environment():
    normalizer = make_normalizer()

    original_median = normalizer.volume_median
    original_iqr = normalizer.volume_iqr

    env = TradingEnvironment(
        market_data=make_market_data(),
        initial_balance=10_000.0,
        execution=ExecutionSimulator(
            ExecutionConfig()
        ),
        quantity=1.0,
        reward_beta=0.0,
    )

    env.set_observation_normalizer(normalizer)
    env.reset()

    assert normalizer.volume_median == original_median
    assert normalizer.volume_iqr == original_iqr


def test_environment_observation_is_deterministic():
    env1 = make_environment()
    env2 = make_environment()

    observation1 = env1.reset()
    observation2 = env2.reset()

    assert np.array_equal(
        observation1,
        observation2,
    )

    for action in [
        Action.HOLD,
        Action.LONG,
        Action.HOLD,
    ]:
        result1 = env1.step(action)
        result2 = env2.step(action)

        observation1, reward1, terminated1, truncated1, _ = result1
        observation2, reward2, terminated2, truncated2, _ = result2

        assert np.array_equal(
            observation1,
            observation2,
        )
        assert reward1 == reward2
        assert terminated1 == terminated2
        assert truncated1 == truncated2
