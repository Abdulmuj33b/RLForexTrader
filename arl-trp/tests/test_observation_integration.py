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
            [0.0, 0.01, -0.01, 0.001, 0.002,
             1000.0 + i * 100,
             0.0, 0.0, 1.0, 0.0]
            for i in range(100)
        ],
        dtype=np.float32,
    )

    return ObservationNormalizer().fit(observations)


def make_environment():
    return TradingEnvironment(
        market_data=make_market_data(),
        initial_balance=10_000.0,
        execution=ExecutionSimulator(
            ExecutionConfig(
                slippage=0.0,
                commission_per_unit=0.0,
            )
        ),
        quantity=1.0,
        reward_beta=0.0,
    )


def test_environment_accepts_observation_normalizer():
    normalizer = make_normalizer()

    env = make_environment()

    # This test defines the integration contract.
    env.set_observation_normalizer(normalizer)

    assert env.observation_normalizer is normalizer


def test_reset_initializes_observation_buffer():
    env = make_environment()
    env.set_observation_normalizer(make_normalizer())

    env.reset()

    assert len(env.observation_buffer) == 1
    assert env.observation_buffer.is_ready is False


def test_observation_buffer_fills_one_bar_at_a_time():
    env = make_environment()
    env.set_observation_normalizer(make_normalizer())

    env.reset()

    for _ in range(10):
        env.step(Action.HOLD)

    assert len(env.observation_buffer) == 11
    assert env.observation_buffer.is_ready is False


def test_environment_produces_64_by_10_observation_after_warmup():
    env = make_environment()
    env.set_observation_normalizer(make_normalizer())

    env.reset()

    for _ in range(63):
        observation, _, terminated, _, _ = env.step(Action.HOLD)

    assert terminated is False
    assert observation.shape == (64, 10)
    assert observation.dtype == np.float32


def test_environment_observation_is_not_ready_during_warmup():
    env = make_environment()
    env.set_observation_normalizer(make_normalizer())

    observation = env.reset()

    assert observation.shape == (10,)


def test_reset_clears_observation_buffer():
    env = make_environment()
    env.set_observation_normalizer(make_normalizer())

    env.reset()

    for _ in range(63):
        env.step(Action.HOLD)

    assert env.observation_buffer.is_ready is True

    env.reset()

    assert len(env.observation_buffer) == 1
    assert env.observation_buffer.is_ready is False


def test_environment_requires_fitted_normalizer():
    env = make_environment()

    normalizer = ObservationNormalizer()

    with pytest.raises(RuntimeError):
        env.set_observation_normalizer(normalizer)


def test_normalizer_statistics_are_not_modified_by_environment():
    normalizer = make_normalizer()

    original_median = normalizer.volume_median
    original_iqr = normalizer.volume_iqr

    env = make_environment()
    env.set_observation_normalizer(normalizer)

    env.reset()

    for _ in range(20):
        env.step(Action.HOLD)

    assert normalizer.volume_median == original_median
    assert normalizer.volume_iqr == original_iqr


def test_environment_observation_is_deterministic():
    normalizer_1 = make_normalizer()
    normalizer_2 = make_normalizer()

    env_1 = make_environment()
    env_2 = make_environment()

    env_1.set_observation_normalizer(normalizer_1)
    env_2.set_observation_normalizer(normalizer_2)

    observation_1 = env_1.reset()
    observation_2 = env_2.reset()

    assert np.array_equal(observation_1, observation_2)

    for _ in range(63):
        observation_1, reward_1, _, _, _ = env_1.step(Action.HOLD)
        observation_2, reward_2, _, _, _ = env_2.step(Action.HOLD)

    assert np.array_equal(observation_1, observation_2)
    assert reward_1 == reward_2
