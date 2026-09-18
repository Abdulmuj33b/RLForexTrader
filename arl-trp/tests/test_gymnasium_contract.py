import numpy as np
import pytest
import gymnasium
from gymnasium import spaces

from arl_trp.domain.market import MarketSnapshot
from arl_trp.execution.simulator import (
    ExecutionConfig,
    ExecutionSimulator,
)
from arl_trp.environment.observation import ObservationNormalizer
from arl_trp.environment.trading_env import TradingEnvironment


def make_market_data(count=70):
    from datetime import datetime, timedelta

    data = []

    for i in range(count):
        close = 100.0 + i

        data.append(
            MarketSnapshot(
                timestamp=datetime(2026, 1, 1)
                + timedelta(minutes=5 * i),
                open=close - 0.5,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                volume=1000.0 + i,
                bid=close - 0.1,
                ask=close + 0.1,
            )
        )

    return data


def make_normalizer():
    observations = np.array(
        [
            [
                0.0,
                0.01,
                -0.01,
                0.001,
                0.002,
                1000.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            [
                0.0,
                0.01,
                -0.01,
                0.001,
                0.002,
                2000.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
        ],
        dtype=np.float32,
    )

    return ObservationNormalizer().fit(observations)


def make_environment():
    env = TradingEnvironment(
        market_data=make_market_data(),
        initial_balance=10_000.0,
        execution=ExecutionSimulator(
            ExecutionConfig()
        ),
        quantity=1.0,
        reward_beta=0.0,
        observation_window=64,
    )

    env.set_observation_normalizer(
        make_normalizer()
    )

    return env


def test_environment_is_gymnasium_environment():
    env = make_environment()

    assert isinstance(env, gymnasium.Env)


def test_action_space_is_discrete_four():
    env = make_environment()

    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 4


def test_observation_space_is_box_64_by_10():
    env = make_environment()

    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (64, 10)


def test_reset_returns_observation_and_info():
    env = make_environment()

    observation, info = env.reset()

    assert observation.shape == (64, 10)
    assert isinstance(info, dict)


def test_reset_observation_is_inside_observation_space():
    env = make_environment()

    observation, _ = env.reset()

    assert env.observation_space.contains(observation)


def test_action_space_contains_all_actions():
    env = make_environment()

    for action in range(4):
        assert env.action_space.contains(action)


def test_step_accepts_gymnasium_action():
    env = make_environment()

    env.reset()

    observation, reward, terminated, truncated, info = env.step(0)

    assert observation.shape == (64, 10)
    assert isinstance(reward, (float, np.floating))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_observation_is_inside_observation_space():
    env = make_environment()

    env.reset()

    observation, _, _, _, _ = env.step(0)

    assert env.observation_space.contains(observation)


def test_reset_is_seed_compatible():
    env = make_environment()

    observation1, _ = env.reset(seed=123)

    observation2, _ = env.reset(seed=123)

    np.testing.assert_array_equal(
        observation1,
        observation2,
    )