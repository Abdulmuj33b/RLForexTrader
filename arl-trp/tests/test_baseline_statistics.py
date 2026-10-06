from datetime import datetime, timedelta

import numpy as np

from arl_trp.domain.market import MarketSnapshot
from arl_trp.environment.observation import ObservationNormalizer
from arl_trp.environment.trading_env import TradingEnvironment
from arl_trp.evaluation.baselines import FlatBaseline
from arl_trp.evaluation.baseline_walk_forward import (
    BaselineWalkForwardEvaluator,
)
from arl_trp.evaluation.statistics import OOSStatistics
from arl_trp.execution.simulator import (
    ExecutionConfig,
    ExecutionSimulator,
)


def make_market_data(length=100):
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
            [0.0, 0.01, -0.01, 0.0, 0.01, 100.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.01, -0.01, 0.01, 0.01, 110.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.01, -0.01, 0.02, 0.01, 120.0, 0.0, 0.0, 1.0, 0.0],
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


def test_oos_statistics_accepts_baseline_walk_forward_result():
    walk_forward = (
        BaselineWalkForwardEvaluator().evaluate(
            policy=FlatBaseline(),
            fold_runs=(
                (make_environment(), 0),
                (make_environment(), 1),
            ),
        )
    )

    statistics = OOSStatistics(
        periods_per_year=1.0,
        block_size=2,
        iterations=10,
        seed=42,
    )

    result = statistics.evaluate(
        walk_forward
    )

    assert result.pooled_returns.size == 72
    assert result.observed_sharpe == 0.0
    assert result.bootstrap.iterations == 10


def test_baseline_statistics_use_the_same_protocol_version():
    walk_forward = (
        BaselineWalkForwardEvaluator().evaluate(
            policy=FlatBaseline(),
            fold_runs=(
                (make_environment(), 0),
                (make_environment(), 1),
            ),
        )
    )

    statistics = OOSStatistics(
        periods_per_year=1.0,
        block_size=2,
        iterations=10,
        seed=42,
    )

    statistics.evaluate(walk_forward)

    assert (
        statistics.PROTOCOL_VERSION
        == "oos_stats_v0.1"
    )
