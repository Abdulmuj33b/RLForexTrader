from datetime import datetime, timedelta

import numpy as np
import pytest

from arl_trp.domain.market import MarketSnapshot
from arl_trp.evaluation.baselines import FlatBaseline
from arl_trp.evaluation.baseline_walk_forward import (
    BaselineWalkForwardEvaluationResult,
    BaselineWalkForwardEvaluator,
)
from arl_trp.execution.simulator import (
    ExecutionConfig,
    ExecutionSimulator,
)
from arl_trp.environment.observation import (
    ObservationNormalizer,
)
from arl_trp.environment.trading_env import (
    TradingEnvironment,
)


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


def test_baseline_walk_forward_returns_result():
    evaluator = BaselineWalkForwardEvaluator()

    fold_runs = (
        (make_environment(), 0),
        (make_environment(), 1),
        (make_environment(), 2),
    )

    result = evaluator.evaluate(
        policy=FlatBaseline(),
        fold_runs=fold_runs,
    )

    assert isinstance(
        result,
        BaselineWalkForwardEvaluationResult,
    )

    assert result.fold_count == 3


def test_baseline_walk_forward_preserves_chronological_order():
    evaluator = BaselineWalkForwardEvaluator()

    fold_runs = (
        (make_environment(), 2),
        (make_environment(), 0),
        (make_environment(), 1),
    )

    result = evaluator.evaluate(
        policy=FlatBaseline(),
        fold_runs=fold_runs,
    )

    assert tuple(
        fold.fold_id
        for fold in result.folds
    ) == (0, 1, 2)


def test_baseline_walk_forward_preserves_fold_metrics():
    evaluator = BaselineWalkForwardEvaluator()

    fold_runs = (
        (make_environment(), 0),
        (make_environment(), 1),
    )

    result = evaluator.evaluate(
        policy=FlatBaseline(),
        fold_runs=fold_runs,
    )

    for fold in result.folds:
        assert fold.report.metrics.initial_equity == 10_000.0
        assert fold.report.metrics.final_equity == 10_000.0
        assert fold.report.metrics.net_pnl == 0.0


def test_baseline_walk_forward_rejects_empty_runs():
    evaluator = BaselineWalkForwardEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(
            policy=FlatBaseline(),
            fold_runs=(),
        )


def test_baseline_walk_forward_requires_unique_fold_ids():
    evaluator = BaselineWalkForwardEvaluator()

    fold_runs = (
        (make_environment(), 0),
        (make_environment(), 0),
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(
            policy=FlatBaseline(),
            fold_runs=fold_runs,
        )
