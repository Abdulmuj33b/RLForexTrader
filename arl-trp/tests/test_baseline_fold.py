from datetime import datetime, timedelta

import numpy as np
import pytest

from arl_trp.domain.market import MarketSnapshot
from arl_trp.evaluation.baseline_fold import BaselineFoldEvaluator
from arl_trp.evaluation.baselines import FlatBaseline
from arl_trp.evaluation.fold import FoldEvaluationResult
from arl_trp.evaluation.report import EvaluationReporter
from arl_trp.execution.simulator import ExecutionConfig, ExecutionSimulator
from arl_trp.environment.observation import ObservationNormalizer
from arl_trp.environment.trading_env import TradingEnvironment

def make_market_data(length=100):
    data = []
    for index in range(length):
        bid = 100.0 + index
        ask = bid + 1.0
        data.append(MarketSnapshot(
            timestamp=datetime(2026, 1, 1) + timedelta(minutes=index),
            open=bid,
            high=ask,
            low=bid - 1.0,
            close=bid + 0.5,
            volume=1000.0,
            bid=bid,
            ask=ask,
        ))
    return data

def make_environment():
    observations = np.array([
        [0.0, 0.01, -0.01, 0.0, 0.01, 100.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.01, -0.01, 0.01, 0.01, 110.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.01, -0.01, 0.02, 0.01, 120.0, 0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    normalizer = ObservationNormalizer().fit(observations)
    environment = TradingEnvironment(
        market_data=make_market_data(),
        initial_balance=10_000.0,
        execution=ExecutionSimulator(ExecutionConfig()),
        quantity=1.0,
    )
    environment.set_observation_normalizer(normalizer)
    return environment

def test_baseline_fold_evaluator_returns_fold_result():
    result = BaselineFoldEvaluator().evaluate(FlatBaseline(), make_environment(), 1)
    assert isinstance(result, FoldEvaluationResult)
    assert result.fold_id == 1

def test_baseline_fold_evaluator_builds_report():
    result = BaselineFoldEvaluator().evaluate(FlatBaseline(), make_environment(), 1)
    assert result.report.result.steps == 36
    assert result.report.result.equity_curve.size == 37
    assert result.report.result.trade_pnls.size == 0
    assert result.report.metrics.initial_equity == 10_000.0
    assert result.report.metrics.final_equity == 10_000.0
    assert result.report.metrics.net_pnl == 0.0

def test_baseline_fold_evaluator_preserves_baseline_actions():
    result = BaselineFoldEvaluator().evaluate(FlatBaseline(), make_environment(), 3)
    assert result.report.result.actions.size == 36
    assert np.all(result.report.result.actions == 0)

def test_baseline_fold_evaluator_rejects_negative_fold_id():
    with pytest.raises(ValueError):
        BaselineFoldEvaluator().evaluate(FlatBaseline(), make_environment(), -1)

def test_baseline_fold_evaluator_uses_supplied_reporter():
    reporter = EvaluationReporter(periods_per_year=12.0)
    evaluator = BaselineFoldEvaluator(reporter=reporter)
    result = evaluator.evaluate(FlatBaseline(), make_environment(), 2)
    assert result.report.metrics is not None

def test_baseline_fold_evaluator_resets_baseline_before_evaluation():
    result = BaselineFoldEvaluator().evaluate(FlatBaseline(), make_environment(), 0)
    assert result.fold_id == 0
    assert result.report.result.steps == 36
