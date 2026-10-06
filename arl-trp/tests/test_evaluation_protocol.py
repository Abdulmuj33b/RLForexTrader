from dataclasses import dataclass

import numpy as np
import pytest

from arl_trp.evaluation.evaluation_protocol import EvaluationProtocol


@dataclass(frozen=True)
class FakeFoldReportResult:
    equity_curve: np.ndarray


@dataclass(frozen=True)
class FakeFoldReport:
    result: FakeFoldReportResult


@dataclass(frozen=True)
class FakeFold:
    fold_id: int
    report: FakeFoldReport


@dataclass(frozen=True)
class FakeWalkForward:
    folds: tuple[FakeFold, ...]


def make_walk_forward(
    equity_curves: tuple[np.ndarray, ...],
) -> FakeWalkForward:
    folds = tuple(
        FakeFold(
            fold_id=index,
            report=FakeFoldReport(
                result=FakeFoldReportResult(equity_curve=curve)
            ),
        )
        for index, curve in enumerate(equity_curves)
    )

    return FakeWalkForward(folds=folds)


def test_protocol_evaluates_candidate_and_all_baselines():
    candidate = make_walk_forward(
        (
            np.array([100.0, 101.0, 102.0]),
            np.array([102.0, 103.0, 105.0]),
        )
    )

    flat = make_walk_forward(
        (
            np.array([100.0, 100.0, 100.0]),
            np.array([100.0, 100.0, 100.0]),
        )
    )

    momentum = make_walk_forward(
        (
            np.array([100.0, 100.5, 101.0]),
            np.array([101.0, 101.5, 102.0]),
        )
    )

    protocol = EvaluationProtocol()

    result = protocol.evaluate(
        candidate_walk_forward=candidate,
        baseline_walk_forwards={
            "flat": flat,
            "momentum": momentum,
        },
        candidate_max_drawdown=0.05,
        baseline_max_drawdowns={
            "flat": 0.00,
            "momentum": 0.03,
        },
    )

    assert result.baseline_count == 2
    assert set(result.baseline_statistics) == {
        "flat",
        "momentum",
    }

    assert result.candidate_fold_count == 2
    assert result.protocol_version == "evaluation_protocol_v0.1"


def test_protocol_preserves_candidate_statistics():
    candidate = make_walk_forward(
        (
            np.array([100.0, 101.0, 102.0]),
            np.array([102.0, 103.0, 105.0]),
        )
    )

    baseline = make_walk_forward(
        (
            np.array([100.0, 100.0, 100.0]),
            np.array([100.0, 100.0, 100.0]),
        )
    )

    result = EvaluationProtocol().evaluate(
        candidate_walk_forward=candidate,
        baseline_walk_forwards={"flat": baseline},
        candidate_max_drawdown=0.05,
        baseline_max_drawdowns={"flat": 0.00},
    )

    assert result.candidate_statistics.observed_sharpe != 0.0


def test_protocol_reports_all_baseline_comparisons():
    candidate = make_walk_forward(
        (
            np.array([100.0, 101.0, 102.0]),
            np.array([102.0, 103.0, 105.0]),
        )
    )

    baseline_a = make_walk_forward(
        (
            np.array([100.0, 100.0, 100.0]),
            np.array([100.0, 100.0, 100.0]),
        )
    )

    baseline_b = make_walk_forward(
        (
            np.array([100.0, 100.2, 100.4]),
            np.array([100.4, 100.6, 100.8]),
        )
    )

    result = EvaluationProtocol().evaluate(
        candidate_walk_forward=candidate,
        baseline_walk_forwards={
            "flat": baseline_a,
            "buy_and_hold": baseline_b,
        },
        candidate_max_drawdown=0.05,
        baseline_max_drawdowns={
            "flat": 0.00,
            "buy_and_hold": 0.02,
        },
    )

    names = {
        comparison.baseline_name
        for comparison in result.comparison.comparisons
    }

    assert names == {"flat", "buy_and_hold"}


def test_protocol_rejects_empty_baselines():
    candidate = make_walk_forward(
        (
            np.array([100.0, 101.0, 102.0]),
        )
    )

    with pytest.raises(ValueError, match="baseline_walk_forwards"):
        EvaluationProtocol().evaluate(
            candidate_walk_forward=candidate,
            baseline_walk_forwards={},
            candidate_max_drawdown=0.05,
            baseline_max_drawdowns={},
        )


def test_protocol_rejects_mismatched_drawdown_names():
    candidate = make_walk_forward(
        (
            np.array([100.0, 101.0, 102.0]),
        )
    )

    baseline = make_walk_forward(
        (
            np.array([100.0, 100.0, 100.0]),
        )
    )

    with pytest.raises(
        ValueError,
        match="same baseline names",
    ):
        EvaluationProtocol().evaluate(
            candidate_walk_forward=candidate,
            baseline_walk_forwards={"flat": baseline},
            candidate_max_drawdown=0.05,
            baseline_max_drawdowns={"momentum": 0.01},
        )


def test_protocol_rejects_negative_candidate_drawdown():
    candidate = make_walk_forward(
        (
            np.array([100.0, 101.0, 102.0]),
        )
    )

    baseline = make_walk_forward(
        (
            np.array([100.0, 100.0, 100.0]),
        )
    )

    with pytest.raises(
        ValueError,
        match="candidate_max_drawdown",
    ):
        EvaluationProtocol().evaluate(
            candidate_walk_forward=candidate,
            baseline_walk_forwards={"flat": baseline},
            candidate_max_drawdown=-0.01,
            baseline_max_drawdowns={"flat": 0.0},
        )