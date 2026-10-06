from __future__ import annotations

import numpy as np

from arl_trp.evaluation.comparison import (
    BaselineComparison,
    BaselineComparisonResult,
)
from arl_trp.evaluation.evaluation_protocol import (
    EvaluationProtocolResult,
)
from arl_trp.evaluation.multi_baseline_promotion import (
    MultiBaselinePromotionDecision,
    MultiBaselinePromotionGate,
)
from arl_trp.evaluation.promotion_adapter import (
    MultiBaselinePromotionEvidenceAdapter,
)
from arl_trp.evaluation.statistics import (
    BootstrapResult,
    OOSStatisticsResult,
)


def make_statistics(sharpe: float) -> OOSStatisticsResult:
    return OOSStatisticsResult(
        pooled_returns=np.array(
            [0.001, 0.002, -0.001],
            dtype=np.float64,
        ),
        observed_sharpe=sharpe,
        bootstrap=BootstrapResult(
            observed_statistic=sharpe,
            bootstrap_statistics=np.array(
                [sharpe] * 10,
                dtype=np.float64,
            ),
            confidence_level=0.95,
            lower_bound=sharpe - 0.5,
            upper_bound=sharpe,
            block_size=288,
            iterations=10,
            seed=42,
        ),
    )


def make_evaluation() -> EvaluationProtocolResult:
    candidate = make_statistics(2.0)

    baseline_sharpes = {
        "flat": 0.1,
        "buy_and_hold": 0.8,
        "momentum": 0.6,
        "random": 0.2,
    }

    baseline_drawdowns = {
        "flat": 0.0,
        "buy_and_hold": 0.12,
        "momentum": 0.15,
        "random": 0.20,
    }

    baselines = {
        name: make_statistics(sharpe)
        for name, sharpe in baseline_sharpes.items()
    }

    comparisons = tuple(
        BaselineComparison(
            baseline_name=name,
            candidate_sharpe=2.0,
            baseline_sharpe=baseline_sharpes[name],
            candidate_max_drawdown=0.10,
            baseline_max_drawdown=baseline_drawdowns[name],
        )
        for name in baseline_sharpes
    )

    return EvaluationProtocolResult(
        candidate_statistics=candidate,
        baseline_statistics=baselines,
        comparison=BaselineComparisonResult(
            comparisons=comparisons,
        ),
        candidate_fold_count=5,
        candidate_max_drawdown=0.10,
        baseline_max_drawdowns=baseline_drawdowns,
    )


def test_complete_evaluation_to_promotion_flow_passes() -> None:
    evaluation = make_evaluation()

    evidence = MultiBaselinePromotionEvidenceAdapter().build(
        evaluation=evaluation,
        dsr_passed=True,
        simulator_validated=True,
    )

    result = MultiBaselinePromotionGate().evaluate(
        evidence
    )

    assert (
        result.decision
        == MultiBaselinePromotionDecision.PASS
    )

    assert result.passed is True
    assert result.reasons == ()


def test_complete_flow_preserves_all_baselines() -> None:
    evaluation = make_evaluation()

    evidence = MultiBaselinePromotionEvidenceAdapter().build(
        evaluation=evaluation,
        dsr_passed=True,
        simulator_validated=True,
    )

    result = MultiBaselinePromotionGate().evaluate(
        evidence
    )

    assert result.comparison.baseline_count == 4

    names = {
        comparison.baseline_name
        for comparison in result.comparison.comparisons
    }

    assert names == {
        "flat",
        "buy_and_hold",
        "momentum",
        "random",
    }


def test_complete_flow_fails_when_candidate_loses_to_baseline() -> None:
    evaluation = make_evaluation()

    comparisons = tuple(
        BaselineComparison(
            baseline_name=comparison.baseline_name,
            candidate_sharpe=0.5,
            baseline_sharpe=comparison.baseline_sharpe,
            candidate_max_drawdown=0.10,
            baseline_max_drawdown=(
                comparison.baseline_max_drawdown
            ),
        )
        for comparison in evaluation.comparison.comparisons
    )

    modified_evaluation = EvaluationProtocolResult(
        candidate_statistics=make_statistics(0.5),
        baseline_statistics=evaluation.baseline_statistics,
        comparison=BaselineComparisonResult(
            comparisons=comparisons,
        ),
        candidate_fold_count=5,
        candidate_max_drawdown=0.10,
        baseline_max_drawdowns=(
            evaluation.baseline_max_drawdowns
        ),
    )

    evidence = MultiBaselinePromotionEvidenceAdapter().build(
        evaluation=modified_evaluation,
        dsr_passed=True,
        simulator_validated=True,
    )

    result = MultiBaselinePromotionGate().evaluate(
        evidence
    )

    assert (
        result.decision
        == MultiBaselinePromotionDecision.FAIL
    )

    assert result.passed is False

    assert any(
        "'buy_and_hold'" in reason
        for reason in result.reasons
    )