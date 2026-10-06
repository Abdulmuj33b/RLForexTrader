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
    MultiBaselinePromotionEvidence,
    MultiBaselinePromotionGate,
)
from arl_trp.evaluation.statistics import (
    BootstrapResult,
    OOSStatisticsResult,
)


def make_statistics(
    sharpe: float,
    lower_bound: float,
) -> OOSStatisticsResult:
    bootstrap = BootstrapResult(
        observed_statistic=sharpe,
        bootstrap_statistics=np.array(
            [sharpe] * 10,
            dtype=np.float64,
        ),
        confidence_level=0.95,
        lower_bound=lower_bound,
        upper_bound=sharpe,
        block_size=288,
        iterations=10,
        seed=42,
    )

    return OOSStatisticsResult(
        pooled_returns=np.array(
            [0.001, 0.002, -0.001],
            dtype=np.float64,
        ),
        observed_sharpe=sharpe,
        bootstrap=bootstrap,
    )


def make_evaluation(
    candidate_sharpe: float,
    candidate_drawdown: float,
    baseline_sharpes: dict[str, float],
    baseline_drawdowns: dict[str, float],
) -> EvaluationProtocolResult:
    candidate_statistics = make_statistics(
        sharpe=candidate_sharpe,
        lower_bound=candidate_sharpe - 0.5,
    )

    baseline_statistics = {
        name: make_statistics(
            sharpe=sharpe,
            lower_bound=sharpe - 0.5,
        )
        for name, sharpe in baseline_sharpes.items()
    }

    comparisons = tuple(
        BaselineComparison(
            baseline_name=name,
            candidate_sharpe=candidate_sharpe,
            baseline_sharpe=baseline_sharpes[name],
            candidate_max_drawdown=candidate_drawdown,
            baseline_max_drawdown=baseline_drawdowns[name],
        )
        for name in baseline_sharpes
    )

    comparison = BaselineComparisonResult(
        comparisons=comparisons,
    )

    return EvaluationProtocolResult(
        candidate_statistics=candidate_statistics,
        baseline_statistics=baseline_statistics,
        comparison=comparison,
        candidate_fold_count=5,
        candidate_max_drawdown=candidate_drawdown,
        baseline_max_drawdowns=baseline_drawdowns,
    )


def make_evidence(
    evaluation: EvaluationProtocolResult,
    *,
    dsr_passed: bool = True,
    simulator_validated: bool = True,
) -> MultiBaselinePromotionEvidence:
    return MultiBaselinePromotionEvidence(
        fold_count=evaluation.candidate_fold_count,
        candidate_sharpe=(
            evaluation.candidate_statistics.observed_sharpe
        ),
        candidate_bootstrap_lower_bound=(
            evaluation.candidate_statistics.bootstrap.lower_bound
        ),
        candidate_max_drawdown=(
            evaluation.candidate_max_drawdown
        ),
        comparison=evaluation.comparison,
        baseline_max_drawdowns=(
            evaluation.baseline_max_drawdowns
        ),
        dsr_passed=dsr_passed,
        simulator_validated=simulator_validated,
        confidence_level=(
            evaluation.candidate_statistics.bootstrap.confidence_level
        ),
    )


def test_multi_baseline_evidence_can_pass_new_promotion_gate():
    evaluation = make_evaluation(
        candidate_sharpe=2.0,
        candidate_drawdown=0.10,
        baseline_sharpes={
            "flat": 0.1,
            "buy_and_hold": 0.8,
            "momentum": 0.6,
            "random": 0.2,
        },
        baseline_drawdowns={
            "flat": 0.00,
            "buy_and_hold": 0.12,
            "momentum": 0.15,
            "random": 0.20,
        },
    )

    evidence = make_evidence(evaluation)

    result = MultiBaselinePromotionGate().evaluate(
        evidence
    )

    assert (
        result.decision
        == MultiBaselinePromotionDecision.PASS
    )
    assert result.passed is True
    assert result.reasons == ()


def test_candidate_must_beat_every_performance_baseline():
    evaluation = make_evaluation(
        candidate_sharpe=0.5,
        candidate_drawdown=0.05,
        baseline_sharpes={
            "flat": 0.1,
            "buy_and_hold": 0.8,
            "momentum": 0.6,
            "random": 0.2,
        },
        baseline_drawdowns={
            "flat": 0.00,
            "buy_and_hold": 0.12,
            "momentum": 0.15,
            "random": 0.20,
        },
    )

    evidence = make_evidence(evaluation)

    result = MultiBaselinePromotionGate().evaluate(
        evidence
    )

    assert result.decision == MultiBaselinePromotionDecision.FAIL
    assert result.passed is False

    assert any(
        "'buy_and_hold'" in reason
        for reason in result.reasons
    )

    assert any(
        "'momentum'" in reason
        for reason in result.reasons
    )


def test_candidate_drawdown_is_checked_against_each_risk_baseline():
    evaluation = make_evaluation(
        candidate_sharpe=2.0,
        candidate_drawdown=0.20,
        baseline_sharpes={
            "flat": 0.1,
            "buy_and_hold": 0.8,
            "momentum": 0.6,
            "random": 0.2,
        },
        baseline_drawdowns={
            "flat": 0.00,
            "buy_and_hold": 0.05,
            "momentum": 0.08,
            "random": 0.03,
        },
    )

    evidence = make_evidence(evaluation)

    result = MultiBaselinePromotionGate().evaluate(
        evidence
    )

    assert result.decision == MultiBaselinePromotionDecision.FAIL
    assert result.passed is False

    assert any(
        "'buy_and_hold'" in reason
        for reason in result.reasons
    )

    assert any(
        "'momentum'" in reason
        for reason in result.reasons
    )

    assert any(
        "'random'" in reason
        for reason in result.reasons
    )


def test_flat_baseline_does_not_impose_zero_drawdown():
    evaluation = make_evaluation(
        candidate_sharpe=2.0,
        candidate_drawdown=0.10,
        baseline_sharpes={
            "flat": 0.1,
            "buy_and_hold": 0.8,
            "momentum": 0.6,
            "random": 0.2,
        },
        baseline_drawdowns={
            "flat": 0.00,
            "buy_and_hold": 0.12,
            "momentum": 0.15,
            "random": 0.20,
        },
    )

    evidence = make_evidence(evaluation)

    result = MultiBaselinePromotionGate().evaluate(
        evidence
    )

    assert result.passed is True


def test_dsr_failure_still_blocks_multi_baseline_promotion():
    evaluation = make_evaluation(
        candidate_sharpe=2.0,
        candidate_drawdown=0.05,
        baseline_sharpes={
            "flat": 0.1,
            "buy_and_hold": 0.8,
            "momentum": 0.6,
            "random": 0.2,
        },
        baseline_drawdowns={
            "flat": 0.00,
            "buy_and_hold": 0.12,
            "momentum": 0.15,
            "random": 0.20,
        },
    )

    evidence = make_evidence(
        evaluation,
        dsr_passed=False,
    )

    result = MultiBaselinePromotionGate().evaluate(
        evidence
    )

    assert result.decision == MultiBaselinePromotionDecision.FAIL
    assert result.passed is False

    assert any(
        "deflated Sharpe ratio" in reason
        for reason in result.reasons
    )


def test_simulator_failure_still_blocks_multi_baseline_promotion():
    evaluation = make_evaluation(
        candidate_sharpe=2.0,
        candidate_drawdown=0.05,
        baseline_sharpes={
            "flat": 0.1,
            "buy_and_hold": 0.8,
            "momentum": 0.6,
            "random": 0.2,
        },
        baseline_drawdowns={
            "flat": 0.00,
            "buy_and_hold": 0.12,
            "momentum": 0.15,
            "random": 0.20,
        },
    )

    evidence = make_evidence(
        evaluation,
        simulator_validated=False,
    )

    result = MultiBaselinePromotionGate().evaluate(
        evidence
    )

    assert result.decision == MultiBaselinePromotionDecision.FAIL
    assert result.passed is False

    assert any(
        "simulator validation" in reason
        for reason in result.reasons
    )


def test_promotion_result_preserves_all_baseline_comparisons():
    evaluation = make_evaluation(
        candidate_sharpe=2.0,
        candidate_drawdown=0.05,
        baseline_sharpes={
            "flat": 0.1,
            "buy_and_hold": 0.8,
            "momentum": 0.6,
            "random": 0.2,
        },
        baseline_drawdowns={
            "flat": 0.00,
            "buy_and_hold": 0.12,
            "momentum": 0.15,
            "random": 0.20,
        },
    )

    evidence = make_evidence(evaluation)

    result = MultiBaselinePromotionGate().evaluate(
        evidence
    )

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

    assert result.comparison.baseline_count == 4