from __future__ import annotations

import pytest

from arl_trp.evaluation.comparison import (
    BaselineComparison,
    BaselineComparisonResult,
)
from arl_trp.evaluation.multi_baseline_promotion import (
    MultiBaselinePromotionConfig,
    MultiBaselinePromotionDecision,
    MultiBaselinePromotionEvidence,
    MultiBaselinePromotionGate,
)


BASELINES = (
    "flat",
    "buy_and_hold",
    "momentum",
    "random",
)


def make_comparison(
    baseline_name: str,
    candidate_sharpe: float = 1.0,
    baseline_sharpe: float = 0.5,
    candidate_dd: float = 0.10,
    baseline_dd: float = 0.10,
) -> BaselineComparison:
    return BaselineComparison(
        baseline_name=baseline_name,
        candidate_sharpe=candidate_sharpe,
        baseline_sharpe=baseline_sharpe,
        candidate_max_drawdown=candidate_dd,
        baseline_max_drawdown=baseline_dd,
    )


def make_evidence(
    *,
    fold_count: int = 5,
    candidate_sharpe: float = 1.0,
    candidate_lower_bound: float = 0.3,
    candidate_dd: float = 0.10,
    baseline_sharpes: dict[str, float] | None = None,
    baseline_drawdowns: dict[str, float] | None = None,
    dsr_passed: bool = True,
    simulator_validated: bool = True,
) -> MultiBaselinePromotionEvidence:
    if baseline_sharpes is None:
        baseline_sharpes = {
            "flat": 0.0,
            "buy_and_hold": 0.5,
            "momentum": 0.6,
            "random": 0.1,
        }

    if baseline_drawdowns is None:
        baseline_drawdowns = {
            "flat": 0.0,
            "buy_and_hold": 0.15,
            "momentum": 0.12,
            "random": 0.20,
        }

    comparisons = tuple(
        make_comparison(
            baseline_name=name,
            candidate_sharpe=candidate_sharpe,
            baseline_sharpe=baseline_sharpes[name],
            candidate_dd=candidate_dd,
            baseline_dd=baseline_drawdowns[name],
        )
        for name in BASELINES
    )

    comparison_result = BaselineComparisonResult(
        comparisons=comparisons
    )

    return MultiBaselinePromotionEvidence(
        fold_count=fold_count,
        candidate_sharpe=candidate_sharpe,
        candidate_bootstrap_lower_bound=(
            candidate_lower_bound
        ),
        candidate_max_drawdown=candidate_dd,
        comparison=comparison_result,
        baseline_max_drawdowns=baseline_drawdowns,
        dsr_passed=dsr_passed,
        simulator_validated=simulator_validated,
        confidence_level=0.95,
    )


def test_passing_candidate_is_promoted() -> None:
    gate = MultiBaselinePromotionGate()

    result = gate.evaluate(make_evidence())

    assert (
        result.decision
        == MultiBaselinePromotionDecision.PASS
    )
    assert result.passed
    assert result.reasons == ()


def test_flat_baseline_does_not_impose_zero_drawdown() -> None:
    gate = MultiBaselinePromotionGate()

    evidence = make_evidence(
        candidate_dd=0.10,
    )

    result = gate.evaluate(evidence)

    assert result.passed


def test_candidate_must_beat_every_performance_baseline() -> None:
    gate = MultiBaselinePromotionGate()

    evidence = make_evidence(
        candidate_sharpe=0.55,
        baseline_sharpes={
            "flat": 0.0,
            "buy_and_hold": 0.5,
            "momentum": 0.60,
            "random": 0.1,
        },
    )

    result = gate.evaluate(evidence)

    assert not result.passed
    assert any(
        "'momentum'" in reason
        for reason in result.reasons
    )


def test_candidate_must_respect_every_risk_baseline() -> None:
    gate = MultiBaselinePromotionGate()

    evidence = make_evidence(
        candidate_dd=0.14,
        baseline_drawdowns={
            "flat": 0.0,
            "buy_and_hold": 0.15,
            "momentum": 0.12,
            "random": 0.20,
        },
    )

    result = gate.evaluate(evidence)

    assert not result.passed
    assert any(
        "'momentum'" in reason
        for reason in result.reasons
    )


def test_drawdown_degradation_is_applied_per_risk_baseline() -> None:
    gate = MultiBaselinePromotionGate()

    evidence = make_evidence(
        candidate_dd=0.132,
        baseline_drawdowns={
            "flat": 0.0,
            "buy_and_hold": 0.12,
            "momentum": 0.12,
            "random": 0.20,
        },
    )

    result = gate.evaluate(evidence)

    assert result.passed


def test_insufficient_folds_fail() -> None:
    gate = MultiBaselinePromotionGate()

    result = gate.evaluate(
        make_evidence(fold_count=4)
    )

    assert not result.passed
    assert any(
        "insufficient OOS folds" in reason
        for reason in result.reasons
    )


def test_missing_required_baseline_fails() -> None:
    gate = MultiBaselinePromotionGate()

    evidence = make_evidence()

    comparisons = tuple(
        comparison
        for comparison in evidence.comparison.comparisons
        if comparison.baseline_name != "momentum"
    )

    modified_comparison = BaselineComparisonResult(
        comparisons=comparisons
    )

    modified_evidence = MultiBaselinePromotionEvidence(
        fold_count=evidence.fold_count,
        candidate_sharpe=evidence.candidate_sharpe,
        candidate_bootstrap_lower_bound=(
            evidence.candidate_bootstrap_lower_bound
        ),
        candidate_max_drawdown=(
            evidence.candidate_max_drawdown
        ),
        comparison=modified_comparison,
        baseline_max_drawdowns=(
            evidence.baseline_max_drawdowns
        ),
        dsr_passed=evidence.dsr_passed,
        simulator_validated=evidence.simulator_validated,
        confidence_level=evidence.confidence_level,
    )

    result = gate.evaluate(modified_evidence)

    assert not result.passed
    assert any(
        "missing required baselines" in reason
        for reason in result.reasons
    )


def test_dsr_failure_blocks_promotion() -> None:
    gate = MultiBaselinePromotionGate()

    result = gate.evaluate(
        make_evidence(dsr_passed=False)
    )

    assert not result.passed
    assert any(
        "deflated Sharpe ratio" in reason
        for reason in result.reasons
    )


def test_simulator_failure_blocks_promotion() -> None:
    gate = MultiBaselinePromotionGate()

    result = gate.evaluate(
        make_evidence(simulator_validated=False)
    )

    assert not result.passed
    assert any(
        "simulator validation" in reason
        for reason in result.reasons
    )


def test_multiple_failures_are_preserved() -> None:
    gate = MultiBaselinePromotionGate()

    result = gate.evaluate(
        make_evidence(
            fold_count=3,
            candidate_sharpe=0.2,
            baseline_sharpes={
                "flat": 0.3,
                "buy_and_hold": 0.5,
                "momentum": 0.6,
                "random": 0.1,
            },
            candidate_dd=0.30,
            baseline_drawdowns={
                "flat": 0.0,
                "buy_and_hold": 0.10,
                "momentum": 0.10,
                "random": 0.10,
            },
            dsr_passed=False,
            simulator_validated=False,
        )
    )

    assert not result.passed
    assert len(result.reasons) >= 4


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        MultiBaselinePromotionConfig(
            risk_baselines=("unknown",)
        )


def test_performance_baselines_must_be_required() -> None:
    with pytest.raises(ValueError):
        MultiBaselinePromotionConfig(
            required_baselines=("flat",),
            performance_baselines=(
                "flat",
                "momentum",
            ),
        )


def test_risk_baselines_must_be_required() -> None:
    with pytest.raises(ValueError):
        MultiBaselinePromotionConfig(
            required_baselines=("flat",),
            risk_baselines=(
                "buy_and_hold",
            ),
        )