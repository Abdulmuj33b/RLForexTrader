from __future__ import annotations

import numpy as np
import pytest

from arl_trp.evaluation.comparison import (
    BaselineComparison,
    BaselineComparisonResult,
)
from arl_trp.evaluation.evaluation_protocol import (
    EvaluationProtocolResult,
)
from arl_trp.evaluation.promotion_adapter import (
    MultiBaselinePromotionEvidenceAdapter,
)
from arl_trp.evaluation.statistics import (
    BootstrapResult,
    OOSStatisticsResult,
)


def make_statistics(
    sharpe: float,
    lower_bound: float,
    confidence_level: float = 0.95,
) -> OOSStatisticsResult:
    bootstrap = BootstrapResult(
        observed_statistic=sharpe,
        bootstrap_statistics=np.array(
            [sharpe] * 10,
            dtype=np.float64,
        ),
        confidence_level=confidence_level,
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


def make_evaluation() -> EvaluationProtocolResult:
    candidate_statistics = make_statistics(
        sharpe=2.0,
        lower_bound=1.2,
    )

    flat_statistics = make_statistics(
        sharpe=0.2,
        lower_bound=-0.1,
    )

    momentum_statistics = make_statistics(
        sharpe=0.8,
        lower_bound=0.2,
    )

    comparison = BaselineComparisonResult(
        comparisons=(
            BaselineComparison(
                baseline_name="flat",
                candidate_sharpe=2.0,
                baseline_sharpe=0.2,
                candidate_max_drawdown=0.05,
                baseline_max_drawdown=0.10,
            ),
            BaselineComparison(
                baseline_name="momentum",
                candidate_sharpe=2.0,
                baseline_sharpe=0.8,
                candidate_max_drawdown=0.05,
                baseline_max_drawdown=0.15,
            ),
        ),
    )

    return EvaluationProtocolResult(
        candidate_statistics=candidate_statistics,
        baseline_statistics={
            "flat": flat_statistics,
            "momentum": momentum_statistics,
        },
        comparison=comparison,
        candidate_fold_count=5,
        candidate_max_drawdown=0.05,
        baseline_max_drawdowns={
            "flat": 0.10,
            "momentum": 0.15,
        },
    )


def test_adapter_uses_candidate_statistics():
    evidence = MultiBaselinePromotionEvidenceAdapter().build(
        make_evaluation(),
        dsr_passed=True,
        simulator_validated=True,
    )

    assert evidence.fold_count == 5
    assert evidence.candidate_sharpe == 2.0
    assert evidence.candidate_bootstrap_lower_bound == 1.2
    assert evidence.candidate_max_drawdown == 0.05


def test_adapter_preserves_every_baseline_comparison():
    evidence = MultiBaselinePromotionEvidenceAdapter().build(
        make_evaluation(),
        dsr_passed=True,
        simulator_validated=True,
    )

    assert evidence.comparison.baseline_count == 2

    comparisons = {
        comparison.baseline_name: comparison
        for comparison in evidence.comparison.comparisons
    }

    assert set(comparisons) == {
        "flat",
        "momentum",
    }

    assert comparisons["flat"].baseline_sharpe == 0.2
    assert comparisons["momentum"].baseline_sharpe == 0.8


def test_adapter_does_not_aggregate_baseline_sharpes():
    evidence = MultiBaselinePromotionEvidenceAdapter().build(
        make_evaluation(),
        dsr_passed=True,
        simulator_validated=True,
    )

    sharpes = {
        comparison.baseline_name:
        comparison.baseline_sharpe
        for comparison in evidence.comparison.comparisons
    }

    assert sharpes == {
        "flat": 0.2,
        "momentum": 0.8,
    }


def test_adapter_preserves_individual_baseline_drawdowns():
    evidence = MultiBaselinePromotionEvidenceAdapter().build(
        make_evaluation(),
        dsr_passed=True,
        simulator_validated=True,
    )

    assert evidence.baseline_max_drawdowns == {
        "flat": 0.10,
        "momentum": 0.15,
    }


def test_adapter_does_not_aggregate_baseline_drawdowns():
    evidence = MultiBaselinePromotionEvidenceAdapter().build(
        make_evaluation(),
        dsr_passed=True,
        simulator_validated=True,
    )

    assert (
        evidence.baseline_max_drawdowns["flat"]
        == 0.10
    )

    assert (
        evidence.baseline_max_drawdowns["momentum"]
        == 0.15
    )


def test_adapter_preserves_dsr_and_simulator_evidence():
    evidence = MultiBaselinePromotionEvidenceAdapter().build(
        make_evaluation(),
        dsr_passed=False,
        simulator_validated=False,
    )

    assert evidence.dsr_passed is False
    assert evidence.simulator_validated is False


def test_adapter_preserves_confidence_level():
    evidence = MultiBaselinePromotionEvidenceAdapter().build(
        make_evaluation(),
        dsr_passed=True,
        simulator_validated=True,
    )

    assert evidence.confidence_level == 0.95


def test_adapter_has_new_protocol_version():
    adapter = MultiBaselinePromotionEvidenceAdapter()

    assert (
        adapter.protocol_version
        == "multi_baseline_promotion_adapter_v0.1"
    )


def test_adapter_rejects_empty_baseline_evidence():
    class EmptyEvaluation:
        baseline_count = 0

    with pytest.raises(
        ValueError,
        match="at least one baseline",
    ):
        MultiBaselinePromotionEvidenceAdapter().build(
            evaluation=EmptyEvaluation(),
            dsr_passed=True,
            simulator_validated=True,
        )