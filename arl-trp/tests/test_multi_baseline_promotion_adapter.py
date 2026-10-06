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


def make_evaluation() -> EvaluationProtocolResult:
    candidate_statistics = OOSStatisticsResult(
        pooled_returns=np.array(
            [0.001, 0.002, -0.001],
            dtype=np.float64,
        ),
        observed_sharpe=2.0,
        bootstrap=BootstrapResult(
            observed_statistic=2.0,
            bootstrap_statistics=np.array(
                [1.8, 1.9, 2.0, 2.1],
                dtype=np.float64,
            ),
            confidence_level=0.95,
            lower_bound=1.8,
            upper_bound=2.1,
            block_size=288,
            iterations=4,
            seed=42,
        ),
    )

    baseline_statistics = {
        "flat": candidate_statistics,
        "buy_and_hold": candidate_statistics,
        "momentum": candidate_statistics,
        "random": candidate_statistics,
    }

    comparisons = tuple(
        BaselineComparison(
            baseline_name=name,
            candidate_sharpe=2.0,
            baseline_sharpe=sharpe,
            candidate_max_drawdown=0.10,
            baseline_max_drawdown=drawdown,
        )
        for name, sharpe, drawdown in (
            ("flat", 0.1, 0.0),
            ("buy_and_hold", 0.8, 0.12),
            ("momentum", 0.6, 0.15),
            ("random", 0.2, 0.20),
        )
    )

    comparison = BaselineComparisonResult(
        comparisons=comparisons,
    )

    return EvaluationProtocolResult(
        candidate_statistics=candidate_statistics,
        baseline_statistics=baseline_statistics,
        comparison=comparison,
        candidate_fold_count=5,
        candidate_max_drawdown=0.10,
        baseline_max_drawdowns={
            "flat": 0.0,
            "buy_and_hold": 0.12,
            "momentum": 0.15,
            "random": 0.20,
        },
    )


def test_adapter_preserves_candidate_statistics() -> None:
    evaluation = make_evaluation()

    evidence = MultiBaselinePromotionEvidenceAdapter().build(
        evaluation=evaluation,
        dsr_passed=True,
        simulator_validated=True,
    )

    assert evidence.fold_count == 5
    assert evidence.candidate_sharpe == 2.0
    assert evidence.candidate_bootstrap_lower_bound == 1.8
    assert evidence.candidate_max_drawdown == 0.10
    assert evidence.confidence_level == 0.95


def test_adapter_preserves_every_baseline() -> None:
    evaluation = make_evaluation()

    evidence = MultiBaselinePromotionEvidenceAdapter().build(
        evaluation=evaluation,
        dsr_passed=True,
        simulator_validated=True,
    )

    names = {
        comparison.baseline_name
        for comparison in evidence.comparison.comparisons
    }

    assert names == {
        "flat",
        "buy_and_hold",
        "momentum",
        "random",
    }

    assert evidence.comparison.baseline_count == 4


def test_adapter_preserves_individual_baseline_sharpes() -> None:
    evaluation = make_evaluation()

    evidence = MultiBaselinePromotionEvidenceAdapter().build(
        evaluation=evaluation,
        dsr_passed=True,
        simulator_validated=True,
    )

    sharpes = {
        comparison.baseline_name:
        comparison.baseline_sharpe
        for comparison in evidence.comparison.comparisons
    }

    assert sharpes == {
        "flat": 0.1,
        "buy_and_hold": 0.8,
        "momentum": 0.6,
        "random": 0.2,
    }


def test_adapter_preserves_individual_baseline_drawdowns() -> None:
    evaluation = make_evaluation()

    evidence = MultiBaselinePromotionEvidenceAdapter().build(
        evaluation=evaluation,
        dsr_passed=True,
        simulator_validated=True,
    )

    assert evidence.baseline_max_drawdowns == {
        "flat": 0.0,
        "buy_and_hold": 0.12,
        "momentum": 0.15,
        "random": 0.20,
    }


def test_adapter_preserves_dsr_and_simulator_evidence() -> None:
    evaluation = make_evaluation()

    evidence = MultiBaselinePromotionEvidenceAdapter().build(
        evaluation=evaluation,
        dsr_passed=False,
        simulator_validated=False,
    )

    assert evidence.dsr_passed is False
    assert evidence.simulator_validated is False


def test_adapter_rejects_empty_baseline_evidence() -> None:
    evaluation = make_evaluation()

    

    # EvaluationProtocolResult itself requires the baseline
    # statistics and comparison names to remain aligned, so
    # construct an evaluation with no baselines by testing the
    # adapter contract through a minimal fake object.
    class EmptyEvaluation:
        baseline_count = 0

    with pytest.raises(ValueError, match="at least one baseline"):
        MultiBaselinePromotionEvidenceAdapter().build(
            evaluation=EmptyEvaluation(),
            dsr_passed=True,
            simulator_validated=True,
        )


def test_adapter_protocol_version_is_nonempty() -> None:
    adapter = MultiBaselinePromotionEvidenceAdapter()

    assert adapter.protocol_version