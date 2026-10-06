import numpy as np
import pytest

from arl_trp.evaluation.comparison import (
    BaselineComparator,
    BaselineComparison,
    BaselineComparisonResult,
)
from arl_trp.evaluation.statistics import (
    BootstrapResult,
    OOSStatisticsResult,
)


def make_statistics(sharpe: float) -> OOSStatisticsResult:
    returns = np.array(
        [0.01, -0.005, 0.02],
        dtype=np.float64,
    )

    bootstrap = BootstrapResult(
        observed_statistic=sharpe,
        bootstrap_statistics=np.array(
            [sharpe, sharpe],
            dtype=np.float64,
        ),
        confidence_level=0.95,
        lower_bound=sharpe - 0.1,
        upper_bound=sharpe + 0.1,
        block_size=2,
        iterations=2,
        seed=42,
    )

    return OOSStatisticsResult(
        pooled_returns=returns,
        observed_sharpe=sharpe,
        bootstrap=bootstrap,
    )


def test_baseline_comparison_records_sharpe_superiority():
    comparison = BaselineComparison(
        baseline_name="flat",
        candidate_sharpe=1.20,
        baseline_sharpe=0.80,
        candidate_max_drawdown=0.10,
        baseline_max_drawdown=0.10,
    )

    assert comparison.sharpe_superior is True
    assert comparison.sharpe_difference == pytest.approx(0.40)
    assert comparison.drawdown_difference == pytest.approx(0.0)


def test_baseline_comparison_detects_non_superiority():
    comparison = BaselineComparison(
        baseline_name="momentum",
        candidate_sharpe=0.70,
        baseline_sharpe=0.80,
        candidate_max_drawdown=0.10,
        baseline_max_drawdown=0.10,
    )

    assert comparison.sharpe_superior is False
    assert comparison.sharpe_difference == pytest.approx(-0.10)


def test_comparison_result_preserves_all_baselines():
    result = BaselineComparisonResult(
        comparisons=(
            BaselineComparison(
                baseline_name="flat",
                candidate_sharpe=1.2,
                baseline_sharpe=0.5,
                candidate_max_drawdown=0.1,
                baseline_max_drawdown=0.1,
            ),
            BaselineComparison(
                baseline_name="momentum",
                candidate_sharpe=1.2,
                baseline_sharpe=0.9,
                candidate_max_drawdown=0.1,
                baseline_max_drawdown=0.1,
            ),
        )
    )

    assert result.baseline_count == 2
    assert result.all_sharpe_superior is True
    assert result.protocol_version == "comparison_v0.1"


def test_comparator_compares_every_baseline():
    comparator = BaselineComparator()

    candidate = make_statistics(1.20)

    baselines = {
        "flat": make_statistics(0.20),
        "momentum": make_statistics(0.80),
        "random": make_statistics(0.50),
    }

    result = comparator.compare(
        candidate=candidate,
        baselines=baselines,
        candidate_max_drawdown=0.10,
        baseline_max_drawdowns={
            "flat": 0.10,
            "momentum": 0.12,
            "random": 0.08,
        },
    )

    assert result.baseline_count == 3
    assert result.all_sharpe_superior is True

    assert tuple(
        comparison.baseline_name
        for comparison in result.comparisons
    ) == (
        "flat",
        "momentum",
        "random",
    )


def test_comparator_rejects_missing_drawdown():
    comparator = BaselineComparator()

    with pytest.raises(ValueError):
        comparator.compare(
            candidate=make_statistics(1.20),
            baselines={
                "flat": make_statistics(0.20),
            },
            candidate_max_drawdown=0.10,
            baseline_max_drawdowns={},
        )


def test_comparator_rejects_empty_baselines():
    comparator = BaselineComparator()

    with pytest.raises(ValueError):
        comparator.compare(
            candidate=make_statistics(1.20),
            baselines={},
            candidate_max_drawdown=0.10,
            baseline_max_drawdowns={},
        )


def test_comparison_result_rejects_duplicate_baselines():
    comparison = BaselineComparison(
        baseline_name="flat",
        candidate_sharpe=1.2,
        baseline_sharpe=0.5,
        candidate_max_drawdown=0.1,
        baseline_max_drawdown=0.1,
    )

    with pytest.raises(ValueError):
        BaselineComparisonResult(
            comparisons=(
                comparison,
                comparison,
            )
        )
