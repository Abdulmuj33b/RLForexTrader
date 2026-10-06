import numpy as np
import pytest

from arl_trp.evaluation.fold import (
    FoldEvaluationResult,
)
from arl_trp.evaluation.metrics import (
    EvaluationMetrics,
)
from arl_trp.evaluation.policy import (
    PolicyEvaluationResult,
)
from arl_trp.evaluation.report import (
    EvaluationReport,
)
from arl_trp.evaluation.statistics import (
    BootstrapResult,
    OOSStatistics,
    OOSStatisticsResult,
)
from arl_trp.evaluation.walk_forward import (
    WalkForwardEvaluationResult,
)


def make_fold(
    fold_id: int,
    equity_curve: list[float],
) -> FoldEvaluationResult:
    equity = np.asarray(
        equity_curve,
        dtype=np.float64,
    )

    trade_pnls = np.array(
        [1.0],
        dtype=np.float64,
    )

    actions = np.zeros(
        equity.size - 1,
        dtype=np.int64,
    )

    result = PolicyEvaluationResult(
        equity_curve=equity,
        trade_pnls=trade_pnls,
        actions=actions,
        steps=equity.size - 1,
    )

    metrics = EvaluationMetrics(
        initial_equity=float(equity[0]),
        final_equity=float(equity[-1]),
        net_pnl=float(
            equity[-1] - equity[0]
        ),
        total_return=float(
            equity[-1] / equity[0] - 1.0
        ),
        sharpe_ratio=0.0,
        maximum_drawdown=0.0,
        number_of_trades=1,
        winning_trades=1,
        losing_trades=0,
        win_rate=1.0,
        profit_factor=float("inf"),
    )

    report = EvaluationReport(
        result=result,
        metrics=metrics,
    )

    return FoldEvaluationResult(
        fold_id=fold_id,
        report=report,
    )


def make_walk_forward_result():
    return WalkForwardEvaluationResult(
        folds=(
            make_fold(
                1,
                [1000.0, 1010.0, 1005.0],
            ),
            make_fold(
                2,
                [1000.0, 1020.0, 1010.0],
            ),
        )
    )


def test_extracts_pooled_returns_from_all_folds():
    statistics = OOSStatistics(
        periods_per_year=1.0,
        block_size=2,
        iterations=10,
        seed=42,
    )

    result = statistics.evaluate(
        make_walk_forward_result()
    )

    expected = np.array(
        [
            0.01,
            -0.00495049504950495,
            0.02,
            -0.00980392156862745,
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        result.pooled_returns,
        expected,
    )


def test_does_not_average_fold_sharpes():
    statistics = OOSStatistics(
        periods_per_year=1.0,
        block_size=2,
        iterations=10,
        seed=42,
    )

    result = statistics.evaluate(
        make_walk_forward_result()
    )

    direct_sharpe = (
        np.mean(result.pooled_returns)
        / np.std(
            result.pooled_returns,
            ddof=0,
        )
    )

    assert result.observed_sharpe == pytest.approx(
        direct_sharpe
    )


def test_bootstrap_result_has_expected_shape():
    statistics = OOSStatistics(
        periods_per_year=1.0,
        block_size=2,
        iterations=100,
        seed=42,
    )

    result = statistics.evaluate(
        make_walk_forward_result()
    )

    assert isinstance(
        result,
        OOSStatisticsResult,
    )

    assert isinstance(
        result.bootstrap,
        BootstrapResult,
    )

    assert (
        result.bootstrap.bootstrap_statistics.size
        == 100
    )

    assert (
        result.bootstrap.iterations
        == 100
    )


def test_bootstrap_is_reproducible_with_same_seed():
    walk_forward = make_walk_forward_result()

    statistics_1 = OOSStatistics(
        periods_per_year=1.0,
        block_size=2,
        iterations=100,
        seed=42,
    )

    statistics_2 = OOSStatistics(
        periods_per_year=1.0,
        block_size=2,
        iterations=100,
        seed=42,
    )

    result_1 = statistics_1.evaluate(
        walk_forward
    )

    result_2 = statistics_2.evaluate(
        walk_forward
    )

    np.testing.assert_array_equal(
        result_1.bootstrap.bootstrap_statistics,
        result_2.bootstrap.bootstrap_statistics,
    )

    assert (
        result_1.bootstrap.lower_bound
        == result_2.bootstrap.lower_bound
    )

    assert (
        result_1.bootstrap.upper_bound
        == result_2.bootstrap.upper_bound
    )


def test_different_seed_changes_bootstrap_sample():
    walk_forward = make_walk_forward_result()

    result_1 = OOSStatistics(
        periods_per_year=1.0,
        block_size=2,
        iterations=100,
        seed=42,
    ).evaluate(walk_forward)

    result_2 = OOSStatistics(
        periods_per_year=1.0,
        block_size=2,
        iterations=100,
        seed=123,
    ).evaluate(walk_forward)

    assert not np.array_equal(
        result_1.bootstrap.bootstrap_statistics,
        result_2.bootstrap.bootstrap_statistics,
    )


def test_confidence_interval_is_ordered():
    result = OOSStatistics(
        periods_per_year=1.0,
        block_size=2,
        iterations=1000,
        seed=42,
    ).evaluate(
        make_walk_forward_result()
    )

    assert (
        result.bootstrap.lower_bound
        <= result.bootstrap.upper_bound
    )


def test_block_size_is_capped_for_short_series():
    result = OOSStatistics(
        periods_per_year=1.0,
        block_size=288,
        iterations=10,
        seed=42,
    ).evaluate(
        make_walk_forward_result()
    )

    assert (
        result.bootstrap.block_size
        == result.pooled_returns.size
    )


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        OOSStatistics(
            periods_per_year=0.0
        )

    with pytest.raises(ValueError):
        OOSStatistics(
            block_size=0
        )

    with pytest.raises(ValueError):
        OOSStatistics(
            iterations=0
        )

    with pytest.raises(ValueError):
        OOSStatistics(
            confidence_level=1.0
        )


def test_non_positive_equity_is_rejected():
    walk_forward = WalkForwardEvaluationResult(
        folds=(
            make_fold(
                1,
                [1000.0, 0.0],
            ),
        )
    )

    with pytest.raises(ValueError):
        OOSStatistics(
            iterations=10,
        ).evaluate(
            walk_forward
        )
