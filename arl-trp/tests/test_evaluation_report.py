import numpy as np
import pytest

from arl_trp.evaluation.metrics import (
    EvaluationMetrics,
)
from arl_trp.evaluation.policy import (
    PolicyEvaluationResult,
)
from arl_trp.evaluation.report import (
    EvaluationReport,
    EvaluationReporter,
)


def make_result():
    return PolicyEvaluationResult(
        equity_curve=np.array(
            [
                1000.0,
                1010.0,
                1005.0,
                1020.0,
            ],
            dtype=np.float64,
        ),
        trade_pnls=np.array(
            [
                10.0,
                -5.0,
                15.0,
            ],
            dtype=np.float64,
        ),
        actions=np.array(
            [1, 0, 3],
            dtype=np.int64,
        ),
        steps=3,
    )


def test_report_contains_result_and_metrics():
    result = make_result()

    report = EvaluationReporter().build(result)

    assert isinstance(
        report,
        EvaluationReport,
    )

    assert report.result is result

    assert isinstance(
        report.metrics,
        EvaluationMetrics,
    )


def test_report_metrics_match_policy_result():
    result = make_result()

    report = EvaluationReporter().build(result)

    assert report.metrics.initial_equity == pytest.approx(
        1000.0
    )

    assert report.metrics.final_equity == pytest.approx(
        1020.0
    )

    assert report.metrics.net_pnl == pytest.approx(
        20.0
    )

    assert report.metrics.number_of_trades == 3

    assert report.metrics.winning_trades == 2

    assert report.metrics.losing_trades == 1

    assert report.metrics.win_rate == pytest.approx(
        2.0 / 3.0
    )


def test_report_preserves_raw_evaluation_data():
    result = make_result()

    report = EvaluationReporter().build(result)

    np.testing.assert_array_equal(
        report.result.equity_curve,
        result.equity_curve,
    )

    np.testing.assert_array_equal(
        report.result.trade_pnls,
        result.trade_pnls,
    )

    np.testing.assert_array_equal(
        report.result.actions,
        result.actions,
    )


def test_report_rejects_invalid_periods_per_year():
    with pytest.raises(ValueError):
        EvaluationReporter(
            periods_per_year=0.0
        )


def test_report_rejects_negative_periods_per_year():
    with pytest.raises(ValueError):
        EvaluationReporter(
            periods_per_year=-1.0
        )


def test_report_uses_configured_periods_per_year():
    result = make_result()

    report = EvaluationReporter(
        periods_per_year=252.0
    ).build(result)

    assert report.metrics.sharpe_ratio != pytest.approx(
        0.0
    )
