import numpy as np
import pytest

from arl_trp.evaluation.metrics import (
    EvaluationMetrics,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_total_return,
    calculate_win_rate,
    evaluate_metrics,
)


def test_total_return_is_calculated_from_equity_curve():
    equity = np.array(
        [1000.0, 1050.0, 1100.0],
        dtype=np.float32,
    )

    result = calculate_total_return(equity)

    assert result == pytest.approx(0.10)


def test_total_return_can_be_negative():
    equity = np.array(
        [1000.0, 950.0, 900.0],
        dtype=np.float32,
    )

    result = calculate_total_return(equity)

    assert result == pytest.approx(-0.10)


def test_maximum_drawdown_is_calculated():
    equity = np.array(
        [1000.0, 1100.0, 1050.0, 900.0, 950.0],
        dtype=np.float64,
    )

    result = calculate_max_drawdown(equity)

    assert result == pytest.approx(200.0 / 1100.0)


def test_maximum_drawdown_is_zero_for_monotonic_equity():
    equity = np.array(
        [1000.0, 1010.0, 1020.0, 1030.0],
        dtype=np.float64,
    )

    result = calculate_max_drawdown(equity)

    assert result == pytest.approx(0.0)


def test_sharpe_ratio_is_zero_for_constant_equity():
    equity = np.array(
        [1000.0, 1000.0, 1000.0],
        dtype=np.float64,
    )

    result = calculate_sharpe_ratio(
        equity,
        periods_per_year=252,
    )

    assert result == pytest.approx(0.0)


def test_sharpe_ratio_is_positive_for_positive_constant_returns():
    equity = np.array(
        [100.0, 101.0, 102.01],
        dtype=np.float64,
    )

    result = calculate_sharpe_ratio(
        equity,
        periods_per_year=1,
    )

    assert np.isinf(result)
    assert result > 0.0


def test_sharpe_ratio_rejects_invalid_annualization():
    equity = np.array(
        [1000.0, 1010.0],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        calculate_sharpe_ratio(
            equity,
            periods_per_year=0.0,
        )


def test_win_rate_counts_only_profitable_and_losing_trades():
    trades = np.array(
        [10.0, -5.0, 20.0, -2.0, 0.0],
        dtype=np.float64,
    )

    result = calculate_win_rate(trades)

    assert result == pytest.approx(0.5)


def test_win_rate_is_zero_without_trades():
    trades = np.array([], dtype=np.float64)

    result = calculate_win_rate(trades)

    assert result == pytest.approx(0.0)


def test_profit_factor_is_gross_profit_divided_by_gross_loss():
    trades = np.array(
        [100.0, -50.0, 25.0, -25.0],
        dtype=np.float64,
    )

    result = calculate_profit_factor(trades)

    assert result == pytest.approx(125.0 / 75.0)


def test_profit_factor_is_infinite_without_losses():
    trades = np.array(
        [100.0, 50.0],
        dtype=np.float64,
    )

    result = calculate_profit_factor(trades)

    assert np.isinf(result)
    assert result > 0.0


def test_profit_factor_is_zero_without_winning_trades():
    trades = np.array(
        [-100.0, -50.0],
        dtype=np.float64,
    )

    result = calculate_profit_factor(trades)

    assert result == pytest.approx(0.0)


def test_complete_evaluation_returns_expected_metrics():
    equity = np.array(
        [1000.0, 1050.0, 1000.0, 1100.0],
        dtype=np.float64,
    )

    trades = np.array(
        [50.0, -50.0, 100.0],
        dtype=np.float64,
    )

    result = evaluate_metrics(
        equity_curve=equity,
        trade_pnls=trades,
        periods_per_year=1,
    )

    assert isinstance(result, EvaluationMetrics)

    assert result.initial_equity == pytest.approx(1000.0)
    assert result.final_equity == pytest.approx(1100.0)
    assert result.net_pnl == pytest.approx(100.0)
    assert result.total_return == pytest.approx(0.10)

    assert result.maximum_drawdown == pytest.approx(
        50.0 / 1050.0
    )

    assert result.number_of_trades == 3
    assert result.winning_trades == 2
    assert result.losing_trades == 1

    assert result.win_rate == pytest.approx(2.0 / 3.0)

    assert result.profit_factor == pytest.approx(3.0)


def test_equity_curve_must_be_one_dimensional():
    equity = np.array(
        [
            [1000.0, 1010.0],
            [1020.0, 1030.0],
        ]
    )

    with pytest.raises(ValueError):
        calculate_total_return(equity)


def test_equity_curve_requires_at_least_two_observations():
    equity = np.array([1000.0], dtype=np.float64)

    with pytest.raises(ValueError):
        calculate_total_return(equity)


def test_equity_curve_cannot_start_at_zero():
    equity = np.array(
        [0.0, 100.0],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        calculate_total_return(equity)


def test_equity_curve_cannot_contain_negative_values():
    equity = np.array(
        [1000.0, 900.0, -100.0],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        calculate_max_drawdown(equity)


def test_trade_pnls_must_be_one_dimensional():
    trades = np.array(
        [
            [10.0, -5.0],
            [20.0, -2.0],
        ]
    )

    with pytest.raises(ValueError):
        calculate_win_rate(trades)


def test_metrics_reject_inconsistent_trade_counts():
    with pytest.raises(ValueError):
        EvaluationMetrics(
            initial_equity=1000.0,
            final_equity=1100.0,
            net_pnl=100.0,
            total_return=0.10,
            sharpe_ratio=1.0,
            maximum_drawdown=0.05,
            number_of_trades=2,
            winning_trades=2,
            losing_trades=1,
            win_rate=1.0,
            profit_factor=2.0,
        )


def test_metrics_reject_invalid_drawdown():
    with pytest.raises(ValueError):
        EvaluationMetrics(
            initial_equity=1000.0,
            final_equity=1100.0,
            net_pnl=100.0,
            total_return=0.10,
            sharpe_ratio=1.0,
            maximum_drawdown=1.5,
            number_of_trades=1,
            winning_trades=1,
            losing_trades=0,
            win_rate=1.0,
            profit_factor=float("inf"),
        )