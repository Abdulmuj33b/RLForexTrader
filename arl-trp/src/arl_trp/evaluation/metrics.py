from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvaluationMetrics:
    """
    Deterministic performance metrics for one evaluation run.

    v0.1 metrics:

        - initial equity
        - final equity
        - net P/L
        - total return
        - Sharpe ratio
        - maximum drawdown
        - number of trades
        - winning trades
        - losing trades
        - win rate
        - profit factor
    """

    initial_equity: float
    final_equity: float
    net_pnl: float
    total_return: float
    sharpe_ratio: float
    maximum_drawdown: float
    number_of_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float

    def __post_init__(self) -> None:
        if self.initial_equity <= 0.0:
            raise ValueError(
                "initial_equity must be positive"
            )

        if self.final_equity < 0.0:
            raise ValueError(
                "final_equity cannot be negative"
            )

        if self.number_of_trades < 0:
            raise ValueError(
                "number_of_trades cannot be negative"
            )

        if self.winning_trades < 0:
            raise ValueError(
                "winning_trades cannot be negative"
            )

        if self.losing_trades < 0:
            raise ValueError(
                "losing_trades cannot be negative"
            )

        if (
            self.winning_trades
            + self.losing_trades
            > self.number_of_trades
        ):
            raise ValueError(
                "winning and losing trades cannot exceed "
                "number_of_trades"
            )

        if not 0.0 <= self.maximum_drawdown <= 1.0:
            raise ValueError(
                "maximum_drawdown must be between 0 and 1"
            )

        if not 0.0 <= self.win_rate <= 1.0:
            raise ValueError(
                "win_rate must be between 0 and 1"
            )


def _validate_equity_curve(
    equity_curve: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        equity_curve,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "equity_curve must be one-dimensional"
        )

    if values.size < 2:
        raise ValueError(
            "equity_curve must contain at least two observations"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "equity_curve must contain only finite values"
        )

    if np.any(values < 0.0):
        raise ValueError(
            "equity_curve cannot contain negative equity"
        )

    if values[0] <= 0.0:
        raise ValueError(
            "initial equity must be positive"
        )

    return values


def _validate_trade_pnls(
    trade_pnls: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        trade_pnls,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "trade_pnls must be one-dimensional"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "trade_pnls must contain only finite values"
        )

    return values


def calculate_total_return(
    equity_curve: np.ndarray,
) -> float:
    """
    Calculate cumulative return.

        total_return =
            (final_equity / initial_equity) - 1
    """

    values = _validate_equity_curve(equity_curve)

    return float(
        (values[-1] / values[0]) - 1.0
    )


def calculate_max_drawdown(
    equity_curve: np.ndarray,
) -> float:
    """
    Calculate maximum peak-to-trough drawdown.
    """

    values = _validate_equity_curve(equity_curve)

    running_peak = np.maximum.accumulate(values)

    drawdowns = (
        running_peak - values
    ) / running_peak

    return float(np.max(drawdowns))


def calculate_sharpe_ratio(
    equity_curve: np.ndarray,
    periods_per_year: float = 1.0,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    v0.1 uses population standard deviation (ddof=0).

    Zero volatility conventions:

        positive mean return -> +inf
        negative mean return -> -inf
        zero mean return     -> 0
    """

    if periods_per_year <= 0.0:
        raise ValueError(
            "periods_per_year must be positive"
        )

    values = _validate_equity_curve(equity_curve)

    returns = (
        values[1:] / values[:-1]
    ) - 1.0

    mean_return = float(
        np.mean(returns)
    )

    standard_deviation = float(
        np.std(
            returns,
            ddof=0,
        )
    )

    if standard_deviation == 0.0:
        if mean_return > 0.0:
            return float("inf")

        if mean_return < 0.0:
            return float("-inf")

        return 0.0

    return float(
        (
            mean_return
            / standard_deviation
        )
        * np.sqrt(periods_per_year)
    )


def calculate_win_rate(
    trade_pnls: np.ndarray,
) -> float:
    """
    Calculate the proportion of profitable trades.

    Zero-P/L trades are excluded from the denominator.
    """

    values = _validate_trade_pnls(
        trade_pnls
    )

    if values.size == 0:
        return 0.0

    wins = np.sum(
        values > 0.0
    )

    losses = np.sum(
        values < 0.0
    )

    decisive_trades = wins + losses

    if decisive_trades == 0:
        return 0.0

    return float(
        wins / decisive_trades
    )


def calculate_profit_factor(
    trade_pnls: np.ndarray,
) -> float:
    """
    Calculate:

        gross_profit / gross_loss
    """

    values = _validate_trade_pnls(
        trade_pnls
    )

    gross_profit = float(
        np.sum(
            values[values > 0.0]
        )
    )

    gross_loss = float(
        -np.sum(
            values[values < 0.0]
        )
    )

    if gross_loss == 0.0:
        if gross_profit > 0.0:
            return float("inf")

        return 0.0

    return float(
        gross_profit / gross_loss
    )


def evaluate_metrics(
    equity_curve: np.ndarray,
    trade_pnls: np.ndarray,
    periods_per_year: float = 1.0,
) -> EvaluationMetrics:
    """
    Calculate the complete v0.1 evaluation metric set.
    """

    equity = _validate_equity_curve(
        equity_curve
    )

    trades = _validate_trade_pnls(
        trade_pnls
    )

    initial_equity = float(
        equity[0]
    )

    final_equity = float(
        equity[-1]
    )

    net_pnl = (
        final_equity
        - initial_equity
    )

    total_return = calculate_total_return(
        equity
    )

    maximum_drawdown = calculate_max_drawdown(
        equity
    )

    sharpe_ratio = calculate_sharpe_ratio(
        equity,
        periods_per_year=periods_per_year,
    )

    number_of_trades = int(
        trades.size
    )

    winning_trades = int(
        np.sum(trades > 0.0)
    )

    losing_trades = int(
        np.sum(trades < 0.0)
    )

    win_rate = calculate_win_rate(
        trades
    )

    profit_factor = calculate_profit_factor(
        trades
    )

    return EvaluationMetrics(
        initial_equity=initial_equity,
        final_equity=final_equity,
        net_pnl=net_pnl,
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        maximum_drawdown=maximum_drawdown,
        number_of_trades=number_of_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
    )
