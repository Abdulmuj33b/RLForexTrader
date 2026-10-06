from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class FoldEvaluationCollection(Protocol):
    """
    Structural contract required by OOSStatistics.

    Any walk-forward evaluation result is compatible if it exposes
    a `folds` collection whose elements contain:

        fold.report.result.equity_curve
    """

    folds: tuple


@dataclass(frozen=True)
class BootstrapResult:
    """
    Result of a moving-block bootstrap over pooled OOS returns.
    """

    observed_statistic: float
    bootstrap_statistics: np.ndarray
    confidence_level: float
    lower_bound: float
    upper_bound: float
    block_size: int
    iterations: int
    seed: int

    def __post_init__(self) -> None:
        if self.bootstrap_statistics.ndim != 1:
            raise ValueError(
                "bootstrap_statistics must be one-dimensional"
            )

        if self.bootstrap_statistics.size != self.iterations:
            raise ValueError(
                "bootstrap_statistics size must equal iterations"
            )

        if self.iterations <= 0:
            raise ValueError(
                "iterations must be positive"
            )

        if self.block_size <= 0:
            raise ValueError(
                "block_size must be positive"
            )

        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                "confidence_level must be between 0 and 1"
            )

        if self.lower_bound > self.upper_bound:
            raise ValueError(
                "lower_bound cannot exceed upper_bound"
            )


@dataclass(frozen=True)
class OOSStatisticsResult:
    """
    Statistical summary of a pooled chronological OOS evaluation.

    Fold-level results remain preserved in the source evaluation
    result. This object contains the pooled return series and
    bootstrap result.
    """

    pooled_returns: np.ndarray
    observed_sharpe: float
    bootstrap: BootstrapResult

    def __post_init__(self) -> None:
        if self.pooled_returns.ndim != 1:
            raise ValueError(
                "pooled_returns must be one-dimensional"
            )

        if self.pooled_returns.size == 0:
            raise ValueError(
                "pooled_returns cannot be empty"
            )


class OOSStatistics:
    """
    v0.1 OOS statistical protocol.

    Primary statistic:
        pooled chronological OOS Sharpe ratio.

    Bootstrap:
        moving block bootstrap
        block size = 288
        iterations = 10,000
        confidence level = 0.95
        percentile interval

    The implementation deliberately does not:
        - calculate DSR
        - compare candidates
        - make promotion decisions
        - discard zero-trade folds
        - average fold Sharpe ratios
    """

    PROTOCOL_VERSION = "oos_stats_v0.1"

    def __init__(
        self,
        periods_per_year: float = 1.0,
        block_size: int = 288,
        iterations: int = 10_000,
        confidence_level: float = 0.95,
        seed: int = 42,
    ) -> None:
        if periods_per_year <= 0.0:
            raise ValueError(
                "periods_per_year must be positive"
            )

        if block_size <= 0:
            raise ValueError(
                "block_size must be positive"
            )

        if iterations <= 0:
            raise ValueError(
                "iterations must be positive"
            )

        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                "confidence_level must be between 0 and 1"
            )

        self.periods_per_year = periods_per_year
        self.block_size = block_size
        self.iterations = iterations
        self.confidence_level = confidence_level
        self.seed = seed

    def evaluate(
        self,
        walk_forward_result: FoldEvaluationCollection,
    ) -> OOSStatisticsResult:
        pooled_returns = self._extract_pooled_returns(
            walk_forward_result
        )

        observed_sharpe = self._calculate_sharpe(
            pooled_returns
        )

        bootstrap = self._bootstrap_sharpe(
            pooled_returns
        )

        return OOSStatisticsResult(
            pooled_returns=pooled_returns,
            observed_sharpe=observed_sharpe,
            bootstrap=bootstrap,
        )

    def _extract_pooled_returns(
        self,
        walk_forward_result: FoldEvaluationCollection,
    ) -> np.ndarray:
        returns: list[np.ndarray] = []

        for fold in walk_forward_result.folds:
            equity_curve = np.asarray(
                fold.report.result.equity_curve,
                dtype=np.float64,
            )

            if equity_curve.ndim != 1:
                raise ValueError(
                    "fold equity curve must be one-dimensional"
                )

            if equity_curve.size < 2:
                raise ValueError(
                    "fold equity curve must contain at least "
                    "two observations"
                )

            if not np.all(
                np.isfinite(equity_curve)
            ):
                raise ValueError(
                    "fold equity curve must contain finite values"
                )

            if np.any(equity_curve <= 0.0):
                raise ValueError(
                    "equity must remain strictly positive "
                    "for return calculation"
                )

            fold_returns = (
                equity_curve[1:]
                / equity_curve[:-1]
                - 1.0
            )

            returns.append(fold_returns)

        if not returns:
            raise ValueError(
                "walk-forward result contains no folds"
            )

        pooled = np.concatenate(returns)

        if not np.all(np.isfinite(pooled)):
            raise ValueError(
                "pooled returns must be finite"
            )

        return pooled

    def _calculate_sharpe(
        self,
        returns: np.ndarray,
    ) -> float:
        returns = np.asarray(
            returns,
            dtype=np.float64,
        )

        if returns.ndim != 1:
            raise ValueError(
                "returns must be one-dimensional"
            )

        if returns.size == 0:
            raise ValueError(
                "returns cannot be empty"
            )

        if not np.all(
            np.isfinite(returns)
        ):
            raise ValueError(
                "returns must be finite"
            )

        mean = float(np.mean(returns))
        standard_deviation = float(
            np.std(
                returns,
                ddof=0,
            )
        )

        if standard_deviation == 0.0:
            if mean > 0.0:
                return float("inf")

            if mean < 0.0:
                return float("-inf")

            return 0.0

        return (
            mean
            / standard_deviation
            * np.sqrt(self.periods_per_year)
        )

    def _bootstrap_sharpe(
        self,
        returns: np.ndarray,
    ) -> BootstrapResult:
        effective_block_size = min(
            self.block_size,
            returns.size,
        )

        rng = np.random.default_rng(
            self.seed
        )

        bootstrap_statistics = np.empty(
            self.iterations,
            dtype=np.float64,
        )

        for iteration in range(
            self.iterations
        ):
            sample = self._sample_blocks(
                returns=returns,
                block_size=effective_block_size,
                rng=rng,
            )

            bootstrap_statistics[iteration] = (
                self._calculate_sharpe(sample)
            )

        alpha = 1.0 - self.confidence_level

        lower_bound = float(
            np.quantile(
                bootstrap_statistics,
                alpha / 2.0,
            )
        )

        upper_bound = float(
            np.quantile(
                bootstrap_statistics,
                1.0 - alpha / 2.0,
            )
        )

        observed_statistic = self._calculate_sharpe(
            returns
        )

        return BootstrapResult(
            observed_statistic=observed_statistic,
            bootstrap_statistics=bootstrap_statistics,
            confidence_level=self.confidence_level,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            block_size=effective_block_size,
            iterations=self.iterations,
            seed=self.seed,
        )

    @staticmethod
    def _sample_blocks(
        returns: np.ndarray,
        block_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        observation_count = returns.size

        if block_size <= 0:
            raise ValueError(
                "block_size must be positive"
            )

        if block_size > observation_count:
            raise ValueError(
                "block_size cannot exceed number of observations"
            )

        block_count = int(
            np.ceil(
                observation_count
                / block_size
            )
        )

        starts = rng.integers(
            0,
            observation_count - block_size + 1,
            size=block_count,
        )

        blocks = [
            returns[
                start:start + block_size
            ]
            for start in starts
        ]

        return np.concatenate(
            blocks
        )[:observation_count]
