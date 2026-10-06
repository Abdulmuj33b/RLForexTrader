from __future__ import annotations

from dataclasses import dataclass

from arl_trp.evaluation.statistics import (
    OOSStatisticsResult,
)


@dataclass(frozen=True)
class BaselineComparison:
    """
    Comparison of a candidate against one baseline.

    The comparison layer does not make a promotion decision.
    """

    baseline_name: str

    candidate_sharpe: float
    baseline_sharpe: float

    candidate_max_drawdown: float
    baseline_max_drawdown: float

    def __post_init__(self) -> None:
        if not self.baseline_name:
            raise ValueError(
                "baseline_name cannot be empty"
            )

        if self.candidate_max_drawdown < 0.0:
            raise ValueError(
                "candidate_max_drawdown cannot be negative"
            )

        if self.baseline_max_drawdown < 0.0:
            raise ValueError(
                "baseline_max_drawdown cannot be negative"
            )

    @property
    def sharpe_superior(self) -> bool:
        return (
            self.candidate_sharpe
            > self.baseline_sharpe
        )

    @property
    def sharpe_difference(self) -> float:
        return (
            self.candidate_sharpe
            - self.baseline_sharpe
        )

    @property
    def drawdown_difference(self) -> float:
        return (
            self.candidate_max_drawdown
            - self.baseline_max_drawdown
        )


@dataclass(frozen=True)
class BaselineComparisonResult:
    """
    Comparison evidence for a candidate against all supplied
    baselines.

    This object records evidence only. It does not make a
    promotion decision.
    """

    comparisons: tuple[BaselineComparison, ...]
    protocol_version: str = "comparison_v0.1"

    def __post_init__(self) -> None:
        if not self.comparisons:
            raise ValueError(
                "comparisons cannot be empty"
            )

        names = tuple(
            comparison.baseline_name
            for comparison in self.comparisons
        )

        if len(set(names)) != len(names):
            raise ValueError(
                "baseline names must be unique"
            )

        if not self.protocol_version:
            raise ValueError(
                "protocol_version cannot be empty"
            )

    @property
    def baseline_count(self) -> int:
        return len(self.comparisons)

    @property
    def all_sharpe_superior(self) -> bool:
        return all(
            comparison.sharpe_superior
            for comparison in self.comparisons
        )


class BaselineComparator:
    """
    Compares candidate OOS statistics against one or more
    baseline OOS statistics.

    v0.1:
        - comparison is based on pooled OOS Sharpe
        - maximum drawdown is retained as secondary evidence
        - no promotion decision is made here
        - every supplied baseline is preserved
    """

    PROTOCOL_VERSION = "comparison_v0.1"

    def compare(
        self,
        candidate: OOSStatisticsResult,
        baselines: dict[str, OOSStatisticsResult],
        candidate_max_drawdown: float,
        baseline_max_drawdowns: dict[str, float],
    ) -> BaselineComparisonResult:
        if not baselines:
            raise ValueError(
                "baselines cannot be empty"
            )

        if candidate_max_drawdown < 0.0:
            raise ValueError(
                "candidate_max_drawdown cannot be negative"
            )

        comparisons: list[BaselineComparison] = []

        for baseline_name, baseline in baselines.items():
            if not baseline_name:
                raise ValueError(
                    "baseline name cannot be empty"
                )

            if baseline_name not in baseline_max_drawdowns:
                raise ValueError(
                    f"missing maximum drawdown for baseline: "
                    f"{baseline_name}"
                )

            baseline_max_drawdown = (
                baseline_max_drawdowns[baseline_name]
            )

            if baseline_max_drawdown < 0.0:
                raise ValueError(
                    "baseline maximum drawdown cannot be negative"
                )

            comparisons.append(
                BaselineComparison(
                    baseline_name=baseline_name,
                    candidate_sharpe=candidate.observed_sharpe,
                    baseline_sharpe=baseline.observed_sharpe,
                    candidate_max_drawdown=(
                        candidate_max_drawdown
                    ),
                    baseline_max_drawdown=(
                        baseline_max_drawdown
                    ),
                )
            )

        return BaselineComparisonResult(
            comparisons=tuple(comparisons),
            protocol_version=self.PROTOCOL_VERSION,
        )
