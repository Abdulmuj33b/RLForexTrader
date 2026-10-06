from __future__ import annotations

from dataclasses import dataclass

from arl_trp.evaluation.comparison import (
    BaselineComparisonResult,
    BaselineComparator,
)
from arl_trp.evaluation.statistics import OOSStatistics, OOSStatisticsResult


@dataclass(frozen=True)
class EvaluationProtocolResult:
    """
    Complete candidate-vs-baseline statistical evaluation.

    v0.1 deliberately stops before promotion.

    Responsibilities:
        1. Evaluate candidate OOS statistics.
        2. Evaluate every supplied baseline OOS result.
        3. Compare candidate Sharpe against every baseline.
        4. Preserve candidate/baseline statistical evidence.
        5. Preserve candidate fold count and drawdown evidence.
        6. Preserve the protocol version.

    Promotion decisions remain the responsibility of PromotionGate.
    """

    candidate_statistics: OOSStatisticsResult
    baseline_statistics: dict[str, OOSStatisticsResult]
    comparison: BaselineComparisonResult

    candidate_fold_count: int
    candidate_max_drawdown: float
    baseline_max_drawdowns: dict[str, float]

    protocol_version: str = "evaluation_protocol_v0.1"

    def __post_init__(self) -> None:
        if not self.baseline_statistics:
            raise ValueError(
                "baseline_statistics cannot be empty"
            )

        if not self.baseline_max_drawdowns:
            raise ValueError(
                "baseline_max_drawdowns cannot be empty"
            )

        if self.candidate_fold_count < 0:
            raise ValueError(
                "candidate_fold_count cannot be negative"
            )

        if self.candidate_max_drawdown < 0.0:
            raise ValueError(
                "candidate_max_drawdown cannot be negative"
            )

        if set(self.baseline_statistics) != set(
            self.baseline_max_drawdowns
        ):
            raise ValueError(
                "baseline statistics and baseline drawdowns "
                "must contain the same baseline names"
            )

        comparison_names = {
            comparison.baseline_name
            for comparison in self.comparison.comparisons
        }

        if set(self.baseline_statistics) != comparison_names:
            raise ValueError(
                "comparison baselines must match baseline statistics"
            )

        if not self.protocol_version:
            raise ValueError(
                "protocol_version cannot be empty"
            )

    @property
    def baseline_count(self) -> int:
        return len(self.baseline_statistics)

    @property
    def all_baselines_sharpe_superior(self) -> bool:
        return self.comparison.all_sharpe_superior


class EvaluationProtocol:
    """
    Builds the frozen v0.1 candidate-vs-baseline evaluation evidence.

    This class does not make a promotion decision.
    """

    PROTOCOL_VERSION = "evaluation_protocol_v0.1"

    def __init__(
        self,
        statistics: OOSStatistics | None = None,
        comparator: BaselineComparator | None = None,
    ) -> None:
        self.statistics = statistics or OOSStatistics()
        self.comparator = comparator or BaselineComparator()

    def evaluate(
        self,
        candidate_walk_forward,
        baseline_walk_forwards: dict[str, object],
        candidate_max_drawdown: float,
        baseline_max_drawdowns: dict[str, float],
    ) -> EvaluationProtocolResult:
        if not baseline_walk_forwards:
            raise ValueError(
                "baseline_walk_forwards cannot be empty"
            )

        if candidate_max_drawdown < 0.0:
            raise ValueError(
                "candidate_max_drawdown cannot be negative"
            )

        if set(baseline_walk_forwards) != set(
            baseline_max_drawdowns
        ):
            raise ValueError(
                "baseline walk-forward results and baseline "
                "drawdowns must contain the same baseline names"
            )

        candidate_fold_count = len(
            candidate_walk_forward.folds
        )

        if candidate_fold_count == 0:
            raise ValueError(
                "candidate walk-forward result contains no folds"
            )

        candidate_statistics = self.statistics.evaluate(
            candidate_walk_forward
        )

        baseline_statistics: dict[str, OOSStatisticsResult] = {}

        for baseline_name, walk_forward_result in (
            baseline_walk_forwards.items()
        ):
            if not baseline_name:
                raise ValueError(
                    "baseline name cannot be empty"
                )

            if not walk_forward_result.folds:
                raise ValueError(
                    f"baseline '{baseline_name}' "
                    "contains no folds"
                )

            baseline_statistics[baseline_name] = (
                self.statistics.evaluate(
                    walk_forward_result
                )
            )

        comparison = self.comparator.compare(
            candidate=candidate_statistics,
            baselines=baseline_statistics,
            candidate_max_drawdown=candidate_max_drawdown,
            baseline_max_drawdowns=baseline_max_drawdowns,
        )

        return EvaluationProtocolResult(
            candidate_statistics=candidate_statistics,
            baseline_statistics=baseline_statistics,
            comparison=comparison,
            candidate_fold_count=candidate_fold_count,
            candidate_max_drawdown=candidate_max_drawdown,
            baseline_max_drawdowns=dict(
                baseline_max_drawdowns
            ),
            protocol_version=self.PROTOCOL_VERSION,
        )