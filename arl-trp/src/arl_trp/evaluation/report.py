from dataclasses import dataclass

from arl_trp.evaluation.metrics import (
    EvaluationMetrics,
    evaluate_metrics,
)
from arl_trp.evaluation.policy import (
    PolicyEvaluationResult,
)


@dataclass(frozen=True)
class EvaluationReport:
    """
    Complete evaluation output for one deterministic policy evaluation.

    The report preserves the raw policy-evaluation result alongside the
    derived performance metrics.

    v0.1:
        - policy execution is performed by PolicyEvaluator
        - metrics are calculated from the resulting equity curve
          and completed-trade P/L
        - no promotion decision is made here
        - no statistical significance test is performed here
    """

    result: PolicyEvaluationResult
    metrics: EvaluationMetrics


class EvaluationReporter:
    """
    Converts a PolicyEvaluationResult into a complete EvaluationReport.

    This class deliberately does not contain:
        - training logic
        - model selection
        - promotion logic
        - walk-forward splitting
        - bootstrap testing
        - Deflated Sharpe calculations

    Those concerns belong to later research/evaluation layers.
    """

    def __init__(
        self,
        periods_per_year: float = 1.0,
    ) -> None:
        if periods_per_year <= 0.0:
            raise ValueError(
                "periods_per_year must be positive"
            )

        self.periods_per_year = periods_per_year

    def build(
        self,
        result: PolicyEvaluationResult,
    ) -> EvaluationReport:
        metrics = evaluate_metrics(
            equity_curve=result.equity_curve,
            trade_pnls=result.trade_pnls,
            periods_per_year=self.periods_per_year,
        )

        return EvaluationReport(
            result=result,
            metrics=metrics,
        )
