from __future__ import annotations

from dataclasses import dataclass

from arl_trp.evaluation.evaluation_protocol import (
    EvaluationProtocolResult,
)
from arl_trp.evaluation.multi_baseline_promotion import (
    MultiBaselinePromotionEvidence,
)


@dataclass(frozen=True)
class MultiBaselinePromotionEvidenceAdapter:
    """
    Converts EvaluationProtocolResult into the evidence contract
    required by MultiBaselinePromotionGate.

    This adapter is intentionally lossless.

    It performs no baseline aggregation and makes no statistical
    promotion decision.

    v0.1:
        EvaluationProtocolResult
            ->
        MultiBaselinePromotionEvidence
    """

    protocol_version: str = (
        "multi_baseline_promotion_adapter_v0.1"
    )

    def __post_init__(self) -> None:
        if not self.protocol_version:
            raise ValueError(
                "protocol_version cannot be empty"
            )

    def build(
        self,
        evaluation: EvaluationProtocolResult,
        dsr_passed: bool,
        simulator_validated: bool,
    ) -> MultiBaselinePromotionEvidence:
        if evaluation.baseline_count == 0:
            raise ValueError(
                "evaluation must contain at least one baseline"
            )

        return MultiBaselinePromotionEvidence(
            fold_count=evaluation.candidate_fold_count,
            candidate_sharpe=(
                evaluation.candidate_statistics.observed_sharpe
            ),
            candidate_bootstrap_lower_bound=(
                evaluation.candidate_statistics
                .bootstrap
                .lower_bound
            ),
            candidate_max_drawdown=(
                evaluation.candidate_max_drawdown
            ),
            comparison=evaluation.comparison,
            baseline_max_drawdowns=dict(
                evaluation.baseline_max_drawdowns
            ),
            dsr_passed=dsr_passed,
            simulator_validated=simulator_validated,
            confidence_level=(
                evaluation.candidate_statistics
                .bootstrap
                .confidence_level
            ),
        )