from __future__ import annotations

from arl_trp.evaluation.baselines import (
    BaselineEvaluator,
    BaselinePolicy,
)
from arl_trp.evaluation.fold import FoldEvaluationResult
from arl_trp.evaluation.report import EvaluationReporter


class BaselineFoldEvaluator:
    """
    Evaluates one baseline policy on one walk-forward test fold.

    This evaluator deliberately mirrors the downstream contract of
    FoldEvaluator without modifying the existing PPO evaluation path.

    Pipeline:

        BaselinePolicy
            ↓
        BaselineEvaluator
            ↓
        PolicyEvaluationResult
            ↓
        EvaluationReporter
            ↓
        FoldEvaluationResult
    """

    def __init__(
        self,
        baseline_evaluator: BaselineEvaluator | None = None,
        reporter: EvaluationReporter | None = None,
    ) -> None:
        self.baseline_evaluator = (
            baseline_evaluator
            if baseline_evaluator is not None
            else BaselineEvaluator()
        )

        self.reporter = (
            reporter
            if reporter is not None
            else EvaluationReporter()
        )

    def evaluate(
        self,
        policy: BaselinePolicy,
        environment,
        fold_id: int,
    ) -> FoldEvaluationResult:
        if fold_id < 0:
            raise ValueError(
                "fold_id cannot be negative"
            )

        result = self.baseline_evaluator.evaluate(
            policy=policy,
            environment=environment,
        )

        report = self.reporter.build(result)

        return FoldEvaluationResult(
            fold_id=fold_id,
            report=report,
        )
