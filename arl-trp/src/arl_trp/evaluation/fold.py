from dataclasses import dataclass

from arl_trp.evaluation.report import (
    EvaluationReport,
    EvaluationReporter,
)
from arl_trp.evaluation.policy import (
    PolicyEvaluator,
)


@dataclass(frozen=True)
class FoldEvaluationResult:
    """
    Evaluation result for one walk-forward test fold.

    A fold evaluation contains:
        - fold identity
        - deterministic policy evaluation
        - derived performance metrics

    This layer does not train models or make promotion decisions.
    """

    fold_id: int
    report: EvaluationReport

    def __post_init__(self) -> None:
        if self.fold_id < 0:
            raise ValueError("fold_id cannot be negative")


class FoldEvaluator:
    """
    Evaluates a trained policy on exactly one evaluation environment.

    The supplied environment must already represent the intended
    walk-forward test fold.

    Training data, fold construction, and model selection remain
    outside this class.
    """

    def __init__(
        self,
        policy_evaluator: PolicyEvaluator | None = None,
        reporter: EvaluationReporter | None = None,
    ) -> None:
        self.policy_evaluator = (
            policy_evaluator
            if policy_evaluator is not None
            else PolicyEvaluator()
        )

        self.reporter = (
            reporter
            if reporter is not None
            else EvaluationReporter()
        )

    def evaluate(
        self,
        model,
        environment,
        fold_id: int,
    ) -> FoldEvaluationResult:
        if fold_id < 0:
            raise ValueError(
                "fold_id cannot be negative"
            )

        result = self.policy_evaluator.evaluate(
            model=model,
            environment=environment,
        )

        report = self.reporter.build(result)

        return FoldEvaluationResult(
            fold_id=fold_id,
            report=report,
        )
