from dataclasses import dataclass

from arl_trp.evaluation.fold import (
    FoldEvaluationResult,
    FoldEvaluator,
)


@dataclass(frozen=True)
class WalkForwardEvaluationResult:
    """
    Complete evaluation output for a walk-forward run.

    Each fold remains independent and retains its own evaluation
    result and metrics.

    v0.1:
        - fold results are preserved in chronological order
        - no averaging of Sharpe ratios
        - no statistical significance testing
        - no model selection
        - no promotion decision
    """

    folds: tuple[FoldEvaluationResult, ...]

    def __post_init__(self) -> None:
        if not self.folds:
            raise ValueError(
                "walk-forward evaluation must contain at least one fold"
            )

        fold_ids = tuple(
            fold.fold_id
            for fold in self.folds
        )

        if fold_ids != tuple(sorted(fold_ids)):
            raise ValueError(
                "fold evaluation results must be ordered by fold_id"
            )

        if len(set(fold_ids)) != len(fold_ids):
            raise ValueError(
                "fold evaluation results must have unique fold_ids"
            )

    @property
    def fold_count(self) -> int:
        return len(self.folds)


class WalkForwardEvaluator:
    """
    Evaluates a sequence of already-constructed walk-forward folds.

    Training remains external to this class. Each supplied item contains
    a model and its corresponding OOS evaluation environment.

    The evaluator deliberately does not aggregate performance metrics.
    """

    def __init__(
        self,
        fold_evaluator: FoldEvaluator | None = None,
    ) -> None:
        self.fold_evaluator = (
            fold_evaluator
            if fold_evaluator is not None
            else FoldEvaluator()
        )

    def evaluate(
        self,
        fold_runs: tuple[tuple[object, object, int], ...],
    ) -> WalkForwardEvaluationResult:
        if not fold_runs:
            raise ValueError(
                "fold_runs cannot be empty"
            )

        results: list[FoldEvaluationResult] = []

        for model, environment, fold_id in fold_runs:
            result = self.fold_evaluator.evaluate(
                model=model,
                environment=environment,
                fold_id=fold_id,
            )

            results.append(result)

        results.sort(
            key=lambda result: result.fold_id
        )

        return WalkForwardEvaluationResult(
            folds=tuple(results)
        )
