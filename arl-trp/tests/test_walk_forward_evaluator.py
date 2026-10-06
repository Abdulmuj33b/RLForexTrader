import numpy as np
import pytest

from arl_trp.evaluation.fold import (
    FoldEvaluationResult,
)
from arl_trp.evaluation.walk_forward import (
    WalkForwardEvaluationResult,
    WalkForwardEvaluator,
)


class FakeModel:
    def predict(
        self,
        observation,
        deterministic=True,
    ):
        return 0, None


class FakeEnvironment:
    def __init__(self):
        self.accounting = type(
            "Accounting",
            (),
            {"balance": 1000.0},
        )()

        self.step_count = 0

    def reset(self):
        self.step_count = 0

        observation = np.zeros(
            (64, 10),
            dtype=np.float32,
        )

        return observation, {}

    def step(self, action):
        self.step_count += 1

        observation = np.zeros(
            (64, 10),
            dtype=np.float32,
        )

        terminated = self.step_count >= 2

        info = {
            "equity": 1000.0,
        }

        return (
            observation,
            0.0,
            terminated,
            False,
            info,
        )


def test_walk_forward_result_requires_at_least_one_fold():
    with pytest.raises(ValueError):
        WalkForwardEvaluationResult(
            folds=()
        )


def test_walk_forward_result_rejects_duplicate_fold_ids():
    evaluator = WalkForwardEvaluator()

    model = FakeModel()
    environment = FakeEnvironment()

    result_1 = evaluator.fold_evaluator.evaluate(
        model=model,
        environment=environment,
        fold_id=1,
    )

    result_2 = evaluator.fold_evaluator.evaluate(
        model=model,
        environment=environment,
        fold_id=1,
    )

    with pytest.raises(ValueError):
        WalkForwardEvaluationResult(
            folds=(result_1, result_2)
        )


def test_walk_forward_result_rejects_unsorted_fold_ids():
    evaluator = WalkForwardEvaluator()

    model = FakeModel()

    result_2 = evaluator.fold_evaluator.evaluate(
        model=model,
        environment=FakeEnvironment(),
        fold_id=2,
    )

    result_1 = evaluator.fold_evaluator.evaluate(
        model=model,
        environment=FakeEnvironment(),
        fold_id=1,
    )

    with pytest.raises(ValueError):
        WalkForwardEvaluationResult(
            folds=(result_2, result_1)
        )


def test_walk_forward_evaluator_returns_all_fold_results():
    evaluator = WalkForwardEvaluator()

    runs = (
        (
            FakeModel(),
            FakeEnvironment(),
            1,
        ),
        (
            FakeModel(),
            FakeEnvironment(),
            2,
        ),
        (
            FakeModel(),
            FakeEnvironment(),
            3,
        ),
    )

    result = evaluator.evaluate(runs)

    assert isinstance(
        result,
        WalkForwardEvaluationResult,
    )

    assert result.fold_count == 3

    assert tuple(
        fold.fold_id
        for fold in result.folds
    ) == (1, 2, 3)


def test_walk_forward_evaluator_orders_results_by_fold_id():
    evaluator = WalkForwardEvaluator()

    runs = (
        (
            FakeModel(),
            FakeEnvironment(),
            3,
        ),
        (
            FakeModel(),
            FakeEnvironment(),
            1,
        ),
        (
            FakeModel(),
            FakeEnvironment(),
            2,
        ),
    )

    result = evaluator.evaluate(runs)

    assert tuple(
        fold.fold_id
        for fold in result.folds
    ) == (1, 2, 3)


def test_walk_forward_evaluator_rejects_empty_runs():
    evaluator = WalkForwardEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(())


def test_each_fold_retains_independent_evaluation_result():
    evaluator = WalkForwardEvaluator()

    runs = (
        (
            FakeModel(),
            FakeEnvironment(),
            1,
        ),
        (
            FakeModel(),
            FakeEnvironment(),
            2,
        ),
    )

    result = evaluator.evaluate(runs)

    assert all(
        isinstance(
            fold,
            FoldEvaluationResult,
        )
        for fold in result.folds
    )

    assert all(
        fold.report.result.steps == 2
        for fold in result.folds
    )
