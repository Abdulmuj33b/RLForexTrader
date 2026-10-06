import numpy as np
import pytest

from arl_trp.evaluation.fold import (
    FoldEvaluationResult,
    FoldEvaluator,
)
from arl_trp.evaluation.metrics import (
    EvaluationMetrics,
)
from arl_trp.evaluation.report import (
    EvaluationReport,
)
from arl_trp.evaluation.policy import (
    PolicyEvaluationResult,
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

        terminated = self.step_count >= 3

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


def test_fold_evaluation_result_rejects_negative_fold_id():
    result = PolicyEvaluationResult(
        equity_curve=np.array(
            [1000.0, 1001.0],
            dtype=np.float64,
        ),
        trade_pnls=np.array(
            [],
            dtype=np.float64,
        ),
        actions=np.array(
            [0],
            dtype=np.int64,
        ),
        steps=1,
    )

    report = EvaluationReport(
        result=result,
        metrics=EvaluationMetrics(
            initial_equity=1000.0,
            final_equity=1001.0,
            net_pnl=1.0,
            total_return=0.001,
            sharpe_ratio=0.0,
            maximum_drawdown=0.0,
            number_of_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
        ),
    )

    with pytest.raises(ValueError):
        FoldEvaluationResult(
            fold_id=-1,
            report=report,
        )


def test_fold_evaluator_returns_fold_result():
    evaluator = FoldEvaluator()

    result = evaluator.evaluate(
        model=FakeModel(),
        environment=FakeEnvironment(),
        fold_id=0,
    )

    assert isinstance(
        result,
        FoldEvaluationResult,
    )

    assert result.fold_id == 0

    assert isinstance(
        result.report,
        EvaluationReport,
    )

    assert isinstance(
        result.report.result,
        PolicyEvaluationResult,
    )

    assert isinstance(
        result.report.metrics,
        EvaluationMetrics,
    )


def test_fold_evaluator_preserves_evaluation_steps():
    evaluator = FoldEvaluator()

    result = evaluator.evaluate(
        model=FakeModel(),
        environment=FakeEnvironment(),
        fold_id=2,
    )

    assert result.fold_id == 2

    assert result.report.result.steps == 3

    assert (
        result.report.result.equity_curve.size
        == 4
    )


def test_fold_evaluator_is_deterministic():
    evaluator = FoldEvaluator()

    result_1 = evaluator.evaluate(
        model=FakeModel(),
        environment=FakeEnvironment(),
        fold_id=0,
    )

    result_2 = evaluator.evaluate(
        model=FakeModel(),
        environment=FakeEnvironment(),
        fold_id=0,
    )

    np.testing.assert_array_equal(
        result_1.report.result.equity_curve,
        result_2.report.result.equity_curve,
    )

    np.testing.assert_array_equal(
        result_1.report.result.actions,
        result_2.report.result.actions,
    )
