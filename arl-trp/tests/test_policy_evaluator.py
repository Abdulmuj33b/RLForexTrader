import numpy as np
import pytest

from arl_trp.domain.order import Action
from arl_trp.domain.transition import TradeTransition
from arl_trp.evaluation.policy import (
    PolicyEvaluationResult,
    PolicyEvaluator,
)


class FakeModel:
    def __init__(self, actions):
        self.actions = list(actions)
        self.index = 0

    def predict(
        self,
        observation,
        deterministic=False,
    ):
        assert deterministic is True

        action = self.actions[self.index]
        self.index += 1

        return np.array(action), None


class FakeAccounting:
    def __init__(self, balance):
        self.balance = balance


class FakeEnvironment:
    def __init__(
        self,
        equities,
        transitions,
    ):
        self.accounting = FakeAccounting(
            equities[0]
        )

        self._equities = list(equities)
        self._transitions = list(transitions)
        self._step = 0

    def reset(self):
        self._step = 0
        self.accounting.balance = (
            self._equities[0]
        )

        return np.array([0.0]), {}

    def step(self, action):
        assert action in {0, 1, 2, 3}

        self._step += 1

        equity = self._equities[
            self._step
        ]

        self.accounting.balance = equity

        info = {
            "equity": equity,
            "transition": self._transitions[
                self._step - 1
            ],
        }

        terminated = (
            self._step
            >= len(self._equities) - 1
        )

        return (
            np.array(
                [float(self._step)]
            ),
            0.0,
            terminated,
            False,
            info,
        )


def make_transition(
    position_before,
    position_after,
    net_pnl,
):
    return TradeTransition(
        timestamp=None,
        action=Action.HOLD,
        position_before=position_before,
        position_after=position_after,
        fill_price=100.0,
        gross_pnl=net_pnl,
        transaction_cost=0.0,
        net_pnl=net_pnl,
        equity_before=1000.0,
        equity_after=1000.0 + net_pnl,
        reward=net_pnl,
    )


def test_policy_evaluator_returns_expected_result():
    transitions = [
        make_transition(
            0.0,
            1.0,
            -2.0,
        ),
        make_transition(
            1.0,
            0.0,
            25.0,
        ),
        make_transition(
            0.0,
            0.0,
            0.0,
        ),
    ]

    environment = FakeEnvironment(
        equities=[
            1000.0,
            998.0,
            1023.0,
            1023.0,
        ],
        transitions=transitions,
    )

    model = FakeModel(
        actions=[
            1,
            3,
            0,
        ]
    )

    result = PolicyEvaluator().evaluate(
        model=model,
        environment=environment,
    )

    assert isinstance(
        result,
        PolicyEvaluationResult,
    )

    np.testing.assert_array_equal(
        result.equity_curve,
        np.array(
            [
                1000.0,
                998.0,
                1023.0,
                1023.0,
            ]
        ),
    )

    np.testing.assert_array_equal(
        result.trade_pnls,
        np.array([25.0]),
    )

    np.testing.assert_array_equal(
        result.actions,
        np.array(
            [1, 3, 0]
        ),
    )

    assert result.steps == 3


def test_policy_evaluator_uses_deterministic_prediction():
    transitions = [
        make_transition(
            0.0,
            0.0,
            0.0,
        ),
    ]

    environment = FakeEnvironment(
        equities=[
            1000.0,
            1010.0,
        ],
        transitions=transitions,
    )

    model = FakeModel(
        actions=[1]
    )

    result = PolicyEvaluator().evaluate(
        model=model,
        environment=environment,
    )

    assert result.actions.tolist() == [1]


def test_policy_evaluator_does_not_count_opening_cost_as_trade():
    transitions = [
        make_transition(
            0.0,
            1.0,
            -5.0,
        ),
    ]

    environment = FakeEnvironment(
        equities=[
            1000.0,
            995.0,
        ],
        transitions=transitions,
    )

    model = FakeModel(
        actions=[1]
    )

    result = PolicyEvaluator().evaluate(
        model=model,
        environment=environment,
    )

    assert result.trade_pnls.size == 0


def test_policy_evaluator_records_completed_long_trade():
    transitions = [
        make_transition(
            0.0,
            1.0,
            -2.0,
        ),
        make_transition(
            1.0,
            0.0,
            30.0,
        ),
    ]

    environment = FakeEnvironment(
        equities=[
            1000.0,
            998.0,
            1028.0,
        ],
        transitions=transitions,
    )

    model = FakeModel(
        actions=[
            1,
            3,
        ]
    )

    result = PolicyEvaluator().evaluate(
        model=model,
        environment=environment,
    )

    np.testing.assert_array_equal(
        result.trade_pnls,
        np.array([30.0]),
    )


def test_policy_evaluator_records_completed_short_trade():
    transitions = [
        make_transition(
            0.0,
            -1.0,
            -2.0,
        ),
        make_transition(
            -1.0,
            0.0,
            40.0,
        ),
    ]

    environment = FakeEnvironment(
        equities=[
            1000.0,
            998.0,
            1038.0,
        ],
        transitions=transitions,
    )

    model = FakeModel(
        actions=[
            2,
            3,
        ]
    )

    result = PolicyEvaluator().evaluate(
        model=model,
        environment=environment,
    )

    np.testing.assert_array_equal(
        result.trade_pnls,
        np.array([40.0]),
    )


def test_policy_evaluator_requires_equity_in_step_info():
    class InvalidEnvironment:
        accounting = FakeAccounting(
            1000.0
        )

        def reset(self):
            return np.array([0.0]), {}

        def step(self, action):
            return (
                np.array([0.0]),
                0.0,
                True,
                False,
                {},
            )

    model = FakeModel([0])

    with pytest.raises(ValueError):
        PolicyEvaluator().evaluate(
            model=model,
            environment=InvalidEnvironment(),
        )


def test_policy_evaluator_rejects_multi_action_prediction():
    class InvalidModel:
        def predict(
            self,
            observation,
            deterministic=False,
        ):
            return np.array(
                [0, 1]
            ), None

    transitions = [
        make_transition(
            0.0,
            0.0,
            0.0,
        ),
    ]

    environment = FakeEnvironment(
        equities=[
            1000.0,
            1005.0,
        ],
        transitions=transitions,
    )

    with pytest.raises(ValueError):
        PolicyEvaluator().evaluate(
            model=InvalidModel(),
            environment=environment,
        )


def test_policy_evaluation_result_rejects_mismatched_action_count():
    with pytest.raises(ValueError):
        PolicyEvaluationResult(
            equity_curve=np.array(
                [1000.0, 1010.0]
            ),
            trade_pnls=np.array([]),
            actions=np.array([]),
            steps=1,
        )


def test_policy_evaluation_result_requires_multiple_equity_points():
    with pytest.raises(ValueError):
        PolicyEvaluationResult(
            equity_curve=np.array(
                [1000.0]
            ),
            trade_pnls=np.array([]),
            actions=np.array([0]),
            steps=1,
        )