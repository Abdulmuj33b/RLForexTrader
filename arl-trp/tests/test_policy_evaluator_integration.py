import numpy as np

from arl_trp.evaluation.policy import (
    PolicyEvaluationResult,
    PolicyEvaluator,
)
from arl_trp.rl.config import PPOConfig
from arl_trp.rl.ppo import build_ppo_model
from tests.test_observation_integration import make_environment


def make_integration_environment():
    return make_environment()


def make_integration_model(environment):
    config = PPOConfig(
        n_steps=64,
        batch_size=32,
    )

    return build_ppo_model(
        environment,
        config,
    )


def test_policy_evaluator_runs_with_real_trading_environment():
    environment = make_integration_environment()

    model = make_integration_model(
        environment
    )

    model.learn(
        total_timesteps=128,
        progress_bar=False,
    )

    result = PolicyEvaluator().evaluate(
        model=model,
        environment=environment,
    )

    assert isinstance(
        result,
        PolicyEvaluationResult,
    )

    assert result.steps > 0

    assert result.equity_curve.ndim == 1

    assert result.equity_curve.size == (
        result.steps + 1
    )

    assert result.actions.size == result.steps

    assert np.all(
        np.isfinite(result.equity_curve)
    )

    assert np.all(
        result.equity_curve >= 0.0
    )

    assert np.all(
        np.isfinite(result.trade_pnls)
    )


def test_policy_evaluator_real_environment_is_deterministic():
    environment = make_integration_environment()

    model = make_integration_model(
        environment
    )

    model.learn(
        total_timesteps=128,
        progress_bar=False,
    )

    evaluator = PolicyEvaluator()

    result_1 = evaluator.evaluate(
        model=model,
        environment=environment,
    )

    result_2 = evaluator.evaluate(
        model=model,
        environment=environment,
    )

    np.testing.assert_array_equal(
        result_1.equity_curve,
        result_2.equity_curve,
    )

    np.testing.assert_array_equal(
        result_1.trade_pnls,
        result_2.trade_pnls,
    )

    np.testing.assert_array_equal(
        result_1.actions,
        result_2.actions,
    )

    assert result_1.steps == result_2.steps
