import numpy as np

from arl_trp.rl.config import PPOConfig
from arl_trp.rl.ppo import build_ppo_model
from arl_trp.rl.feature_extractor import TradingFeatureExtractor
from tests.test_observation_integration import make_environment


def make_ppo_environment():
    return make_environment()


def make_ppo_model(env):
    config = PPOConfig(
        n_steps=64,
        batch_size=32,
    )
    return build_ppo_model(env, config)


def test_ppo_model_constructs_with_trading_environment():
    env = make_ppo_environment()
    model = make_ppo_model(env)

    assert isinstance(
        model.policy.features_extractor,
        TradingFeatureExtractor,
    )


def test_ppo_predict_returns_valid_action():
    env = make_ppo_environment()
    model = make_ppo_model(env)

    observation, _ = env.reset()

    action, _ = model.predict(
        observation,
        deterministic=True,
    )

    assert env.action_space.contains(action)


def test_ppo_short_training_smoke_test():
    env = make_ppo_environment()
    model = make_ppo_model(env)

    model.learn(
        total_timesteps=128,
        progress_bar=False,
    )


def test_ppo_prediction_is_deterministic():
    env = make_ppo_environment()
    model = make_ppo_model(env)

    observation, _ = env.reset()

    action_1, _ = model.predict(
        observation,
        deterministic=True,
    )

    action_2, _ = model.predict(
        observation,
        deterministic=True,
    )

    assert np.array_equal(action_1, action_2)