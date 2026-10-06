from stable_baselines3 import PPO

from arl_trp.rl.config import PPOConfig
from arl_trp.rl.ppo import build_ppo_model
from arl_trp.rl.feature_extractor import TradingFeatureExtractor
from tests.test_observation_integration import make_environment


def test_build_ppo_model_uses_configuration():
    env = make_environment()
    config = PPOConfig()

    model = build_ppo_model(env, config)

    assert isinstance(model, PPO)
    assert isinstance(
        model.policy.features_extractor,
        TradingFeatureExtractor,
    )


def test_build_ppo_model_uses_configured_feature_dimension():
    env = make_environment()
    config = PPOConfig(output_dim=64)

    model = build_ppo_model(env, config)

    assert model.policy.features_extractor.features_dim == 64


def test_build_ppo_model_rejects_observation_shape_mismatch():
    env = make_environment()
    config = PPOConfig(
        observation_window=32,
    )

    try:
        build_ppo_model(env, config)
    except ValueError as exc:
        assert "observation shape" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for observation shape mismatch"
        )