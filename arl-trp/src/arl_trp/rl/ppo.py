from stable_baselines3 import PPO

from arl_trp.environment.trading_env import TradingEnvironment
from arl_trp.rl.config import PPOConfig
from arl_trp.rl.feature_extractor import TradingFeatureExtractor


def build_ppo_model(
    env: TradingEnvironment,
    config: PPOConfig,
) -> PPO:
    """
    Build the versioned PPO baseline from PPOConfig.

    The configuration is the single source of truth for
    PPO hyperparameters and feature-extractor dimensions.
    """

    if env.observation_space.shape != config.observation_shape:
        raise ValueError(
            "environment observation shape does not match PPOConfig"
        )

    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        seed=config.seed,
        device=config.device,
        policy_kwargs={
            "features_extractor_class": TradingFeatureExtractor,
            "features_extractor_kwargs": {
                "features_dim": config.output_dim,
            },
        },
        verbose=0,
    )