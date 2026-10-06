import pytest

from arl_trp.rl.config import PPOConfig
import tomllib
from pathlib import Path



def test_default_configuration():
    config = PPOConfig()

    assert config.algorithm == "PPO"
    assert config.version == "0.1"

    assert config.observation_window == 64
    assert config.observation_features == 10

    assert config.hidden_dim == 256
    assert config.output_dim == 128

    assert config.learning_rate == 0.0003
    assert config.n_steps == 2048
    assert config.batch_size == 64
    assert config.n_epochs == 10
    assert config.gamma == 0.99
    assert config.gae_lambda == 0.95
    assert config.clip_range == 0.2
    assert config.ent_coef == 0.0
    assert config.vf_coef == 0.5
    assert config.max_grad_norm == 0.5

    assert config.seed == 42
    assert config.device == "cpu"


def test_observation_shape():
    config = PPOConfig()

    assert config.observation_shape == (64, 10)


def test_invalid_observation_window():
    with pytest.raises(ValueError):
        PPOConfig(observation_window=0)


def test_invalid_observation_features():
    with pytest.raises(ValueError):
        PPOConfig(observation_features=0)


def test_invalid_hidden_dimension():
    with pytest.raises(ValueError):
        PPOConfig(hidden_dim=0)


def test_invalid_output_dimension():
    with pytest.raises(ValueError):
        PPOConfig(output_dim=0)


def test_invalid_learning_rate():
    with pytest.raises(ValueError):
        PPOConfig(learning_rate=0.0)


def test_invalid_rollout_steps():
    with pytest.raises(ValueError):
        PPOConfig(n_steps=0)


def test_batch_size_cannot_exceed_rollout_steps():
    with pytest.raises(ValueError):
        PPOConfig(n_steps=32, batch_size=64)


def test_invalid_gamma():
    with pytest.raises(ValueError):
        PPOConfig(gamma=0.0)

    with pytest.raises(ValueError):
        PPOConfig(gamma=1.1)


def test_invalid_gae_lambda():
    with pytest.raises(ValueError):
        PPOConfig(gae_lambda=0.0)

    with pytest.raises(ValueError):
        PPOConfig(gae_lambda=1.1)


def test_invalid_seed():
    with pytest.raises(ValueError):
        PPOConfig(seed=-1)



from arl_trp.rl.config import PPOConfig


def test_toml_matches_default_configuration():
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "ppo_baseline.toml"
    )

    data = tomllib.loads(config_path.read_text())

    config = PPOConfig()

    assert data["algorithm"]["name"] == config.algorithm
    assert data["algorithm"]["version"] == config.version

    assert data["observation"]["window"] == config.observation_window
    assert data["observation"]["features"] == config.observation_features
    assert tuple(data["observation"]["shape"]) == config.observation_shape

    assert data["feature_extractor"]["hidden_dim"] == config.hidden_dim
    assert data["feature_extractor"]["output_dim"] == config.output_dim

    ppo = data["ppo"]

    assert ppo["learning_rate"] == config.learning_rate
    assert ppo["n_steps"] == config.n_steps
    assert ppo["batch_size"] == config.batch_size
    assert ppo["n_epochs"] == config.n_epochs
    assert ppo["gamma"] == config.gamma
    assert ppo["gae_lambda"] == config.gae_lambda
    assert ppo["clip_range"] == config.clip_range
    assert ppo["ent_coef"] == config.ent_coef
    assert ppo["vf_coef"] == config.vf_coef
    assert ppo["max_grad_norm"] == config.max_grad_norm

    assert data["reproducibility"]["seed"] == config.seed
    assert data["reproducibility"]["device"] == config.device

def test_load_configuration_from_toml():
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "ppo_baseline.toml"
    )

    config = PPOConfig.from_toml(config_path)
    default = PPOConfig()

    assert config == default