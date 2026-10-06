import numpy as np
import torch
from gymnasium import spaces

from arl_trp.rl.feature_extractor import TradingFeatureExtractor


def make_observation_space():
    return spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(64, 10),
        dtype=np.float32,
    )


def test_feature_extractor_output_dimension():
    extractor = TradingFeatureExtractor(
        observation_space=make_observation_space(),
        features_dim=128,
    )

    observation = torch.zeros((1, 64, 10), dtype=torch.float32)

    output = extractor(observation)

    assert output.shape == (1, 128)
    assert extractor.features_dim == 128


def test_feature_extractor_accepts_batch_observations():
    extractor = TradingFeatureExtractor(
        observation_space=make_observation_space(),
        features_dim=128,
    )

    observation = torch.zeros((8, 64, 10), dtype=torch.float32)

    output = extractor(observation)

    assert output.shape == (8, 128)


def test_feature_extractor_output_is_finite():
    extractor = TradingFeatureExtractor(
        observation_space=make_observation_space(),
        features_dim=128,
    )

    observation = torch.randn(
        (4, 64, 10),
        dtype=torch.float32,
    )

    output = extractor(observation)

    assert torch.isfinite(output).all()


def test_feature_extractor_is_deterministic():
    torch.manual_seed(42)

    extractor = TradingFeatureExtractor(
        observation_space=make_observation_space(),
        features_dim=128,
    )

    observation = torch.randn(
        (2, 64, 10),
        dtype=torch.float32,
    )

    output_1 = extractor(observation)
    output_2 = extractor(observation)

    assert torch.equal(output_1, output_2)