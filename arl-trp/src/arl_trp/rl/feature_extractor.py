import torch
from torch import nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TradingFeatureExtractor(BaseFeaturesExtractor):
    """
    Feed-forward feature extractor for the v0.1 trading observation.

    Input:
        (batch_size, 64, 10)

    Output:
        (batch_size, features_dim)

    The temporal observation window is flattened internally.
    No recurrent state is used.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
    ) -> None:
        if not isinstance(observation_space, spaces.Box):
            raise TypeError(
                "observation_space must be a gymnasium.spaces.Box"
            )

        if observation_space.shape != (64, 10):
            raise ValueError(
                "observation_space must have shape (64, 10)"
            )

        super().__init__(
            observation_space,
            features_dim=features_dim,
        )

        input_dim = 64 * 10

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)