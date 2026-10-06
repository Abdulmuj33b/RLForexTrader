from dataclasses import dataclass
import tomllib
from pathlib import Path


@dataclass(frozen=True)
class PPOConfig:
    """
    Versioned baseline configuration for PPO v0.1.

    This configuration defines the initial feed-forward PPO
    experiment baseline. It is not a tuned configuration.
    """

    algorithm: str = "PPO"
    version: str = "0.1"

    observation_window: int = 64
    observation_features: int = 10

    hidden_dim: int = 256
    output_dim: int = 128

    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2

    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.algorithm != "PPO":
            raise ValueError("algorithm must be PPO")

        if not self.version:
            raise ValueError("version cannot be empty")

        if self.observation_window <= 0:
            raise ValueError(
                "observation_window must be positive"
            )

        if self.observation_features <= 0:
            raise ValueError(
                "observation_features must be positive"
            )

        if self.hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive"
            )

        if self.output_dim <= 0:
            raise ValueError(
                "output_dim must be positive"
            )

        if self.learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.n_steps <= 0:
            raise ValueError(
                "n_steps must be positive"
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        if self.batch_size > self.n_steps:
            raise ValueError(
                "batch_size cannot exceed n_steps"
            )

        if self.n_epochs <= 0:
            raise ValueError(
                "n_epochs must be positive"
            )

        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(
                "gamma must be in (0, 1]"
            )

        if not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError(
                "gae_lambda must be in (0, 1]"
            )

        if self.clip_range <= 0.0:
            raise ValueError(
                "clip_range must be positive"
            )

        if self.ent_coef < 0.0:
            raise ValueError(
                "ent_coef cannot be negative"
            )

        if self.vf_coef < 0.0:
            raise ValueError(
                "vf_coef cannot be negative"
            )

        if self.max_grad_norm <= 0.0:
            raise ValueError(
                "max_grad_norm must be positive"
            )

        if self.seed < 0:
            raise ValueError(
                "seed cannot be negative"
            )
    @classmethod
    def from_toml(cls, path: str | Path) -> "PPOConfig":
        """
        Load a PPO configuration from a TOML file.
        """

        path = Path(path)

        with path.open("rb") as file:
            data = tomllib.load(file)

        algorithm = data["algorithm"]
        observation = data["observation"]
        feature_extractor = data["feature_extractor"]
        ppo = data["ppo"]
        reproducibility = data["reproducibility"]

        return cls(
            algorithm=algorithm["name"],
            version=algorithm["version"],
            observation_window=observation["window"],
            observation_features=observation["features"],
            hidden_dim=feature_extractor["hidden_dim"],
            output_dim=feature_extractor["output_dim"],
            learning_rate=ppo["learning_rate"],
            n_steps=ppo["n_steps"],
            batch_size=ppo["batch_size"],
            n_epochs=ppo["n_epochs"],
            gamma=ppo["gamma"],
            gae_lambda=ppo["gae_lambda"],
            clip_range=ppo["clip_range"],
            ent_coef=ppo["ent_coef"],
            vf_coef=ppo["vf_coef"],
            max_grad_norm=ppo["max_grad_norm"],
            seed=reproducibility["seed"],
            device=reproducibility["device"],
        )

    @property
    def observation_shape(self) -> tuple[int, int]:
        return (
            self.observation_window,
            self.observation_features,
        )