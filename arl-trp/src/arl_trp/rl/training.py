import json
from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import PPO

from arl_trp.data.dataset import DatasetSpec
from arl_trp.research.governance import TrialGovernance
from arl_trp.research.trial import TrialRecord
from arl_trp.rl.config import PPOConfig
from arl_trp.rl.ppo import build_ppo_model



@dataclass(frozen=True)
class TrainingRunConfig:
    """
    Reproducibility metadata for one PPO training run.

    This identifies the trial, experiment, dataset, PPO
    configuration, and output artifact for the run.

    Research-governance metadata is optional at construction
    time so existing training and smoke-test workflows remain
    compatible. A TrialRecord can only be created explicitly
    through to_trial_record().
    """

    trial_id: str
    experiment_id: str
    dataset: DatasetSpec
    output_dir: Path
    total_timesteps: int
    ppo_config: PPOConfig

    def __post_init__(self) -> None:
        if not self.trial_id:
            raise ValueError("trial_id cannot be empty")

        if not self.experiment_id:
            raise ValueError("experiment_id cannot be empty")

        if self.total_timesteps <= 0:
            raise ValueError(
                "total_timesteps must be positive"
            )

    @property
    def model_path(self) -> Path:
        return self.output_dir / f"{self.experiment_id}.zip"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / f"{self.experiment_id}.json"

    def metadata(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "experiment_id": self.experiment_id,
            "dataset": {
                "version": self.dataset.version,
                "instrument": self.dataset.instrument,
                "timeframe": self.dataset.timeframe,
                "source": self.dataset.source,
                "start": self.dataset.start.isoformat(),
                "end": self.dataset.end.isoformat(),
            },
            "total_timesteps": self.total_timesteps,
            "ppo": {
                "algorithm": self.ppo_config.algorithm,
                "version": self.ppo_config.version,
                "observation_window": (
                    self.ppo_config.observation_window
                ),
                "observation_features": (
                    self.ppo_config.observation_features
                ),
                "hidden_dim": self.ppo_config.hidden_dim,
                "output_dim": self.ppo_config.output_dim,
                "learning_rate": self.ppo_config.learning_rate,
                "n_steps": self.ppo_config.n_steps,
                "batch_size": self.ppo_config.batch_size,
                "n_epochs": self.ppo_config.n_epochs,
                "gamma": self.ppo_config.gamma,
                "gae_lambda": self.ppo_config.gae_lambda,
                "clip_range": self.ppo_config.clip_range,
                "ent_coef": self.ppo_config.ent_coef,
                "vf_coef": self.ppo_config.vf_coef,
                "max_grad_norm": self.ppo_config.max_grad_norm,
                "seed": self.ppo_config.seed,
                "device": self.ppo_config.device,
            },
        }

    def to_trial_record(
        self,
        governance: TrialGovernance,
    ) -> TrialRecord:
        """
        Convert this training configuration into an immutable
        research TrialRecord.

        Governance participation is explicit. Calling this method
        does not register the trial in a ledger.
        """

        return TrialRecord(
            trial_id=self.trial_id,
            experiment_family=governance.experiment_family,
            instrument=self.dataset.instrument,
            model_version=self.ppo_config.version,
            dataset_version=self.dataset.version,
            feature_version=governance.feature_version,
            reward_version=governance.reward_version,
            simulator_version=governance.simulator_version,
            seed=self.ppo_config.seed,
            selection_eligible=(
                governance.selection_eligible
            ),
            selection_stage=governance.selection_stage,
            trigger_reason=governance.trigger_reason,
            protocol_version=governance.protocol_version,
            split_protocol_version=(
                governance.split_protocol_version
            ),
            parent_trial_id=governance.parent_trial_id,
        )


def train_ppo(
    env,
    run_config: TrainingRunConfig,
) -> PPO:
    """
    Train one PPO experiment and save its model artifact
    together with its run metadata.

    The environment is supplied by the caller so that dataset,
    split, and environment construction remain explicit.
    """

    run_config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = build_ppo_model(
        env=env,
        config=run_config.ppo_config,
    )

    model.learn(
        total_timesteps=run_config.total_timesteps,
        progress_bar=False,
    )

    model.save(run_config.model_path)

    with run_config.metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_config.metadata(),
            file,
            indent=2,
            sort_keys=True,
        )

    return model
