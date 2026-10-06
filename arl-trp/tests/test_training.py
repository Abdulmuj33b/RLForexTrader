from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from arl_trp.data.dataset import DatasetSpec
from arl_trp.rl.config import PPOConfig
from arl_trp.rl.training import TrainingRunConfig, train_ppo
from tests.test_observation_integration import make_environment
from arl_trp.research.governance import TrialGovernance

def make_dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        version="xauusd_5m_v1",
        instrument="XAUUSD",
        timeframe="5m",
        source="historical_broker_data",
        start=datetime(
            2018,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        end=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )


def test_training_run_configuration():
    config = TrainingRunConfig(
        trial_id="trial_001",
        experiment_id="ppo_baseline_001",
        dataset=make_dataset_spec(),
        output_dir=Path("artifacts"),
        total_timesteps=100_000,
        ppo_config=PPOConfig(),
    )

    assert config.trial_id == "trial_001"
    assert config.experiment_id == "ppo_baseline_001"
    assert config.dataset.version == "xauusd_5m_v1"
    assert config.dataset.instrument == "XAUUSD"
    assert config.total_timesteps == 100_000
    assert config.ppo_config == PPOConfig()


def test_model_path():
    config = TrainingRunConfig(
        trial_id="trial_001",
        experiment_id="ppo_baseline_001",
        dataset=make_dataset_spec(),
        output_dir=Path("artifacts"),
        total_timesteps=100_000,
        ppo_config=PPOConfig(),
    )

    assert config.model_path == Path(
        "artifacts/ppo_baseline_001.zip"
    )


def test_empty_trial_id_rejected():
    with pytest.raises(ValueError):
        TrainingRunConfig(
            trial_id="",
            experiment_id="ppo_baseline_001",
            dataset=make_dataset_spec(),
            output_dir=Path("artifacts"),
            total_timesteps=100_000,
            ppo_config=PPOConfig(),
        )


def test_empty_experiment_id_rejected():
    with pytest.raises(ValueError):
        TrainingRunConfig(
            trial_id="trial_001",
            experiment_id="",
            dataset=make_dataset_spec(),
            output_dir=Path("artifacts"),
            total_timesteps=100_000,
            ppo_config=PPOConfig(),
        )


def test_invalid_timesteps_rejected():
    with pytest.raises(ValueError):
        TrainingRunConfig(
            trial_id="trial_001",
            experiment_id="ppo_baseline_001",
            dataset=make_dataset_spec(),
            output_dir=Path("artifacts"),
            total_timesteps=0,
            ppo_config=PPOConfig(),
        )


def test_train_ppo_smoke_run(tmp_path):
    env = make_environment()

    run_config = TrainingRunConfig(
        trial_id="trial_smoke_001",
        experiment_id="ppo_smoke_001",
        dataset=make_dataset_spec(),
        output_dir=tmp_path,
        total_timesteps=128,
        ppo_config=PPOConfig(
            n_steps=64,
            batch_size=32,
        ),
    )

    model = train_ppo(
        env=env,
        run_config=run_config,
    )

    assert model is not None
    assert run_config.model_path.exists()
    assert run_config.model_path.stat().st_size > 0


def test_metadata_path():
    config = TrainingRunConfig(
        trial_id="trial_001",
        experiment_id="ppo_baseline_001",
        dataset=make_dataset_spec(),
        output_dir=Path("artifacts"),
        total_timesteps=100_000,
        ppo_config=PPOConfig(),
    )

    assert config.metadata_path == Path(
        "artifacts/ppo_baseline_001.json"
    )


def test_metadata_contains_run_identity():
    config = TrainingRunConfig(
        trial_id="trial_001",
        experiment_id="ppo_baseline_001",
        dataset=make_dataset_spec(),
        output_dir=Path("artifacts"),
        total_timesteps=100_000,
        ppo_config=PPOConfig(),
    )

    metadata = config.metadata()

    assert metadata["trial_id"] == "trial_001"
    assert metadata["experiment_id"] == "ppo_baseline_001"
    assert metadata["total_timesteps"] == 100_000

    dataset = metadata["dataset"]

    assert dataset["version"] == "xauusd_5m_v1"
    assert dataset["instrument"] == "XAUUSD"
    assert dataset["timeframe"] == "5m"
    assert dataset["source"] == "historical_broker_data"
    assert dataset["start"] == "2018-01-01T00:00:00+00:00"
    assert dataset["end"] == "2026-01-01T00:00:00+00:00"


def test_metadata_contains_ppo_configuration():
    config = TrainingRunConfig(
        trial_id="trial_001",
        experiment_id="ppo_baseline_001",
        dataset=make_dataset_spec(),
        output_dir=Path("artifacts"),
        total_timesteps=100_000,
        ppo_config=PPOConfig(),
    )

    metadata = config.metadata()
    ppo = metadata["ppo"]

    assert ppo["algorithm"] == "PPO"
    assert ppo["version"] == "0.1"
    assert ppo["observation_window"] == 64
    assert ppo["observation_features"] == 10
    assert ppo["hidden_dim"] == 256
    assert ppo["output_dim"] == 128
    assert ppo["learning_rate"] == 0.0003
    assert ppo["seed"] == 42


def test_train_ppo_persists_metadata(tmp_path):
    env = make_environment()

    run_config = TrainingRunConfig(
        trial_id="trial_metadata_001",
        experiment_id="ppo_metadata_001",
        dataset=make_dataset_spec(),
        output_dir=tmp_path,
        total_timesteps=128,
        ppo_config=PPOConfig(
            n_steps=64,
            batch_size=32,
        ),
    )

    train_ppo(
        env=env,
        run_config=run_config,
    )

    assert run_config.metadata_path.exists()

    metadata = json.loads(
        run_config.metadata_path.read_text(
            encoding="utf-8"
        )
    )

    assert metadata["trial_id"] == "trial_metadata_001"
    assert metadata["experiment_id"] == "ppo_metadata_001"

    dataset = metadata["dataset"]

    assert dataset["version"] == "xauusd_5m_v1"
    assert dataset["instrument"] == "XAUUSD"
    assert dataset["timeframe"] == "5m"

def make_governance() -> TrialGovernance:
    return TrialGovernance(
        experiment_family="ppo_baseline",
        selection_eligible=True,
        selection_stage="candidate",
        trigger_reason="manual",
        feature_version="features_v0.1",
        reward_version="combined_v0.1",
        simulator_version="simulator_v0.1",
        protocol_version="research_v0.1",
        split_protocol_version="walk_forward_v0.1",
    )


def test_training_config_converts_to_trial_record():
    config = TrainingRunConfig(
        trial_id="trial_001",
        experiment_id="ppo_baseline_001",
        dataset=make_dataset_spec(),
        output_dir=Path("artifacts"),
        total_timesteps=100_000,
        ppo_config=PPOConfig(),
    )

    trial = config.to_trial_record(
        make_governance()
    )

    assert trial.trial_id == "trial_001"
    assert trial.experiment_family == "ppo_baseline"
    assert trial.instrument == "XAUUSD"
    assert trial.model_version == "0.1"
    assert trial.dataset_version == "xauusd_5m_v1"
    assert trial.feature_version == "features_v0.1"
    assert trial.reward_version == "combined_v0.1"
    assert trial.simulator_version == "simulator_v0.1"
    assert trial.seed == 42
    assert trial.selection_eligible is True
    assert trial.selection_stage == "candidate"
    assert trial.trigger_reason == "manual"
    assert trial.protocol_version == "research_v0.1"
    assert trial.split_protocol_version == (
        "walk_forward_v0.1"
    )


def test_training_config_does_not_register_trial_implicitly():
    config = TrainingRunConfig(
        trial_id="trial_001",
        experiment_id="ppo_baseline_001",
        dataset=make_dataset_spec(),
        output_dir=Path("artifacts"),
        total_timesteps=100_000,
        ppo_config=PPOConfig(),
    )

    governance = make_governance()

    trial = config.to_trial_record(
        governance
    )

    assert trial.trial_id == "trial_001"


def test_training_config_preserves_parent_trial_lineage():
    config = TrainingRunConfig(
        trial_id="trial_002",
        experiment_id="ppo_baseline_002",
        dataset=make_dataset_spec(),
        output_dir=Path("artifacts"),
        total_timesteps=100_000,
        ppo_config=PPOConfig(),
    )

    governance = TrialGovernance(
        experiment_family="ppo_baseline",
        selection_eligible=True,
        selection_stage="candidate",
        trigger_reason="challenger",
        feature_version="features_v0.1",
        reward_version="combined_v0.1",
        simulator_version="simulator_v0.1",
        protocol_version="research_v0.1",
        split_protocol_version="walk_forward_v0.1",
        parent_trial_id="trial_001",
    )

    trial = config.to_trial_record(
        governance
    )

    assert trial.parent_trial_id == "trial_001"