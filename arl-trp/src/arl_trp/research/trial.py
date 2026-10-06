from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TrialRecord:
    """
    Immutable research-governance record for one trial attempt.

    A TrialRecord contains the immutable identity and governance
    metadata established when the trial enters the research
    universe.

    Lifecycle changes are represented by TrialEvent records rather
    than by mutating this object.
    """

    trial_id: str
    experiment_family: str
    instrument: str
    model_version: str
    dataset_version: str
    feature_version: str
    reward_version: str
    simulator_version: str
    seed: int

    selection_eligible: bool
    selection_stage: str
    trigger_reason: str
    protocol_version: str
    split_protocol_version: str

    parent_trial_id: Optional[str] = None

    def __post_init__(self) -> None:
        required_fields = {
            "trial_id": self.trial_id,
            "experiment_family": self.experiment_family,
            "instrument": self.instrument,
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "reward_version": self.reward_version,
            "simulator_version": self.simulator_version,
            "selection_stage": self.selection_stage,
            "trigger_reason": self.trigger_reason,
            "protocol_version": self.protocol_version,
            "split_protocol_version": self.split_protocol_version,
        }

        for name, value in required_fields.items():
            if not value:
                raise ValueError(
                    f"{name} cannot be empty"
                )

        if self.seed < 0:
            raise ValueError(
                "seed cannot be negative"
            )


@dataclass(frozen=True)
class TrialEvent:
    """
    Immutable lifecycle event associated with a trial.

    Events are append-only. They record what happened to a trial
    without changing the original TrialRecord.
    """

    trial_id: str
    event_type: str
    reason: str

    def __post_init__(self) -> None:
        if not self.trial_id:
            raise ValueError(
                "trial_id cannot be empty"
            )

        if not self.event_type:
            raise ValueError(
                "event_type cannot be empty"
            )

        if not self.reason:
            raise ValueError(
                "reason cannot be empty"
            )
