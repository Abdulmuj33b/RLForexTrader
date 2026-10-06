from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TrialGovernance:
    """
    Immutable governance metadata for one research trial.

    This object describes how a trial participates in the
    project's research-selection process. It does not contain
    training hyperparameters or evaluation metrics.
    """

    experiment_family: str
    selection_eligible: bool
    selection_stage: str
    trigger_reason: str

    feature_version: str
    reward_version: str
    simulator_version: str

    protocol_version: str
    split_protocol_version: str

    parent_trial_id: Optional[str] = None

    def __post_init__(self) -> None:
        required_fields = {
            "experiment_family": self.experiment_family,
            "selection_stage": self.selection_stage,
            "trigger_reason": self.trigger_reason,
            "feature_version": self.feature_version,
            "reward_version": self.reward_version,
            "simulator_version": self.simulator_version,
            "protocol_version": self.protocol_version,
            "split_protocol_version": self.split_protocol_version,
        }

        for name, value in required_fields.items():
            if not value:
                raise ValueError(
                    f"{name} cannot be empty"
                )