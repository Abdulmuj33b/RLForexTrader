

import pytest

from arl_trp.research.governance import TrialGovernance


def make_governance(
    *,
    selection_eligible: bool = True,
) -> TrialGovernance:
    return TrialGovernance(
        experiment_family="ppo_baseline",
        selection_eligible=selection_eligible,
        selection_stage="candidate",
        trigger_reason="manual",
        feature_version="features_v0.1",
        reward_version="combined_v0.1",
        simulator_version="simulator_v0.1",
        protocol_version="research_v0.1",
        split_protocol_version="walk_forward_v0.1",
    )


def test_governance_stores_research_metadata():
    governance = make_governance()

    assert governance.experiment_family == "ppo_baseline"
    assert governance.selection_eligible is True
    assert governance.selection_stage == "candidate"
    assert governance.trigger_reason == "manual"
    assert governance.feature_version == "features_v0.1"
    assert governance.reward_version == "combined_v0.1"
    assert governance.simulator_version == "simulator_v0.1"
    assert governance.protocol_version == "research_v0.1"
    assert governance.split_protocol_version == (
        "walk_forward_v0.1"
    )


def test_governance_is_immutable():
    governance = make_governance()

    with pytest.raises(AttributeError):
        governance.experiment_family = "other_family"


def test_non_selection_eligible_governance_is_supported():
    governance = make_governance(
        selection_eligible=False,
    )

    assert governance.selection_eligible is False


@pytest.mark.parametrize(
    "field",
    [
        "experiment_family",
        "selection_stage",
        "trigger_reason",
        "feature_version",
        "reward_version",
        "simulator_version",
        "protocol_version",
        "split_protocol_version",
    ],
)
def test_empty_governance_fields_are_rejected(field):
    values = {
        "experiment_family": "ppo_baseline",
        "selection_eligible": True,
        "selection_stage": "candidate",
        "trigger_reason": "manual",
        "feature_version": "features_v0.1",
        "reward_version": "combined_v0.1",
        "simulator_version": "simulator_v0.1",
        "protocol_version": "research_v0.1",
        "split_protocol_version": "walk_forward_v0.1",
    }

    values[field] = ""

    with pytest.raises(ValueError):
        TrialGovernance(**values)