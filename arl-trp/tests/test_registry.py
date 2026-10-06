import pytest

from arl_trp.research.registry import ResearchRegistry
from arl_trp.research.trial import (
    TrialEvent,
    TrialRecord,
)
from arl_trp.research.trial_ledger import TrialLedger

def make_trial(
    trial_id: str = "trial_001",
    *,
    selection_eligible: bool = True,
    parent_trial_id: str | None = None,
) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        experiment_family="ppo_baseline",
        instrument="XAUUSD",
        model_version="ppo_v0.1",
        dataset_version="xauusd_5m_v1",
        feature_version="features_v0.1",
        reward_version="combined_v0.1",
        simulator_version="simulator_v0.1",
        seed=42,
        selection_eligible=selection_eligible,
        selection_stage="candidate",
        trigger_reason="manual",
        protocol_version="research_v0.1",
        split_protocol_version="walk_forward_v0.1",
        parent_trial_id=parent_trial_id,
    )


def make_event(
    trial_id: str = "trial_001",
) -> TrialEvent:
    return TrialEvent(
        trial_id=trial_id,
        event_type="TRAINING_COMPLETED",
        reason="training finished",
    )


def test_trial_ledger_is_a_research_registry():
    ledger = TrialLedger()

    assert isinstance(
        ledger,
        ResearchRegistry,
    )


def test_register_trial_through_registry_contract():
    registry: ResearchRegistry = TrialLedger()

    trial = make_trial()

    registry.register_trial(trial)

    assert registry.contains_trial("trial_001")
    assert registry.get_trial("trial_001") == trial


def test_registry_rejects_duplicate_trial():
    registry: ResearchRegistry = TrialLedger()

    registry.register_trial(
        make_trial("trial_001")
    )

    with pytest.raises(ValueError):
        registry.register_trial(
            make_trial("trial_001")
        )


def test_registry_records_and_retrieves_events():
    registry: ResearchRegistry = TrialLedger()

    registry.register_trial(
        make_trial()
    )

    event = make_event()

    registry.record_event(event)

    assert registry.events("trial_001") == (
        event,
    )


def test_registry_preserves_event_order():
    registry: ResearchRegistry = TrialLedger()

    registry.register_trial(
        make_trial()
    )

    first = TrialEvent(
        trial_id="trial_001",
        event_type="TRAINING_COMPLETED",
        reason="training finished",
    )

    second = TrialEvent(
        trial_id="trial_001",
        event_type="EVALUATION_COMPLETED",
        reason="evaluation finished",
    )

    registry.record_event(first)
    registry.record_event(second)

    assert registry.events("trial_001") == (
        first,
        second,
    )


def test_registry_counts_only_selection_eligible_trials():
    registry: ResearchRegistry = TrialLedger()

    registry.register_trial(
        make_trial(
            "trial_001",
            selection_eligible=True,
        )
    )

    registry.register_trial(
        make_trial(
            "trial_002",
            selection_eligible=False,
        )
    )

    assert registry.dsr_trial_count == 1


def test_registry_returns_selection_eligible_trials():
    registry: ResearchRegistry = TrialLedger()

    registry.register_trial(
        make_trial(
            "trial_001",
            selection_eligible=True,
        )
    )

    registry.register_trial(
        make_trial(
            "trial_002",
            selection_eligible=False,
        )
    )

    assert registry.selection_eligible_trials() == (
        make_trial("trial_001"),
    )


def test_registry_preserves_rejected_trials():
    registry: ResearchRegistry = TrialLedger()

    trial = make_trial()

    registry.register_trial(trial)

    registry.record_event(
        TrialEvent(
            trial_id="trial_001",
            event_type="PROMOTION_REJECTED",
            reason="failed_dsr",
        )
    )

    assert registry.get_trial("trial_001") == trial
    assert registry.dsr_trial_count == 1
    assert registry.events("trial_001")[0].event_type == (
        "PROMOTION_REJECTED"
    )