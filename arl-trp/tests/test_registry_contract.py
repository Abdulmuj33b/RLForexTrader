from __future__ import annotations

import pytest

from arl_trp.research.registry import ResearchRegistry
from arl_trp.research.trial import TrialEvent, TrialRecord
from arl_trp.research.trial_ledger import TrialLedger


def make_trial(
    trial_id: str,
    *,
    selection_eligible: bool = True,
    parent_trial_id: str | None = None,
) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        experiment_family="ppo_baseline",
        instrument="XAUUSD",
        model_version="ppo-0.1",
        dataset_version="dataset-0.1",
        feature_version="features-0.1",
        reward_version="reward-0.1",
        simulator_version="simulator-0.1",
        seed=42,
        selection_eligible=selection_eligible,
        selection_stage="candidate",
        trigger_reason="manual",
        protocol_version="protocol-0.1",
        split_protocol_version="split-0.1",
        parent_trial_id=parent_trial_id,
    )


@pytest.fixture
def registry() -> ResearchRegistry:
    return TrialLedger()


def test_registry_implements_research_registry_contract(
    registry: ResearchRegistry,
) -> None:
    assert isinstance(registry, ResearchRegistry)


def test_register_and_retrieve_trial(
    registry: ResearchRegistry,
) -> None:
    trial = make_trial("trial-001")

    registry.register_trial(trial)

    assert registry.get_trial("trial-001") == trial
    assert registry.contains_trial("trial-001")


def test_duplicate_trial_registration_is_rejected(
    registry: ResearchRegistry,
) -> None:
    trial = make_trial("trial-001")

    registry.register_trial(trial)

    with pytest.raises(ValueError):
        registry.register_trial(trial)


def test_missing_parent_trial_is_rejected(
    registry: ResearchRegistry,
) -> None:
    trial = make_trial(
        "trial-child",
        parent_trial_id="trial-missing",
    )

    with pytest.raises(ValueError, match="parent_trial_id does not exist"):
        registry.register_trial(trial)

def test_parent_trial_must_exist_before_child_registration(
    registry: ResearchRegistry,
) -> None:
    parent = make_trial("trial-parent")
    child = make_trial(
        "trial-child",
        parent_trial_id="trial-parent",
    )

    registry.register_trial(parent)
    registry.register_trial(child)

    assert registry.get_trial("trial-child") == child


def test_trial_record_is_immutable_after_registration(
    registry: ResearchRegistry,
) -> None:
    trial = make_trial("trial-001")

    registry.register_trial(trial)

    with pytest.raises(Exception):
        trial.selection_stage = "promoted"  # type: ignore[misc]

    assert registry.get_trial("trial-001").selection_stage == "candidate"


def test_events_are_append_only_and_ordered(
    registry: ResearchRegistry,
) -> None:
    trial = make_trial("trial-001")
    registry.register_trial(trial)

    registry.record_event(
        TrialEvent(
            trial_id="trial-001",
            event_type="TRAINING_COMPLETED",
            reason="training finished",
        )
    )
    registry.record_event(
        TrialEvent(
            trial_id="trial-001",
            event_type="EVALUATION_COMPLETED",
            reason="evaluation finished",
        )
    )

    events = registry.events("trial-001")

    assert [event.event_type for event in events] == [
        "TRAINING_COMPLETED",
        "EVALUATION_COMPLETED",
    ]


def test_events_cannot_be_recorded_for_unknown_trial(
    registry: ResearchRegistry,
) -> None:
    event = TrialEvent(
        trial_id="trial-missing",
        event_type="TRAINING_COMPLETED",
        reason="training finished",
    )

    with pytest.raises(KeyError):
        registry.record_event(event)


def test_selection_eligible_trial_count_is_derived_from_trials(
    registry: ResearchRegistry,
) -> None:
    registry.register_trial(
        make_trial("trial-001", selection_eligible=True)
    )
    registry.register_trial(
        make_trial("trial-002", selection_eligible=False)
    )
    registry.register_trial(
        make_trial("trial-003", selection_eligible=True)
    )

    assert registry.dsr_trial_count == 2


def test_selection_eligible_trials_returns_only_eligible_trials(
    registry: ResearchRegistry,
) -> None:
    eligible = make_trial("trial-001", selection_eligible=True)
    rejected = make_trial("trial-002", selection_eligible=False)

    registry.register_trial(eligible)
    registry.register_trial(rejected)

    assert registry.selection_eligible_trials() == (eligible,)


def test_rejected_trials_are_retained(
    registry: ResearchRegistry,
) -> None:
    rejected = make_trial(
        "trial-rejected",
        selection_eligible=True,
    )

    registry.register_trial(rejected)
    registry.record_event(
        TrialEvent(
            trial_id="trial-rejected",
            event_type="PROMOTION_REJECTED",
            reason="candidate failed OOS superiority",
        )
    )

    assert registry.contains_trial("trial-rejected")
    assert registry.get_trial("trial-rejected") == rejected
    assert registry.dsr_trial_count == 1
    assert registry.events("trial-rejected")[0].event_type == "PROMOTION_REJECTED"


def test_all_trials_preserves_registration_order(
    registry: ResearchRegistry,
) -> None:
    first = make_trial("trial-001")
    second = make_trial("trial-002")
    third = make_trial("trial-003")

    registry.register_trial(first)
    registry.register_trial(second)
    registry.register_trial(third)

    assert registry.all_trials() == (
        first,
        second,
        third,
    )


def test_get_unknown_trial_raises_key_error(
    registry: ResearchRegistry,
) -> None:
    with pytest.raises(KeyError):
        registry.get_trial("trial-missing")


def test_event_history_is_independent_per_trial(
    registry: ResearchRegistry,
) -> None:
    first = make_trial("trial-001")
    second = make_trial("trial-002")

    registry.register_trial(first)
    registry.register_trial(second)

    registry.record_event(
        TrialEvent(
            trial_id="trial-001",
            event_type="TRAINING_COMPLETED",
            reason="first training completed",
        )
    )

    registry.record_event(
        TrialEvent(
            trial_id="trial-002",
            event_type="TRAINING_COMPLETED",
            reason="second training completed",
        )
    )

    assert len(registry.events("trial-001")) == 1
    assert len(registry.events("trial-002")) == 1

    assert registry.events("trial-001")[0].reason == (
        "first training completed"
    )
    assert registry.events("trial-002")[0].reason == (
        "second training completed"
    )

def test_register_trial_with_event_is_atomic(
    registry: ResearchRegistry,
) -> None:
    trial = make_trial("trial-atomic")
    event = TrialEvent(
        trial_id="trial-atomic",
        event_type="CREATED",
        reason="trial registered",
    )

    registry.register_trial_with_event(trial, event)

    assert registry.get_trial("trial-atomic") == trial
    assert registry.events("trial-atomic") == (event,)


def test_register_trial_with_event_requires_matching_trial_ids(
    registry: ResearchRegistry,
) -> None:
    trial = make_trial("trial-atomic")
    event = TrialEvent(
        trial_id="different-trial",
        event_type="CREATED",
        reason="trial registered",
    )

    with pytest.raises(
        ValueError,
        match="event.trial_id must match trial.trial_id",
    ):
        registry.register_trial_with_event(trial, event)

    assert not registry.contains_trial("trial-atomic")
    assert not registry.contains_trial("different-trial")


def test_register_trial_with_event_rejects_duplicate_trial(
    registry: ResearchRegistry,
) -> None:
    trial = make_trial("trial-duplicate")
    registry.register_trial(trial)

    event = TrialEvent(
        trial_id="trial-duplicate",
        event_type="CREATED",
        reason="duplicate registration",
    )

    with pytest.raises(ValueError, match="trial_id already exists"):
        registry.register_trial_with_event(trial, event)

    assert registry.events("trial-duplicate") == ()


def test_register_trial_with_event_rejects_missing_parent(
    registry: ResearchRegistry,
) -> None:
    trial = make_trial(
        "trial-child",
        parent_trial_id="trial-missing",
    )
    event = TrialEvent(
        trial_id="trial-child",
        event_type="CREATED",
        reason="child trial registered",
    )

    with pytest.raises(
        ValueError,
        match="parent_trial_id does not exist",
    ):
        registry.register_trial_with_event(trial, event)

    assert not registry.contains_trial("trial-child")


def test_register_trial_with_event_places_event_first(
    registry: ResearchRegistry,
) -> None:
    trial = make_trial("trial-history")
    created = TrialEvent(
        trial_id="trial-history",
        event_type="CREATED",
        reason="trial registered",
    )
    training = TrialEvent(
        trial_id="trial-history",
        event_type="TRAINING_COMPLETED",
        reason="training finished",
    )

    registry.register_trial_with_event(trial, created)
    registry.record_event(training)

    assert registry.events("trial-history") == (
        created,
        training,
    )

def test_register_trial_with_event_is_atomic(
    registry: ResearchRegistry,
) -> None:
    trial = make_trial("trial-atomic")
    event = TrialEvent(
        trial_id="trial-atomic",
        event_type="CREATED",
        reason="trial registered",
    )

    registry.register_trial_with_event(trial, event)

    assert registry.get_trial("trial-atomic") == trial
    assert registry.events("trial-atomic") == (event,)


def test_register_trial_with_event_requires_matching_trial_ids(
    registry: ResearchRegistry,
) -> None:
    trial = make_trial("trial-atomic")
    event = TrialEvent(
        trial_id="different-trial",
        event_type="CREATED",
        reason="trial registered",
    )

    with pytest.raises(
        ValueError,
        match="event.trial_id must match trial.trial_id",
    ):
        registry.register_trial_with_event(trial, event)

    assert not registry.contains_trial("trial-atomic")
    assert not registry.contains_trial("different-trial")


def test_register_trial_with_event_rejects_duplicate_trial(
    registry: ResearchRegistry,
) -> None:
    trial = make_trial("trial-duplicate")
    registry.register_trial(trial)

    event = TrialEvent(
        trial_id="trial-duplicate",
        event_type="CREATED",
        reason="duplicate registration",
    )

    with pytest.raises(ValueError, match="trial_id already exists"):
        registry.register_trial_with_event(trial, event)

    assert registry.events("trial-duplicate") == ()


def test_register_trial_with_event_rejects_missing_parent(
    registry: ResearchRegistry,
) -> None:
    trial = make_trial(
        "trial-child",
        parent_trial_id="trial-missing",
    )
    event = TrialEvent(
        trial_id="trial-child",
        event_type="CREATED",
        reason="child trial registered",
    )

    with pytest.raises(
        ValueError,
        match="parent_trial_id does not exist",
    ):
        registry.register_trial_with_event(trial, event)

    assert not registry.contains_trial("trial-child")


def test_register_trial_with_event_places_event_first(
    registry: ResearchRegistry,
) -> None:
    trial = make_trial("trial-history")
    created = TrialEvent(
        trial_id="trial-history",
        event_type="CREATED",
        reason="trial registered",
    )
    training = TrialEvent(
        trial_id="trial-history",
        event_type="TRAINING_COMPLETED",
        reason="training finished",
    )

    registry.register_trial_with_event(trial, created)
    registry.record_event(training)

    assert registry.events("trial-history") == (
        created,
        training,
    )