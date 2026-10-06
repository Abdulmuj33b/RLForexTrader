import pytest

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


def test_trial_record_stores_governance_metadata():
    trial = make_trial()

    assert trial.trial_id == "trial_001"
    assert trial.experiment_family == "ppo_baseline"
    assert trial.instrument == "XAUUSD"
    assert trial.model_version == "ppo_v0.1"
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


def test_trial_record_is_immutable():
    trial = make_trial()

    with pytest.raises(AttributeError):
        trial.trial_id = "trial_999"


def test_negative_seed_is_rejected():
    with pytest.raises(ValueError):
        TrialRecord(
            trial_id="trial_001",
            experiment_family="ppo_baseline",
            instrument="XAUUSD",
            model_version="ppo_v0.1",
            dataset_version="xauusd_5m_v1",
            feature_version="features_v0.1",
            reward_version="combined_v0.1",
            simulator_version="simulator_v0.1",
            seed=-1,
            selection_eligible=True,
            selection_stage="candidate",
            trigger_reason="manual",
            protocol_version="research_v0.1",
            split_protocol_version="walk_forward_v0.1",
        )


def test_ledger_registers_trial():
    ledger = TrialLedger()
    trial = make_trial()

    ledger.register(trial)

    assert ledger.contains("trial_001")
    assert ledger.get("trial_001") == trial
    assert ledger.trial_count == 1


def test_duplicate_trial_id_is_rejected():
    ledger = TrialLedger()

    ledger.register(make_trial("trial_001"))

    with pytest.raises(ValueError):
        ledger.register(make_trial("trial_001"))


def test_unknown_trial_id_is_rejected():
    ledger = TrialLedger()

    with pytest.raises(KeyError):
        ledger.get("missing_trial")


def test_parent_trial_must_exist():
    ledger = TrialLedger()

    with pytest.raises(ValueError):
        ledger.register(
            make_trial(
                "trial_002",
                parent_trial_id="trial_001",
            )
        )


def test_parent_trial_can_be_registered():
    ledger = TrialLedger()

    ledger.register(make_trial("trial_001"))

    ledger.register(
        make_trial(
            "trial_002",
            parent_trial_id="trial_001",
        )
    )

    assert (
        ledger.get("trial_002").parent_trial_id
        == "trial_001"
    )


def test_dsr_trial_count_counts_only_selection_eligible_trials():
    ledger = TrialLedger()

    ledger.register(
        make_trial(
            "trial_001",
            selection_eligible=True,
        )
    )

    ledger.register(
        make_trial(
            "trial_002",
            selection_eligible=False,
        )
    )

    ledger.register(
        make_trial(
            "trial_003",
            selection_eligible=True,
        )
    )

    assert ledger.trial_count == 3
    assert ledger.dsr_trial_count == 2


def test_calibration_trial_does_not_increase_dsr_count():
    ledger = TrialLedger()

    ledger.register(
        make_trial(
            "calibration_001",
            selection_eligible=False,
        )
    )

    assert ledger.trial_count == 1
    assert ledger.dsr_trial_count == 0


def test_trial_status_is_recorded_as_event():
    ledger = TrialLedger()

    ledger.register(make_trial())

    ledger.record_event(
        TrialEvent(
            trial_id="trial_001",
            event_type="TRAINING_COMPLETED",
            reason="training finished",
        )
    )

    events = ledger.events("trial_001")

    assert len(events) == 1
    assert events[0].event_type == "TRAINING_COMPLETED"
    assert events[0].reason == "training finished"


def test_rejection_is_recorded_as_event():
    ledger = TrialLedger()

    ledger.register(make_trial())

    ledger.record_event(
        TrialEvent(
            trial_id="trial_001",
            event_type="PROMOTION_REJECTED",
            reason="failed_dsr",
        )
    )

    events = ledger.events("trial_001")

    assert len(events) == 1
    assert events[0].event_type == "PROMOTION_REJECTED"
    assert events[0].reason == "failed_dsr"


def test_events_are_append_only():
    ledger = TrialLedger()

    ledger.register(make_trial())

    ledger.record_event(
        TrialEvent(
            trial_id="trial_001",
            event_type="TRAINING_COMPLETED",
            reason="training finished",
        )
    )

    ledger.record_event(
        TrialEvent(
            trial_id="trial_001",
            event_type="EVALUATION_COMPLETED",
            reason="evaluation finished",
        )
    )

    events = ledger.events("trial_001")

    assert [
        event.event_type
        for event in events
    ] == [
        "TRAINING_COMPLETED",
        "EVALUATION_COMPLETED",
    ]


def test_event_for_unknown_trial_is_rejected():
    ledger = TrialLedger()

    with pytest.raises(KeyError):
        ledger.record_event(
            TrialEvent(
                trial_id="missing_trial",
                event_type="TRAINING_COMPLETED",
                reason="training finished",
            )
        )


def test_event_trial_id_is_immutable():
    event = TrialEvent(
        trial_id="trial_001",
        event_type="TRAINING_COMPLETED",
        reason="training finished",
    )

    with pytest.raises(AttributeError):
        event.trial_id = "trial_999"


def test_all_trials_preserves_registration_order():
    ledger = TrialLedger()

    ledger.register(make_trial("trial_001"))
    ledger.register(make_trial("trial_002"))
    ledger.register(make_trial("trial_003"))

    assert [
        trial.trial_id
        for trial in ledger.all_trials()
    ] == [
        "trial_001",
        "trial_002",
        "trial_003",
    ]


def test_rejected_trial_remains_in_ledger():
    ledger = TrialLedger()

    trial = make_trial("trial_001")

    ledger.register(trial)

    ledger.record_event(
        TrialEvent(
            trial_id="trial_001",
            event_type="PROMOTION_REJECTED",
            reason="failed_risk_gate",
        )
    )

    assert ledger.contains("trial_001")
    assert ledger.get("trial_001") == trial
    assert ledger.dsr_trial_count == 1
    assert ledger.events("trial_001")[0].event_type == (
        "PROMOTION_REJECTED"
    )


def test_selection_eligible_trials_returns_only_eligible_trials():
    ledger = TrialLedger()

    ledger.register(
        make_trial(
            "trial_001",
            selection_eligible=True,
        )
    )

    ledger.register(
        make_trial(
            "trial_002",
            selection_eligible=False,
        )
    )

    assert [
        trial.trial_id
        for trial in ledger.selection_eligible_trials()
    ] == ["trial_001"]
