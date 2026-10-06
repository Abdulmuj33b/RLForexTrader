from arl_trp.research.registry import ResearchRegistry
from arl_trp.research.trial import (
    TrialEvent,
    TrialRecord,
)


class TrialLedger(ResearchRegistry):
    """
    In-memory research trial registry.

    Trial records are immutable after registration.

    Lifecycle changes are represented as append-only TrialEvent
    records. This preserves the historical research record and
    prevents promotion outcomes or other governance facts from
    silently rewriting the original trial definition.

    Trial registration with an initial event is atomic: all
    validation occurs before either the trial or event is stored.

    The shorter register/get/contains methods are retained for
    backward compatibility. The ResearchRegistry interface uses
    register_trial/get_trial/contains_trial.
    """

    def __init__(self) -> None:
        self._records: dict[str, TrialRecord] = {}
        self._events: dict[str, list[TrialEvent]] = {}

    def register(
        self,
        trial: TrialRecord,
    ) -> None:
        """
        Backward-compatible alias for register_trial().
        """

        self.register_trial(trial)

    def register_trial(
        self,
        trial: TrialRecord,
    ) -> None:
        """
        Register a new immutable trial.

        Trial IDs must be unique within the ledger.

        If the trial declares a parent, that parent must already
        exist in the ledger.
        """

        self._validate_trial_registration(trial)

        self._records[trial.trial_id] = trial
        self._events[trial.trial_id] = []

    def register_trial_with_event(
        self,
        trial: TrialRecord,
        event: TrialEvent,
    ) -> None:
        """
        Atomically register a trial and its initial lifecycle event.

        All validation occurs before the ledger is mutated.

        Therefore, if validation fails, neither the trial nor the
        event is stored.
        """

        self._validate_trial_registration(trial)

        if event.trial_id != trial.trial_id:
            raise ValueError(
                "event.trial_id must match trial.trial_id: "
                f"{event.trial_id} != {trial.trial_id}"
            )

        self._records[trial.trial_id] = trial
        self._events[trial.trial_id] = [event]

    def _validate_trial_registration(
        self,
        trial: TrialRecord,
    ) -> None:
        """
        Validate all conditions required before trial registration.
        """

        if trial.trial_id in self._records:
            raise ValueError(
                f"trial_id already exists: {trial.trial_id}"
            )

        if trial.parent_trial_id is not None:
            if trial.parent_trial_id not in self._records:
                raise ValueError(
                    "parent_trial_id does not exist: "
                    f"{trial.parent_trial_id}"
                )

    def get(
        self,
        trial_id: str,
    ) -> TrialRecord:
        """
        Backward-compatible alias for get_trial().
        """

        return self.get_trial(trial_id)

    def get_trial(
        self,
        trial_id: str,
    ) -> TrialRecord:
        """
        Return a registered trial by ID.
        """

        try:
            return self._records[trial_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown trial_id: {trial_id}"
            ) from exc

    def contains(
        self,
        trial_id: str,
    ) -> bool:
        """
        Backward-compatible alias for contains_trial().
        """

        return self.contains_trial(trial_id)

    def contains_trial(
        self,
        trial_id: str,
    ) -> bool:
        """
        Return whether a trial is registered.
        """

        return trial_id in self._records

    def record_event(
        self,
        event: TrialEvent,
    ) -> None:
        """
        Append an immutable lifecycle event to a registered trial.
        """

        if event.trial_id not in self._records:
            raise KeyError(
                f"unknown trial_id: {event.trial_id}"
            )

        self._events[event.trial_id].append(event)

    def events(
        self,
        trial_id: str,
    ) -> tuple[TrialEvent, ...]:
        """
        Return all lifecycle events for a trial in append order.
        """

        if trial_id not in self._records:
            raise KeyError(
                f"unknown trial_id: {trial_id}"
            )

        return tuple(self._events[trial_id])

    def all_trials(
        self,
    ) -> tuple[TrialRecord, ...]:
        """
        Return all trials in deterministic registration order.
        """

        return tuple(self._records.values())

    @property
    def trial_count(self) -> int:
        """
        Return the total number of registered trials.
        """

        return len(self._records)

    @property
    def dsr_trial_count(self) -> int:
        """
        Return the number of selection-eligible trials.

        This is the N used by the project's initial DSR
        trial-count convention.
        """

        return sum(
            trial.selection_eligible
            for trial in self._records.values()
        )

    def selection_eligible_trials(
        self,
    ) -> tuple[TrialRecord, ...]:
        """
        Return only trials eligible for model selection.
        """

        return tuple(
            trial
            for trial in self._records.values()
            if trial.selection_eligible
        )