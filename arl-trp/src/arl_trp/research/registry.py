from abc import ABC, abstractmethod

from arl_trp.research.trial import (
    TrialEvent,
    TrialRecord,
)


class ResearchRegistry(ABC):
    """
    Persistence contract for the research experiment registry.

    Implementations must preserve the project's research
    invariants:

    - trial identity is immutable;
    - trials cannot be silently overwritten;
    - lifecycle events are append-only;
    - rejected trials remain queryable;
    - DSR trial count includes only selection-eligible trials;
    - trial registration with its initial event is atomic.
    """

    @abstractmethod
    def register_trial(
        self,
        trial: TrialRecord,
    ) -> None:
        """
        Register a new immutable trial.
        """
        raise NotImplementedError

    @abstractmethod
    def register_trial_with_event(
        self,
        trial: TrialRecord,
        event: TrialEvent,
    ) -> None:
        """
        Atomically register a new trial and its initial lifecycle event.

        Implementations must guarantee that either both the trial
        and event are persisted, or neither is persisted.
        """
        raise NotImplementedError

    @abstractmethod
    def get_trial(
        self,
        trial_id: str,
    ) -> TrialRecord:
        """
        Retrieve a registered trial.
        """
        raise NotImplementedError

    @abstractmethod
    def contains_trial(
        self,
        trial_id: str,
    ) -> bool:
        """
        Return whether a trial exists.
        """
        raise NotImplementedError

    @abstractmethod
    def record_event(
        self,
        event: TrialEvent,
    ) -> None:
        """
        Append a lifecycle event to a registered trial.
        """
        raise NotImplementedError

    @abstractmethod
    def events(
        self,
        trial_id: str,
    ) -> tuple[TrialEvent, ...]:
        """
        Return lifecycle events in append order.
        """
        raise NotImplementedError

    @abstractmethod
    def all_trials(
        self,
    ) -> tuple[TrialRecord, ...]:
        """
        Return all registered trials in deterministic order.
        """
        raise NotImplementedError

    @abstractmethod
    def selection_eligible_trials(
        self,
    ) -> tuple[TrialRecord, ...]:
        """
        Return trials eligible for model selection.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dsr_trial_count(self) -> int:
        """
        Return the number of selection-eligible trials.
        """
        raise NotImplementedError