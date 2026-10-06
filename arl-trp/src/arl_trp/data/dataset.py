from dataclasses import dataclass
from datetime import datetime

from arl_trp.domain.market import MarketSnapshot


@dataclass(frozen=True)
class DatasetSpec:
    """
    Immutable specification identifying one market-data dataset.

    This describes the dataset and its temporal boundaries.
    It does not load or download market data.
    """

    version: str
    instrument: str
    timeframe: str
    source: str
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("version cannot be empty")

        if not self.instrument:
            raise ValueError("instrument cannot be empty")

        if not self.timeframe:
            raise ValueError("timeframe cannot be empty")

        if not self.source:
            raise ValueError("source cannot be empty")

        if self.start >= self.end:
            raise ValueError("start must be earlier than end")

    @property
    def duration_days(self) -> float:
        return (self.end - self.start).total_seconds() / 86_400


@dataclass(frozen=True)
class ValidatedDataset:
    """
    Validated market dataset together with its immutable specification.
    """

    spec: DatasetSpec
    snapshots: tuple[MarketSnapshot, ...]

    def __post_init__(self) -> None:
        if not self.snapshots:
            raise ValueError("snapshots cannot be empty")

    @property
    def start(self) -> datetime:
        return self.snapshots[0].timestamp

    @property
    def end(self) -> datetime:
        return self.snapshots[-1].timestamp

    @property
    def size(self) -> int:
        return len(self.snapshots)