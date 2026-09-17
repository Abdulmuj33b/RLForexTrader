from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MarketSnapshot:
    """
    Immutable snapshot of market information available at a
    specific timestamp.
    """

    timestamp: datetime

    open: float
    high: float
    low: float
    close: float

    volume: float

    bid: float
    ask: float

    def __post_init__(self):
        # OHLC validation
        if self.high < max(self.open, self.close):
            raise ValueError(
                "high must be greater than or equal to open and close"
            )

        if self.low > min(self.open, self.close):
            raise ValueError(
                "low must be less than or equal to open and close"
            )

        # Bid/ask validation
        if self.bid > self.ask:
            raise ValueError(
                "bid cannot exceed ask"
            )

        # Volume validation
        if self.volume < 0:
            raise ValueError(
                "volume cannot be negative"
            )

    @property
    def spread(self) -> float:
        """Return the bid/ask spread."""

        return self.ask - self.bid