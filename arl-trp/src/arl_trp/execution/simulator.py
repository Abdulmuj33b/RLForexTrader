from dataclasses import dataclass

from arl_trp.domain.market import MarketSnapshot
from arl_trp.domain.order import Action


@dataclass(frozen=True)
class ExecutionConfig:
    """
    Configuration for the deterministic execution simulator.
    """

    slippage: float = 0.0
    commission_per_unit: float = 0.0

    def __post_init__(self):
        if self.slippage < 0.0:
            raise ValueError("slippage cannot be negative")

        if self.commission_per_unit < 0.0:
            raise ValueError(
                "commission_per_unit cannot be negative"
            )


class ExecutionSimulator:
    """
    Simulates order execution against bid/ask prices.

    v0.1 execution rules:

        LONG entry   -> ask + slippage
        SHORT entry  -> bid - slippage

        LONG close   -> bid - slippage
        SHORT close  -> ask + slippage

    Spread is represented by bid/ask and therefore must not
    be charged separately.
    """

    def __init__(self, config: ExecutionConfig):
        self.config = config

    def fill_price(
        self,
        action: Action,
        market: MarketSnapshot,
    ) -> float:
        """
        Determine the execution price for a new position.
        """

        if action == Action.LONG:
            return market.ask + self.config.slippage

        if action == Action.SHORT:
            return market.bid - self.config.slippage

        if action == Action.CLOSE:
            raise ValueError(
                "close price requires the current position direction"
            )

        raise ValueError(
            "only LONG or SHORT can open a position"
        )

    def close_price(
        self,
        quantity: float,
        market: MarketSnapshot,
    ) -> float:
        """
        Determine the execution price for closing a position.

        Positive quantity = long position.
        Negative quantity = short position.
        """

        if quantity > 0.0:
            return market.bid - self.config.slippage

        if quantity < 0.0:
            return market.ask + self.config.slippage

        raise ValueError(
            "cannot close a flat position"
        )

    def commission(self, quantity: float) -> float:
        """
        Calculate commission for an executed quantity.
        """

        return (
            abs(quantity)
            * self.config.commission_per_unit
        )