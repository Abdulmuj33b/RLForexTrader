from dataclasses import dataclass

# from arl_trp.domain.instrument import InstrumentSpec
from arl_trp.domain.market import MarketSnapshot
from arl_trp.domain.position import Position


@dataclass
class PortfolioAccounting:
    """
    Handles account and portfolio valuation.

    Responsibilities:
    - Track cash/balance
    - Track current position
    - Calculate unrealized P/L
    - Calculate equity
    - Track peak equity
    - Calculate drawdown

    This class does NOT decide whether to buy or sell.
    It only accounts for the financial consequences.
    """

    initial_balance: float

    balance: float = 0.0
    peak_equity: float = 0.0

    def __post_init__(self):
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")

        self.balance = self.initial_balance
        self.peak_equity = self.initial_balance

    def reset(self) -> None:
        """Reset the account to its initial state."""
        self.balance = self.initial_balance
        self.peak_equity = self.initial_balance

    def unrealized_pnl(
        self,
        position: Position,
        market: MarketSnapshot,
    ) -> float:
        """
        Calculate current unrealized P/L.

        Long positions are marked at the bid.
        Short positions are marked at the ask.
        """

        if position.is_flat:
            return 0.0

        if position.is_long:
            mark_price = market.bid
        else:
            mark_price = market.ask

        return position.quantity * (
            mark_price - position.entry_price
        )

    def equity(
        self,
        position: Position,
        market: MarketSnapshot,
    ) -> float:
        """Calculate current account equity."""

        unrealized = self.unrealized_pnl(
            position,
            market,
        )

        current_equity = self.balance + unrealized

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        return current_equity

    def drawdown(
        self,
        position: Position,
        market: MarketSnapshot,
    ) -> float:
        """Calculate current percentage drawdown from peak equity."""

        current_equity = self.equity(
            position,
            market,
        )

        if self.peak_equity <= 0:
            return 0.0

        return (
            self.peak_equity - current_equity
        ) / self.peak_equity

    def apply_realized_pnl(self, pnl: float) -> None:
        """
        Apply realized net P/L to account balance.

        Transaction costs should already be included
        in the supplied net P/L.
        """

        self.balance += pnl

        if self.balance > self.peak_equity:
            self.peak_equity = self.balance

    def snapshot(
        self,
        position: Position,
        market: MarketSnapshot,
    ) -> dict:
        """
        Return a complete accounting snapshot.

        Useful for debugging, logging and environment info.
        """

        unrealized = self.unrealized_pnl(
            position,
            market,
        )

        current_equity = self.balance + unrealized

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        current_drawdown = 0.0

        if self.peak_equity > 0:
            current_drawdown = (
                self.peak_equity - current_equity
            ) / self.peak_equity

        return {
            "balance": self.balance,
            "unrealized_pnl": unrealized,
            "equity": current_equity,
            "peak_equity": self.peak_equity,
            "drawdown": current_drawdown,
            "position_quantity": position.quantity,
            "position_entry_price": position.entry_price,
        }