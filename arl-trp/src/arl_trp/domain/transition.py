from dataclasses import dataclass
from datetime import datetime

from .order import Action


@dataclass(frozen=True)
class TradeTransition:
    timestamp: datetime

    action: Action

    position_before: float
    position_after: float

    fill_price: float

    gross_pnl: float
    transaction_cost: float
    net_pnl: float

    equity_before: float
    equity_after: float

    reward: float