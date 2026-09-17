from dataclasses import dataclass


@dataclass
class AccountState:
    balance: float
    equity: float
    margin_used: float
    available_margin: float
    peak_equity: float

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0

        return (
            self.peak_equity - self.equity
        ) / self.peak_equity