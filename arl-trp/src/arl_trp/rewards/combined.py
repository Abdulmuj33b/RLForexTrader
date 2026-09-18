from dataclasses import dataclass

from arl_trp.rewards.differential_sharpe import (
    DifferentialSharpeReward,
)
from arl_trp.rewards.economic import EconomicReward


@dataclass
class CombinedReward:
    """
    Combines economic reward with a differential-Sharpe-style
    risk-adjusted learning signal.

    v0.1:

        R_t = ΔEquity_t + β × R_t^DS

    The DifferentialSharpeReward handles its own burn-in.
    """

    beta: float = 0.5
    differential_sharpe: DifferentialSharpeReward | None = None

    def __post_init__(self):
        if self.beta < 0.0:
            raise ValueError(
                "beta cannot be negative"
            )

        if self.differential_sharpe is None:
            self.differential_sharpe = (
                DifferentialSharpeReward()
            )

    def reset(self) -> None:
        """
        Reset the differential-Sharpe estimator.

        This must happen at the beginning of every
        independent episode/fold.
        """

        self.differential_sharpe.reset()

    def calculate(
        self,
        equity_before: float,
        equity_after: float,
    ) -> float:
        """
        Calculate the combined reward.
        """

        economic_reward = EconomicReward.calculate(
            equity_before,
            equity_after,
        )

        differential_reward = (
            self.differential_sharpe.update(
                economic_reward
            )
        )

        return (
            economic_reward
            + self.beta * differential_reward
        )
