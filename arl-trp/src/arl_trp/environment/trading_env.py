from arl_trp.environment.observation import (
    ObservationBuffer,
    ObservationNormalizer,
    StateEncoder,
)
from typing import Sequence

import numpy as np

from arl_trp.domain.market import MarketSnapshot
from arl_trp.domain.order import Action
from arl_trp.domain.position import Position
from arl_trp.domain.transition import TradeTransition
from arl_trp.execution.simulator import ExecutionSimulator
from arl_trp.portfolio.accounting import PortfolioAccounting
from arl_trp.rewards.combined import CombinedReward


class TradingEnvironment:
    """
    Minimal deterministic trading environment for ARL-TRP v0.1.

    The environment provides market mechanics, accounting,
    and reward calculation.

    It does NOT contain a trading strategy.

    Supported actions:
        HOLD
        LONG
        SHORT
        CLOSE

    Opposite-direction actions close the existing position.
    They do not immediately reverse the position.
    """

    def set_observation_normalizer(
        self,
        normalizer: ObservationNormalizer,
    ) -> None:
        if not normalizer.is_fitted:
            raise RuntimeError(
            "ObservationNormalizer must be fitted before use"
            )

        self.observation_normalizer = normalizer

    def __init__(
        self,
        market_data: Sequence[MarketSnapshot],
        initial_balance: float,
        execution: ExecutionSimulator,
        quantity: float = 1.0,
        reward_beta: float = 0.5,
    ):
        if not market_data:
            raise ValueError(
                "market_data cannot be empty"
            )

        if initial_balance <= 0.0:
            raise ValueError(
                "initial_balance must be positive"
            )

        if quantity <= 0.0:
            raise ValueError(
                "quantity must be positive"
            )

        self.market_data = list(market_data)
        self.initial_balance = initial_balance
        self.execution = execution
        self.quantity = quantity

        self.position = Position()

        self.accounting = PortfolioAccounting(
            initial_balance=initial_balance
        )

        self.reward = CombinedReward(
            beta=reward_beta
        )

        self.state_encoder = StateEncoder(
            initial_balance=initial_balance
        )

        self.observation_normalizer: ObservationNormalizer | None = None

        self.observation_buffer = ObservationBuffer(
            window_size=64,
            feature_count=10,
        )

        self.previous_close = self.market_data[0].close
        self.index = 0
        self.previous_equity = initial_balance

    def reset(self):
        """Reset the environment to the beginning."""

        self.index = 0
        self.position = Position()
        self.accounting.reset()
        self.reward.reset()

        self.previous_equity = self.initial_balance

        self.previous_close = self.market_data[0].close

        self.observation_buffer.reset()

        return self._observation()

    def _observation(self) -> np.ndarray:
        """
        Encode, normalize, and buffer the current market observation.

        During warm-up:
            returns the current normalized 10-feature vector.

        Once the buffer contains 64 observations:
            returns the chronological (64, 10) observation window.
        """

        if self.observation_normalizer is None:
            raise RuntimeError(
                "ObservationNormalizer must be set before generating observations"
            )

        market = self.market_data[self.index]

        raw_observation = self.state_encoder.encode(
            market=market,
            previous_close=self.previous_close,
            position=self.position,
            accounting=self.accounting,
        )

        normalized_observation = self.observation_normalizer.transform(
            raw_observation.reshape(1, -1)
        )[0]

        self.observation_buffer.append(normalized_observation)

        if self.observation_buffer.is_ready:
            return self.observation_buffer.get()

        return normalized_observation.copy()

    def _close_position(
        self,
        market: MarketSnapshot,
    ) -> tuple[float, float, float]:
        """
        Close the current position.

        Returns:
            fill_price,
            gross_pnl,
            transaction_cost
        """

        if self.position.is_flat:
            return market.close, 0.0, 0.0

        quantity = self.position.quantity

        fill_price = self.execution.close_price(
            quantity,
            market,
        )

        gross_pnl = quantity * (
            fill_price - self.position.entry_price
        )

        transaction_cost = self.execution.commission(
            quantity
        )

        net_pnl = gross_pnl - transaction_cost

        self.accounting.apply_realized_pnl(
            net_pnl
        )

        self.position.close()

        return (
            fill_price,
            gross_pnl,
            transaction_cost,
        )

    def step(self, action: Action):
        """
        Execute one environment transition.
        """

        if not isinstance(action, Action):
            try:
                action = Action(action)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Unknown action: {action}"
                ) from exc

        market = self.market_data[self.index]

        equity_before = self.accounting.equity(
            self.position,
            market,
        )

        position_before = self.position.quantity

        gross_pnl = 0.0
        transaction_cost = 0.0
        fill_price = market.close

        # -------------------------------------------------
        # HOLD
        # -------------------------------------------------

        if action == Action.HOLD:
            pass

        # -------------------------------------------------
        # LONG
        # -------------------------------------------------

        elif action == Action.LONG:

            if self.position.is_flat:

                fill_price = self.execution.fill_price(
                    Action.LONG,
                    market,
                )

                self.position.open(
                    self.quantity,
                    fill_price,
                )

                transaction_cost = (
                    self.execution.commission(
                        self.quantity
                    )
                )

                self.accounting.apply_realized_pnl(
                    -transaction_cost
                )

            elif self.position.is_short:

                (
                    fill_price,
                    gross_pnl,
                    transaction_cost,
                ) = self._close_position(market)

        # -------------------------------------------------
        # SHORT
        # -------------------------------------------------

        elif action == Action.SHORT:

            if self.position.is_flat:

                fill_price = self.execution.fill_price(
                    Action.SHORT,
                    market,
                )

                self.position.open(
                    -self.quantity,
                    fill_price,
                )

                transaction_cost = (
                    self.execution.commission(
                        self.quantity
                    )
                )

                self.accounting.apply_realized_pnl(
                    -transaction_cost
                )

            elif self.position.is_long:

                (
                    fill_price,
                    gross_pnl,
                    transaction_cost,
                ) = self._close_position(market)

        # -------------------------------------------------
        # CLOSE
        # -------------------------------------------------

        elif action == Action.CLOSE:

            (
                fill_price,
                gross_pnl,
                transaction_cost,
            ) = self._close_position(market)

        # -------------------------------------------------
        # ADVANCE MARKET
        # -------------------------------------------------

        if self.index < len(self.market_data) - 1:
            self.index += 1

        next_market = self.market_data[self.index]

        previous_close = market.close

        equity_after = self.accounting.equity(
            self.position,
            next_market,
        )

        reward = self.reward.calculate(
            equity_before,
            equity_after,
        )

        transition = TradeTransition(
            timestamp=market.timestamp,
            action=action,
            position_before=position_before,
            position_after=self.position.quantity,
            fill_price=fill_price,
            gross_pnl=gross_pnl,
            transaction_cost=transaction_cost,
            net_pnl=(
                gross_pnl - transaction_cost
            ),
            equity_before=equity_before,
            equity_after=equity_after,
            reward=reward,
        )

        terminated = (
            self.index
            >= len(self.market_data) - 1
        )

        info = {
            "transition": transition,
            "drawdown": self.accounting.drawdown(
                self.position,
                next_market,
            ),
            "equity": equity_after,
        }

        self.previous_close = previous_close

        return (
            self._observation(),
            reward,
            terminated,
            False,
            info,
        )