from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arl_trp.domain.order import Action
from arl_trp.evaluation.policy import PolicyEvaluationResult


class BaselinePolicy:
    """
    Minimal interface for deterministic baseline policies.

    Baselines intentionally do not implement their own execution,
    accounting, or reward logic. They only select an Action.

    TradingEnvironment remains the single source of truth for:
        - execution
        - spread
        - slippage
        - commission
        - position constraints
        - equity
        - reward
    """

    name: str = "baseline"

    def predict(self, observation, deterministic: bool = True):
        """
        SB3-compatible prediction interface.

        Returns:
            action, state
        """
        raise NotImplementedError


@dataclass(frozen=True)
class FlatBaseline(BaselinePolicy):
    """
    Never enters a position.
    """

    name: str = "flat"

    def predict(self, observation, deterministic: bool = True):
        return Action.HOLD, None


@dataclass
class BuyAndHoldBaseline(BaselinePolicy):
    """
    Enters long on the first decision and then holds.

    This is deliberately simple:
        first decision -> LONG
        subsequent decisions -> HOLD
    """

    name: str = "buy_and_hold"
    entered: bool = False

    def reset(self) -> None:
        self.entered = False

    def predict(self, observation, deterministic: bool = True):
        if not self.entered:
            self.entered = True
            return Action.LONG, None

        return Action.HOLD, None


@dataclass(frozen=True)
class MomentumBaseline(BaselinePolicy):
    """
    Simple direction-of-recent-return baseline.

    v0.1 definition:

        lookback = N bars

        recent_return =
            (close[t] - close[t-N]) / close[t-N]

        recent_return > 0 -> LONG
        recent_return < 0 -> SHORT
        recent_return == 0 -> HOLD

    The environment interprets an opposite action as CLOSE rather
    than an immediate reversal. Therefore this baseline does not
    bypass the environment's position constraints.
    """

    lookback_bars: int = 20
    name: str = "momentum"

    def __post_init__(self) -> None:
        if self.lookback_bars <= 0:
            raise ValueError(
                "lookback_bars must be positive"
            )

    def predict(self, observation, deterministic: bool = True):
        raise RuntimeError(
            "MomentumBaseline requires environment-aware evaluation"
        )

    def action(self, environment) -> Action:
        index = environment.index

        if index < self.lookback_bars:
            return Action.HOLD

        current_close = environment.market_data[index].close
        previous_close = environment.market_data[
            index - self.lookback_bars
        ].close

        if previous_close <= 0.0:
            raise ValueError(
                "momentum reference price must be positive"
            )

        recent_return = (
            current_close - previous_close
        ) / previous_close

        if recent_return > 0.0:
            return Action.LONG

        if recent_return < 0.0:
            return Action.SHORT

        return Action.HOLD


@dataclass
class ConstrainedRandomBaseline(BaselinePolicy):
    """
    Seeded random baseline.

    Actions are sampled from HOLD/LONG/SHORT/CLOSE using fixed
    probabilities.

    The environment remains responsible for deciding whether an
    action actually changes the position.

    The random generator is resettable so repeated evaluations with
    the same seed are deterministic.
    """

    seed: int = 42
    hold_probability: float = 0.70
    long_probability: float = 0.10
    short_probability: float = 0.10
    close_probability: float = 0.10
    name: str = "constrained_random"

    def __post_init__(self) -> None:
        probabilities = (
            self.hold_probability,
            self.long_probability,
            self.short_probability,
            self.close_probability,
        )

        if any(
            probability < 0.0
            for probability in probabilities
        ):
            raise ValueError(
                "action probabilities cannot be negative"
            )

        total = sum(probabilities)

        if not np.isclose(total, 1.0):
            raise ValueError(
                "action probabilities must sum to 1.0"
            )

        self._rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def predict(self, observation, deterministic: bool = True):
        probabilities = np.array(
            [
                self.hold_probability,
                self.long_probability,
                self.short_probability,
                self.close_probability,
            ],
            dtype=np.float64,
        )

        action = self._rng.choice(
            np.array(
                [
                    Action.HOLD,
                    Action.LONG,
                    Action.SHORT,
                    Action.CLOSE,
                ],
                dtype=np.int64,
            ),
            p=probabilities,
        )

        return Action(int(action)), None


class BaselineEvaluator:
    """
    Evaluates a baseline through the same environment interface used
    by PolicyEvaluator.

    Returns the same PolicyEvaluationResult type used by PPO
    evaluation so the downstream reporting and statistics pipeline
    has one evaluation contract.

    This class exists because MomentumBaseline needs access to the
    current market index to calculate its historical return.

    Completed trades are defined identically to PolicyEvaluator:
        position_before != 0
        AND
        position_after == 0
    """

    def evaluate(
        self,
        policy: BaselinePolicy,
        environment,
    ) -> PolicyEvaluationResult:
        observation, _ = environment.reset()

        self._reset_policy(policy)

        initial_equity = self._extract_equity(
            environment,
            fallback=environment.initial_balance,
        )

        equity_curve = [initial_equity]
        trade_pnls: list[float] = []
        actions: list[int] = []

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = self._select_action(
                policy,
                environment,
                observation,
            )

            observation, _reward, terminated, truncated, info = (
                environment.step(action)
            )

            equity = self._extract_equity(
                environment,
                info.get("equity"),
            )

            equity_curve.append(equity)
            actions.append(int(action))

            transition = info.get("transition")

            if transition is not None:
                if (
                    transition.position_before != 0.0
                    and transition.position_after == 0.0
                ):
                    trade_pnls.append(
                        float(transition.net_pnl)
                    )

        return PolicyEvaluationResult(
            equity_curve=np.asarray(
                equity_curve,
                dtype=np.float64,
            ),
            trade_pnls=np.asarray(
                trade_pnls,
                dtype=np.float64,
            ),
            actions=np.asarray(
                actions,
                dtype=np.int64,
            ),
            steps=len(actions),
        )

    @staticmethod
    def _reset_policy(policy: BaselinePolicy) -> None:
        reset = getattr(policy, "reset", None)

        if reset is not None:
            reset()

    @staticmethod
    def _select_action(
        policy: BaselinePolicy,
        environment,
        observation,
    ) -> Action:
        action_method = getattr(policy, "action", None)

        if action_method is not None:
            action = action_method(environment)
            return Action(action)

        action, _ = policy.predict(
            observation,
            deterministic=True,
        )

        return Action(action)

    @staticmethod
    def _extract_equity(
        environment,
        fallback: float | None,
    ) -> float:
        if fallback is not None:
            return float(fallback)

        market = environment.market_data[
            environment.index
        ]

        return float(
            environment.accounting.equity(
                environment.position,
                market,
            )
        )