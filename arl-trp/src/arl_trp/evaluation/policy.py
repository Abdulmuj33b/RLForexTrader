from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyEvaluationResult:
    """
    Raw result of running one policy through one evaluation environment.

    The result contains the time series required by the metrics layer.

    trade_pnls contains only completed-trade P/L values.
    Opening commissions and unrealized P/L are not treated as
    completed trades.
    """

    equity_curve: np.ndarray
    trade_pnls: np.ndarray
    actions: np.ndarray
    steps: int

    def __post_init__(self) -> None:
        if self.equity_curve.ndim != 1:
            raise ValueError(
                "equity_curve must be one-dimensional"
            )

        if self.trade_pnls.ndim != 1:
            raise ValueError(
                "trade_pnls must be one-dimensional"
            )

        if self.actions.ndim != 1:
            raise ValueError(
                "actions must be one-dimensional"
            )

        if self.equity_curve.size < 2:
            raise ValueError(
                "equity_curve must contain at least two observations"
            )

        if self.steps <= 0:
            raise ValueError(
                "steps must be positive"
            )

        if self.actions.size != self.steps:
            raise ValueError(
                "number of actions must equal steps"
            )


class PolicyEvaluator:
    """
    Evaluates a trained policy on an environment.

    The policy is always evaluated deterministically:

        model.predict(
            observation,
            deterministic=True,
        )

    Expected Gymnasium environment contract:

        reset() ->
            observation, info

        step(action) ->
            observation,
            reward,
            terminated,
            truncated,
            info

    Required step information:

        info["equity"]
        info["transition"]

    A completed trade is identified from the transition:

        position_before != 0
        position_after == 0

    Only then is transition.net_pnl recorded in trade_pnls.

    The evaluator does not:
        - train the model
        - modify model parameters
        - fit normalization statistics
        - modify market data
        - make promotion decisions
    """

    def evaluate(
        self,
        model,
        environment,
    ) -> PolicyEvaluationResult:
        observation, _ = environment.reset()

        initial_equity = self._extract_initial_equity(
            environment
        )

        equity_curve = [
            initial_equity
        ]

        trade_pnls: list[float] = []
        actions: list[int] = []

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            action_value = self._extract_action(
                action
            )

            (
                observation,
                _reward,
                terminated,
                truncated,
                info,
            ) = environment.step(
                action_value
            )

            equity = self._extract_equity(
                info
            )

            equity_curve.append(equity)
            actions.append(action_value)

            transition = info.get(
                "transition"
            )

            if transition is not None:
                if self._is_completed_trade(
                    transition
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
    def _extract_action(action) -> int:
        """
        Convert an SB3 scalar/NumPy scalar action into an int.
        """

        array = np.asarray(action)

        if array.size != 1:
            raise ValueError(
                "policy must return exactly one action"
            )

        return int(
            array.reshape(-1)[0]
        )

    @staticmethod
    def _extract_initial_equity(
        environment,
    ) -> float:
        """
        Extract initial equity after environment reset.

        v0.1 TradingEnvironment exposes its accounting object.
        """

        accounting = getattr(
            environment,
            "accounting",
            None,
        )

        if accounting is None:
            raise ValueError(
                "evaluation environment must expose "
                "'accounting'"
            )

        equity = float(
            accounting.balance
        )

        if not np.isfinite(equity):
            raise ValueError(
                "initial evaluation equity must be finite"
            )

        if equity < 0.0:
            raise ValueError(
                "initial evaluation equity cannot be negative"
            )

        return equity

    @staticmethod
    def _extract_equity(
        info: dict,
    ) -> float:
        """
        Extract equity from the environment step info.
        """

        if "equity" not in info:
            raise ValueError(
                "evaluation environment step info must "
                "contain 'equity'"
            )

        equity = float(
            info["equity"]
        )

        if not np.isfinite(equity):
            raise ValueError(
                "evaluation equity must be finite"
            )

        if equity < 0.0:
            raise ValueError(
                "evaluation equity cannot be negative"
            )

        return equity

    @staticmethod
    def _is_completed_trade(
        transition,
    ) -> bool:
        """
        Determine whether a transition completed a trade.

        A completed trade occurs when an existing position
        becomes flat.

        Examples:

            long  -> flat   = completed trade
            short -> flat   = completed trade
            flat  -> long   = not completed
            flat  -> short  = not completed
            long  -> long   = not completed
            short -> short  = not completed
        """

        return (
            transition.position_before != 0.0
            and transition.position_after == 0.0
        )