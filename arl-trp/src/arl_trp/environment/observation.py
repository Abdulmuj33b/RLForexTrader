from dataclasses import dataclass

import numpy as np

from arl_trp.domain.market import MarketSnapshot
from arl_trp.domain.position import Position
from arl_trp.portfolio.accounting import PortfolioAccounting


@dataclass(frozen=True)
class StateEncoder:
    """
    Converts the current market and portfolio state into
    a deterministic 10-feature vector.

    v0.1 feature order:
        0. relative_open
        1. relative_high
        2. relative_low
        3. close_return
        4. relative_spread
        5. volume
        6. position_direction
        7. unrealized_return
        8. relative_equity
        9. drawdown

    Normalization of volume is intentionally handled elsewhere.
    """

    initial_balance: float

    def __post_init__(self):
        if self.initial_balance <= 0.0:
            raise ValueError("initial_balance must be positive")

    def encode(
        self,
        market: MarketSnapshot,
        previous_close: float,
        position: Position,
        accounting: PortfolioAccounting,
    ) -> np.ndarray:
        if previous_close <= 0.0:
            raise ValueError("previous_close must be positive")

        close = market.close

        relative_open = (market.open - close) / close
        relative_high = (market.high - close) / close
        relative_low = (market.low - close) / close
        close_return = (close - previous_close) / previous_close
        relative_spread = market.spread / close

        position_direction = (
            1.0 if position.is_long
            else -1.0 if position.is_short
            else 0.0
        )

        if position.is_flat:
            unrealized_return = 0.0
        elif position.is_long:
            unrealized_return = (
                market.bid - position.entry_price
            ) / position.entry_price
        else:
            unrealized_return = (
                position.entry_price - market.ask
            ) / position.entry_price

        equity = accounting.equity(position, market)
        relative_equity = equity / self.initial_balance
        drawdown = accounting.drawdown(position, market)

        return np.array(
            [
                relative_open,
                relative_high,
                relative_low,
                close_return,
                relative_spread,
                market.volume,
                position_direction,
                unrealized_return,
                relative_equity,
                drawdown,
            ],
            dtype=np.float32,
        )

@dataclass
class ObservationNormalizer:
    """
    Training-fold-fitted normalizer for the v0.1 observation vector.

    Only volume is statistically normalized in v0.1.

    Volume normalization:
        normalized_volume =
            (volume - training_median) / training_iqr

    All other features are already dimensionless or bounded and are
    therefore passed through unchanged.

    The normalizer must be fitted on training data before it can
    transform observations.
    """

    volume_median: float | None = None
    volume_iqr: float | None = None

    FEATURE_COUNT: int = 10
    VOLUME_INDEX: int = 5

    @property
    def is_fitted(self) -> bool:
        return (
            self.volume_median is not None
            and self.volume_iqr is not None
        )

    def fit(self, observations: np.ndarray) -> "ObservationNormalizer":
        observations = np.asarray(observations, dtype=np.float32)

        if observations.size == 0:
            raise ValueError("observations cannot be empty")

        if observations.ndim != 2 or observations.shape[1] != self.FEATURE_COUNT:
            raise ValueError(
                f"observations must contain {self.FEATURE_COUNT} features"
            )

        volume = observations[:, self.VOLUME_INDEX]

        self.volume_median = float(np.median(volume))

        q1, q3 = np.percentile(volume, [25.0, 75.0])
        self.volume_iqr = float(q3 - q1)

        # Constant-volume training data must not create a
        # division-by-zero problem.
        if self.volume_iqr == 0.0:
            self.volume_iqr = 1.0

        return self

    def transform(self, observations: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(
                "ObservationNormalizer must be fitted before transform"
            )

        observations = np.asarray(observations, dtype=np.float32)

        if observations.ndim != 2 or observations.shape[1] != self.FEATURE_COUNT:
            raise ValueError(
                f"observations must contain {self.FEATURE_COUNT} features"
            )

        normalized = observations.copy()

        normalized[:, self.VOLUME_INDEX] = (
            normalized[:, self.VOLUME_INDEX] - self.volume_median
        ) / self.volume_iqr

        return normalized.astype(np.float32)

    def fit_transform(self, observations: np.ndarray) -> np.ndarray:
        self.fit(observations)
        return self.transform(observations)

from collections import deque
@dataclass
class ObservationBuffer:
    """
    Maintains a fixed-size chronological observation window.

    v0.1:
        - window size: 64 bars
        - features per bar: 10
        - no zero-padding
        - buffer becomes ready only after 64 observations
        - once full, the oldest observation is discarded first
    """

    window_size: int = 64
    feature_count: int = 10

    def __post_init__(self):
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")

        if self.feature_count <= 0:
            raise ValueError("feature_count must be positive")

        self._buffer = deque(maxlen=self.window_size)

    def append(self, observation: np.ndarray) -> None:
        observation = np.asarray(observation, dtype=np.float32)

        if observation.ndim != 1:
            raise ValueError(
                "observation must be a one-dimensional array"
            )

        if observation.shape[0] != self.feature_count:
            raise ValueError(
                f"observation must contain {self.feature_count} features"
            )

        self._buffer.append(observation.copy())

    @property
    def is_ready(self) -> bool:
        return len(self._buffer) == self.window_size

    def __len__(self) -> int:
        return len(self._buffer)

    def get(self) -> np.ndarray:
        if not self.is_ready:
            raise RuntimeError(
                "ObservationBuffer is not ready"
            )

        return np.array(self._buffer, dtype=np.float32, copy=True)

    def reset(self) -> None:
        self._buffer.clear()
