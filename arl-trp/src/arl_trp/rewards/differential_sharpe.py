from dataclasses import dataclass


@dataclass
class DifferentialSharpeReward:
    """
    Online differential-Sharpe-style reward estimator.

    The estimator is reset for every independent episode/fold.

    v0.1 protocol:
        - minimum observations: 500
        - variance floor: 1e-8
        - observations before burn-in return 0.0
    """

    min_observations: int = 500
    variance_floor: float = 1e-8

    count: int = 0
    mean: float = 0.0
    variance: float = 0.0

    def __post_init__(self):
        if self.min_observations <= 0:
            raise ValueError(
                "min_observations must be positive"
            )

        if self.variance_floor <= 0.0:
            raise ValueError(
                "variance_floor must be positive"
            )

    def reset(self) -> None:
        """Reset estimator state."""

        self.count = 0
        self.mean = 0.0
        self.variance = 0.0

    def update(self, reward: float) -> float:
        """
        Update the online return statistics and return
        the differential-Sharpe-style signal.

        Before min_observations are available, the signal
        is zero and only the statistics are accumulated.
        """

        self.count += 1

        if self.count == 1:
            self.mean = reward
            self.variance = 0.0
            return 0.0

        previous_mean = self.mean

        delta = reward - previous_mean

        self.mean += delta / self.count

        self.variance = (
            (self.count - 1)
            * self.variance
            + delta
            * (reward - self.mean)
        ) / self.count

        if self.count < self.min_observations:
            return 0.0

        variance = max(
            self.variance,
            self.variance_floor,
        )

        standard_deviation = variance ** 0.5

        if standard_deviation <= 0.0:
            return 0.0

        # Differential-Sharpe-style incremental signal.
        #
        # This is intentionally a learning signal rather
        # than the sole economic objective.
        differential = (
            reward - previous_mean
        ) / standard_deviation

        return differential
