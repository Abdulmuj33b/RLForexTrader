from dataclasses import dataclass

import numpy as np

from arl_trp.data.fold import FoldDataset
from arl_trp.environment.observation import ObservationNormalizer
from arl_trp.domain.market import MarketSnapshot


@dataclass(frozen=True)
class NormalizedFoldDataset:
    """
    A walk-forward fold after applying a training-fitted normalizer.

    The normalizer is fitted exclusively on training observations.
    The same fitted normalizer is then applied to both training
    and test observations.
    """

    fold: FoldDataset
    train_observations: np.ndarray
    test_observations: np.ndarray
    normalizer: ObservationNormalizer

    def __post_init__(self) -> None:
        if self.train_observations.ndim != 2:
            raise ValueError(
                "train_observations must be two-dimensional"
            )

        if self.test_observations.ndim != 2:
            raise ValueError(
                "test_observations must be two-dimensional"
            )

        if len(self.train_observations) != self.fold.train_size:
            raise ValueError(
                "train observation count does not match fold"
            )

        if len(self.test_observations) != self.fold.test_size:
            raise ValueError(
                "test observation count does not match fold"
            )

    @property
    def train_size(self) -> int:
        return len(self.train_observations)

    @property
    def test_size(self) -> int:
        return len(self.test_observations)


class FoldObservationNormalizer:
    """
    Fits an ObservationNormalizer using only one fold's training data.

    v0.1 normalization protocol:

        volume_normalized =
            (volume - training_median)
            / training_IQR

    No test observation is used during fitting.
    """

    def __init__(self) -> None:
        self._normalizer = ObservationNormalizer()

    @property
    def normalizer(self) -> ObservationNormalizer:
        return self._normalizer

    def fit(
        self,
        train_observations: np.ndarray,
    ) -> "FoldObservationNormalizer":
        self._normalizer.fit(train_observations)
        return self

    def transform(
        self,
        observations: np.ndarray,
    ) -> np.ndarray:
        return self._normalizer.transform(observations)

    def fit_transform_fold(
        self,
        fold: FoldDataset,
        train_observations: np.ndarray,
        test_observations: np.ndarray,
    ) -> NormalizedFoldDataset:
        if len(train_observations) != fold.train_size:
            raise ValueError(
                "train observations do not match fold size"
            )

        if len(test_observations) != fold.test_size:
            raise ValueError(
                "test observations do not match fold size"
            )

        self.fit(train_observations)

        normalized_train = self.transform(
            train_observations
        )

        normalized_test = self.transform(
            test_observations
        )

        return NormalizedFoldDataset(
            fold=fold,
            train_observations=normalized_train,
            test_observations=normalized_test,
            normalizer=self.normalizer,
        )