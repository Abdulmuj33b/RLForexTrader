from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from arl_trp.data.dataset import (
    DatasetSpec,
    ValidatedDataset,
)
from arl_trp.data.fold import (
    WalkForwardDatasetBuilder,
)
from arl_trp.data.normalized_fold import (
    FoldObservationNormalizer,
    NormalizedFoldDataset,
)
from arl_trp.data.splitter import (
    PurgedWalkForwardSplitter,
    WalkForwardConfig,
)
from arl_trp.domain.market import MarketSnapshot


def make_dataset(size: int = 6000) -> ValidatedDataset:
    snapshots = tuple(
        MarketSnapshot(
            timestamp=(
                datetime(
                    2020,
                    1,
                    1,
                    tzinfo=timezone.utc,
                )
                + timedelta(minutes=5 * index)
            ),
            open=1500.0,
            high=1501.0,
            low=1499.0,
            close=1500.0,
            volume=100.0 + index,
            bid=1499.9,
            ask=1500.1,
        )
        for index in range(size)
    )

    spec = DatasetSpec(
        version="test_v1",
        instrument="XAUUSD",
        timeframe="5m",
        source="test",
        start=snapshots[0].timestamp,
        end=snapshots[-1].timestamp,
    )

    return ValidatedDataset(
        spec=spec,
        snapshots=snapshots,
    )


def make_fold():
    dataset = make_dataset()

    fold = PurgedWalkForwardSplitter(
        WalkForwardConfig()
    ).split(dataset)[0]

    return WalkForwardDatasetBuilder.build(
        dataset,
        fold,
    )


def make_observations(size: int) -> np.ndarray:
    observations = np.zeros(
        (size, 10),
        dtype=np.float32,
    )

    observations[:, 0] = 0.01
    observations[:, 1] = 0.02
    observations[:, 2] = -0.01
    observations[:, 3] = 0.005
    observations[:, 4] = 0.001

    observations[:, 5] = np.arange(
        size,
        dtype=np.float32,
    )

    observations[:, 6] = 0.0
    observations[:, 7] = 0.0
    observations[:, 8] = 1.0
    observations[:, 9] = 0.0

    return observations


def test_fit_uses_training_data():
    train = make_observations(100)

    normalizer = FoldObservationNormalizer()

    normalizer.fit(train)

    assert normalizer.normalizer.is_fitted
    assert normalizer.normalizer.volume_median == 49.5


def test_transform_preserves_feature_count():
    train = make_observations(100)

    normalizer = FoldObservationNormalizer()
    normalizer.fit(train)

    transformed = normalizer.transform(train)

    assert transformed.shape == train.shape
    assert transformed.shape[1] == 10


def test_volume_is_normalized():
    train = make_observations(100)

    normalizer = FoldObservationNormalizer()
    normalizer.fit(train)

    transformed = normalizer.transform(train)

    assert not np.array_equal(
        transformed[:, 5],
        train[:, 5],
    )


def test_non_volume_features_are_unchanged():
    train = make_observations(100)

    normalizer = FoldObservationNormalizer()
    normalizer.fit(train)

    transformed = normalizer.transform(train)

    for index in range(10):
        if index != 5:
            np.testing.assert_array_equal(
                transformed[:, index],
                train[:, index],
            )


def test_test_data_is_transformed_without_refitting():
    train = make_observations(100)

    test = make_observations(20)

    test[:, 5] += 10_000

    normalizer = FoldObservationNormalizer()
    normalizer.fit(train)

    transformed = normalizer.transform(test)

    expected = (
        test[:, 5]
        - normalizer.normalizer.volume_median
    ) / normalizer.normalizer.volume_iqr

    np.testing.assert_allclose(
        transformed[:, 5],
        expected,
    )


def test_test_data_cannot_change_training_statistics():
    train = make_observations(100)

    test = make_observations(100)

    test[:, 5] = 1_000_000

    normalizer = FoldObservationNormalizer()
    normalizer.fit(train)

    median_before = normalizer.normalizer.volume_median
    iqr_before = normalizer.normalizer.volume_iqr

    normalizer.transform(test)

    assert normalizer.normalizer.volume_median == median_before
    assert normalizer.normalizer.volume_iqr == iqr_before


def test_transform_before_fit_rejected():
    normalizer = FoldObservationNormalizer()

    observations = make_observations(10)

    with pytest.raises(RuntimeError):
        normalizer.transform(observations)


def test_fold_normalization_returns_expected_type():
    fold = make_fold()

    train = make_observations(fold.train_size)
    test = make_observations(fold.test_size)

    result = FoldObservationNormalizer().fit_transform_fold(
        fold,
        train,
        test,
    )

    assert isinstance(
        result,
        NormalizedFoldDataset,
    )


def test_fold_normalization_preserves_sizes():
    fold = make_fold()

    train = make_observations(fold.train_size)
    test = make_observations(fold.test_size)

    result = FoldObservationNormalizer().fit_transform_fold(
        fold,
        train,
        test,
    )

    assert result.train_size == fold.train_size
    assert result.test_size == fold.test_size


def test_fold_normalization_uses_training_statistics():
    fold = make_fold()

    train = make_observations(fold.train_size)
    test = make_observations(fold.test_size)

    test[:, 5] += 1_000_000

    normalizer = FoldObservationNormalizer()

    result = normalizer.fit_transform_fold(
        fold,
        train,
        test,
    )

    expected_test_volume = (
        test[:, 5]
        - result.normalizer.volume_median
    ) / result.normalizer.volume_iqr

    np.testing.assert_allclose(
        result.test_observations[:, 5],
        expected_test_volume,
    )

    assert result.normalizer.volume_median != np.median(
        test[:, 5]
    )


def test_train_size_mismatch_rejected():
    fold = make_fold()

    train = make_observations(
        fold.train_size - 1
    )

    test = make_observations(
        fold.test_size
    )

    with pytest.raises(ValueError):
        FoldObservationNormalizer().fit_transform_fold(
            fold,
            train,
            test,
        )


def test_test_size_mismatch_rejected():
    fold = make_fold()

    train = make_observations(
        fold.train_size
    )

    test = make_observations(
        fold.test_size - 1
    )

    with pytest.raises(ValueError):
        FoldObservationNormalizer().fit_transform_fold(
            fold,
            train,
            test,
        )