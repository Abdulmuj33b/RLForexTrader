from datetime import datetime, timedelta, timezone

import pytest

from arl_trp.data.dataset import (
    DatasetSpec,
    ValidatedDataset,
)
from arl_trp.data.fold import (
    FoldDataset,
    WalkForwardDatasetBuilder,
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
            open=1500.0 + index,
            high=1501.0 + index,
            low=1499.0 + index,
            close=1500.0 + index,
            volume=100.0 + index,
            bid=1499.9 + index,
            ask=1500.1 + index,
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

    folds = PurgedWalkForwardSplitter(
        WalkForwardConfig()
    ).split(dataset)

    return dataset, folds[0]


def test_build_returns_fold_dataset():
    dataset, fold = make_fold()

    result = WalkForwardDatasetBuilder.build(
        dataset,
        fold,
    )

    assert isinstance(result, FoldDataset)


def test_train_size_matches_fold():
    dataset, fold = make_fold()

    result = WalkForwardDatasetBuilder.build(
        dataset,
        fold,
    )

    assert result.train_size == fold.train_size


def test_test_size_matches_fold():
    dataset, fold = make_fold()

    result = WalkForwardDatasetBuilder.build(
        dataset,
        fold,
    )

    assert result.test_size == fold.test_size


def test_training_data_matches_fold_indices():
    dataset, fold = make_fold()

    result = WalkForwardDatasetBuilder.build(
        dataset,
        fold,
    )

    expected = dataset.snapshots[
        fold.train_start:fold.train_end
    ]

    assert result.train == expected


def test_test_data_matches_fold_indices():
    dataset, fold = make_fold()

    result = WalkForwardDatasetBuilder.build(
        dataset,
        fold,
    )

    expected = dataset.snapshots[
        fold.test_start:fold.test_end
    ]

    assert result.test == expected


def test_purge_and_embargo_are_excluded():
    dataset, fold = make_fold()

    result = WalkForwardDatasetBuilder.build(
        dataset,
        fold,
    )

    train_indices = {
        snapshot.timestamp
        for snapshot in result.train
    }

    test_indices = {
        snapshot.timestamp
        for snapshot in result.test
    }

    excluded = dataset.snapshots[
        fold.purge_start:fold.embargo_end
    ]

    for snapshot in excluded:
        assert snapshot.timestamp not in train_indices
        assert snapshot.timestamp not in test_indices


def test_train_and_test_are_disjoint():
    dataset, fold = make_fold()

    result = WalkForwardDatasetBuilder.build(
        dataset,
        fold,
    )

    train_timestamps = {
        snapshot.timestamp
        for snapshot in result.train
    }

    test_timestamps = {
        snapshot.timestamp
        for snapshot in result.test
    }

    assert train_timestamps.isdisjoint(
        test_timestamps
    )


def test_train_remains_chronological():
    dataset, fold = make_fold()

    result = WalkForwardDatasetBuilder.build(
        dataset,
        fold,
    )

    timestamps = [
        snapshot.timestamp
        for snapshot in result.train
    ]

    assert timestamps == sorted(timestamps)


def test_test_remains_chronological():
    dataset, fold = make_fold()

    result = WalkForwardDatasetBuilder.build(
        dataset,
        fold,
    )

    timestamps = [
        snapshot.timestamp
        for snapshot in result.test
    ]

    assert timestamps == sorted(timestamps)


def test_builder_does_not_modify_source_dataset():
    dataset, fold = make_fold()

    original = dataset.snapshots

    WalkForwardDatasetBuilder.build(
        dataset,
        fold,
    )

    assert dataset.snapshots == original


def test_invalid_train_start_rejected():
    dataset, fold = make_fold()

    invalid_fold = type(fold)(
        fold_id=fold.fold_id,
        train_start=-1,
        train_end=fold.train_end,
        purge_start=fold.purge_start,
        purge_end=fold.purge_end,
        embargo_start=fold.embargo_start,
        embargo_end=fold.embargo_end,
        test_start=fold.test_start,
        test_end=fold.test_end,
    )

    with pytest.raises(ValueError):
        WalkForwardDatasetBuilder.build(
            dataset,
            invalid_fold,
        )


def test_train_end_beyond_dataset_rejected():
    dataset, fold = make_fold()

    invalid_fold = type(fold)(
        fold_id=fold.fold_id,
        train_start=fold.train_start,
        train_end=dataset.size + 1,
        purge_start=fold.purge_start,
        purge_end=fold.purge_end,
        embargo_start=fold.embargo_start,
        embargo_end=fold.embargo_end,
        test_start=fold.test_start,
        test_end=fold.test_end,
    )

    with pytest.raises(ValueError):
        WalkForwardDatasetBuilder.build(
            dataset,
            invalid_fold,
        )


def test_test_end_beyond_dataset_rejected():
    dataset, fold = make_fold()

    invalid_fold = type(fold)(
        fold_id=fold.fold_id,
        train_start=fold.train_start,
        train_end=fold.train_end,
        purge_start=fold.purge_start,
        purge_end=fold.purge_end,
        embargo_start=fold.embargo_start,
        embargo_end=fold.embargo_end,
        test_start=fold.test_start,
        test_end=dataset.size + 1,
    )

    with pytest.raises(ValueError):
        WalkForwardDatasetBuilder.build(
            dataset,
            invalid_fold,
        )