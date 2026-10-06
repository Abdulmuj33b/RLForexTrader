from datetime import datetime, timedelta, timezone

import pytest

from arl_trp.data.dataset import (
    DatasetSpec,
    ValidatedDataset,
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
            volume=100.0,
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


def test_default_configuration():
    config = WalkForwardConfig()

    assert config.feature_lookback_bars == 500
    assert config.label_horizon_bars == 100
    assert config.initial_train_bars == 601
    assert config.folds == 5


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        WalkForwardConfig(
            feature_lookback_bars=0,
        )

    with pytest.raises(ValueError):
        WalkForwardConfig(
            label_horizon_bars=0,
        )

    with pytest.raises(ValueError):
        WalkForwardConfig(
            initial_train_bars=0,
        )

    with pytest.raises(ValueError):
        WalkForwardConfig(
            folds=0,
        )


def test_initial_training_window_must_cover_protocol_lookback():
    with pytest.raises(ValueError):
        WalkForwardConfig(
            initial_train_bars=600,
        )


def test_creates_requested_number_of_folds():
    dataset = make_dataset()

    folds = PurgedWalkForwardSplitter(
        WalkForwardConfig()
    ).split(dataset)

    assert len(folds) == 5


def test_folds_are_deterministic():
    dataset = make_dataset()

    splitter = PurgedWalkForwardSplitter(
        WalkForwardConfig()
    )

    first = splitter.split(dataset)
    second = splitter.split(dataset)

    assert first == second


def test_train_and_test_are_chronological():
    dataset = make_dataset()

    folds = PurgedWalkForwardSplitter(
        WalkForwardConfig()
    ).split(dataset)

    for fold in folds:
        assert fold.train_start < fold.train_end
        assert fold.train_end == fold.purge_start
        assert fold.purge_start < fold.purge_end
        assert fold.purge_end == fold.embargo_start
        assert fold.embargo_start < fold.embargo_end
        assert fold.embargo_end == fold.test_start
        assert fold.test_start < fold.test_end


def test_training_data_never_overlaps_test_data():
    dataset = make_dataset()

    folds = PurgedWalkForwardSplitter(
        WalkForwardConfig()
    ).split(dataset)

    for fold in folds:
        train = set(fold.train_indices)
        test = set(fold.test_indices)

        assert train.isdisjoint(test)


def test_purge_uses_label_horizon():
    dataset = make_dataset()

    config = WalkForwardConfig(
        feature_lookback_bars=500,
        label_horizon_bars=100,
        initial_train_bars=601,
        folds=5,
    )

    folds = PurgedWalkForwardSplitter(
        config
    ).split(dataset)

    for fold in folds:
        assert fold.purge_size == 100


def test_embargo_uses_feature_lookback():
    dataset = make_dataset()

    config = WalkForwardConfig(
        feature_lookback_bars=500,
        label_horizon_bars=100,
        initial_train_bars=601,
        folds=5,
    )

    folds = PurgedWalkForwardSplitter(
        config
    ).split(dataset)

    for fold in folds:
        assert fold.embargo_size == 500


def test_training_ends_at_purge_boundary():
    dataset = make_dataset()

    folds = PurgedWalkForwardSplitter(
        WalkForwardConfig()
    ).split(dataset)

    for fold in folds:
        assert fold.train_end == fold.purge_start


def test_purge_precedes_embargo():
    dataset = make_dataset()

    folds = PurgedWalkForwardSplitter(
        WalkForwardConfig()
    ).split(dataset)

    for fold in folds:
        assert fold.purge_end == fold.embargo_start


def test_embargo_precedes_test():
    dataset = make_dataset()

    folds = PurgedWalkForwardSplitter(
        WalkForwardConfig()
    ).split(dataset)

    for fold in folds:
        assert fold.embargo_end == fold.test_start


def test_test_indices_are_contiguous():
    dataset = make_dataset()

    folds = PurgedWalkForwardSplitter(
        WalkForwardConfig()
    ).split(dataset)

    for fold in folds:
        indices = list(fold.test_indices)

        assert indices == list(
            range(
                fold.test_start,
                fold.test_end,
            )
        )


def test_fold_ids_are_sequential():
    dataset = make_dataset()

    folds = PurgedWalkForwardSplitter(
        WalkForwardConfig()
    ).split(dataset)

    assert [
        fold.fold_id
        for fold in folds
    ] == [1, 2, 3, 4, 5]


def test_fold_sizes_are_positive():
    dataset = make_dataset()

    folds = PurgedWalkForwardSplitter(
        WalkForwardConfig()
    ).split(dataset)

    for fold in folds:
        assert fold.train_size > 0
        assert fold.test_size > 0
        assert fold.purge_size == 100
        assert fold.embargo_size == 500


def test_fold_training_window_expands():
    dataset = make_dataset()

    folds = PurgedWalkForwardSplitter(
        WalkForwardConfig()
    ).split(dataset)

    train_sizes = [
        fold.train_size
        for fold in folds
    ]

    assert train_sizes == sorted(train_sizes)
    assert len(set(train_sizes)) == len(train_sizes)


def test_test_windows_have_equal_size():
    dataset = make_dataset()

    folds = PurgedWalkForwardSplitter(
        WalkForwardConfig()
    ).split(dataset)

    test_sizes = [
        fold.test_size
        for fold in folds
    ]

    assert len(set(test_sizes)) == 1


def test_final_fold_stays_inside_dataset():
    dataset = make_dataset()

    folds = PurgedWalkForwardSplitter(
        WalkForwardConfig()
    ).split(dataset)

    assert folds[-1].test_end <= dataset.size


def test_insufficient_dataset_rejected():
    dataset = make_dataset(size=1000)

    splitter = PurgedWalkForwardSplitter(
        WalkForwardConfig()
    )

    with pytest.raises(ValueError):
        splitter.split(dataset)