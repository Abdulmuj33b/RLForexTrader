from datetime import datetime, timezone
from pathlib import Path

import pytest

from arl_trp.data.dataset import DatasetSpec
from arl_trp.data.loader import (
    DatasetLoadError,
    MarketDatasetLoader,
)


def make_dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        version="xauusd_5m_v1",
        instrument="XAUUSD",
        timeframe="5m",
        source="historical_broker_data",
        start=datetime(
            2020,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        end=datetime(
            2020,
            1,
            2,
            tzinfo=timezone.utc,
        ),
    )


def write_csv(
    tmp_path: Path,
    content: str,
) -> Path:
    path = tmp_path / "market.csv"

    path.write_text(
        content,
        encoding="utf-8",
    )

    return path


def test_load_valid_csv(tmp_path):
    path = write_csv(
        tmp_path,
        """timestamp,open,high,low,close,volume,bid,ask
2020-01-01T00:00:00+00:00,1500,1502,1498,1501,100,1500.9,1501.1
2020-01-01T00:05:00+00:00,1501,1504,1500,1503,120,1502.9,1503.1
""",
    )

    dataset = MarketDatasetLoader().load(
        path,
        make_dataset_spec(),
    )

    snapshots = dataset.snapshots

    assert len(snapshots) == 2
    assert snapshots[0].close == 1501
    assert snapshots[1].volume == 120


def test_load_preserves_source_order(tmp_path):
    path = write_csv(
        tmp_path,
        """timestamp,open,high,low,close,volume,bid,ask
2020-01-01T00:05:00+00:00,1501,1504,1500,1503,120,1502.9,1503.1
2020-01-01T00:10:00+00:00,1503,1505,1501,1504,130,1503.9,1504.1
""",
    )

    dataset = MarketDatasetLoader().load(
        path,
        make_dataset_spec(),
    )

    snapshots = dataset.snapshots

    assert snapshots[0].timestamp < snapshots[1].timestamp


def test_loader_returns_validated_dataset(tmp_path):
    path = write_csv(
        tmp_path,
        """timestamp,open,high,low,close,volume,bid,ask
2020-01-01T00:00:00+00:00,1500,1502,1498,1501,100,1500.9,1501.1
2020-01-01T00:05:00+00:00,1501,1504,1500,1503,120,1502.9,1503.1
""",
    )

    dataset_spec = make_dataset_spec()

    dataset = MarketDatasetLoader().load(
        path,
        dataset_spec,
    )

    assert dataset.spec == dataset_spec
    assert dataset.size == 2

    assert dataset.start == datetime(
        2020,
        1,
        1,
        tzinfo=timezone.utc,
    )

    assert dataset.end == datetime(
        2020,
        1,
        1,
        0,
        5,
        tzinfo=timezone.utc,
    )


def test_missing_required_column_rejected(tmp_path):
    path = write_csv(
        tmp_path,
        """timestamp,open,high,low,close,volume,bid
2020-01-01T00:00:00+00:00,1500,1502,1498,1501,100,1500.9
""",
    )

    with pytest.raises(DatasetLoadError):
        MarketDatasetLoader().load(
            path,
            make_dataset_spec(),
        )


def test_invalid_numeric_value_rejected(tmp_path):
    path = write_csv(
        tmp_path,
        """timestamp,open,high,low,close,volume,bid,ask
2020-01-01T00:00:00+00:00,bad,1502,1498,1501,100,1500.9,1501.1
""",
    )

    with pytest.raises(DatasetLoadError):
        MarketDatasetLoader().load(
            path,
            make_dataset_spec(),
        )


def test_timezone_naive_timestamp_rejected(tmp_path):
    path = write_csv(
        tmp_path,
        """timestamp,open,high,low,close,volume,bid,ask
2020-01-01T00:00:00,1500,1502,1498,1501,100,1500.9,1501.1
""",
    )

    with pytest.raises(DatasetLoadError):
        MarketDatasetLoader().load(
            path,
            make_dataset_spec(),
        )


def test_non_monotonic_timestamps_rejected(tmp_path):
    path = write_csv(
        tmp_path,
        """timestamp,open,high,low,close,volume,bid,ask
2020-01-01T00:05:00+00:00,1501,1504,1500,1503,120,1502.9,1503.1
2020-01-01T00:00:00+00:00,1500,1502,1498,1501,100,1500.9,1501.1
""",
    )

    with pytest.raises(DatasetLoadError):
        MarketDatasetLoader().load(
            path,
            make_dataset_spec(),
        )


def test_missing_file_rejected(tmp_path):
    path = tmp_path / "missing.csv"

    with pytest.raises(DatasetLoadError):
        MarketDatasetLoader().load(
            path,
            make_dataset_spec(),
        )


def test_data_before_declared_start_rejected(tmp_path):
    path = write_csv(
        tmp_path,
        """timestamp,open,high,low,close,volume,bid,ask
2019-12-31T23:55:00+00:00,1500,1502,1498,1501,100,1500.9,1501.1
""",
    )

    with pytest.raises(DatasetLoadError):
        MarketDatasetLoader().load(
            path,
            make_dataset_spec(),
        )


def test_data_after_declared_end_rejected(tmp_path):
    path = write_csv(
        tmp_path,
        """timestamp,open,high,low,close,volume,bid,ask
2020-01-02T00:05:00+00:00,1500,1502,1498,1501,100,1500.9,1501.1
""",
    )

    with pytest.raises(DatasetLoadError):
        MarketDatasetLoader().load(
            path,
            make_dataset_spec(),
        )