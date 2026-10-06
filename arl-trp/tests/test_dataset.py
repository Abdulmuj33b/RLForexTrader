from datetime import datetime, timezone

import pytest

from arl_trp.data.dataset import DatasetSpec


def make_dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        version="xauusd_5m_v1",
        instrument="XAUUSD",
        timeframe="5m",
        source="historical_broker_data",
        start=datetime(
            2018,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        end=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )


def test_dataset_spec():
    dataset = make_dataset_spec()

    assert dataset.version == "xauusd_5m_v1"
    assert dataset.instrument == "XAUUSD"
    assert dataset.timeframe == "5m"
    assert dataset.source == "historical_broker_data"


def test_dataset_temporal_boundary():
    dataset = make_dataset_spec()

    assert dataset.start < dataset.end


def test_dataset_duration():
    dataset = make_dataset_spec()

    assert dataset.duration_days == pytest.approx(
        2922.0
    )


def test_empty_version_rejected():
    with pytest.raises(ValueError):
        DatasetSpec(
            version="",
            instrument="XAUUSD",
            timeframe="5m",
            source="historical_broker_data",
            start=datetime(
                2018,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            end=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
        )


def test_invalid_temporal_boundary_rejected():
    start = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    with pytest.raises(ValueError):
        DatasetSpec(
            version="xauusd_5m_v1",
            instrument="XAUUSD",
            timeframe="5m",
            source="historical_broker_data",
            start=start,
            end=start,
        )