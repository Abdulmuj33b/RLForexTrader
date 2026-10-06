from datetime import datetime, timedelta, timezone

from arl_trp.data.validation import validate_market_data
from arl_trp.domain.market import MarketSnapshot


def make_snapshot(
    timestamp: datetime,
) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=timestamp,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=10.0,
        bid=100.4,
        ask=100.6,
    )


def make_start() -> datetime:
    return datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )


def test_valid_market_data():
    start = make_start()

    market_data = [
        make_snapshot(start),
        make_snapshot(start + timedelta(minutes=5)),
        make_snapshot(start + timedelta(minutes=10)),
    ]

    result = validate_market_data(
        market_data,
        expected_interval=timedelta(minutes=5),
    )

    assert result.valid
    assert result.errors == ()
    assert result.gaps == ()


def test_empty_market_data_rejected():
    result = validate_market_data([])

    assert not result.valid
    assert "market_data cannot be empty" in result.errors


def test_timestamps_must_be_increasing():
    start = make_start()

    market_data = [
        make_snapshot(start),
        make_snapshot(start),
    ]

    result = validate_market_data(market_data)

    assert not result.valid
    assert any(
        "timestamps must be strictly increasing" in error
        for error in result.errors
    )


def test_duplicate_timestamps_are_rejected():
    start = make_start()

    market_data = [
        make_snapshot(start),
        make_snapshot(start),
    ]

    result = validate_market_data(market_data)

    assert not result.valid
    assert len(result.errors) == 1


def test_timezone_naive_timestamp_is_rejected():
    start = datetime(
        2026,
        1,
        1,
    )

    market_data = [
        make_snapshot(start),
    ]

    result = validate_market_data(market_data)

    assert not result.valid
    assert any(
        "timezone-aware" in error
        for error in result.errors
    )


def test_unexpected_gap_is_reported():
    start = make_start()

    market_data = [
        make_snapshot(start),
        make_snapshot(start + timedelta(minutes=5)),
        make_snapshot(start + timedelta(minutes=15)),
    ]

    result = validate_market_data(
        market_data,
        expected_interval=timedelta(minutes=5),
    )

    assert result.valid
    assert result.errors == ()
    assert result.gaps == (2,)


def test_multiple_validation_errors_are_reported():
    start = make_start()

    market_data = [
        make_snapshot(start),
        make_snapshot(start),
    ]

    result = validate_market_data(market_data)

    assert not result.valid
    assert len(result.errors) >= 1