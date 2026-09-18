from datetime import datetime

import numpy as np
import pytest

from arl_trp.domain.market import MarketSnapshot
from arl_trp.domain.position import Position
from arl_trp.environment.observation import StateEncoder
from arl_trp.portfolio.accounting import PortfolioAccounting


def make_market(
    *,
    open_price: float = 100.0,
    high: float = 105.0,
    low: float = 95.0,
    close: float = 100.0,
    volume: float = 1000.0,
    bid: float = 99.9,
    ask: float = 100.1,
) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=datetime(2026, 1, 1),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        bid=bid,
        ask=ask,
    )


def make_accounting(initial_balance: float = 10_000.0) -> PortfolioAccounting:
    return PortfolioAccounting(initial_balance=initial_balance)


def test_encoder_returns_ten_features():
    market = make_market()
    position = Position()
    accounting = make_accounting()
    encoder = StateEncoder(initial_balance=10_000.0)

    observation = encoder.encode(
        market=market,
        previous_close=98.0,
        position=position,
        accounting=accounting,
    )

    assert observation.shape == (10,)


def test_encoder_returns_float32():
    market = make_market()
    position = Position()
    accounting = make_accounting()
    encoder = StateEncoder(initial_balance=10_000.0)

    observation = encoder.encode(
        market=market,
        previous_close=98.0,
        position=position,
        accounting=accounting,
    )

    assert observation.dtype == np.float32


def test_flat_position_encoding():
    market = make_market()
    position = Position()
    accounting = make_accounting()
    encoder = StateEncoder(initial_balance=10_000.0)

    observation = encoder.encode(
        market=market,
        previous_close=98.0,
        position=position,
        accounting=accounting,
    )

    assert observation[6] == pytest.approx(0.0)
    assert observation[7] == pytest.approx(0.0)
    assert observation[8] == pytest.approx(1.0)
    assert observation[9] == pytest.approx(0.0)


def test_relative_price_features():
    market = make_market(
        open_price=102.0,
        high=105.0,
        low=95.0,
        close=100.0,
    )
    position = Position()
    accounting = make_accounting()
    encoder = StateEncoder(initial_balance=10_000.0)

    observation = encoder.encode(
        market=market,
        previous_close=98.0,
        position=position,
        accounting=accounting,
    )

    assert observation[0] == pytest.approx(0.02)
    assert observation[1] == pytest.approx(0.05)
    assert observation[2] == pytest.approx(-0.05)
    assert observation[3] == pytest.approx((100.0 - 98.0) / 98.0)


def test_relative_spread():
    market = make_market(
        close=100.0,
        bid=99.5,
        ask=100.5,
    )
    position = Position()
    accounting = make_accounting()
    encoder = StateEncoder(initial_balance=10_000.0)

    observation = encoder.encode(
        market=market,
        previous_close=100.0,
        position=position,
        accounting=accounting,
    )

    assert observation[4] == pytest.approx(0.01)


def test_volume_is_preserved_for_later_normalization():
    market = make_market(volume=2500.0)
    position = Position()
    accounting = make_accounting()
    encoder = StateEncoder(initial_balance=10_000.0)

    observation = encoder.encode(
        market=market,
        previous_close=100.0,
        position=position,
        accounting=accounting,
    )

    assert observation[5] == pytest.approx(2500.0)


def test_long_position_encoding():
    market = make_market(
        bid=102.0,
        ask=102.2,
        close=102.1,
    )
    position = Position()
    position.open(quantity=1.0, price=100.0)
    accounting = make_accounting()
    encoder = StateEncoder(initial_balance=10_000.0)

    observation = encoder.encode(
        market=market,
        previous_close=100.0,
        position=position,
        accounting=accounting,
    )

    assert observation[6] == pytest.approx(1.0)
    assert observation[7] == pytest.approx(0.02)


def test_short_position_encoding():
    market = make_market(
        bid=97.8,
        ask=98.0,
        close=97.9,
    )
    position = Position()
    position.open(quantity=-1.0, price=100.0)
    accounting = make_accounting()
    encoder = StateEncoder(initial_balance=10_000.0)

    observation = encoder.encode(
        market=market,
        previous_close=100.0,
        position=position,
        accounting=accounting,
    )

    assert observation[6] == pytest.approx(-1.0)
    assert observation[7] == pytest.approx(0.02)


def test_relative_equity():
    market = make_market()
    position = Position()
    accounting = make_accounting(initial_balance=10_000.0)
    accounting.apply_realized_pnl(500.0)
    encoder = StateEncoder(initial_balance=10_000.0)

    observation = encoder.encode(
        market=market,
        previous_close=100.0,
        position=position,
        accounting=accounting,
    )

    assert observation[8] == pytest.approx(1.05)


def test_invalid_previous_close_is_rejected():
    market = make_market()
    position = Position()
    accounting = make_accounting()
    encoder = StateEncoder(initial_balance=10_000.0)

    with pytest.raises(ValueError, match="previous_close must be positive"):
        encoder.encode(
            market=market,
            previous_close=0.0,
            position=position,
            accounting=accounting,
        )


def test_invalid_initial_balance_is_rejected():
    with pytest.raises(
        ValueError,
        match="initial_balance must be positive",
    ):
        StateEncoder(initial_balance=0.0)