import pytest
from datetime import datetime

from arl_trp.domain.market import MarketSnapshot
from arl_trp.domain.order import Action
from arl_trp.execution.simulator import (
    ExecutionConfig,
    ExecutionSimulator,
)


def make_market(
    bid: float = 100.0,
    ask: float = 101.0,
) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=datetime(2026, 1, 1),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000.0,
        bid=bid,
        ask=ask,
    )


def test_long_entry_uses_ask():
    simulator = ExecutionSimulator(
        ExecutionConfig()
    )

    market = make_market(
        bid=100.0,
        ask=101.0,
    )

    price = simulator.fill_price(
        Action.LONG,
        market,
    )

    assert price == 101.0


def test_short_entry_uses_bid():
    simulator = ExecutionSimulator(
        ExecutionConfig()
    )

    market = make_market(
        bid=100.0,
        ask=101.0,
    )

    price = simulator.fill_price(
        Action.SHORT,
        market,
    )

    assert price == 100.0


def test_long_close_uses_bid():
    simulator = ExecutionSimulator(
        ExecutionConfig()
    )

    market = make_market(
        bid=100.0,
        ask=101.0,
    )

    price = simulator.close_price(
        quantity=1.0,
        market=market,
    )

    assert price == 100.0


def test_short_close_uses_ask():
    simulator = ExecutionSimulator(
        ExecutionConfig()
    )

    market = make_market(
        bid=100.0,
        ask=101.0,
    )

    price = simulator.close_price(
        quantity=-1.0,
        market=market,
    )

    assert price == 101.0


def test_slippage_increases_long_entry_price():
    simulator = ExecutionSimulator(
        ExecutionConfig(
            slippage=0.5
        )
    )

    market = make_market()

    price = simulator.fill_price(
        Action.LONG,
        market,
    )

    assert price == 101.5


def test_slippage_decreases_short_entry_price():
    simulator = ExecutionSimulator(
        ExecutionConfig(
            slippage=0.5
        )
    )

    market = make_market()

    price = simulator.fill_price(
        Action.SHORT,
        market,
    )

    assert price == 99.5


def test_slippage_reduces_long_close_price():
    simulator = ExecutionSimulator(
        ExecutionConfig(
            slippage=0.5
        )
    )

    market = make_market()

    price = simulator.close_price(
        quantity=1.0,
        market=market,
    )

    assert price == 99.5


def test_slippage_increases_short_close_price():
    simulator = ExecutionSimulator(
        ExecutionConfig(
            slippage=0.5
        )
    )

    market = make_market()

    price = simulator.close_price(
        quantity=-1.0,
        market=market,
    )

    assert price == 101.5


def test_commission_is_based_on_absolute_quantity():
    simulator = ExecutionSimulator(
        ExecutionConfig(
            commission_per_unit=2.0
        )
    )

    assert simulator.commission(3.0) == 6.0
    assert simulator.commission(-3.0) == 6.0


def test_flat_position_cannot_be_closed():
    simulator = ExecutionSimulator(
        ExecutionConfig()
    )

    market = make_market()

    with pytest.raises(ValueError):
        simulator.close_price(
            quantity=0.0,
            market=market,
        )


def test_close_action_requires_position_direction():
    simulator = ExecutionSimulator(
        ExecutionConfig()
    )

    market = make_market()

    with pytest.raises(ValueError):
        simulator.fill_price(
            Action.CLOSE,
            market,
        )


def test_negative_slippage_is_rejected():
    with pytest.raises(ValueError):
        ExecutionConfig(
            slippage=-0.1
        )


def test_negative_commission_is_rejected():
    with pytest.raises(ValueError):
        ExecutionConfig(
            commission_per_unit=-1.0
        )