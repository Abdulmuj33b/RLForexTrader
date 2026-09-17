import pytest

from arl_trp.domain.instrument import InstrumentSpec


def make_instrument() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        tick_size=0.01,
        tick_value=1.0,
        base_currency="XAU",
        quote_currency="USD",
    )


def test_valid_instrument_spec():
    instrument = make_instrument()

    assert instrument.symbol == "XAUUSD"
    assert instrument.contract_size == 100.0
    assert instrument.tick_size == 0.01
    assert instrument.tick_value == 1.0
    assert instrument.base_currency == "XAU"
    assert instrument.quote_currency == "USD"


def test_instrument_is_immutable():
    instrument = make_instrument()

    with pytest.raises(Exception):
        instrument.symbol = "EURUSD"


def test_zero_contract_size_is_rejected():
    with pytest.raises(ValueError):
        InstrumentSpec(
            symbol="XAUUSD",
            contract_size=0.0,
            tick_size=0.01,
            tick_value=1.0,
            base_currency="XAU",
            quote_currency="USD",
        )


def test_negative_contract_size_is_rejected():
    with pytest.raises(ValueError):
        InstrumentSpec(
            symbol="XAUUSD",
            contract_size=-100.0,
            tick_size=0.01,
            tick_value=1.0,
            base_currency="XAU",
            quote_currency="USD",
        )


def test_zero_tick_size_is_rejected():
    with pytest.raises(ValueError):
        InstrumentSpec(
            symbol="XAUUSD",
            contract_size=100.0,
            tick_size=0.0,
            tick_value=1.0,
            base_currency="XAU",
            quote_currency="USD",
        )


def test_negative_tick_size_is_rejected():
    with pytest.raises(ValueError):
        InstrumentSpec(
            symbol="XAUUSD",
            contract_size=100.0,
            tick_size=-0.01,
            tick_value=1.0,
            base_currency="XAU",
            quote_currency="USD",
        )


def test_zero_tick_value_is_rejected():
    with pytest.raises(ValueError):
        InstrumentSpec(
            symbol="XAUUSD",
            contract_size=100.0,
            tick_size=0.01,
            tick_value=0.0,
            base_currency="XAU",
            quote_currency="USD",
        )


def test_negative_tick_value_is_rejected():
    with pytest.raises(ValueError):
        InstrumentSpec(
            symbol="XAUUSD",
            contract_size=100.0,
            tick_size=0.01,
            tick_value=-1.0,
            base_currency="XAU",
            quote_currency="USD",
        )