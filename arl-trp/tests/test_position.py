import pytest

from arl_trp.domain.position import Position


def test_new_position_is_flat():
    position = Position()

    assert position.is_flat
    assert not position.is_long
    assert not position.is_short


def test_open_long_position():
    position = Position()

    position.open(
        quantity=1.0,
        price=100.0,
    )

    assert position.quantity == 1.0
    assert position.entry_price == 100.0
    assert position.is_long
    assert not position.is_short
    assert not position.is_flat


def test_open_short_position():
    position = Position()

    position.open(
        quantity=-1.0,
        price=100.0,
    )

    assert position.quantity == -1.0
    assert position.entry_price == 100.0
    assert position.is_short
    assert not position.is_long
    assert not position.is_flat


def test_zero_quantity_cannot_open():
    position = Position()

    with pytest.raises(ValueError):
        position.open(
            quantity=0.0,
            price=100.0,
        )


def test_negative_price_cannot_open():
    position = Position()

    with pytest.raises(ValueError):
        position.open(
            quantity=1.0,
            price=-100.0,
        )


def test_cannot_open_second_position():
    position = Position()

    position.open(
        quantity=1.0,
        price=100.0,
    )

    with pytest.raises(ValueError):
        position.open(
            quantity=-1.0,
            price=105.0,
        )


def test_close_position():
    position = Position()

    position.open(
        quantity=1.0,
        price=100.0,
    )

    position.close()

    assert position.is_flat
    assert position.quantity == 0.0
    assert position.entry_price == 0.0