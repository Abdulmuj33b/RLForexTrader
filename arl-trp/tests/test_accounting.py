import pytest
from datetime import datetime

from arl_trp.domain.market import MarketSnapshot
from arl_trp.domain.position import Position
from arl_trp.portfolio.accounting import PortfolioAccounting


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


def test_initial_account_state():
    account = PortfolioAccounting(
        initial_balance=10_000.0
    )

    assert account.balance == 10_000.0
    assert account.peak_equity == 10_000.0


def test_flat_position_has_zero_unrealized_pnl():
    account = PortfolioAccounting(
        initial_balance=10_000.0
    )

    position = Position()
    market = make_market()

    assert account.unrealized_pnl(
        position,
        market,
    ) == 0.0


def test_long_position_is_marked_at_bid():
    account = PortfolioAccounting(
        initial_balance=10_000.0
    )

    position = Position()
    position.open(
        quantity=1.0,
        price=101.0,
    )

    market = make_market(
        bid=105.0,
        ask=106.0,
    )

    pnl = account.unrealized_pnl(
        position,
        market,
    )

    assert pnl == 4.0


def test_short_position_is_marked_at_ask():
    account = PortfolioAccounting(
        initial_balance=10_000.0
    )

    position = Position()
    position.open(
        quantity=-1.0,
        price=101.0,
    )

    market = make_market(
        bid=95.0,
        ask=96.0,
    )

    pnl = account.unrealized_pnl(
        position,
        market,
    )

    assert pnl == 5.0


def test_long_position_with_spread_has_immediate_loss():
    account = PortfolioAccounting(
        initial_balance=10_000.0
    )

    position = Position()
    position.open(
        quantity=1.0,
        price=101.0,
    )

    market = make_market(
        bid=100.0,
        ask=101.0,
    )

    pnl = account.unrealized_pnl(
        position,
        market,
    )

    assert pnl == -1.0


def test_short_position_with_spread_has_immediate_loss():
    account = PortfolioAccounting(
        initial_balance=10_000.0
    )

    position = Position()
    position.open(
        quantity=-1.0,
        price=100.0,
    )

    market = make_market(
        bid=100.0,
        ask=101.0,
    )

    pnl = account.unrealized_pnl(
        position,
        market,
    )

    assert pnl == -1.0


def test_equity_equals_balance_plus_unrealized_pnl():
    account = PortfolioAccounting(
        initial_balance=10_000.0
    )

    position = Position()
    position.open(
        quantity=2.0,
        price=100.0,
    )

    market = make_market(
        bid=105.0,
        ask=106.0,
    )

    equity = account.equity(
        position,
        market,
    )

    assert equity == 10_010.0


def test_realized_pnl_changes_balance():
    account = PortfolioAccounting(
        initial_balance=10_000.0
    )

    account.apply_realized_pnl(250.0)

    assert account.balance == 10_250.0


def test_realized_loss_reduces_balance():
    account = PortfolioAccounting(
        initial_balance=10_000.0
    )

    account.apply_realized_pnl(-300.0)

    assert account.balance == 9_700.0


def test_peak_equity_updates_when_equity_increases():
    account = PortfolioAccounting(
        initial_balance=10_000.0
    )

    account.apply_realized_pnl(500.0)

    assert account.balance == 10_500.0
    assert account.peak_equity == 10_500.0


def test_drawdown_is_calculated_from_peak_equity():
    account = PortfolioAccounting(
        initial_balance=10_000.0
    )

    account.apply_realized_pnl(1_000.0)

    position = Position()

    market = make_market()

    account.apply_realized_pnl(-500.0)

    drawdown = account.drawdown(
        position,
        market,
    )

    assert drawdown == pytest.approx(
        500.0 / 11_000.0
    )


def test_snapshot_contains_accounting_state():
    account = PortfolioAccounting(
        initial_balance=10_000.0
    )

    position = Position()
    market = make_market()

    snapshot = account.snapshot(
        position,
        market,
    )

    assert snapshot["balance"] == 10_000.0
    assert snapshot["unrealized_pnl"] == 0.0
    assert snapshot["equity"] == 10_000.0
    assert snapshot["peak_equity"] == 10_000.0
    assert snapshot["drawdown"] == 0.0
    assert snapshot["position_quantity"] == 0.0
    assert snapshot["position_entry_price"] == 0.0