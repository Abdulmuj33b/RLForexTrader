from dataclasses import dataclass


@dataclass
class Position:
    quantity: float = 0.0
    entry_price: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0.0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0.0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0.0

    def open(self, quantity: float, price: float) -> None:
        if not self.is_flat:
            raise ValueError("position already exists")

        if quantity == 0.0:
            raise ValueError("quantity cannot be zero")

        if price <= 0.0:
            raise ValueError("entry price must be positive")

        self.quantity = quantity
        self.entry_price = price

    def close(self) -> None:
        self.quantity = 0.0
        self.entry_price = 0.0