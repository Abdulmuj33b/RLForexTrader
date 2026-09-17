from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    contract_size: float
    tick_size: float
    tick_value: float
    base_currency: str
    quote_currency: str

    def __post_init__(self):
        if self.contract_size <= 0:
            raise ValueError(
                "contract_size must be positive"
            )

        if self.tick_size <= 0:
            raise ValueError(
                "tick_size must be positive"
            )

        if self.tick_value <= 0:
            raise ValueError(
                "tick_value must be positive"
            )