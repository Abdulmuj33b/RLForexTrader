from dataclasses import dataclass
from datetime import timedelta

from arl_trp.domain.market import MarketSnapshot


@dataclass(frozen=True)
class DatasetValidationResult:
    """
    Result of validating a sequence of market snapshots.
    """

    valid: bool
    errors: tuple[str, ...]
    gaps: tuple[int, ...] = ()


def validate_market_data(
    market_data: list[MarketSnapshot],
    expected_interval: timedelta | None = None,
) -> DatasetValidationResult:
    """
    Validate chronological market data before it enters
    the RL environment.

    If expected_interval is supplied, unexpected timestamp
    gaps are reported separately from validation errors.
    """

    errors: list[str] = []
    gaps: list[int] = []

    if not market_data:
        errors.append("market_data cannot be empty")
        return DatasetValidationResult(
            valid=False,
            errors=tuple(errors),
            gaps=(),
        )

    for index, snapshot in enumerate(market_data):
        if snapshot.timestamp.tzinfo is None:
            errors.append(
                f"timestamp must be timezone-aware at index {index}"
            )

        if index > 0:
            previous = market_data[index - 1]

            if snapshot.timestamp.tzinfo != previous.timestamp.tzinfo:
                errors.append(
                    f"timestamps must use a consistent timezone "
                    f"at index {index}"
                )

            if snapshot.timestamp <= previous.timestamp:
                errors.append(
                    f"timestamps must be strictly increasing "
                    f"at index {index}"
                )

            if (
                expected_interval is not None
                and snapshot.timestamp - previous.timestamp
                != expected_interval
            ):
                gaps.append(index)

    return DatasetValidationResult(
        valid=not errors,
        errors=tuple(errors),
        gaps=tuple(gaps),
    )