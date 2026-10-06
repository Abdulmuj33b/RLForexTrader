import csv
from datetime import datetime
from pathlib import Path

from arl_trp.data.dataset import DatasetSpec, ValidatedDataset
from arl_trp.data.validation import validate_market_data
from arl_trp.domain.market import MarketSnapshot


REQUIRED_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bid",
    "ask",
)


class DatasetLoadError(ValueError):
    """Raised when a dataset cannot be loaded safely."""


class MarketDatasetLoader:
    """
    Loads historical market data from CSV into MarketSnapshot objects.

    The loader is intentionally deterministic and does not modify,
    sort, repair, or interpolate the source data.
    """

    def load(
        self,
        path: Path,
        dataset: DatasetSpec,
    ) -> ValidatedDataset:
        path = Path(path)

        if not path.exists():
            raise DatasetLoadError(
                f"dataset file does not exist: {path}"
            )

        if not path.is_file():
            raise DatasetLoadError(
                f"dataset path is not a file: {path}"
            )

        snapshots: list[MarketSnapshot] = []

        try:
            with path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                reader = csv.DictReader(file)

                if reader.fieldnames is None:
                    raise DatasetLoadError(
                        "dataset is missing a header row"
                    )

                missing = [
                    column
                    for column in REQUIRED_COLUMNS
                    if column not in reader.fieldnames
                ]

                if missing:
                    raise DatasetLoadError(
                        f"dataset is missing required columns: {missing}"
                    )

                for row_number, row in enumerate(
                    reader,
                    start=2,
                ):
                    try:
                        snapshots.append(
                            self._parse_row(row)
                        )
                    except (TypeError, ValueError) as exc:
                        raise DatasetLoadError(
                            f"invalid dataset row {row_number}: {exc}"
                        ) from exc

        except OSError as exc:
            raise DatasetLoadError(
                f"could not read dataset: {path}"
            ) from exc

        result = validate_market_data(snapshots)

        if not result.valid:
            raise DatasetLoadError(
                "dataset validation failed: "
                + "; ".join(result.errors)
            )

        self._validate_dataset_boundaries(
            snapshots,
            dataset,
        )

        return ValidatedDataset(
            spec=dataset,
            snapshots=tuple(snapshots),
        )

    @staticmethod
    def _parse_row(
        row: dict[str, str | None],
    ) -> MarketSnapshot:
        timestamp_value = row["timestamp"]

        if timestamp_value is None or not timestamp_value.strip():
            raise ValueError("timestamp cannot be empty")

        timestamp = datetime.fromisoformat(
            timestamp_value
        )

        values: dict[str, float] = {}

        for column in REQUIRED_COLUMNS[1:]:
            value = row[column]

            if value is None or not value.strip():
                raise ValueError(
                    f"{column} cannot be empty"
                )

            values[column] = float(value)

        return MarketSnapshot(
            timestamp=timestamp,
            open=values["open"],
            high=values["high"],
            low=values["low"],
            close=values["close"],
            volume=values["volume"],
            bid=values["bid"],
            ask=values["ask"],
        )

    @staticmethod
    def _validate_dataset_boundaries(
        snapshots: list[MarketSnapshot],
        dataset: DatasetSpec,
    ) -> None:
        first_timestamp = snapshots[0].timestamp
        last_timestamp = snapshots[-1].timestamp

        if first_timestamp < dataset.start:
            raise DatasetLoadError(
                "dataset contains data before declared start: "
                f"{first_timestamp.isoformat()} < "
                f"{dataset.start.isoformat()}"
            )

        if last_timestamp > dataset.end:
            raise DatasetLoadError(
                "dataset contains data after declared end: "
                f"{last_timestamp.isoformat()} > "
                f"{dataset.end.isoformat()}"
            )