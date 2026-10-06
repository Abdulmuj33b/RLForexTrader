from dataclasses import dataclass

from arl_trp.data.dataset import ValidatedDataset
from arl_trp.data.splitter import WalkForwardFold
from arl_trp.domain.market import MarketSnapshot


@dataclass(frozen=True)
class FoldDataset:
    """
    Immutable data view for one walk-forward fold.

    The fold contains only the observations explicitly assigned
    to training and testing. Purge and embargo observations are
    intentionally excluded.
    """

    fold: WalkForwardFold
    train: tuple[MarketSnapshot, ...]
    test: tuple[MarketSnapshot, ...]

    def __post_init__(self) -> None:
        if not self.train:
            raise ValueError(
                "train dataset cannot be empty"
            )

        if not self.test:
            raise ValueError(
                "test dataset cannot be empty"
            )

    @property
    def train_size(self) -> int:
        return len(self.train)

    @property
    def test_size(self) -> int:
        return len(self.test)


class WalkForwardDatasetBuilder:
    """
    Builds isolated train/test dataset views from a validated dataset.

    The builder does not modify, sort, interpolate, or normalize
    observations.
    """

    @staticmethod
    def build(
        dataset: ValidatedDataset,
        fold: WalkForwardFold,
    ) -> FoldDataset:
        if fold.train_start < 0:
            raise ValueError(
                "train_start cannot be negative"
            )

        if fold.test_start < 0:
            raise ValueError(
                "test_start cannot be negative"
            )

        if fold.train_end > dataset.size:
            raise ValueError(
                "train_end exceeds dataset size"
            )

        if fold.test_end > dataset.size:
            raise ValueError(
                "test_end exceeds dataset size"
            )

        if fold.train_start >= fold.train_end:
            raise ValueError(
                "training interval must be non-empty"
            )

        if fold.test_start >= fold.test_end:
            raise ValueError(
                "test interval must be non-empty"
            )

        train = tuple(
            dataset.snapshots[
                fold.train_start:fold.train_end
            ]
        )

        test = tuple(
            dataset.snapshots[
                fold.test_start:fold.test_end
            ]
        )

        return FoldDataset(
            fold=fold,
            train=train,
            test=test,
        )