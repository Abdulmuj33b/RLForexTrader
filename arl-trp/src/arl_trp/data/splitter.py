from dataclasses import dataclass

from arl_trp.data.dataset import ValidatedDataset


@dataclass(frozen=True)
class WalkForwardConfig:
    """
    Configuration for purged walk-forward evaluation.

    v0.1 protocol:

        feature_lookback_bars = 500
        label_horizon_bars = 100
        initial_train_bars = 601
        folds = 5

    The splitter operates on observation indices.

    Each fold has the structure:

        TRAIN → PURGE → EMBARGO → TEST
    """

    feature_lookback_bars: int = 500
    label_horizon_bars: int = 100
    initial_train_bars: int = 601
    folds: int = 5

    def __post_init__(self) -> None:
        if self.feature_lookback_bars <= 0:
            raise ValueError(
                "feature_lookback_bars must be positive"
            )

        if self.label_horizon_bars <= 0:
            raise ValueError(
                "label_horizon_bars must be positive"
            )

        if self.initial_train_bars <= 0:
            raise ValueError(
                "initial_train_bars must be positive"
            )

        if self.folds <= 0:
            raise ValueError(
                "folds must be positive"
            )

        minimum_initial_train = (
            self.feature_lookback_bars
            + self.label_horizon_bars
            + 1
        )

        if self.initial_train_bars < minimum_initial_train:
            raise ValueError(
                "initial_train_bars must be at least "
                "feature_lookback_bars + label_horizon_bars + 1"
            )


@dataclass(frozen=True)
class WalkForwardFold:
    """
    One chronological walk-forward fold.

    All index ranges are half-open:

        train_start <= index < train_end
        purge_start <= index < purge_end
        embargo_start <= index < embargo_end
        test_start <= index < test_end
    """

    fold_id: int

    train_start: int
    train_end: int

    purge_start: int
    purge_end: int

    embargo_start: int
    embargo_end: int

    test_start: int
    test_end: int

    @property
    def train_indices(self) -> range:
        return range(
            self.train_start,
            self.train_end,
        )

    @property
    def test_indices(self) -> range:
        return range(
            self.test_start,
            self.test_end,
        )

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start

    @property
    def purge_size(self) -> int:
        return self.purge_end - self.purge_start

    @property
    def embargo_size(self) -> int:
        return self.embargo_end - self.embargo_start


class PurgedWalkForwardSplitter:
    """
    Generates deterministic expanding-window walk-forward folds.

    Dataset layout:

        INITIAL TRAIN
             │
             ├── PURGE
             ├── EMBARGO
             └── TEST 1
                    │
                    ├── PURGE
                    ├── EMBARGO
                    └── TEST 2
                           │
                           └── ...

    Training expands after each completed test period.

    The total dataset allocation is:

        N = initial_train
            + folds × (
                purge
                + embargo
                + test
            )
    """

    def __init__(
        self,
        config: WalkForwardConfig,
    ) -> None:
        self.config = config

    def split(
        self,
        dataset: ValidatedDataset,
    ) -> tuple[WalkForwardFold, ...]:
        observation_count = dataset.size

        if observation_count <= 0:
            raise ValueError(
                "dataset must contain observations"
            )

        gap_size = (
            self.config.label_horizon_bars
            + self.config.feature_lookback_bars
        )

        required_without_test = (
            self.config.initial_train_bars
            + (
                self.config.folds
                * gap_size
            )
        )

        if observation_count <= required_without_test:
            raise ValueError(
                "dataset does not contain enough observations "
                "for the requested walk-forward configuration"
            )

        remaining_observations = (
            observation_count
            - required_without_test
        )

        test_size = (
            remaining_observations
            // self.config.folds
        )

        if test_size <= 0:
            raise ValueError(
                "test fold size must be positive"
            )

        folds: list[WalkForwardFold] = []

        train_end = self.config.initial_train_bars

        for fold_number in range(
            1,
            self.config.folds + 1,
        ):
            purge_start = train_end

            purge_end = (
                purge_start
                + self.config.label_horizon_bars
            )

            embargo_start = purge_end

            embargo_end = (
                embargo_start
                + self.config.feature_lookback_bars
            )

            test_start = embargo_end

            test_end = (
                test_start
                + test_size
            )

            if test_end > observation_count:
                raise ValueError(
                    "walk-forward fold exceeds dataset boundary"
                )

            folds.append(
                WalkForwardFold(
                    fold_id=fold_number,
                    train_start=0,
                    train_end=train_end,
                    purge_start=purge_start,
                    purge_end=purge_end,
                    embargo_start=embargo_start,
                    embargo_end=embargo_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

            train_end = test_end

        return tuple(folds)