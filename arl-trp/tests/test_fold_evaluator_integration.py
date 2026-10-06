import numpy as np

from arl_trp.data.fold import WalkForwardDatasetBuilder
from arl_trp.data.splitter import (
    PurgedWalkForwardSplitter,
    WalkForwardConfig,
)
from arl_trp.domain.market import MarketSnapshot
from arl_trp.environment.observation import ObservationNormalizer
from arl_trp.evaluation.fold import (
    FoldEvaluationResult,
    FoldEvaluator,
)
from arl_trp.execution.simulator import (
    ExecutionConfig,
    ExecutionSimulator,
)
from arl_trp.rl.config import PPOConfig
from arl_trp.rl.ppo import build_ppo_model
from arl_trp.environment.trading_env import TradingEnvironment

from tests.test_fold import make_dataset


def make_raw_observations(
    snapshots: tuple[MarketSnapshot, ...],
) -> np.ndarray:
    """
    Build the raw observation matrix required to fit the
    training-fold observation normalizer.

    Only the volume feature is relevant to the current
    ObservationNormalizer protocol.
    """
    observations = np.zeros(
        (len(snapshots), 10),
        dtype=np.float32,
    )

    observations[:, 5] = np.asarray(
        [snapshot.volume for snapshot in snapshots],
        dtype=np.float32,
    )

    return observations


def make_fold_dataset():
    dataset = make_dataset()

    folds = PurgedWalkForwardSplitter(
        WalkForwardConfig()
    ).split(dataset)

    fold = folds[0]

    fold_dataset = WalkForwardDatasetBuilder.build(
        dataset,
        fold,
    )

    return fold_dataset


def make_training_normalizer(
    train: tuple[MarketSnapshot, ...],
) -> ObservationNormalizer:
    normalizer = ObservationNormalizer()

    train_observations = make_raw_observations(
        train
    )

    normalizer.fit(train_observations)

    return normalizer


def make_environment(
    market_data: tuple[MarketSnapshot, ...],
    initial_balance: float,
    normalizer: ObservationNormalizer,
):
    environment = TradingEnvironment(
        market_data=market_data,
        initial_balance=initial_balance,
        execution=ExecutionSimulator(
            ExecutionConfig()
        ),
        quantity=1.0,
        reward_beta=0.5,
        observation_window=64,
    )

    environment.set_observation_normalizer(
        normalizer
    )

    return environment


def make_model(environment):
    config = PPOConfig(
        n_steps=64,
        batch_size=32,
    )

    return build_ppo_model(
        environment,
        config,
    )


def test_real_ppo_evaluates_on_walk_forward_test_fold():
    fold_dataset = make_fold_dataset()

    normalizer = make_training_normalizer(
        fold_dataset.train
    )

    training_environment = make_environment(
        market_data=fold_dataset.train,
        initial_balance=10_000.0,
        normalizer=normalizer,
    )

    test_environment = make_environment(
        market_data=fold_dataset.test,
        initial_balance=10_000.0,
        normalizer=normalizer,
    )

    model = make_model(
        training_environment
    )

    model.learn(
        total_timesteps=128,
        progress_bar=False,
    )

    result = FoldEvaluator().evaluate(
        model=model,
        environment=test_environment,
        fold_id=fold_dataset.fold.fold_id,
    )

    assert isinstance(
        result,
        FoldEvaluationResult,
    )

    assert result.fold_id == (
        fold_dataset.fold.fold_id
    )

    assert result.report.result.steps > 0

    assert (
        result.report.result.steps
        == result.report.result.actions.size
    )

    assert (
        result.report.result.equity_curve.size
        == result.report.result.steps + 1
    )

    assert np.all(
        np.isfinite(
            result.report.result.equity_curve
        )
    )

    assert np.all(
        result.report.result.equity_curve >= 0.0
    )

    assert np.all(
        np.isfinite(
            result.report.result.trade_pnls
        )
    )


def test_walk_forward_test_fold_is_chronologically_after_training():
    fold_dataset = make_fold_dataset()

    training_end = fold_dataset.train[-1].timestamp
    test_start = fold_dataset.test[0].timestamp

    assert test_start > training_end

    assert (
        fold_dataset.train[-1].timestamp
        < fold_dataset.test[0].timestamp
    )


def test_training_and_test_environments_use_same_fitted_normalizer():
    fold_dataset = make_fold_dataset()

    normalizer = make_training_normalizer(
        fold_dataset.train
    )

    training_environment = make_environment(
        market_data=fold_dataset.train,
        initial_balance=10_000.0,
        normalizer=normalizer,
    )

    test_environment = make_environment(
        market_data=fold_dataset.test,
        initial_balance=10_000.0,
        normalizer=normalizer,
    )

    assert (
        training_environment.observation_normalizer
        is test_environment.observation_normalizer
    )

    assert (
        training_environment.observation_normalizer
        is normalizer
    )

    assert (
        test_environment.observation_normalizer
        is normalizer
    )


def test_ppo_is_trained_only_on_training_environment():
    fold_dataset = make_fold_dataset()

    normalizer = make_training_normalizer(
        fold_dataset.train
    )

    training_environment = make_environment(
        market_data=fold_dataset.train,
        initial_balance=10_000.0,
        normalizer=normalizer,
    )

    test_environment = make_environment(
        market_data=fold_dataset.test,
        initial_balance=10_000.0,
        normalizer=normalizer,
    )

    model = make_model(
        training_environment
    )

    model.learn(
        total_timesteps=128,
        progress_bar=False,
    )

    assert model.get_env() is not None

    trained_environment = model.get_env()

    wrapped_environment = trained_environment.envs[0]

    underlying_environment = wrapped_environment.unwrapped

    assert (
        underlying_environment.market_data
        == list(fold_dataset.train)
    )

    assert (
        underlying_environment.market_data
        != list(fold_dataset.test)
    )

    # The test environment has not been used for training.
    assert test_environment.index == 0