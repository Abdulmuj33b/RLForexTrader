import numpy as np

from arl_trp.data.fold import WalkForwardDatasetBuilder
from arl_trp.data.splitter import (
    PurgedWalkForwardSplitter,
    WalkForwardConfig,
)
from arl_trp.environment.observation import ObservationNormalizer
from arl_trp.environment.trading_env import TradingEnvironment
from arl_trp.evaluation.walk_forward import (
    WalkForwardEvaluationResult,
    WalkForwardEvaluator,
)
from arl_trp.execution.simulator import (
    ExecutionConfig,
    ExecutionSimulator,
)
from arl_trp.rl.config import PPOConfig
from arl_trp.rl.ppo import build_ppo_model

from tests.test_fold import make_dataset


def make_raw_observations(snapshots):
    observations = np.zeros(
        (len(snapshots), 10),
        dtype=np.float32,
    )

    observations[:, 5] = np.asarray(
        [snapshot.volume for snapshot in snapshots],
        dtype=np.float32,
    )

    return observations


def make_normalizer(train):
    normalizer = ObservationNormalizer()

    normalizer.fit(
        make_raw_observations(train)
    )

    return normalizer


def make_environment(
    market_data,
    normalizer,
):
    environment = TradingEnvironment(
        market_data=market_data,
        initial_balance=10_000.0,
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


def make_fold_runs():
    dataset = make_dataset()

    folds = PurgedWalkForwardSplitter(
        WalkForwardConfig()
    ).split(dataset)

    fold_runs = []

    for fold in folds:
        fold_dataset = (
            WalkForwardDatasetBuilder.build(
                dataset,
                fold,
            )
        )

        normalizer = make_normalizer(
            fold_dataset.train
        )

        training_environment = make_environment(
            fold_dataset.train,
            normalizer,
        )

        test_environment = make_environment(
            fold_dataset.test,
            normalizer,
        )

        model = make_model(
            training_environment
        )

        model.learn(
            total_timesteps=128,
            progress_bar=False,
        )

        fold_runs.append(
            (
                model,
                test_environment,
                fold.fold_id,
            )
        )

    return tuple(fold_runs)


def test_real_five_fold_walk_forward_evaluation():
    fold_runs = make_fold_runs()

    result = WalkForwardEvaluator().evaluate(
        fold_runs
    )

    assert isinstance(
        result,
        WalkForwardEvaluationResult,
    )

    assert result.fold_count == 5

    assert tuple(
        fold.fold_id
        for fold in result.folds
    ) == (1, 2, 3, 4, 5)

    for fold_result in result.folds:
        evaluation = fold_result.report.result

        assert evaluation.steps > 0

        assert (
            evaluation.equity_curve.size
            == evaluation.steps + 1
        )

        assert np.all(
            np.isfinite(
                evaluation.equity_curve
            )
        )

        assert np.all(
            evaluation.equity_curve >= 0.0
        )

        assert np.all(
            np.isfinite(
                evaluation.trade_pnls
            )
        )


def test_each_walk_forward_fold_has_independent_test_evaluation():
    fold_runs = make_fold_runs()

    result = WalkForwardEvaluator().evaluate(
        fold_runs
    )

    assert len(result.folds) == 5

    equity_curves = [
        fold.report.result.equity_curve
        for fold in result.folds
    ]

    for equity_curve in equity_curves:
        assert equity_curve.size >= 2

    fold_ids = [
        fold.fold_id
        for fold in result.folds
    ]

    assert len(set(fold_ids)) == 5


def test_walk_forward_evaluation_preserves_fold_order():
    fold_runs = make_fold_runs()

    result = WalkForwardEvaluator().evaluate(
        tuple(reversed(fold_runs))
    )

    assert tuple(
        fold.fold_id
        for fold in result.folds
    ) == (1, 2, 3, 4, 5)
