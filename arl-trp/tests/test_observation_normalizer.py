import numpy as np
import pytest

from arl_trp.environment.observation import ObservationNormalizer


def test_normalizer_can_be_created():
    normalizer = ObservationNormalizer()

    assert normalizer is not None


def test_transform_before_fit_is_rejected():
    normalizer = ObservationNormalizer()

    observations = np.array(
        [
            [0.01, 0.02, -0.01, 0.005, 0.001, 100.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(RuntimeError, match="must be fitted"):
        normalizer.transform(observations)


def test_fit_stores_training_statistics():
    normalizer = ObservationNormalizer()

    observations = np.array(
        [
            [0.01, 0.02, -0.01, 0.005, 0.001, 100.0, 0.0, 0.0, 1.0, 0.0],
            [0.02, 0.03, -0.02, 0.010, 0.002, 200.0, 1.0, 0.01, 1.01, 0.0],
            [0.03, 0.04, -0.03, 0.015, 0.003, 300.0, -1.0, -0.01, 0.99, 0.02],
        ],
        dtype=np.float32,
    )

    normalizer.fit(observations)

    assert normalizer.volume_median == pytest.approx(200.0)
    assert normalizer.volume_iqr == pytest.approx(100.0)


def test_volume_is_normalized_using_training_median_and_iqr():
    normalizer = ObservationNormalizer()

    training = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    normalizer.fit(training)

    new_observation = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    transformed = normalizer.transform(new_observation)

    assert transformed[0, 5] == pytest.approx(1.0)


def test_other_features_are_not_changed():
    normalizer = ObservationNormalizer()

    training = np.array(
        [
            [0.01, 0.02, -0.01, 0.005, 0.001, 100.0, 0.0, 0.0, 1.0, 0.0],
            [0.02, 0.03, -0.02, 0.010, 0.002, 200.0, 1.0, 0.01, 1.01, 0.0],
            [0.03, 0.04, -0.03, 0.015, 0.003, 300.0, -1.0, -0.01, 0.99, 0.02],
        ],
        dtype=np.float32,
    )

    normalizer.fit(training)

    transformed = normalizer.transform(training)

    np.testing.assert_allclose(
        transformed[:, 0:5],
        training[:, 0:5],
    )
    np.testing.assert_allclose(
        transformed[:, 6:10],
        training[:, 6:10],
    )


def test_transform_does_not_change_training_statistics():
    normalizer = ObservationNormalizer()

    training = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    normalizer.fit(training)

    median_before = normalizer.volume_median
    iqr_before = normalizer.volume_iqr

    normalizer.transform(
        np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 999999.0, 0.0, 0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
    )

    assert normalizer.volume_median == pytest.approx(median_before)
    assert normalizer.volume_iqr == pytest.approx(iqr_before)


def test_fit_rejects_wrong_feature_count():
    normalizer = ObservationNormalizer()

    invalid = np.zeros((10, 9), dtype=np.float32)

    with pytest.raises(ValueError, match="10 features"):
        normalizer.fit(invalid)


def test_transform_rejects_wrong_feature_count():
    normalizer = ObservationNormalizer()

    training = np.zeros((10, 10), dtype=np.float32)
    normalizer.fit(training)

    invalid = np.zeros((1, 9), dtype=np.float32)

    with pytest.raises(ValueError, match="10 features"):
        normalizer.transform(invalid)


def test_fit_rejects_empty_training_data():
    normalizer = ObservationNormalizer()

    empty = np.empty((0, 10), dtype=np.float32)

    with pytest.raises(ValueError, match="cannot be empty"):
        normalizer.fit(empty)


def test_fit_transform_matches_fit_then_transform():
    training = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    normalizer_a = ObservationNormalizer()
    expected = normalizer_a.fit(training).transform(training)

    normalizer_b = ObservationNormalizer()
    actual = normalizer_b.fit_transform(training)

    np.testing.assert_allclose(actual, expected)


def test_transform_returns_float32():
    normalizer = ObservationNormalizer()

    training = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    normalizer.fit(training)

    transformed = normalizer.transform(training)

    assert transformed.dtype == np.float32