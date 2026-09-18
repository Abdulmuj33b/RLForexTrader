import numpy as np
import pytest

from arl_trp.environment.observation import ObservationBuffer


def make_observation(value: float) -> np.ndarray:
    return np.full(10, value, dtype=np.float32)


def test_buffer_can_be_created():
    buffer = ObservationBuffer(window_size=64)

    assert buffer.window_size == 64
    assert buffer.is_ready is False


def test_new_buffer_is_empty():
    buffer = ObservationBuffer(window_size=64)

    assert len(buffer) == 0


def test_buffer_accepts_ten_features():
    buffer = ObservationBuffer(window_size=64)

    buffer.append(make_observation(1.0))

    assert len(buffer) == 1


def test_buffer_rejects_wrong_feature_count():
    buffer = ObservationBuffer(window_size=64)

    with pytest.raises(ValueError):
        buffer.append(np.zeros(9, dtype=np.float32))


def test_buffer_is_not_ready_before_64_observations():
    buffer = ObservationBuffer(window_size=64)

    for i in range(63):
        buffer.append(make_observation(float(i)))

    assert buffer.is_ready is False


def test_buffer_becomes_ready_at_64_observations():
    buffer = ObservationBuffer(window_size=64)

    for i in range(64):
        buffer.append(make_observation(float(i)))

    assert buffer.is_ready is True


def test_ready_window_has_shape_64_by_10():
    buffer = ObservationBuffer(window_size=64)

    for i in range(64):
        buffer.append(make_observation(float(i)))

    window = buffer.get()

    assert window.shape == (64, 10)


def test_observations_remain_in_chronological_order():
    buffer = ObservationBuffer(window_size=64)

    for i in range(64):
        buffer.append(make_observation(float(i)))

    window = buffer.get()

    assert np.all(window[0] == 0.0)
    assert np.all(window[-1] == 63.0)


def test_65th_observation_removes_oldest_observation():
    buffer = ObservationBuffer(window_size=64)

    for i in range(65):
        buffer.append(make_observation(float(i)))

    window = buffer.get()

    assert window.shape == (64, 10)
    assert np.all(window[0] == 1.0)
    assert np.all(window[-1] == 64.0)


def test_buffer_never_exceeds_window_size():
    buffer = ObservationBuffer(window_size=64)

    for i in range(100):
        buffer.append(make_observation(float(i)))

    assert len(buffer) == 64


def test_reset_clears_buffer():
    buffer = ObservationBuffer(window_size=64)

    for i in range(64):
        buffer.append(make_observation(float(i)))

    assert buffer.is_ready is True

    buffer.reset()

    assert len(buffer) == 0
    assert buffer.is_ready is False


def test_buffer_can_be_reused_after_reset():
    buffer = ObservationBuffer(window_size=64)

    for i in range(64):
        buffer.append(make_observation(float(i)))

    buffer.reset()

    for i in range(64):
        buffer.append(make_observation(float(i + 100)))

    window = buffer.get()

    assert buffer.is_ready is True
    assert np.all(window[0] == 100.0)
    assert np.all(window[-1] == 163.0)


def test_get_returns_float32():
    buffer = ObservationBuffer(window_size=64)

    for i in range(64):
        buffer.append(make_observation(float(i)))

    window = buffer.get()

    assert window.dtype == np.float32


def test_get_returns_copy_not_internal_buffer():
    buffer = ObservationBuffer(window_size=64)

    for i in range(64):
        buffer.append(make_observation(float(i)))

    window = buffer.get()
    window[0, 0] = 999.0

    fresh_window = buffer.get()

    assert fresh_window[0, 0] == 0.0


def test_invalid_window_size_is_rejected():
    with pytest.raises(ValueError):
        ObservationBuffer(window_size=0)


def test_get_before_ready_is_rejected():
    buffer = ObservationBuffer(window_size=64)

    buffer.append(make_observation(1.0))

    with pytest.raises(RuntimeError):
        buffer.get()
