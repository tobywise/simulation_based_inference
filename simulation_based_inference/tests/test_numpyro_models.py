import pytest
from simulation_based_inference.simulation import simulate_RL
from simulation_based_inference.numpyro_models import (
    rescorla_wagner_trial_iterator,
    rescorla_wagner_model_vmap,
    softmax
)
import numpy as np


def test_rescorla_wagner_iterator_output_shape():

    alpha_p = np.ones(1) * 0.5
    alpha_n = np.ones(1) * 0.5
    temperature = np.ones(1)

    outcomes, _, choices, value_estimates = simulate_RL(alpha_p, alpha_n, temperature)

    value = rescorla_wagner_trial_iterator(outcomes, choices[0, ...])

    assert value.shape == outcomes.shape


def test_rescorla_wagner_iterator_output_values():

    rng = np.random.RandomState(42)

    alpha_p = rng.uniform(0, 1, 100)
    alpha_n = rng.uniform(0, 1, 100)
    temperature = np.ones(100)

    outcomes, _, choices, value_estimates = simulate_RL(alpha_p, alpha_n, temperature)
    value = rescorla_wagner_model_vmap(outcomes, choices, 0.5, alpha_p, alpha_n)

    assert np.all(np.isclose(value, value_estimates))


@pytest.fixture
def q_values_fixture():
    q_vals = np.zeros((7, 4))  # 5 states, 4 actions
    q_vals[0, :] = np.array([4.5, 3.5, 2.5, 1.5])
    q_vals[1, :] = np.array([0, 0, 0, 0])
    q_vals[2, :] = np.array([1, 1, 0, 0])
    q_vals[3, :] = np.array([200, 50, 2, 1])
    q_vals[4, :] = np.array([-20, 0, 20, 1])
    q_vals[5, :] = np.array([10003434, 0, 100000, -100])
    q_vals[6, :] = np.array([-100000, 0, -5000, -500000000000])
    return q_vals

@pytest.fixture()
def q_values_increasing():
    q_values = np.zeros((2, 5))
    q_values[0, :] = np.arange(0, 5)
    q_values[1, :] = np.arange(4, -1, -1)

    return q_values

@pytest.fixture()
def q_values_equal():
    q_values = np.ones((2, 5))
    return q_values


def test_softmax_action_selector_action_p(q_values_fixture):

    action_p = softmax(q_values_fixture, temperature=1)

    assert np.all(np.isclose(action_p.sum(axis=1), 1))
    assert np.all(np.diff(action_p[0, :]) < 0)
    assert np.all(action_p[1, :] == action_p[1, 0])
    assert np.all(action_p >= 0)
    assert np.all(action_p <= 1)

def test_softmax_action_selector_temperature_action_p(q_values_increasing):

    temp_action_p = np.zeros((3, 2, 5))

    for n, temp in enumerate([0.5, 1, 5]):
        action_p = softmax(q_values_increasing, temperature=temp)

        assert np.all(np.diff(action_p[0, :]) > 0)
        assert np.all(np.diff(action_p[1, :]) < 0)
        assert np.all(np.isclose(action_p[0, ::-1], action_p[1]))

        temp_action_p[n, ...] = action_p

    assert np.all(np.diff(temp_action_p[:, 0, 1]) > 0)
    assert np.all(np.diff(temp_action_p[:, 1, 1]) > 0)

