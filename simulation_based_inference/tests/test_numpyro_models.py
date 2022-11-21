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

