import pytest
from simulation_based_inference.simulation import simulate_rescorla_wagner_dual, generate_block_bandit_outcomes
from simulation_based_inference.numpyro_models import (
    rescorla_wagner_trial_iterator,
    rescorla_wagner_model_vmap_blocks,
    rescorla_wagner_model_vmap_observations
)
import numpy as np


def test_rescorla_wagner_iterator_output_shape():

    alpha_p = np.ones(1) * 0.5
    alpha_n = np.ones(1) * 0.5
    temperature = np.ones(1)

    outcomes, _ = generate_block_bandit_outcomes(
        100, 3, 4
    )

    _, choices, _ = simulate_rescorla_wagner_dual(alpha_p, alpha_n, temperature, outcomes)

    value = rescorla_wagner_trial_iterator(outcomes[0, ...], choices[0, 0, ...])

    assert value.shape == outcomes[0, ...].shape

def test_rescorla_wagner_blocks_iterator_output_shape():

    alpha_p = np.ones(1) * 0.5
    alpha_n = np.ones(1) * 0.5
    temperature = np.ones(1)

    outcomes, _ = generate_block_bandit_outcomes(
        100, 3, 4
    )

    _, choices, _ = simulate_rescorla_wagner_dual(alpha_p, alpha_n, temperature, outcomes)

    value = rescorla_wagner_model_vmap_blocks(outcomes, choices[0, ...], 0.5, alpha_p, alpha_n)

    assert value.shape == outcomes.shape


def test_rescorla_wagner_iterator_output_values():

    rng = np.random.RandomState(42)

    alpha_p = rng.uniform(0, 1, 100)
    alpha_n = rng.uniform(0, 1, 100)
    temperature = np.ones(100)

    outcomes, _ = generate_block_bandit_outcomes(
        100, 3, 4
    )

    _, choices, value_estimates = simulate_rescorla_wagner_dual(alpha_p, alpha_n, temperature, outcomes)
    value = rescorla_wagner_model_vmap_observations(outcomes, choices, 0.5, alpha_p, alpha_n)

    assert np.all(np.isclose(value, value_estimates))

