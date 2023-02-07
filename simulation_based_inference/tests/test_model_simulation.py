import pytest
import numpy as np
from simulation_based_inference.simulation.rescorla_wagner import *
from simulation_based_inference.simulation.model_based import *
from simulation_based_inference.simulation.tasks import *
from scipy.stats import bernoulli

rw_parameters = [
    (0.5, 0.5),
    (0.2, 0.8),
    (0.8, 0.2)
]

@pytest.mark.parametrize("alpha_p,alpha_n", rw_parameters)
def test_rescorla_wagner_update_learning_rates(alpha_p, alpha_n):

    outcomes = np.array([0, 1, 0])

    v = 0.5

    for trial in range(len(outcomes)):

        v_new = rescorla_wagner_update(np.expand_dims(v, 0), np.expand_dims(1, 0), np.expand_dims(outcomes[trial], 0), alpha_p, alpha_n)

        if outcomes[trial] > v:
            assert np.isclose(v_new, v + alpha_p * (outcomes[trial] - v))
        else:
            assert np.isclose(v_new, v - alpha_n * (v - outcomes[trial]))
        
        v = v_new


def test_rescorla_wagner_only_chosen_updated():

    outcomes = np.array([1, 0, 1])[:, None].repeat(2, axis=1)

    v = np.array([0.5, 0.5])
    choices = np.array([[0, 1], [1, 0], [0, 0]])

    for trial in range(outcomes.shape[1]):

        v_new = rescorla_wagner_update(v, choices[trial, :], outcomes[trial, :], 0.5, 0.5)

        choice = np.argmax(choices[trial, :])

        assert v_new[1 - choice] == v[1 - choice]
        assert v_new[choice] != v[choice]


@pytest.fixture
def q_values_fixture():
    q_vals = np.zeros((7, 4))  # 5 states, 4 actions
    q_vals[0, :] = np.array([4.5, 3.5, 2.5, 1.5])
    q_vals[1, :] = np.array([0, 0, 0, 0])
    q_vals[2, :] = np.array([1, 1, 0, 0])
    q_vals[3, :] = np.array([20, 5, 2, 1])
    q_vals[4, :] = np.array([-20, 0, 20, 1])
    q_vals[5, :] = np.array([34, 0, 10, -10])
    q_vals[6, :] = np.array([-10, 0, -5, -50])
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


def test_softmax_random_values():

    # Try a load of random values to make sure it doesn't break

    for _ in range(10000):
            
        q_values = np.random.randn(10, 10)
        temperature = np.random.rand() * 10

        action_p = softmax(q_values, temperature=temperature)

        assert np.all(np.isclose(action_p.sum(axis=1), 1))
        assert np.all(action_p >= 0)
        assert np.all(action_p <= 1)
        assert ~np.isnan(action_p).any()


action_probs = [
    np.array([0.25, 0.25, 0.25, 0.25]),
    np.array([0.1, 0.1, 0.1, 0.7]),
    np.array([1., 0., 0., 0.]),
    np.array([0., 0., 0., 1.]),
    np.array([0.5, 0.5, 0., 0., 0., 0.]),
    np.array([0.1, 0.9, 0., 0., 0., 0.]),
]

@pytest.mark.parametrize("action_p", action_probs)
def test_choice_from_action_p(action_p):

    actions = []

    key = jax.random.PRNGKey(42)

    keys = jax.random.split(key, 10000)

    for i in range(10000):
        actions.append(choice_from_action_p_jax(keys[i], action_p, len(action_p)))

    observed_action_prob = np.zeros(len(action_p))

    for i in range(len(action_p)):
        observed_action_prob[i] = np.sum(np.array(actions) == i) / len(actions)

    assert observed_action_prob == pytest.approx(action_p, abs=0.01)

def test_learning_rates():

    rng = np.random.RandomState(42)

    n_obs = 100

    # Positive bias
    alpha_p_1 = rng.uniform(0.5, 1, n_obs)
    alpha_n_1 = rng.uniform(0, 0.5, n_obs)
    temperature_1 = rng.uniform(0, 1, n_obs)

    # Negative bias
    alpha_p_2 = rng.uniform(0, 0.5, n_obs)
    alpha_n_2 = rng.uniform(0.5, 1, n_obs)
    temperature_2 = rng.uniform(0, 1, n_obs)

    outcomes, _ = generate_block_bandit_outcomes(
        100, 3, 4
    )

    _, _, value_estimates_1 = simulate_rescorla_wagner_dual(alpha_p_1, alpha_n_1, temperature_1, outcomes)
    _, _, value_estimates_2 = simulate_rescorla_wagner_dual(alpha_p_2, alpha_n_2, temperature_2, outcomes)

    assert value_estimates_1.mean() > value_estimates_2.mean()

def test_choices_not_the_same():

    rng = np.random.RandomState(42)

    n_obs = 100

    # Positive bias
    alpha_p = rng.uniform(0, 1, n_obs)
    alpha_n = rng.uniform(0, 0, n_obs)
    temperature = rng.uniform(0, 1, n_obs)

    outcomes, _ = generate_block_bandit_outcomes(
        100, 3, 4
    )

    _, choices, _ = simulate_rescorla_wagner_dual(alpha_p, alpha_n, temperature, outcomes, choice_format='index')

    # check that all values are represented in choices
    assert len(np.unique(choices)) == 4


def test_get_choice_MB_choices_valid(q_values_fixture):

    key = jax.random.PRNGKey(42)

    for i in range(q_values_fixture.shape[0]):

        choice_array, choice_p, choice = get_choice(q_values_fixture[i, :], key, 4)

        assert choice in [0, 1, 2, 3]
        assert choice_array[choice] == 1
        assert choice_array.sum() == 1
        assert np.all(choice_array >= 0)
        assert np.all(choice_array <= 1)
        
        assert np.isclose(choice_p.sum(), 1)
        if i == 0:
            assert np.all(np.diff(choice_p) < 0)
        if i == 1:
            assert np.all(choice_p== choice_p[0])
        assert np.all(choice_p >= 0)
        assert np.all(choice_p <= 1)




def test_delta_rule_update_learning_rate():

    rng = np.random.RandomState(42)

    for i in range(10):

        q = np.ones((2, 2)) * 0.5
        q_value_selector = np.zeros((2, 2))
        q_value_selector[rng.randint(0, 2), rng.randint(0, 2)] = 1

        outcomes = np.random.uniform(size=(2, 2))
        outcome_selector = np.zeros((2, 2))
        outcome_selector[rng.randint(0, 2), rng.randint(0, 2)] = 1

        alpha = rng.uniform(0, 1)

        q_updated = delta_rule_update(q, q_value_selector, outcomes, outcome_selector, alpha)

        assert np.isclose(q_updated[q_value_selector.astype(bool)], outcomes[outcome_selector.astype(bool)] + 0.5 * (1 - alpha))
        assert np.isclose(q_updated[~q_value_selector.astype(bool)], 0.5 * (1 - alpha)).all()


def test_mb_learner_learning_rates_choice_probs():

    outcomes, probs = generate_block_bandit_outcomes(100, 1, (2, 2), outcome_generator='random_walk')

    alpha = np.array([0.3, 0.7])
    beta = np.array([1, 1])
    choice_p, choices_one_hot, value_estimates = simulate_mb_learner(
        alpha,
        beta * 0,
        beta,
        beta,
        beta * 0,
        beta,
        outcomes,
        np.array([[0.7, 0.3], [0.3, 0.7]]),
        return_choice_probabilities=True
    )

    # Choice prob should be more variable with higher learning rate (doesn't work for stage 2 as choice_p switches between end states)
    assert np.mean(np.abs(np.diff(choice_p[0][1, 0, :, :], axis=0))) > np.mean(np.abs(np.diff(choice_p[0][0, 0, :, :], axis=0)))

    # Choice probs should be higher for 1 than 0
    assert (choice_p[0] * choices_one_hot[0]).mean() > (choice_p[0] * (1 - choices_one_hot[0])).mean()
    assert (choice_p[1] * choices_one_hot[1]).mean() > (choice_p[1] * (1 - choices_one_hot[1])).mean()


    assert (
        bernoulli.logpmf(choices_one_hot[0], choice_p[0]).sum()
        > bernoulli.logpmf(choices_one_hot[0], np.ones_like(choice_p[0]) * 0.5).sum()
    )  # choices shouldn't be random
    assert (
        bernoulli.logpmf(choices_one_hot[0], choice_p[0]).sum()
        > bernoulli.logpmf(choices_one_hot[0], np.ones_like(choice_p[0])).sum()
    )  # choices shouldn't be 1
    assert (
        bernoulli.logpmf(choices_one_hot[0], choice_p[0]).sum()
        > bernoulli.logpmf(choices_one_hot[0], np.zeros_like(choice_p[0])).sum()
    )  # choices shouldn't be 0

    assert (
        bernoulli.logpmf(choices_one_hot[1], choice_p[1]).sum()
        > bernoulli.logpmf(choices_one_hot[1], np.ones_like(choice_p[1]) * 0.5).sum()
    )  # choices shouldn't be random
    assert (
        bernoulli.logpmf(choices_one_hot[1], choice_p[1]).sum()
        > bernoulli.logpmf(choices_one_hot[1], np.ones_like(choice_p[1])).sum()
    )  # choices shouldn't be 1
    assert (
        bernoulli.logpmf(choices_one_hot[1], choice_p[1]).sum()
        > bernoulli.logpmf(choices_one_hot[1], np.zeros_like(choice_p[1])).sum()
    )  # choices shouldn't be 0