import pytest
import numpy as np
from simulation_based_inference.simulation import *

# Need long blocks to get decent probability estimates
bandit_settings = [
    (5000, 1, [0.2, 0.5, 0.8], (250, 251)),
    (5000, 3, [0.1, 0.5, 0.9], (400, 401)),
    (5000, 1, [0.3, 0.7], (300, 301)),
]

@pytest.mark.parametrize("n_trials,n_bandits,outcome_probability_levels,outcome_probability_duration", bandit_settings)
def test_generate_bandit_outcomes_prob_levels(n_trials, n_bandits, outcome_probability_levels, outcome_probability_duration):

    # Test that the outcome probabilities alternate between the levels specified
    # in outcome_probability_levels

    seed = 42

    # Generate outcomes
    outcomes, _ = generate_bandit_outcomes(
        n_trials,
        n_bandits,
        outcome_probability_levels,
        outcome_probability_duration,
        seed,
    )

    for block in range(0, n_trials, outcome_probability_duration[0]):

        for bandit in range(n_bandits):

            outcome_prob = outcomes[
                block : block + outcome_probability_duration[0], bandit
            ].mean()

            assert outcome_prob in outcome_probability_levels

bandit_settings2 = [
    (5000, 1, [0.2, 0.5, 0.8], (20, 50)),
    (5000, 3, [0.1, 0.5, 0.9], (30, 60)),
    (5000, 1, [0.3, 0.7], (80, 130)),
]

@pytest.mark.parametrize("n_trials,n_bandits,outcome_probability_levels,outcome_probability_duration", bandit_settings2)
def test_generate_bandit_outcome_prob_levels_returned(n_trials, n_bandits, outcome_probability_levels, outcome_probability_duration):

    seed = 42

    # Generate outcomes
    outcomes, trial_outcome_probability_levels = generate_bandit_outcomes(
        n_trials,
        n_bandits,
        outcome_probability_levels,
        outcome_probability_duration,
        seed,
    )

    for block in range(0, n_trials, outcome_probability_duration[0]):

        for bandit in range(n_bandits):

            breakpoints = np.hstack([np.array([0]), np.where(np.diff(trial_outcome_probability_levels[:, bandit]) != 0)[0] + 1])

            for n in range(len(breakpoints) - 1):
                true_outcome_prob = outcomes[
                                    breakpoints[n]:breakpoints[n+1], bandit
                                ].mean()
                reported_trial_outcome_prob = trial_outcome_probability_levels[breakpoints[n]:breakpoints[n+1], bandit]
                print(n, breakpoints[n], breakpoints[n+1])
                assert np.all(reported_trial_outcome_prob == reported_trial_outcome_prob[0])
                reported_outcome_prob = reported_trial_outcome_prob.mean()

                assert np.isclose(true_outcome_prob, reported_outcome_prob, atol=0.05)
                assert any([np.isclose(true_outcome_prob, p, atol=0.05) for p in outcome_probability_levels])



@pytest.mark.parametrize("n_trials,n_bandits,outcome_probability_levels,outcome_probability_duration", bandit_settings)
def test_generate_bandit_outcomes_block_lengths(n_trials, n_bandits, outcome_probability_levels, outcome_probability_duration):

    # Test that the outcome probabilities alternate between the levels specified
    # in outcome_probability_levels

    # Set parameters
    seed = 42

    # Generate outcomes
    outcomes, _ = generate_bandit_outcomes(
        n_trials,
        n_bandits,
        outcome_probability_levels,
        outcome_probability_duration,
        seed,
    )

    for block in range(outcome_probability_duration[0], n_trials, outcome_probability_duration[0]):

        for bandit in range(n_bandits):

            outcome_prob_before = outcomes[
                block - int(outcome_probability_duration[0] / 2) : block, bandit
            ].mean()

            outcome_prob_after = outcomes[
                block : block + int(outcome_probability_duration[0] / 4), bandit
            ].mean()
            print(block, outcome_prob_before, outcome_prob_after)
            assert (
                ~np.isclose(outcome_prob_before, outcome_prob_after, atol=0.1)
            )  # Prob before and after block changes should be different


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

    _, _, _, value_estimates_1 = simulate_RL(alpha_p_1, alpha_n_1, temperature_1)
    _, _, _, value_estimates_2 = simulate_RL(alpha_p_2, alpha_n_2, temperature_2)

    assert value_estimates_1.mean() > value_estimates_2.mean()

def test_choices_not_the_same():

    rng = np.random.RandomState(42)

    n_obs = 100

    # Positive bias
    alpha_p = rng.uniform(0, 1, n_obs)
    alpha_n = rng.uniform(0, 0, n_obs)
    temperature = rng.uniform(0, 1, n_obs)

    _, _, choices, _ = simulate_RL(alpha_p, alpha_n, temperature)

    # check that all values are represented in choices
    assert len(np.unique(choices)) == 4
