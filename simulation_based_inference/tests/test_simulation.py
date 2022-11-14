import pytest
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
            assert v_new == v + alpha_p * (outcomes[trial] - v)
        else:
            assert v_new == v - alpha_n * (v - outcomes[trial])
        
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