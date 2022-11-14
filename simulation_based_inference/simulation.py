import numpy as np
from maMDP.algorithms.action_selection import SoftmaxActionSelector
from typing import Tuple, List
import jax


def generate_bandit_outcomes(
    n_trials: int,
    n_bandits: int,
    outcome_probability_levels: List[int] = [0.2, 0.5, 0.8],
    outcome_probability_duration: Tuple[int, int] = (20, 40),
    seed: int = 42,
) -> np.ndarray:
    """
    Generate outcomes for a multi-armed bandit task with n_bandits arms.

    Args:
        n_trials (int): Number of trials
        n_bandits (int): Number of bandits
        outcome_probability_levels (List[int], optional): Outcome probability levels to alternate between. Defaults to [0.2, 0.5, 0.8].
        outcome_probability_duration (Tuple[int, int], optional): Minimum and maximum number of trials to use each outcome probability level.
        Defaults to (20, 40).
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        np.ndarray: Array of outcomes for each trial and each bandit, of shape (n_trials, n_bandits)
    """

    # Array of outcomes - same outcomes for each subject for now
    outcomes = np.zeros((n_trials, n_bandits))

    # Trial/bandit outcome probability levels
    trial_outcome_probability_levels = np.zeros((n_trials, n_bandits))

    # Generate outcomes - outcome probabilities change every 20-40 trials
    rng = np.random.RandomState(seed)

    # Generate outcome probabilities for each bandit
    for bandit in range(n_bandits):

        # Start of the block
        current_breakpoint = 0

        outcome_probability = None

        while True:
            # Determine when the outcome probability is going to change
            next_breakpoint = rng.randint(*outcome_probability_duration)

            # Get outcome probability for this block
            outcome_probability = rng.choice(
                [x for x in outcome_probability_levels if x != outcome_probability]
            )

            # Generate an array of ones and zeros with the correct number of ones
            block_endpoint = current_breakpoint + next_breakpoint
            if block_endpoint > n_trials:
                next_breakpoint = n_trials - current_breakpoint
            block_outcomes = np.zeros(next_breakpoint, dtype=int)
            block_outcomes[: int(next_breakpoint * outcome_probability)] = 1

            # Shuffle the array
            rng.shuffle(block_outcomes)

            # Add the block to the outcomes array
            outcomes[current_breakpoint:block_endpoint, bandit] = block_outcomes

            # Store the probabilities for this block/bandit
            trial_outcome_probability_levels[
                current_breakpoint:block_endpoint, bandit
            ] = outcome_probability

            # Update the current breakpoint
            current_breakpoint += next_breakpoint

            # End if we've reached the end of the trials
            if block_endpoint > n_trials:
                break

    return outcomes, trial_outcome_probability_levels

# TODO wrap the RW function so that choices and outcomes are passed in as a single tuple

# @jax.jit
def rescorla_wagner_update(
    value_estimate: np.ndarray,
    choices: np.ndarray,
    outcomes: np.ndarray,
    alpha_p: float,
    alpha_n: float,
) -> np.ndarray:
    """
    Update value estimates using Rescorla-Wagner learning rule.

    Args:
        value_estimates (np.ndarray): Value estimates for this trial. Should have as many entries as there are bandits.
        choices (np.ndarray): Choices made in this trial. Should have as many entries as there are bandits, with
        zeros for non-chosen bandits and ones for chosen bandits.
        outcomes (np.ndarray): Outcomes for this trial. Should have as many entries as there are bandits.
        alpha_p (float): Learning rate for prediction errors > 0.
        alpha_n (float): Learning rate for prediction errors <= 0.

    Returns:
        np.ndarray: Updated value estimates for this trial, with one entry per bandit.
    """

    # Calculate prediction error
    prediction_error = (outcomes - value_estimate) * choices

    # Get chosen bandit
    choice = choices.argmax()

    # Update value estimates according to the relevant learning rate
    if prediction_error[choice] > 0:
        value_estimate = value_estimate + alpha_p * prediction_error
    else:
        value_estimate = value_estimate + alpha_n * prediction_error

    return value_estimate


def simulate_RL(
    alpha_p: np.ndarray,
    alpha_n: np.ndarray,
    temperature: np.ndarray,
    n_trials: int = 300,
    n_bandits: int = 4,
    outcome_probability_levels: List[int] = [0.2, 0.5, 0.8],
    outcome_probability_duration: Tuple[int, int] = (20, 40),
    starting_value_estimate: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate choices a multi-armed bandit task with n_bandits arms.

    Args:
        alpha_p (np.ndarray): Learning rate for prediction errors > 0. Should have as many entries as desired observations (subjects).
        alpha_n (np.ndarray): Learning rate for prediction errors <= 0. Should have as many entries as desired observations (subjects).
        temperature (np.ndarray): Temperatures. Should have as many entries as desired observations (subjects).
        n_trials (int): Number of trials to simulate. Defaults to 300.
        n_bandits (int, optional): Number of bandits. Defaults to 4.
        outcome_probability_levels (List[int], optional): Outcome probability levels to alternate between. Defaults to [0.2, 0.5, 0.8].
        outcome_probability_duration (Tuple[int, int], optional): Minimum and maximum number of trials to use each outcome probability
        level. Defaults to (20, 40).
        starting_value_estimate (float, optional): Starting value estimate for each bandit. Defaults to 0.5.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        np.ndarray: _description_
    """

    assert (
        alpha_p.shape == alpha_n.shape == temperature.shape
    ), "alpha_p, alpha_n and temperature must have the same shape"

    n_observations = len(alpha_p)

    # Preallocate array for storing the value estimates
    value_estimates = np.ones((n_trials, n_observations, n_bandits)) * starting_value_estimate

    # Preallocate array for choices
    choices = np.zeros((n_observations, n_trials, n_bandits))

    # Generate outcomes
    outcomes, trial_outcome_probability_levels = generate_bandit_outcomes(
        n_trials,
        n_bandits,
        outcome_probability_levels,
        outcome_probability_duration,
        seed,
    )

    # Loop over observations and trials
    for obs in range(n_observations):

        # Action selector - softmax
        softmax = SoftmaxActionSelector(1 / temperature[obs], seed=seed)

        for trial in range(n_trials):

            # Select one of the bandits according to the choice probabilities
            choice = softmax.get_pi(value_estimates[trial, obs, :][None, :]).squeeze()

            # Add choice to choices array
            choices[obs, trial, choice] = 1

            # Update value estimate
            if trial < n_trials - 1:
                value_estimates[trial + 1, obs, :] = rescorla_wagner_update(
                    value_estimates[trial, obs, :],
                    choices[obs, trial, choice],
                    outcomes[trial, :],
                    alpha_p[obs],
                    alpha_n[obs],
                )

    return outcomes, trial_outcome_probability_levels, choices, value_estimates
