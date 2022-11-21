import numpy as np
from functools import partial
from typing import Tuple, List, Union
import jax
import jax.numpy as jnp


@jax.jit
def softmax(value: np.ndarray, temperature: float = 1) -> np.ndarray:
    """
    Softmax function, with optional temperature parameter.

    Args:
        value (np.ndarray): Array of values to apply softmax to, of shape (n_trials, n_bandits)
        temperature (float, optional): Softmax temperature, in range 0 > inf. Defaults to 1.

    Returns:
        np.ndarray: Choice probabilities, of shape (n_trials, n_bandits)
    """
    # Subtract max value to avoid overflow
    return (jnp.exp((value - value.max(axis=1)[:, None]) / temperature)) / (
        jnp.sum(jnp.exp((value - value.max(axis=1)[:, None]) / temperature), axis=1)[
            :, None
        ]
    )


@jax.jit
def choice_from_action_p_jax(key: int, probs: jnp.ndarray, n_actions: int) -> int:
    """
    Choose an action from a set of action probabilities.

    Args:
        probs (np.ndarray): 1D array of action probabilities, of shape (n_bandits)
        rng (np.random.RandomState): Random number generator

    Returns:
        int: Chosen action
    """

    # Make sure the input is a numpy array (jax arrays cause problems)
    n_actions = len(probs)

    # Deal with zero values etc
    probs = probs + 1e-6 / jnp.sum(probs)

    return jax.random.choice(key, jnp.arange(n_actions, dtype=int), p=probs)


def rescorla_wagner_update_choice_wrapper(
    value_estimate: np.ndarray,
    outcomes_key: Tuple[np.ndarray, jax.random.PRNGKey],
    alpha_p: float,
    alpha_n: float,
    temperature: float,
    n_actions: int,
) -> np.ndarray:
    """
    Wrapper function for rescorla_wagner_update to use in jax.lax.scan, which produces choices for each trial
    alongside value estimates.

    Args:
        value_estimate (np.ndarray): Current value estimate
        choices_outcomes (Tuple[np.ndarray, jax.random.PRNGKey]): Tuple of outcomes and Jax RNG keys
        alpha_p (float): Positive learning rate
        alpha_n (float): Negative learning rate
        temperature (float): Softmax temperature
        n_actions (int): Number of actions

    Returns:
        np.ndarray: Updated value estimate

    """

    # Unpack trial outcomes and RNG key
    outcomes, key = outcomes_key

    # Make a choice
    choice_p = softmax(value_estimate[None, :], temperature).squeeze()
    choice = choice_from_action_p_jax(key, choice_p, n_actions)

    # Convert it to the right format
    choice_array = jnp.zeros(n_actions, dtype=jnp.int16)
    choice_array = choice_array.at[choice].set(1)

    # Get the outcome and update the value estimate
    updated_value = rescorla_wagner_update(
        value_estimate, choice_array, outcomes, alpha_p, alpha_n
    )

    return updated_value, (value_estimate, choice_p, choice)


rescorla_wagner_update_choice_wrapper_jit = jax.jit(
    rescorla_wagner_update_choice_wrapper, static_argnums=(5)
)


def rescorla_wagner_trial_choice_iterator(
    key: jax.random.PRNGKey,
    outcomes: np.ndarray,
    n_actions: int,
    n_trials: int,
    starting_value_estimate: float = 0.5,
    alpha_p: float = 0.5,
    alpha_n: float = 0.5,
    temperature: float = 0.5,
) -> np.ndarray:
    """
    Iterate over trials and update value estimates, generating choices for each trial. Used for model-fitting.

    Args:
        key (jax.random.PRNGKey): Jax random number generator key
        outcomes (np.ndarray): Trial outcomes for each bandit of shape (n_trials, n_bandits)
        n_actions (int, optional): Number of actions.
        n_trials (int): Number of trials
        starting_value_estimate (float, optional): Starting value estimate. Defaults to 0.5.
        alpha_p (float, optional): Learning rate for prediction errors > 0. Defaults to 0.5.
        alpha_n (float, optional): Learning rate for prediction errors < 0. Defaults to 0.5.
        temperature (float, optional): Temperature parameter for softmax. Defaults to 0.5.

    Returns:
        np.ndarray: Value estimates for each trial and each bandit
    """

    # Use functools.partial to create a function that uses the same alpha_p and alpha_n for all trials
    rescorla_wagner_update_partial = partial(
        rescorla_wagner_update_choice_wrapper,
        alpha_p=alpha_p,
        alpha_n=alpha_n,
        temperature=temperature,
        n_actions=n_actions,
    )

    # Initial values
    v_start = jnp.ones(n_actions) * starting_value_estimate

    # Jax random keys for choices
    keys = jax.random.split(key, n_trials)

    # use jax.lax.scan to iterate over trials
    _, (v, choice_p, choices) = jax.lax.scan(
        rescorla_wagner_update_partial, v_start, (outcomes, keys)
    )

    return v, choice_p, choices


rescorla_wagner_trial_choice_iterator_jit = jax.jit(
    rescorla_wagner_trial_choice_iterator, static_argnums=(2, 3)
)

# Vmap to allow iterating over multiple parameter values
rescorla_wagner_simulate_vmap = jax.vmap(
    rescorla_wagner_trial_choice_iterator_jit,
    in_axes=(None, None, None, None, None, 0, 0, 0),
)


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


@jax.jit
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

    # Update value estimates according to the relevant learning rate
    value_estimate = (
        value_estimate + alpha_p * prediction_error * (prediction_error > 0) * choices
    )
    value_estimate = (
        value_estimate + alpha_n * prediction_error * (prediction_error < 0) * choices
    )

    return value_estimate


def simulate_rescorla_wagner_dual(
    alpha_p: np.ndarray,
    alpha_n: np.ndarray,
    temperature: np.ndarray,
    outcomes: Union[None, np.ndarray] = None,
    n_trials: int = 300,
    n_bandits: int = 4,
    outcome_probability_levels: List[int] = [0.2, 0.5, 0.8],
    outcome_probability_duration: Tuple[int, int] = (20, 40),
    starting_value_estimate: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate choices a multi-armed bandit task with n_bandits arms.

    Args:
        alpha_p (np.ndarray): Learning rate for prediction errors > 0. Should have as many entries as desired observations (subjects).
        alpha_n (np.ndarray): Learning rate for prediction errors <= 0. Should have as many entries as desired observations (subjects).
        temperature (np.ndarray): Temperatures. Should have as many entries as desired observations (subjects).
        outcomes (Union[None, np.ndarray], optional): Outcomes for each trial and each bandit. If None, will be generated. Defaults to None.
        n_trials (int): Number of trials to simulate. Defaults to 300.
        n_bandits (int, optional): Number of bandits. Defaults to 4.
        outcome_probability_levels (List[int], optional): Outcome probability levels to alternate between. Defaults to [0.2, 0.5, 0.8].
        outcome_probability_duration (Tuple[int, int], optional): Minimum and maximum number of trials to use each outcome probability
        level. Defaults to (20, 40).
        starting_value_estimate (float, optional): Starting value estimate for each bandit. Defaults to 0.5.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Outcomes, trial outcome probability levels, choice_probabiliies,
        choices, value estimates.
    """

    assert (
        alpha_p.shape == alpha_n.shape == temperature.shape
    ), "alpha_p, alpha_n and temperature must have the same shape"

    if outcomes is None:
        # Generate outcomes
        outcomes, trial_outcome_probability_levels = generate_bandit_outcomes(
            n_trials,
            n_bandits,
            outcome_probability_levels,
            outcome_probability_duration,
            seed,
        )

    # Run simulation
    key = jax.random.PRNGKey(42)
    value_estimates, choice_p, choices = rescorla_wagner_simulate_vmap(
        key,
        outcomes,
        n_bandits,
        n_trials,
        starting_value_estimate,
        alpha_p,
        alpha_n,
        temperature,
    )

    return outcomes, trial_outcome_probability_levels, choice_p, choices, value_estimates


def simulate_rescorla_wagner_single(
    alpha: np.ndarray,
    temperature: np.ndarray,
    outcomes: Union[None, np.ndarray] = None,
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
        alpha_p (np.ndarray): Learning rate. Should have as many entries as desired observations (subjects).
        temperature (np.ndarray): Temperatures. Should have as many entries as desired observations (subjects).
        outcomes (Union[None, np.ndarray], optional): Outcomes for each trial and each bandit. If None, will be generated. Defaults to None.
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

    return simulate_rescorla_wagner_dual(
        alpha,
        alpha,
        temperature,
        outcomes,
        n_trials,
        n_bandits,
        outcome_probability_levels,
        outcome_probability_duration,
        starting_value_estimate,
        seed
    )