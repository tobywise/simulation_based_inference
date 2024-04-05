import numpy as np
from functools import partial
from typing import Tuple, List, Union, Dict, Any
import jax
import jax.numpy as jnp
from .utils import (
    softmax,
    choice_from_action_p_jax_missed,
    choice_from_action_p_jax_noise,
)


def rescorla_wagner_update_choice_wrapper(
    value_estimate: np.ndarray,
    outcomes_key: Tuple[np.ndarray, jax.random.PRNGKey],
    alpha_p: float,
    alpha_n: float,
    temperature: float,
    n_actions: int,
    lapse: float = 0.0,
    lapse_type: str = "noise",
    update_unchosen: bool = False,
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
        lapse (float, optional): Lapse rate. Defaults to 0.0.
        lapse_type (str, optional): Lapse type. If 'noise', an action is selected uniformly at random on lapse
        trials. If 'missed', lapse trials indicate that the subject failed to make a response. This is represented
        as the subject choosing action `n_actions + 1`. Defaults to "noise".
        update_unchosen (bool, optional): Whether to update the value estimate of unchosen actions. Defaults to False.

    Returns:
        np.ndarray: Updated value estimate

    """

    # Unpack trial outcomes and RNG key
    outcomes, key = outcomes_key

    # Make a choice
    choice_p = softmax(value_estimate[None, :], temperature).squeeze()

    if lapse_type == "noise":
        choice = choice_from_action_p_jax_noise(key, choice_p, n_actions, lapse)
    elif lapse_type == "missed":
        choice = choice_from_action_p_jax_missed(key, choice_p, n_actions, lapse)
        n_actions += 1

    # Convert it to the right format
    choice_array = jnp.zeros(n_actions, dtype=jnp.int16)
    choice_array = choice_array.at[choice].set(1)

    # Get the outcome and update the value estimate
    updated_value = rescorla_wagner_update(
        value_estimate,
        choice_array[0: n_actions - int(lapse_type == "missed")],
        outcomes,
        alpha_p,
        alpha_n,
        update_unchosen,
    )

    return updated_value, (value_estimate, choice_p, choice, choice_array)


rescorla_wagner_update_choice_wrapper_jit = jax.jit(
    rescorla_wagner_update_choice_wrapper, static_argnums=(5, 7)
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
    lapse: float = 0.0,
    lapse_type: str = "noise",
    update_unchosen: bool = False,
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
        lapse (float, optional): Lapse rate. Defaults to 0.0.
        lapse_type (str, optional): Lapse type. If 'noise', an action is selected uniformly at random on lapse
        trials. If 'missed', lapse trials indicate that the subject failed to make a response. This is represented
        as the subject choosing action `n_actions + 1`. Defaults to "noise".
        update_unchosen (bool, optional): Whether to update the value estimate of unchosen actions. Defaults to False.

    Returns:
        np.ndarray: Value estimates for each trial and each bandit
    """

    # Use functools.partial to create a function that uses the same alpha_p and alpha_n for all trials
    rescorla_wagner_update_partial = partial(
        rescorla_wagner_update_choice_wrapper_jit,
        alpha_p=alpha_p,
        alpha_n=alpha_n,
        temperature=temperature,
        n_actions=n_actions,
        lapse=lapse,
        lapse_type=lapse_type,
        update_unchosen=update_unchosen
    )

    # Initial values
    v_start = jnp.ones(n_actions) * starting_value_estimate

    # Jax random keys for choices
    keys = jax.random.split(key, n_trials)

    # use jax.lax.scan to iterate over trials
    _, (v, choice_p, choices, choices_one_hot) = jax.lax.scan(
        rescorla_wagner_update_partial, v_start, (outcomes, keys)
    )

    return v, choice_p, choices, choices_one_hot


# Set up jax JIT and vmaps
rescorla_wagner_trial_choice_iterator_jit = jax.jit(
    rescorla_wagner_trial_choice_iterator, static_argnums=(2, 3, 9)
)

# Vmap to iterate over blocks
rescorla_wagner_simulate_vmap_blocks = jax.vmap(
    rescorla_wagner_trial_choice_iterator_jit,
    in_axes=(0, 0, None, None, None, None, None, None, None, None, None),
)

# Vmap to iterate over observations (subjects)
rescorla_wagner_simulate_vmap_observations = jax.vmap(
    rescorla_wagner_simulate_vmap_blocks,
    in_axes=(None, 0, None, None, None, 0, 0, 0, None, None, None),
)


@jax.jit
def rescorla_wagner_update(
    value_estimate: np.ndarray,
    choices: np.ndarray,
    outcomes: np.ndarray,
    alpha_p: float,
    alpha_n: float,
    update_unchosen: bool = False,
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
        update_unchosen (bool, optional): Whether to update the value estimate for unchosen bandits. Defaults to False.

    Returns:
        np.ndarray: Updated value estimates for this trial, with one entry per bandit.
    """

    # Calculate prediction error
    prediction_error = (outcomes - value_estimate) * (
        (1 - update_unchosen) * choices + update_unchosen * 1
    )

    # Update value estimates according to the relevant learning rate
    value_estimate = value_estimate + alpha_p * prediction_error * (
        prediction_error > 0
    )
    value_estimate = value_estimate + alpha_n * prediction_error * (
        prediction_error < 0
    )

    return value_estimate


def simulate_rescorla_wagner_dual(
    alpha_p: np.ndarray,
    alpha_n: np.ndarray,
    temperature: np.ndarray,
    outcomes: np.ndarray,
    choice_format: str = "one_hot",
    starting_value_estimate: float = 0.5,
    lapse: Union[float, None] = 0.0,
    lapse_type: str = "noise",
    update_unchosen: bool = False,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate choices a multi-armed bandit task with n_bandits arms.

    Args:
        alpha_p (np.ndarray): Learning rate for prediction errors > 0. Should have as many entries as desired observations (subjects).
        alpha_n (np.ndarray): Learning rate for prediction errors <= 0. Should have as many entries as desired observations (subjects).
        temperature (np.ndarray): Temperatures. Should have as many entries as desired observations (subjects).
        outcomes (np.ndarray): Outcomes for each block, trial and each bandit. An array of shape (n_blocks, n_trials, n_bandits).
        choice_format (str, optional): Format of the choices. Can be either 'one_hot' or 'index'. Defaults to "one_hot".
        starting_value_estimate (float, optional): Starting value estimate for each bandit. Defaults to 0.5.
        lapse (Union[float, None], optional): Maxmimum level of noise to add to the value estimates. Defaults to 0.0.
        lapse_type (str, optional): Lapse type. If 'noise', an action is selected uniformly at random on lapse
        trials. If 'missed', lapse trials indicate that the subject failed to make a response. This is represented
        as the subject choosing action `n_actions + 1`. Defaults to "noise".
        update_unchosen (bool, optional): Whether to update the value estimate for unchosen bandits. Defaults to False.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Choice probabilities, choices, value estimates.
    """

    assert (
        alpha_p.shape == alpha_n.shape == temperature.shape
    ), "alpha_p, alpha_n and temperature must have the same shape"

    # Extract dimensions
    _, n_blocks, n_trials, n_bandits = outcomes.shape

    # Run simulation
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_blocks)
    # subject_keys = jax.random.split(key, alpha_p.shape[0])
    (
        value_estimates,
        choice_p,
        choices,
        choices_one_hot,
    ) = rescorla_wagner_simulate_vmap_observations(
        keys,
        outcomes,
        n_bandits,
        n_trials,
        starting_value_estimate,
        alpha_p,
        alpha_n,
        temperature,
        lapse,
        lapse_type,
        update_unchosen
    )

    if choice_format == "one_hot":
        return choice_p, choices_one_hot, value_estimates
    elif choice_format == "index":
        return choice_p, choices, value_estimates


def simulate_rescorla_wagner_single(
    alpha: np.ndarray,
    temperature: np.ndarray,
    outcomes: Union[None, np.ndarray] = None,
    choice_format: str = "one_hot",
    starting_value_estimate: float = 0.5,
    lapse: float = 0.0,
    lapse_type: str = "noise",
    update_unchosen: bool = False,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate choices a multi-armed bandit task with n_bandits arms.

    Args:
        alpha_p (np.ndarray): Learning rate. Should have as many entries as desired observations (subjects).
        temperature (np.ndarray): Temperatures. Should have as many entries as desired observations (subjects).
        outcomes (Union[None, np.ndarray], optional): Outcomes for each trial and each bandit. If None, will be generated. Defaults to None.
        choice_format (str, optional): Format of the choices. Can be either 'one_hot' or 'index'. Defaults to "one_hot".
        starting_value_estimate (float, optional): Starting value estimate for each bandit. Defaults to 0.5.
        lapse (float, optional): Noise to add to the value estimates. Defaults to 0.0.
        lapse_type (str, optional): Lapse type. If 'noise', an action is selected uniformly at random on lapse
        trials. If 'missed', lapse trials indicate that the subject failed to make a response. This is represented
        as the subject choosing action `n_actions + 1`. Defaults to "noise".
        update_unchosen (bool, optional): Whether to update the value estimate for unchosen bandits. Defaults to False.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Choice probabilities, choices, value estimates.
    """

    return simulate_rescorla_wagner_dual(
        alpha,
        alpha,
        temperature,
        outcomes,
        choice_format,
        starting_value_estimate,
        lapse,
        lapse_type,
        update_unchosen,
        seed,
    )


def simulate_multiple_rescorla_wagner_models(
    outcomes: np.ndarray,
    models: List[Tuple[str, callable, List[np.ndarray]]],
    model_func_kwargs: Dict[str, Any] = {},
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Simulate behaviour from multiple Rescorla-Wagner style models.

    Args:
        outcomes (np.ndarray): Outcomes for each block, trial and each bandit. An array of shape (n_blocks, n_trials, n_bandits).
        models (List[Tuple[str, callable, List[np.ndarray]]]): List of models to simulate. Each model is a tuple with the name of
        the model, the function to simulate the model and a list of paramater values (as numpy arrays) to pass to the function.
        model_func_kwargs (Dict[str, Any], optional): Keyword arguments to pass to the model function, used for all
        models. Defaults to {}.

    Returns:
        Dict[Dict]: Dictionary of dictionaries with the results of the simulations.
    """

    model_output = {}

    # Loop over models
    for (model_name, model_func, params) in models:

        # Simulate using provided model function
        choice_prob, choices, value_estimate = model_func(
            *params, outcomes=outcomes, **model_func_kwargs
        )

        # Store results
        model_output[model_name] = {}
        model_output[model_name]["choices"] = choices
        model_output[model_name]["choice_prob"] = choice_prob
        model_output[model_name]["value_estimate"] = value_estimate
        model_output[model_name]["params"] = np.stack(params).T

    return model_output
