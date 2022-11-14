import numpyro
import numpy as np
import jax.numpy as jnp
import jax
from jax import random
import numpyro.distributions as dist
from typing import Union, Tuple
from simulation_based_inference.simulation import rescorla_wagner_update
from functools import partial
from numpyro.infer import MCMC, NUTS

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
    return (jnp.exp((value) / temperature)) / (
        jnp.sum(jnp.exp((value) / temperature), axis=1)[
            :, None
        ]
    )

# vmap softmax over observations
softmax_vmap = jax.vmap(softmax, in_axes=(0, 0))


def create_subject_params(
    name: str, n_subs: int
) -> Union[dist.Normal, dist.HalfNormal, dist.Normal]:
    """
    Creates group mean, group sd and subject-level offset parameters.
    Args:
        name (str): Name of the parameter
        n_subs (int): Number of subjects
    Returns:
        Union[dist.Normal, dist.HalfNormal, dist.Normal]: Group mean, group sd, and subject-level offset parameters
    """

    group_mean = numpyro.sample("{0}_group_mean".format(name), dist.Normal(0, 1))
    group_sd = numpyro.sample("{0}_group_sd".format(name), dist.HalfNormal(1))
    offset = numpyro.sample(
        "{0}_offset".format(name), dist.Normal(0, 1), sample_shape=(n_subs,)
    )

    return group_mean, group_sd, offset


@jax.jit
def rescorla_wagner_update_wrapper(
    value_estimate: np.ndarray,
    choices_outcomes: Tuple[np.ndarray, np.ndarray],
    alpha_p: float,
    alpha_n: float,
) -> np.ndarray:
    """
    Wrapper function for rescorla_wagner_update to use in jax.lax.scan
    Args:
        value_estimate (np.ndarray): Current value estimate
        choices_outcomes (Tuple[np.ndarray, np.ndarray]): Tuple of choices and outcomes
        alpha_p (float): Positive learning rate
        alpha_n (float): Negative learning rate

    Returns:
        np.ndarray: Updated value estimate

    """

    choices, outcomes = choices_outcomes

    updated_value = rescorla_wagner_update(
        value_estimate, choices, outcomes, alpha_p, alpha_n
    )

    return updated_value, value_estimate


@jax.jit
def rescorla_wagner_trial_iterator(
    outcomes: np.ndarray,
    choices: np.ndarray,
    starting_value_estimate: float = 0.5,
    alpha_p: float = 0.5,
    alpha_n: float = 0.5,
) -> np.ndarray:
    """
    Iterate over trials and update value estimates, based on predetermined choices. Used for model-fitting.

    Args:
        outcomes (np.ndarray): Trial outcomes for each bandit of shape (n_trials, n_bandits)
        choices (np.ndarray): Observed choices, as a 2D array of shape (n_trials, n_bandits) where
        the chosen bandit is 1 and the others are 0.
        starting_value_estimate (float, optional): Starting value estimate. Defaults to 0.5.
        alpha_p (float, optional): Learning rate for prediction errors > 0. Defaults to 0.5.
        alpha_n (float, optional): Learning rate for prediction errors > 0. Defaults to 0.5.

    Returns:
        np.ndarray: Value estimates for each trial and each bandit
    """

    # Use functools.partial to create a function that uses the same alpha_p and alpha_n for all trials
    rescorla_wagner_update_partial = partial(
        rescorla_wagner_update_wrapper,
        alpha_p=alpha_p,
        alpha_n=alpha_n,
    )

    # Initial values
    v = np.ones(outcomes.shape[1]) * starting_value_estimate

    # use jax.lax.scan to iterate over trials
    _, v = jax.lax.scan(rescorla_wagner_update_partial, v, (choices, outcomes))

    return v


# vmap iterator over subjects
rescorla_wagner_model_vmap = jax.vmap(
    rescorla_wagner_trial_iterator, in_axes=(None, 0, None, 0, 0)
)


def rescorla_wagner_model(
    outcomes: np.ndarray, choices: np.ndarray, starting_value_estimate: float = 0.5
):
    """
    Rescorla-Wagner model, using non-centered hierarchical parameterisation

    Args:
        outcomes (np.ndarray): Rewards received for each bandit on each trial, as a 3D array of shape (n_observations, n_trials, n_bandits)
        choices (np.ndarray): Observed choices, as a 3D array of shape (n_observations, n_trials, n_bandits) where
        the chosen bandit is 1 and the others are 0.
        starting_value_estimate (float, optional): _description_. Defaults to 0.5.

    """

    # Get shapes
    n_obs, _, _ = choices.shape

    # Prior on learning rates
    alpha_p_group_mean, alpha_p_group_sd, alpha_p_offset = create_subject_params(
        "alpha_p", n_obs
    )
    alpha_p_subject_transformed = numpyro.deterministic(
        "alpha_p_subject_transformed",
        jax.scipy.special.expit(alpha_p_group_mean + alpha_p_group_sd * alpha_p_offset)
    )

    alpha_n_group_mean, alpha_n_group_sd, alpha_n_offset = create_subject_params(
        "alpha_n", n_obs
    )
    alpha_n_subject_transformed = numpyro.deterministic(
        "alpha_n_subject_transformed",
        jax.scipy.special.expit(alpha_n_group_mean + alpha_n_group_sd * alpha_n_offset)
    )

    # Prior on temperature
    (
        temperature_group_mean,
        temperature_group_sd,
        temperature_offset,
    ) = create_subject_params("temperature", n_obs)

    temperature_subject_transformed = numpyro.deterministic(
        "temperature_subject_transformed",
        jax.scipy.special.expit(
            temperature_group_mean + temperature_group_sd * temperature_offset
        )
    )

    # Update the values
    v = numpyro.deterministic(
        "value",
        rescorla_wagner_model_vmap(
            outcomes,
            choices,
            starting_value_estimate,
            alpha_p_subject_transformed,
            alpha_n_subject_transformed,
        ),
    )

    # Get action probabilities
    p = numpyro.deterministic("p", softmax_vmap(v, temperature_subject_transformed))

    # Bernoulli likelihood
    numpyro.sample(
        "obs",
        dist.Bernoulli(probs=p),
        obs=choices,
    )

    return v


def run_rescorla_wagner_inference(
    outcomes: np.ndarray,
    choices: np.ndarray,
    starting_value_estimate: float = 0.5,
    n_samples: int = 4000,
    n_warmup: int = 2000,
    num_chains: int = 1,
    seed: int = 42,
) -> MCMC:
    """
    Samples from the posterior distribution of the model using NUTS, implemented in NumPyro.

    Args:
        model_func (callable): Model function
        outcomes (np.ndarray): Rewards received for each bandit on each trial, as a 3D array of shape (n_observations, n_trials, n_bandits)
        choices (np.ndarray): Observed choices, as a 3D array of shape (n_observations, n_trials, n_bandits) where
        the chosen bandit is 1 and the others are 0.
        n_samples (int, optional): Number of samples. Defaults to 4000.
        n_warmup (int, optional): Number of warmup iterations. Defaults to 2000.
        num_chains (int, optional): Number of chains to run. Defaults to 1.
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        MCMC: Results of sampling.
    """

    nuts_kernel = NUTS(rescorla_wagner_model)

    mcmc = MCMC(
        nuts_kernel, num_samples=n_samples, num_warmup=n_warmup, num_chains=num_chains
    )

    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, outcomes, choices, starting_value_estimate)

    return mcmc
