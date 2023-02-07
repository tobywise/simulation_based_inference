import numpy as np
from functools import partial
from typing import Tuple, List, Union, Dict
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
def choice_from_action_p_jax_noise(
    key: int, probs: jnp.ndarray, n_actions: int, lapse: float = 0.0
) -> int:
    """
    Choose an action from a set of action probabilities.

    Noise is added to the choice, with probability lapse. This means that
    on "lapse" trials, the subject will choose an action uniformly at random.

    Args:
        key (int): Jax random key
        probs (np.ndarray): 1D array of action probabilities, of shape (n_bandits)
        n_actions (int): Number of actions
        lapse (float, optional): Probability of lapse. Defaults to 0.0.

    Returns:
        int: Chosen action
    """

    n_actions = len(probs)

    # Deal with zero values etc
    probs = probs + 1e-6 / jnp.sum(probs)

    noise = jax.random.uniform(key) < lapse

    choice = (1 - noise) * jax.random.choice(
        key, jnp.arange(n_actions, dtype=int), p=probs
    ) + noise * jax.random.randint(key, shape=(), minval=0, maxval=n_actions)

    return choice

@jax.jit
def choice_from_action_p_jax_missed(
    key: int, probs: jnp.ndarray, n_actions: int, lapse: float = 0.0
) -> int:
    """
    Choose an action from a set of action probabilities.

    The lapse parameter is interpreted as the probability of making no 
    response at all. This means that on "lapse" trials, the subject will
    choose action `n_actions + 1`, which is used to indicate a missed
    response.

    Args:
        key (int): Jax random key
        probs (np.ndarray): 1D array of action probabilities, of shape (n_bandits)
        n_actions (int): Number of actions
        lapse (float, optional): Probability of lapse. Defaults to 0.0.

    Returns:
        int: Chosen action
    """

    n_actions = len(probs)

    # Deal with zero values etc
    probs = probs + 1e-6 / jnp.sum(probs)

    # Add a final action, which is the "missed" action with probability 0
    probs = jnp.concatenate([probs, jnp.zeros(1)])

    noise = jax.random.uniform(key) < lapse

    # Create a noise vector, with probability 1 for the last action
    noise_probs = jnp.zeros_like(probs)
    noise_probs = noise_probs.at[-1].set(1)

    probs = (1 - noise) * probs + noise * noise_probs

    choice = jax.random.choice(
        key, jnp.arange(n_actions + 1, dtype=int), p=probs
    ) 

    return choice