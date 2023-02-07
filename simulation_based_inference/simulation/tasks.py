import numpy as np
from functools import partial
from typing import Tuple, List, Union, Dict
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

class OutcomeGenerator(ABC):

    @abstractmethod
    def generate(self, n_trials: int, n_bandits:int, seed:int) -> Tuple[np.ndarray, np.ndarray]:
        pass


class ProbabilityBlockGenerator(OutcomeGenerator):
    """Class to outcomes for a multi-armed bandit task with n_bandits arms, using
    discrete probability levels that alternate across trials."""

    def __init__(
        self,
        outcome_probability_levels: List[int] = [0.2, 0.5, 0.8],
        outcome_probability_duration: Tuple[int, int] = (20, 40),
        generate_outcomes: bool = True,
    ) -> None:
        """
        Initialise the generator, setting the probability levels and duration of each level.

        Args:
            outcome_probability_levels (List[int], optional): Outcome probability levels to alternate between. Defaults to [0.2, 0.5, 0.8].
            outcome_probability_duration (Tuple[int, int], optional): Minimum and maximum number of trials to use each outcome probability level.
            Defaults to (20, 40).
            generate_outcomes (bool, optional): Whether to generate outcomes. If False, probability levels are returned but outcomes are
            set to None. Can be used for returning continuous values that are not probabilities. Defaults to True.
        """

        self.outcome_probability_levels = outcome_probability_levels
        self.outcome_probability_duration = outcome_probability_duration
        self.generate_outcomes = generate_outcomes

    def generate(
        self, n_trials: int, n_bandits: int, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate outcomes for a multi-armed bandit task with n_bandits arms.

        Args:
            n_trials (int): Number of trials.
            n_bandits (int): Number of bandits.
            seed (int, optional): Random seed. Defaults to 42.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Array of outcomes, of shape (n_trials, n_bandits),
            and array of outcome probabilities, of shape (n_trials, n_bandits)
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
                next_breakpoint = rng.randint(*self.outcome_probability_duration)

                # Get outcome probability for this block
                outcome_probability = rng.choice(
                    [
                        x
                        for x in self.outcome_probability_levels
                        if x != outcome_probability
                    ]
                )

                # Generate an array of ones and zeros with the correct number of ones
                block_endpoint = current_breakpoint + next_breakpoint
                if block_endpoint > n_trials:
                    next_breakpoint = n_trials - current_breakpoint

                if self.generate_outcomes:
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


class RandomWalkGenerator(OutcomeGenerator):
    def __init__(
        self, mu: float = 0, sigma: float = 0.1, min_value: int = 0, max_value: int = 1, generate_outcomes: bool = True
    ) -> None:
        """
        Initialise the generator, setting the mean and standard deviation of the innovations along with the maximum
        and minimum values.

        Args:
            mu (float, optional): Mean of innovations. Defaults to 0.
            sigma (float, optional): SD of innovations. Defaults to 0.1.
            min_value (int, optional): Minimum value. Defaults to 0.
            max_value (int, optional): Maximum value. Defaults to 1.
            generate_outcomes (bool, optional): Whether to generate outcomes. If False, probability levels are returned but outcomes are
            set to None. Can be used for returning continuous values that are not probabilities. Defaults to True.
        """

        self.mu = mu
        self.sigma = sigma
        self.min_value = min_value
        self.max_value = max_value
        self.generate_outcomes = generate_outcomes

    
    def generate(
        self, n_trials: int, n_bandits: int, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates outcomes according to a random walk with Gaussian innovations.

        Args:
            n_trials (int): Number of trials.
            n_bandits (int): Number of bandits.
            seed (int, optional): Random seed. Defaults to 42.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Array of outcomes, of shape (n_trials, n_bandits),
            and array of outcome probabilities, of shape (n_trials, n_bandits)
        """

        rng = np.random.RandomState(seed)

        # Random starting value
        prob = np.ones((n_trials, n_bandits)) * rng.uniform(self.min_value, self.max_value, size=(1, n_bandits))

        # Loop through bandits
        for b in range(n_bandits):

            # Loop through trials and get next value
            for t in range(1, n_trials):

                # Gaussian innovation
                prob[t, b] = prob[t - 1, b] + rng.normal(self.mu, self.sigma)

                # Make sure values stay within bounds
                if prob[t, b] > self.max_value:
                    prob[t, b] = self.max_value
                elif prob[t, b] < self.min_value:
                    prob[t, b] = self.min_value

        # Generate outcomes based on prob
        if self.generate_outcomes:
            outcomes = rng.binomial(1, prob)
        else:
            outcomes = np.zeros_like(prob)

        return outcomes, prob


def generate_block_bandit_outcomes(
    n_trials_per_block: int,
    n_blocks: int,
    n_bandits: Union[int, Tuple[int]],
    outcome_generator: Union[str, OutcomeGenerator]='probability_block',
    seed: int = 42,
):
    """
    Generate outcomes for a multi-armed bandit task with n_bandits arms over a number of blocks.

    Args:
        n_trials_per_block (int): Number of trials per block
        n_blocks (int): Number of blocks
        n_bandits (int): Number of bandits. Can also be a tuple of any shape, in which case the the number of bandits
        will be split across entries in the tuple.
        outcome_generator (Union[str, OutcomeGenerator]): Outcome generator used to determine how outcomes
        are generated. Can be either a string (one of 'random_walk' or 'probability_block') in which outcomes
        are generated using default settings for each of these methods, or an instance of an OutcomeGenerator class.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        np.ndarray: Array of outcomes for each trial and each bandit, of shape (n_blocks, n_trials, n_bandits) if 
        n_bandits is an int, or (n_blocks, n_trials, *n_bandits) if n_bandits is a tuple.
    """

    # Set up outcome generator classes
    if isinstance(outcome_generator, str):
        if outcome_generator == "random_walk":
            outcome_generator = RandomWalkGenerator()
        elif outcome_generator == "probability_block":
            outcome_generator = ProbabilityBlockGenerator()
        else:
            raise ValueError(f"Invalid outcome generator: {outcome_generator}")

    # If n_bandits is a tuple, store its shape and set n_bandits to the product of its entries
    if isinstance(n_bandits, tuple):
        n_bandits_shape = n_bandits
        n_bandits = np.prod(n_bandits)
    else:
        n_bandits_shape = None

    block_outcomes = []
    block_trial_probability_levels = []

    # Generate seeds for each block
    rng = np.random.RandomState(seed)
    block_seeds = rng.randint(0, 100000, n_blocks)

    # Generate outcomes for each block
    for i in range(n_blocks):

        outcomes, trial_probability_levels = outcome_generator.generate(
            n_trials=n_trials_per_block,
            n_bandits=n_bandits,
            seed=block_seeds[i],
        )

        # Add to the list of outcomes
        block_outcomes.append(outcomes)
        block_trial_probability_levels.append(trial_probability_levels)

    # Stack outcomes and trial probability levels
    outcomes = np.stack(block_outcomes)
    trial_probability_levels = np.stack(block_trial_probability_levels)

    # If n_bandits is a tuple, reshape outcomes and trial probability levels
    if n_bandits_shape is not None:
        outcomes = outcomes.reshape(n_blocks, n_trials_per_block, *n_bandits_shape)
        trial_probability_levels = trial_probability_levels.reshape(
            n_blocks, n_trials_per_block, *n_bandits_shape
        )

    return outcomes, trial_probability_levels
