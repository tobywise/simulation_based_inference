from typing import Tuple
import numpy as np
from npe import NPEModel
from nle import NLEModel
from utils import process_X
import pandas as pd

def elpd_summary_stats(
    model: NLEModel, X: np.ndarray, y: np.ndarray, outcomes: np.ndarray
) -> Tuple[float, float]:
    """
    Computes the sum and standard error of the expected log predictive density of the choice data (X) under the fitted NLE model, given
    samples from the posterior over parameters (y) and outcomes.

    Args:
        model (NLEModel): A trained likelihood estimator.
        X (np.ndarray): Observed choices, as a 3D array with shape (n_subjects, n_blocks, n_trials, n_options)
        y (np.ndarray): Samples from the posterior over parameters, as a 2D array with shape (n_samples, n_params)
        outcomes (np.ndarray): Task outcomes

    Returns:
        Tuple[float, float]: The sum and standard error of the ELPD of observations under the model
    """

    elpd = model.elpd(X, y, outcomes)

    elpd_sum = np.sum(elpd)
    elpd_mean = np.mean(elpd)
    elpd_se = np.sqrt(
        np.var(elpd.flatten())
    )  

    return elpd_sum, elpd_mean, elpd_se


def cv_block(
    block: int,
    train_X: np.ndarray,
    train_y: np.ndarray,
    outcomes: np.ndarray,
    observed_X: np.ndarray,
    npe_model: NPEModel = None,
) -> pd.DataFrame:
    """
    Runs cross-validation for a single block, estimating the Expected Log Predictive Density (ELPD) on the 
    held-out block based on the model parameters estimated on the remaining blocks.

    This function involves the following steps:
    1. Use neural posterior estimation (NPE) to estimate parameter values for each subject on the training
    blocks. This involves training the NPE model on the training blocks, and then sampling from the posterior
    over parameters for the observed data within this block.
    2. Use neural likelihood estimation (NLE) to infer the likelihood of the observed choices on the held-out
    block, given the parameter values estimated in step 1. This involves training the NLE model on the held out
    block, and then computing the ELPD of the observed choices on this block, given the parameter values 
    estimated in step 1.
    3. Compute the sum and standard error of the ELPD for the held-out block.

    This function should be run iteratively to gain an estimate of the ELPD for each block, which can then be
    used to compute the cross-validated ELPD across blocks.

    Args:
        block (int): Block to leave out.
        train_X (np.ndarray): Array of simulated choices used as training data for the models, with 4D shape
        (n_subjects, n_blocks, n_trials, n_options) or 3D shape (n_subjects, n_blocks, n_trials). 
        train_y (np.ndarray): Array of simulated parameter values used as training data for the models, with
        shape (n_subjects, n_params).
        outcomes (np.ndarray): Array of task outcomes, first axis should be blocks.
        observed_X (np.ndarray): Array of observed choices, with shape
        (n_subjects, n_blocks, n_trials, n_options)
        npe_model (NPEModel, optional): An untrained instance of NPEModel. If None, the model will use
        default parameters. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe containing ELPD sum and standard error for the left out block.
    """

    # Make sure X is in the right format
    train_X = process_X(train_X, format='one_hot')
    observed_X = process_X(observed_X, format='one_hot')

    # Get the number of blocks
    n_blocks = train_X.shape[1]

    cv_output = {}

    print("Block {0} of {1}".format(block, n_blocks))

    # Get the blocks to train on - all blocks except the current one
    train_blocks = [i for i in range(n_blocks) if i != block]
    print("Training on blocks: {}".format(train_blocks))

    # Train the NPE model on data for this block
    if npe_model is None:
        npe_model = NPEModel()
    npe_model.fit(train_X[:, train_blocks, ...], train_y)

    # Sample from the posterior given observed behaviour
    npe_samples = npe_model.sample(observed_X[:, train_blocks, ...])

    # Fit the NLE model for the held out block
    nle_model = NLEModel()
    nle_model.fit(
        train_X[:, None, block, ...],
        train_y,
        outcomes[None, block, ...],
    )

    # Compute the ELPD
    elpd_sum, elpd_mean, elpd_se = elpd_summary_stats(
        nle_model,
        observed_X[:, None, block, ...],
        npe_samples,
        outcomes[None, block, ...],
    )

    # Collate the results
    cv_output["test_block"] = block
    cv_output["elpd_sum"] = elpd_sum
    cv_output["elpd_mean"] = elpd_mean
    cv_output["elpd_se"] = elpd_se

    # Convert cv output to a dataframe
    cv_output = pd.DataFrame(cv_output, index=[0])

    return cv_output


def blockwise_cv(
    train_X: np.ndarray,
    train_y: np.ndarray,
    outcomes: np.ndarray,
    observed_X: np.ndarray,
) -> pd.DataFrame:
    """
    Performs cross-validation across task blocks as a measure of model fit. This iteratively runs
    `cv_block` for each block, and returns the sum and standard error of the ELPD for each block
    as a dataframe.

    Args:
        train_X (np.ndarray): Array of simulated choices used as training data for the models, with shape
        (n_subjects, n_blocks, n_trials, n_options)
        train_y (np.ndarray): Array of simulated parameter values used as training data for the models, with
        shape (n_subjects, n_params).
        outcomes (np.ndarray): Array of task outcomes, first axis should be blocks.
        observed_X (np.ndarray): Array of observed choices, with shape
        (n_subjects, n_blocks, n_trials, n_options)

    Returns:
        pd.DataFrame: Dataframe containing the sum and standard error of the ELPD for each block.

    """

    # Make sure X is in the right format
    train_X = process_X(train_X, format='one_hot')
    observed_X = process_X(observed_X, format='one_hot')

    # Get the number of blocks
    n_blocks = train_X.shape[1]

    # Iterate over blocks and get the ELPD for each
    cv_output = []

    for block in range(n_blocks):
        cv_output.append(cv_block(block, train_X, train_y, outcomes, observed_X))

    return pd.concat(cv_output)
