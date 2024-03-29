from typing import List
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from torch import tensor

def process_X(X: np.ndarray, format: str = "one_hot") -> tensor:
    """
    Process X (choice data) to be in the right format for NPE. 

    X data is expected to have either 3 or 4 dimensions. The first two dimensions represent
    the number of observations and the number of blocks, respectively. If the data has 
    3 dimensions, the data is assumed to be in numerical format (i.e., the last dimension 
    represents the index of the chosen option). If 4 dimensions, the data is assumed to be 
    in one-hot format (i.e., the last dimension represents the one-hot encoding 
    of the chosen option).
    
    The `format` argument allows for recoding of the data in the desired format. 

    Args:
        X (np.ndarray): Array of shape (num_observations, n_blocks, ...)
        format (str): Desired format of X. One of 'one_hot' or 'numerical'.

    Returns:
        X (torch.tensor): Tensor of shape (num_observations, num_features)

    """

    assert X.ndim in [
        3,
        4,
    ], "X must have either 3 or 4 dimensions, but has {} dimensions".format(X.ndim)

    if format == "one_hot" and X.ndim == 3:
        X = one_hot_encode_choices(X)
    elif format == "numerical" and X.ndim == 4:
        X = numerical_encode_choices(X)

    X = np.array(X.squeeze().reshape((X.shape[0], -1))).astype(np.float32)
    X = torch.from_numpy(X)
    return X


def process_Y(y: np.ndarray) -> tensor:
    """
    Process y (parameter data) to be in the right format for NPE

    Args:
        y (np.ndarray): Array of shape (num_observations, n_params)

    Returns:
        y (torch.tensor): Tensor of shape (num_observations, n_params)

    """
    y = torch.from_numpy(y.astype(np.float32))
    return y


def process_Y_outcomes(y: np.ndarray, outcomes: np.ndarray) -> tensor:
    """
    Process y (parameter data) to be in the right format for NLE by concatenating outcomes

    Args:
        y (np.ndarray): Array of shape (num_observations, n_params)
        outcomes (np.ndarray): Array of any shape.

    Returns:
        y_outcomes (torch.tensor): Tensor of shape (num_observations, n_params + n_outcomes), where
        n_outcomes is the shape of the flattened outcome array

    """

    outcomes = np.array(outcomes.squeeze().reshape((outcomes.shape[0], -1))).astype(
        np.float32
    )

    y_outcomes = torch.from_numpy(
        np.hstack([y, outcomes.flatten()[None, :].repeat(y.shape[0], axis=0)]).astype(
            np.float32
        )
    )

    return y_outcomes

def summary_df(samples:np.ndarray, param_names:List[str], hdpi_level:float=95) -> pd.DataFrame:
    """
    Creates a summary dataframe of the posterior distribution.

    Args:
        samples (np.ndarray): Samples from the posterior distribution, shape (n_obs, n_samples, n_params).
        param_names (List[str]): List of parameter names.
        hdpi_level (float, optional): Level of HDPI to calculate. Defaults to 95.

    Returns:
        pd.DataFrame: Summary dataframe, with columns for mean, std, variance, HDPI lower, HDPI upper.
    """

    # Calculate HDPI levels
    hdpi_lower_level = (100 - hdpi_level) / 2
    hdpi_upper_level = 100 - hdpi_lower_level

    # Convert to pandas dataframe 
    df_data = {
        'param': [],
        'name': [],
        'mean': [],
        'std': [],
        'var': [],
        'hdpi_{0}'.format(hdpi_lower_level): [],
        'hdpi_{0}'.format(hdpi_upper_level): [],
    }

    for n, param in enumerate(param_names):
        param_samples = samples[..., n]

        # Calculate mean
        mean = param_samples.mean(axis=1)

        # Standard deviation and variance
        std = param_samples.std(axis=1)
        var = param_samples.var(axis=1)

        # Calculate HDPI
        hdpi = np.percentile(param_samples, [hdpi_lower_level, hdpi_upper_level], axis=1)
        hdpi_lower = hdpi[0]
        hdpi_upper = hdpi[1]

        # Get list of of param names for each observation
        param_names = ['{0}__[{1}]'.format(param, i) for i in range(param_samples.shape[0])]

        # Add to dataframe
        df_data['param'] += [param] * param_samples.shape[0]
        df_data['name'] += param_names
        df_data['mean'] += mean.tolist()
        df_data['std'] += std.tolist()
        df_data['var'] += var.tolist()
        df_data['hdpi_{0}'.format(hdpi_lower_level)] += hdpi_lower.tolist()
        df_data['hdpi_{0}'.format(hdpi_upper_level)] += hdpi_upper.tolist()

    # Convert to dataframe
    df = pd.DataFrame(df_data)

    return df


def one_hot_encode_choices(choices:np.ndarray, n_options:int) -> np.ndarray:
    """
    Converts a vector of choices to a one-hot encoding.

    Args:
        choices (np.ndarray): Vector of choices, shape = (n_observations, n_blocks, n_trials)
        n_options (int): Number of options.

    Returns:
        np.ndarray: One-hot encoding of choices, shape = (n_observations, n_blocks, n_trials, n_options)
    """

    choices_binary = np.zeros(choices.shape + (n_options, ))
    choices_binary[np.arange(choices.shape[0])[:, None], np.arange(choices.shape[1]), choices] = 1

    return choices_binary


def numerical_encode_choices(choices:np.ndarray) -> np.ndarray:
    """
    Converts a one-hot encoding to an array of choices in numerical encoding.

    Args:
        choices_binary (np.ndarray): One-hot encoding of choices, shape = (n_observations, n_blocks, n_trials, n_options)

    Returns:
        np.ndarray: Array of choices, shape = (n_observations, n_blocks, n_trials)
    """

    choices = np.argmax(choices, axis=-1)

    return choices


def plot_recovery(true: np.ndarray, estimated:np.ndarray, epoch:int, save_path:str=None, param_names:List[str]=None):
    """
    Plots recovered parameter values against true ones.

    Args:
        true (np.ndarray): True parameter values, shape (n_observations, n_params).
        estimated (np.ndarray): Estimated parameter values, shape (n_observations, n_params).
        epoch (int): Epoch number.
        save_path (str): Path to save the plot to.

    """

    f, ax = plt.subplots(1, true.shape[1], figsize=(2.333 * true.shape[1], 2.8))

    for i in range(true.shape[1]):
        ax[i].scatter(true[:, i], estimated[:, i])

        if i == 0:
            ax[i].set_ylabel("Estimated")

        ax[i].set_xlabel("True")

        # make the title of the subplot the pearson correlation
        if param_names is not None:
            param_name_prefix = param_names[i] + '\n' 
        else:
            param_name_prefix = ''
        ax[i].set_title(
            param_name_prefix + 
            "r = {}".format(
                np.round(np.corrcoef(true[:, i], estimated[:, i])[0, 1], 2)
            )
        )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "recovery_epoch_{0}.png".format(epoch)))



def pp_plot(true: np.ndarray, samples:np.ndarray, epoch:int, save_path:str=None, param_names:List[str]=None):
    """
    Probability-probability plot.

    Args:
        true (np.ndarray): True parameter values, shape (n_observations, n_params).
        samples (np.ndarray): Samples from posterior, shape (n_observations, n_samples, n_params).
        epoch (int): Epoch number.
        save_path (str): Path to save the plot to.
    """

    n_params = true.shape[1]

    f, ax = plt.subplots(1, n_params, figsize=(2.333 * true.shape[1], 2.5))

    for param in range(n_params):

        if param == 0:
            ax[param].set_ylabel("Proportion of samples\nin CI")

        ax[param].set_xlabel("CI probability")

        ps = []
        
        n_samples = samples.shape[1]

        for i in range(samples.shape[0]):

            obs_samples = samples[i, :, param]
            true_value = true[i, param]
            ps.append(np.sum(obs_samples > true_value) / float(n_samples))

        ax[param].plot(np.linspace(0, 1, len(ps)), np.sort(ps))
        ax[param].plot([0, 1], [0, 1], color='black', linestyle='--')

        if param_names is not None:
            ax[param].set_title(param_names[param])

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "pp_epoch_{0}.png".format(epoch)))


