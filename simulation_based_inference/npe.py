import torch
import numpy as np
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE
from sbi.utils.user_input_checks import check_data_device
from sbi.utils.sbiutils import standardizing_net
import torch
from tqdm import tqdm
from torch import Tensor, nn
from typing import Optional, Tuple
from pyknos.nflows import distributions as distributions_
from pyknos.nflows import flows, transforms
from torch import Tensor, nn, tanh, tensor
from warnings import warn
from sbi.utils.sbiutils import (
    standardizing_net,
    standardizing_transform,
    z_score_parser,
)
from sbi.utils.user_input_checks import check_data_device, check_embedding_net_device
from utils import process_X, process_Y


def build_logit_maf(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    num_blocks: int = 2,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False,
    **kwargs,
) -> nn.Module:
    """Builds MAF p(x|y), including a logit transform so that parameters are estimated
    in the range [0, 1].

    Very slightly modified version of the MAF implementation in sbi.

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network, can be one of:
            - `none`, or None: do not z-score.
            - `independent`: z-score each dimension independently.
            - `structured`: treat dimensions as related, therefore compute mean and std
            over the entire batch, instead of per-dimension. Should be used when each
            sample is, for example, a time series or an image.
        z_score_y: Whether to z-score ys passing into the network, same options as
            z_score_x.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms.
        embedding_net: Optional embedding network for y.
        num_blocks: number of blocks used for residual net for context embedding.
        dropout_probability: dropout probability for regularization in residual net.
        use_batch_norm: whether to use batch norm in residual net.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.

    Returns:
        Neural network.
    """

    x_numel = batch_x[0].numel()
    # Infer the output dimensionality of the embedding_net by making a forward pass.
    check_data_device(batch_x, batch_y)
    check_embedding_net_device(embedding_net=embedding_net, datum=batch_y)
    y_numel = embedding_net(batch_y[:1]).numel()

    if x_numel == 1:
        warn("In one-dimensional output space, this flow is limited to Gaussians")

    transform_list = []

    for _ in range(num_transforms):
        block = [
            transforms.MaskedAffineAutoregressiveTransform(
                features=x_numel,
                hidden_features=hidden_features,
                context_features=y_numel,
                num_blocks=num_blocks,
                use_residual_blocks=False,
                random_mask=False,
                activation=tanh,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
            ),
            transforms.RandomPermutation(features=x_numel),
        ]
        transform_list += block

    z_score_x_bool, structured_x = z_score_parser(z_score_x)
    if z_score_x_bool:
        transform_list = [
            standardizing_transform(batch_x, structured_x)
        ] + transform_list

    z_score_y_bool, structured_y = z_score_parser(z_score_y)
    if z_score_y_bool:
        embedding_net = nn.Sequential(
            standardizing_net(batch_y, structured_y), embedding_net
        )

    # Add logit transform at the start - ensures that parameters are estimated in bounded space [0, 1]
    transform_list = [transforms.Logit()] + transform_list

    # Combine transforms.
    transform = transforms.CompositeTransform(transform_list)

    distribution = distributions_.StandardNormal((x_numel,))
    neural_net = flows.Flow(transform, distribution, embedding_net)

    return neural_net


class NPEModel:
    """
    Class for fitting a neural posterior estimator (NPE) to data.
    """

    def __init__(self, density_estimator: str = "logit_maf", choice_format:str='one_hot') -> None:
        """
        Initialises the NPE model.

        Args:
            density_estimator (str, optional): The density estimator to use. By default, uses masked autoregressive
            flows, with parameters first logit transformed to ensure estimated parameters lie in the range
            [0, 1]. Defaults to "logit_maf".
            choice_format (str, optional): The format of the choice data. One of 'one_hot' or 'numerical'. Defaults to 'one_hot'.
        """

        if density_estimator == "logit_maf":
            self.density_estimator = build_logit_maf
        else:
            self.density_estimator = density_estimator
        self.npe_posterior = None
        self.npe_inference = None
        self.choice_format = choice_format

    def preprocess_data(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> Tuple[tensor, tensor]:
        """
        Preprocesses the data for the NPE model.

        X data is expected to have either 3 or 4 dimensions. The first two dimensions represent
        the number of observations and the number of blocks, respectively. If the data has 
        3 dimensions, the data is assumed to be in numerical format (i.e., the last dimension 
        represents the index of the chosen option). If 4 dimensions, the data is assumed to be 
        in one-hot format (i.e., the last dimension represents the one-hot encoding 
        of the chosen option).
        
        The `format` argument allows for recoding of the data in the desired format. 

        Args:
            X (np.ndarray): Observed behaviour (e.g., choices). The first two dimensions represent
            the number of observations and the number of blocks, respectively. The final dimensions
            represent choices, and can be either 3D (numerical format) or 4D (one-hot format).
            y (np.ndarray): True parameter values, as a 2D array with shape (n_subjects, n_params).

        Returns:
            Tuple[tensor, tensor]: Processed X and y data
        """

        self.X = process_X(X, format=self.choice_format)
        self.y = process_Y(y)

        return self.X, self.y

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 0) -> None:
        """
        Fits the NPE estimator to the data.

        X data is expected to have either 3 or 4 dimensions. The first two dimensions represent
        the number of observations and the number of blocks, respectively. If the data has 
        3 dimensions, the data is assumed to be in numerical format (i.e., the last dimension 
        represents the index of the chosen option). If 4 dimensions, the data is assumed to be 
        in one-hot format (i.e., the last dimension represents the one-hot encoding 
        of the chosen option).
        seed (int, optional): Random seed. Defaults to 0.
        
        The `format` argument allows for recoding of the data in the desired format. 

        Args:
            X (np.ndarray): Observed behaviour (e.g., choices). The first two dimensions represent
            the number of observations and the number of blocks, respectively. The final dimensions
            represent choices, and can be either 3D (numerical format) or 4D (one-hot format).
            y (np.ndarray): True parameter values, as a 2D array with shape (n_subjects, n_params)

        """

        torch.manual_seed(seed);

        self.n_params = y.shape[1]

        # Box prior - parameters must lie in [0, 1]
        prior = utils.BoxUniform(
            low=torch.zeros(self.n_params), high=torch.ones(self.n_params)
        )

        # Set up data
        simulations, params = self.preprocess_data(X, y)

        # Train NPE
        self.npe_inference = SNPE(prior=prior, density_estimator=self.density_estimator)
        self.npe_trained = self.npe_inference.append_simulations(
            params, simulations, exclude_invalid_x=False
        ).train()

    def sample(
        self, X: np.ndarray, n_samples: int = 1000, progress_bar: bool = True, seed: int = 0
    ) -> np.ndarray:
        """
        Sample from the estimated posterior.

        Args:
            X (np.ndarray): Observed behaviour (e.g., choices). The first two dimensions represent
            the number of observations and the number of blocks, respectively. The final dimensions
            represent choices, and can be either 3D (numerical format) or 4D (one-hot format).
            n_samples (int, optional): Number of samples to draw. Defaults to 1000.
            progress_bar (bool): Whether to show a progress bar
            seed (int, optional): Random seed. Defaults to 0.

        Returns:
            np.ndarray: Samples from the posterior. Array of shape (num_observations, n_samples, n_params)
        """

        torch.manual_seed(seed);

        X = process_X(X, format=self.choice_format)

        # Box prior - parameters must lie in [0, 1]
        prior = utils.BoxUniform(
            low=torch.zeros(self.n_params), high=torch.ones(self.n_params)
        )

        # Build posterior
        self.npe_posterior = self.npe_inference.build_posterior(self.npe_trained, prior)

        # Sample from the posterior
        all_param_samples = []

        if progress_bar:
            pbar = tqdm(range(X.shape[0]))
        else:
            pbar = range(X.shape[0])

        for i in pbar:
            observation = np.array(X.squeeze()[i, ...]).flatten()

            # Sample params
            param_samples = self.npe_posterior.sample(
                (n_samples,), x=observation, show_progress_bars=False
            )
            all_param_samples.append(param_samples)

        # Combine samples into a single array
        all_param_samples = np.stack(all_param_samples)

        # Swap the first and second dimensions
        all_param_samples = np.swapaxes(all_param_samples, 0, 1)

        return all_param_samples


