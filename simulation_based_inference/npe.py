import torch
import numpy as np
import pandas as pd
from simulation_based_inference.utils import (
    numerical_encode_choices,
    one_hot_encode_choices,
)
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE, SNLE
from sbi.utils.user_input_checks import check_data_device
from sbi.utils.sbiutils import standardizing_net
import torch
from tqdm import tqdm
from torch import Tensor, nn, unique
from typing import Optional, Tuple
from torch.distributions import Categorical
from torch.nn import Sigmoid, Softmax
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


def process_X(X: np.ndarray, format: str = "one_hot") -> tensor:
    """
    Process X (choice data) to be in the right format for NPE

    Args:
        X (np.ndarray): Array of shape (num_observations, ...)
        format (str): Format of X. One of 'one_hot' or 'numerical'

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


class ConditionalCategoricalNet(nn.Module):
    """
    Class to perform conditional density (mass) estimation for a categorical RV.
    Takes as input parameters theta and learns the parameters p of a Categorical.
    Defines log prob and sample functions.

    Modified from sbi's CategoricalNet class but designed to be used 1) on its own,
    rather than with MNLE, and 2) with information about the task, which the likelihood
    is conditional on.
    """

    def __init__(
        self,
        num_input: int = 4,
        num_categories: int = 2,
        num_trials: int = 1,
        num_hidden: int = 20,
        num_layers: int = 2,
        embedding: Optional[nn.Module] = None,
    ):
        """Initialize the neural net.
        Args:
            num_input: number of input units, i.e., dimensionality of parameters.
            num_categories: number of output units, i.e., number of categories.
            num_hidden: number of hidden units per layer.
            num_layers: number of hidden layers.
            embedding: emebedding net for parameters, e.g., a z-scoring transform.
        """
        super(ConditionalCategoricalNet, self).__init__()

        self.num_hidden = num_hidden
        self.num_input = num_input
        self.activation = Sigmoid()
        self.softmax = Softmax(dim=1)
        self.num_categories = num_categories

        # Maybe add z-score embedding for parameters.
        if embedding is not None:
            self.input_layer = nn.Sequential(
                embedding, nn.Linear(num_input, num_hidden)
            )
        else:
            self.input_layer = nn.Linear(num_input, num_hidden)

        # Repeat hidden units hidden layers times.
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(num_hidden, num_hidden))

        self.output_layer = nn.Linear(num_hidden, (num_trials * num_categories))

    def forward(self, theta: Tensor) -> Tensor:
        """Return categorical probability predicted from a batch of parameters.
        Args:
            theta: batch of input parameters for the net.
        Returns:
            Tensor: batch of predicted categorical probabilities.
        """

        assert theta.dim() == 2, "input needs to have a batch dimension."
        assert (
            theta.shape[1] == self.num_input
        ), f"input dimensions must match num_input {self.num_input}"

        # forward path
        theta = self.activation(self.input_layer(theta))

        # iterate n hidden layers, input x and calculate tanh activation
        for layer in self.hidden_layers:
            theta = self.activation(layer(theta))

        out = self.output_layer(theta)
        out = out.reshape(theta.shape[0], -1, self.num_categories)
        out = self.softmax(out)

        return out

    def log_prob(self, x: Tensor, context: Tensor) -> Tensor:
        """Return categorical log probability of categories x, given parameters theta.
        Args:
            theta: parameters.
            x: categories to evaluate.
        Returns:
            Tensor: log probs with shape (x.shape[0],)
        """
        # Predict categorical ps and evaluate.
        ps = self.forward(context)
        return Categorical(probs=ps).log_prob(x.squeeze())

    def sample(self, num_samples: int, theta: Tensor) -> Tensor:
        """Returns samples from categorical random variable with probs predicted from
        the neural net.
        Args:
            theta: batch of parameters for prediction.
            num_samples: number of samples to obtain.
        Returns:
            Tensor: Samples with shape (num_samples, 1)
        """

        # Predict Categorical ps and sample.
        ps = self.forward(theta)
        return Categorical(probs=ps).sample(torch.Size((num_samples,)))


def build_categorical_net(
    batch_y: Tensor,
    batch_x: Tensor,
    z_score_x: Optional[str] = "independent",
    z_score_y: Optional[str] = "independent",
    num_transforms: int = 2,
    num_bins: int = 5,
    hidden_features: int = 128,
    hidden_layers: int = 3,
    tail_bound: float = 10.0,
    log_transform_x: bool = True,
    **kwargs,
):
    """Returns a density estimator for categorical data.
    Uses a categorical net to model discrete data.

    Modified from sbi.

    Args:
        batch_x: batch of data
        batch_y: batch of parameters
        z_score_x: whether to z-score x.
        z_score_y: whether to z-score y.
        num_transforms: number of transforms in the NSF
        num_bins: bins per spline for NSF.
        hidden_features: number of hidden features used in both nets.
        hidden_layers: number of hidden layers in the categorical net.
        tail_bound: spline tail bound for NSF.
        log_transform_x: whether to apply a log-transform to x to move it to unbounded
            space, e.g., in case x consists of reaction time data (bounded by zero).
    Returns:
        MixedDensityEstimator: nn.Module for performing MNLE.
    """

    check_data_device(batch_x, batch_y)
    if z_score_y == "independent":
        embedding = standardizing_net(batch_y)
    else:
        embedding = None

    # Infer input and output dims.
    dim_parameters = batch_y[0].numel()
    num_categories = unique(batch_x).numel()
    num_trials = batch_x.shape[1]

    # Set up a categorical RV neural net for modelling the discrete data.
    disc_nle = ConditionalCategoricalNet(
        num_input=dim_parameters,
        num_categories=num_categories,
        num_trials=num_trials,
        num_hidden=hidden_features,
        num_layers=hidden_layers,
        embedding=embedding,
    )

    return disc_nle


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


class NLEModel:
    """
    Class for fitting a neural likelihood estimator (NLE) to data.
    """

    def __init__(self, density_estimator: str = "categorical") -> None:
        """
        Initalises the NLE model.

        Args:
            density_estimator (str, optional): Density estimator to use. By default uses a
            neural network designed to estimate categorical distributions. Defaults to 'categorical'.
        """

        if density_estimator == "categorical":
            self.density_estimator = build_categorical_net
        else:
            self.density_estimator = density_estimator
        self.nle_posterior = None
        self.nle_inference = None

    def preprocess_data(
        self, X: np.ndarray, y: np.ndarray, outcomes: np.ndarray
    ) -> Tuple[tensor, tensor]:
        """
        Preprocesses the data to be in the right format for NLE.

        Args:
            X (np.ndarray): Observed behaviour (e.g., choices). First dimension must have as many entries
            as there are subjects, but the remaining dimensions can be arbitrary.
            y (np.ndarray): True parameter values, as a 2D array with shape (n_subjects, n_params)
            outcomes (np.ndarray): Array of any shape, representing task outcomes.

        Returns:
            Tuple[tensor, tensor]: Processed X and y data
        """

        self.X = process_X(X, format="numerical")
        self.y = process_Y_outcomes(y, outcomes)

        return self.X, self.y

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        outcomes: np.ndarray,
        train_kwargs: dict = {},
    ) -> None:
        """
        Fits the NLE estimator to the data.

        Note: Assumes task outcomes are the same for all subjects.

        Args:
            X (np.ndarray): Observed behaviour (e.g., choices). First dimension must have as many entries
            as there are subjects, but the remaining dimensions can be arbitrary.
            y (np.ndarray): True parameter values, as a 2D array with shape (n_subjects, n_params).
            outcomes (np.ndarray): Array of any shape, representing task outcomes.

        """

        self.n_params = y.shape[1]  

        # Set a box prior to constrain parameters to be between 0 and 1
        prior = utils.BoxUniform(
            low=torch.zeros(self.n_params), high=torch.ones(self.n_params)
        )

        # Set up data
        simulations, params_outcomes = self.preprocess_data(X, y, outcomes)

        # Train NLE
        self.nle_inference = SNLE(prior=prior, density_estimator=build_categorical_net)
        self.nle_trained = self.nle_inference.append_simulations(
            params_outcomes, simulations
        ).train(**train_kwargs)

    def sample(
        self, y: np.ndarray, outcomes: np.ndarray, n_samples: int = 1000
    ) -> np.ndarray:
        """
        Samples from the trained model (generating a sample of choices for each subject), given
        a set of parameter values and outcomes.

        Args:
            y (np.ndarray): Parameter values, as a 2D array with shape (n_subjects, n_params)
            outcomes (np.ndarray): Outcomes, of any shape.
            n_samples (int, optional): _description_. Defaults to 1000.

        Returns:
            np.ndarray: Samples from the trained model, as an array with shape (n_samples, n_subjects, n_blocks, n_trials)
        """

        y_outcomes = process_Y_outcomes(y, outcomes)

        nle_samples = self.nle_trained.sample(n_samples, y_outcomes)

        return nle_samples

    def log_prob(
        self, X: np.ndarray, y: np.ndarray, outcomes: np.ndarray
    ) -> np.ndarray:
        """
        Computes the log probability of the choice data (X) under the fitted NLE model, given
        samples from the posterior over parameters (y) and outcomes.

        Args:
            X (np.ndarray): Observed choices, as a 3D array with shape (n_subjects, n_blocks, n_trials, n_options)
            y (np.ndarray): Samples from the posterior over parameters, as a 2D array with shape (n_samples, n_params)
            outcomes (np.ndarray): Task outcomes

        Returns:
            np.ndarray: Log probability of observations under the model. Array of shape
            (n_samples, n_subjects, n_blocks, n_trials)
        """

        X = process_X(X, format="numerical")

        n_samples = y.shape[0]

        ll_samples = []

        for i in tqdm(range(n_samples)):

            sample_y_outcomes = process_Y_outcomes(y[i, ...], outcomes)
            ll = self.nle_trained.log_prob(X, context=sample_y_outcomes)
            ll_samples.append(ll.detach().numpy())

        ll = np.stack(ll_samples)

        return ll

    def elpd(self, X: np.ndarray, y: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
        """
        Computes the expected log predictive density of the choice data (X) under the fitted NLE model, given
        samples from the posterior over parameters (y) and outcomes.

        Args:
            X (np.ndarray): Observed choices, as a 3D array with shape (n_subjects, n_blocks, n_trials, n_options)
            y (np.ndarray): Samples from the posterior over parameters, as a 2D array with shape (n_samples, n_params)
            outcomes (np.ndarray): Task outcomes

        Returns:
            np.ndarray: ELPD of observations under the model
        """

        ll = self.log_prob(X, y, outcomes)
        ps = np.exp(ll)

        elpd = np.log(np.sum(ps, axis=0))  # sum across samples

        return elpd


class NPEModel:
    """
    Class for fitting a neural posterior estimator (NPE) to data.
    """

    def __init__(self, density_estimator: str = "logit_maf") -> None:
        """
        Initialises the NPE model.

        Args:
            density_estimator (str, optional): The density estimator to use. By default, uses masked autoregressive
            flows, with parameters first logit transformed to ensure estimated parameters lie in the range
            [0, 1]. Defaults to "logit_maf".
        """

        if density_estimator == "logit_maf":
            self.density_estimator = build_logit_maf
        else:
            self.density_estimator = density_estimator
        self.npe_posterior = None
        self.npe_inference = None

    def preprocess_data(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> Tuple[tensor, tensor]:
        """
        Preprocesses the data for the NPE model.

        Args:
            X (np.ndarray): Observed behaviour (e.g., choices). First dimension must have as many entries
            as there are subjects, but the remaining dimensions can be arbitrary.
            y (np.ndarray): True parameter values, as a 2D array with shape (n_subjects, n_params).

        Returns:
            Tuple[tensor, tensor]: Processed X and y data
        """

        self.X = process_X(X)
        self.y = process_Y(y)

        return self.X, self.y

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the NPE estimator to the data.

        Args:
            X (np.ndarray): Observed behaviour (e.g., choices). First dimension must have as many entries
            as there are subjects, but the remaining dimensions can be arbitrary.
            y (np.ndarray): True parameter values, as a 2D array with shape (n_subjects, n_params)

        """

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
            params, simulations
        ).train()

    def sample(
        self, X: np.ndarray, n_samples: int = 1000, progress_bar: bool = True
    ) -> np.ndarray:
        """
        Sample from the estimated posterior.

        Args:
            X (np.ndarray): Array of shape (num_observations, ...). Each observation must have the same
            shape as the X data used to fit the model.
            n_samples (int, optional): Number of samples to draw. Defaults to 1000.
            progress_bar (bool): Whether to show a progress bar

        Returns:
            np.ndarray: Samples from the posterior. Array of shape (num_observations, n_samples, n_params)
        """

        X = process_X(X)

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
) -> pd.DataFrame:
    """
    Runs cross-validation for a single block.

    Args:
        block (int): Block to leave out.
        train_X (np.ndarray): Array of simulated choices used as training data for the models, with shape
        (n_subjects, n_blocks, n_trials, n_options)
        train_y (np.ndarray): Array of simulated parameter values used as training data for the models, with
        shape (n_subjects, n_params).
        outcomes (np.ndarray): Array of task outcomes, first axis should be blocks.
        observed_X (np.ndarray): Array of observed choices, with shape
        (n_subjects, n_blocks, n_trials, n_options)

    Returns:
        pd.DataFrame: Dataframe containing ELPD sum and standard error for the left out block.
    """

    n_blocks = train_X.shape[1]

    cv_output = {}

    print("Block {0} of {1}".format(block, n_blocks))

    # Get the blocks to train on - all blocks except the current one
    train_blocks = [i for i in range(n_blocks) if i != block]
    print("Training on blocks: {}".format(train_blocks))

    # Train the NPE model on data for this block
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
    Performs cross-validation across task blocks as a measure of model fit.

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

    n_blocks = train_X.shape[1]

    cv_output = []

    for block in range(n_blocks):
        cv_output.append(cv_block(block, train_X, train_y, outcomes, observed_X))

    return pd.concat(cv_output)
