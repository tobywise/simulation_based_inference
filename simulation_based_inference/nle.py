from typing import Optional, Tuple
import torch
from torch import tensor, Tensor, nn, unique
from torch.nn import Softmax, Sigmoid
from torch.distributions import Categorical
import numpy as np
from sbi.utils.sbiutils import standardizing_net
from sbi.utils.user_input_checks import check_data_device
from .utils import process_X, process_Y_outcomes
from sbi import utils as utils
from sbi.inference import SNLE
from tqdm import tqdm

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


class NLEModel:
    """
    Class for fitting a neural likelihood estimator (NLE) to data.
    """

    def __init__(self, density_estimator: str = "categorical") -> None:
        """
        Initalises the NLE model.

        Args:
            density_estimator (str, optional): Density estimator to use. By default uses a
            neural network designed to estimate categorical distributions (e.g., a distribution
            representing choice likelihood). Defaults to 'categorical'.
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
