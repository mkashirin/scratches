from typing import Optional

import numpy as np

from ._base import BaseLayer
from ..operators._base import BaseOperator
from ..operators.concrete import (
    WeightedMultiplicationOperator,
    BiasAdditionOperator,
    DropoutOperator,
    ConvolutionOperator,
    FlattenOperator,
)
from ...._typing import WeightsOption


class DenseLayer(BaseLayer):
    """Class for defining a dense (fully connected) layer in a 
    neural network.
    """

    def __init__(
        self,
        n_neurons,
        activation_function: BaseOperator,
        weight_initialization: WeightsOption = "standard",
        dropout_rate: float = 1.0,
        random_seed: Optional[int] = None,
    ):
        """Initialize the DenseLayer with the number of neurons and an
        activation function.
        """
        super().__init__(n_neurons, weight_initialization, random_seed)
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

    def _setup_layer(self, n_features: int) -> None:
        """Set up the dense layer with the specified input shape and 
        initialize parameters and operators.
        """
        self._set_parameters(n_features)

        self.operators = {
            "weighted_multiplier": WeightedMultiplicationOperator(
                self.parameters["weights"]
            ),
            "bias_adder": BiasAdditionOperator(self.parameters["bias"]),
            "activation_function": self.activation_function,
        }
        if 0 < self.dropout_rate < 1:
            self.operators["dropout"] = DropoutOperator(self.dropout_rate)

    def _set_parameters(self, n_features: int) -> None:
        self.parameters = dict()

        scale = super()._get_scale(n_features)
        self.parameters["weights"] = np.random.normal(
            loc=0, scale=scale, size=(n_features, self.n_neurons)
        )
        self.parameters["bias"] = np.random.normal(
            loc=0, scale=scale, size=(1, self.n_neurons)
        )


class ConvolutionalLayer(BaseLayer):
    """Class for defining a convolutional layer 
    (especially good for computer vision) in a neural network.
    """

    def __init__(
        self,
        output_channels: int,
        parameter_size: int,
        activation_function: BaseOperator,
        weight_initialization: WeightsOption = "standard",
        apply_flatten: bool = False,
        dropout_rate: float = 1.0,
        random_seed: Optional[int] = None,
    ):
        """Initialize the ConvolutionalLayer with the essential parameters.
        """
        super().__init__(output_channels, weight_initialization, random_seed)
        self.parameter_size = parameter_size
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate
        self.apply_flatten = apply_flatten

    def _setup_layer(self, n_features: int) -> None:
        """Set up the conolutional layer such that the `n_features` argument
        represents the number of channels used for the convolutions.
        """
        self._set_parameters(n_features)

        self.operators = {
            "convolver": ConvolutionOperator(self.parameters["weights"]),
            "activation_function": self.activation_function,
        }
        if self.apply_flatten:
            self.operators["flattener"] = FlattenOperator()
        if 0 < self.dropout_rate < 1:
            self.operators["dropout"] = DropoutOperator(self.dropout_rate)

    def _set_parameters(self, input_channels: int) -> None:
        self.parameters = dict()

        scale = super()._get_scale(input_channels)
        weights = np.random.normal(
            # fmt: off
            loc=0, scale=scale,
            size=(
                input_channels, self.n_neurons,
                self.parameter_size, self.parameter_size,
            ),
            # fmt: on
        )
        self.parameters["weights"] = weights
