# pyright: reportAttributeAccessIssue = false

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

from numpy import ndarray
from numpy.random import seed

from ..operators._base import BaseOperator, ParameterizedOperator
from ...._typing import ArraysMap, WeightsOption


class BaseLayer(ABC):
    """Abstract base class for defining neural network layers."""

    def __init__(
        self,
        n_neurons: int,
        weight_initialization: WeightsOption,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the :class:`BaseLayer` with the number of neurons and
        default values for instance variables.
        """
        self.n_neurons = n_neurons
        self.root = True
        self.weight_initialization = weight_initialization
        self.random_seed = random_seed
        self.parameters: ArraysMap
        self.parameters_gradients: ArraysMap
        self.operators: Dict[str, Union[BaseOperator, ParameterizedOperator]]

        if self.random_seed is not None:
            seed(self.random_seed)

    @abstractmethod
    def _setup_layer(self, n_features: int) -> None:
        """Abstract method to set up the layer with the specified number
        of neurons.
        """
        message = "Every layer should implement the `_setup_layer()` method."
        raise NotImplementedError(message)

    def feed_forward(self, input_: ndarray) -> ndarray:
        """Passes the input forward through the layer and returns the
        output.

        :parameter input_: The input to the layer.
            :type input_: :class:`ndarray`

        :returns: The output of the layer.
            :rtype: :class:`ndarray`
        """
        if self.root:
            self._setup_layer(input_.shape[1])
            self.root = False

        output = input_
        for operator in self.operators.values():
            output = operator.feed_forward(output)

        return output

    def propagate_backwards(self, output_gradients: ndarray) -> ndarray:
        """Propagates the output gradients backward through the layer and
        returns the input gradients.

        :parameter output_gradients: The gradients of the output.
            :type output_gradients: :class:`ndarray`

        :returns: The gradients of the input.
            :rtype: :class:`ndarray`
        """
        input_gradients = output_gradients
        for operator in reversed(self.operators.values()):
            input_gradients = operator.propagate_backwards(input_gradients)

        self._compute_parameterized_gradients()

        return input_gradients

    def _compute_parameterized_gradients(self) -> None:
        """Computes the gradients of the parameterized operators in the
        layer.
        """
        self.parameters_gradients = dict()
        for key, operator in zip(
            self.operators.keys(), self.operators.values()
        ):
            if issubclass(type(operator), ParameterizedOperator):
                self.parameters_gradients[key] = (
                    operator.parameterized_gradient  # pyright: ignore[reportAttributeAccessIssue]
                )

    def _get_parameters(self) -> None:
        """Retrieves the parameters from the parameterized operators in the
        layer.
        """
        self.parameters = dict()
        for key, operator in zip(
            self.operators.keys(), self.operators.values()
        ):
            if issubclass(type(operator), ParameterizedOperator):
                self.parameters[key] = operator.parameter

    def _get_scale(self, n_features: int) -> float:
        match self.weight_initialization:
            case "standard":
                scale = 1
            case "Glorot":
                scale = 2 / (n_features + self.n_neurons)
            case _:
                message = (
                    f"Invalid weight initialization: "
                    f"{self.weight_initialization}"
                )
                raise ValueError(message)
        return scale
