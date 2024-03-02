from numpy import ndarray

from typing import Dict, Optional

from ..layers._base import BaseLayer
from ..evaluators._base import BaseEvaluator


class NeuralNetwork:
    """Class for defining a neural network."""

    def __init__(
        self,
        layers: Dict[str, BaseLayer],
        loss_function: BaseEvaluator,
        *,
        random_seed: Optional[int] = None
    ) -> None:
        """Initialize the NeuralNetwork with layers, loss function, and an
        optional random seed.

        :parameter layers: The layers of the neural network;
            :type layers: Dict[str, BaseLayer]
        :parameter loss_function: The loss function used for training the
            network.
            :type loss_function: BaseEvaluator

        :keyword random_seed: The random seed for reproducibility, defaults
            to ``None``.
            :type random_seed: int
        """
        self.layers = layers
        self.loss_function = loss_function
        self.random_seed = random_seed

        if random_seed is not None:
            for layer in self.layers.values():
                setattr(layer, "random_seed", random_seed)

    def feed_forward(self, x_input: ndarray) -> ndarray:
        """Passes the input batch forward through the neural network and 
        returns the output.

        :parameter x_input: The input features to pass to the neural network.
            :type x_input: ndarray

        :returns: The output of the neural network (predictions).
            :rtype: ndarray
        """
        x_output = x_input
        for layer in self.layers.values():
            x_output = layer.feed_forward(x_output)
        return x_output

    def propagate_backwards(self, loss_gradient: ndarray) -> None:
        """Propagates the loss gradient backward through the neural network.

        :parameter loss_gradient: The gradient of the loss function.
            :type loss_gradient: ndarray
        """
        gradient = loss_gradient
        for layer in reversed(self.layers.values()):
            gradient = layer.propagate_backwards(gradient)

    def train(self, x_batch: ndarray, y_batch: ndarray) -> float:
        """Trains the neural network on the input batch and returns the
        loss value.

        :parameter x_batch: The input batch for training;
            :type x_batch: ndarray
        :parameter y_batch: The target output batch for training.
            :type y_batch: ndarray

        :returns: The loss value after training
            :rtype: float
        """
        predicted = self.feed_forward(x_batch)
        loss_value = self.loss_function.feed_forward(y_batch, predicted)
        self.propagate_backwards(self.loss_function.propagate_backwards())

        return loss_value

    def get_parameters(self):
        """Generator to yield the parameters of the neural network layers."""
        for layer in self.layers.values():
            yield from layer.parameters.values()

    def get_parameterized_gradients(self):
        """Generator to yield the parameterized gradients of the neural
        network layers.
        """
        for layer in self.layers.values():
            yield from layer.parameters_gradients.values()
