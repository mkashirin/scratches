# pyright: reportArgumentType = false

# The class built here (LayeredRegressionManualModel) is just pure
# boilerplate, that mean to show how You should never do. Overcomplicated
# computational process demonstrates the amount of work that needs to be
# done for gradient descent to work properly when hard-coded.
# Thus, the manual class solution is not always the best.
# More abstract concepts and approaches are built within
# the **abstract** module.


from typing import Any, Dict, Optional

import numpy as np
from numpy import ndarray

from .linear import LinearRegressionManualModel
from ..evaluation.metrics import compute_mean_squared_error
from ..._typing import WeightsMap, ComputationalMetadata


class LayeredRegressionManualModel(LinearRegressionManualModel):
    """Layered regression model which uses gradient descent to adjust weights.
    Prototype of neural networks build using manually calculated derivatives.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(
        self,
        x_train: ndarray,
        y_train: ndarray,
        *args,
        n_iterations: int = 1000,
        learning_rate: float = 0.001,
        batch_size: int = 100,
        random_seed: Optional[int] = None,
        **kwargs
    ) -> Any:
        """Train a layered regression model for a specified number of 
        iterations using gradient descent to adjust the curve. There is 
        also a possibility to change the number of neurons in hidden layer by
        passing additional :keyword:`hidden_size` keyword argument.

        :parameter x_train: Train selection of features values.
            :type x_train: :class:`ndarray`
        :parameter y_train: Train selection of target values.
            :type y_train: :class:`ndarray`

        :keyword n_iterations: Number of iterations that model will
        cycle through.
            :type n_iterations: :class:`int`
        :keyword learning_rate: Multiplicative coefficient of learning rate
        (describes how fast model will move towards minima).
            :type learning_rate: :class:`float`
        :keyword batch_size: Size of one single batch that model will account
        throughout one iteration.
            :type batch_size: :class:`int`
        :keyword random_seed: Random seed defined for the train process.
            :type random_seed: :class:`Optional[int]`
        :keyword hidden_size: Number of hidden layers in the model
        (equal to the number of x by default).
            :type hidden_size: :class:`int`

        :returns: Training statistics if :data:`True` is passed to one of the
        :keyword:`keep_loss` or :keyword:`keep_weights`, :data:`None` 
        otherwise.
            :rtype: :class:`Any`

        :raises ValueError: If train selections have different length.
        """

        super().fit(
            x_train,
            y_train,
            n_iterations=n_iterations,
            learning_rate=learning_rate,
            batch_size=batch_size,
            random_seed=random_seed,
            **kwargs
        )

    def predict(self, x_test: ndarray) -> ndarray:
        """Make predictions with the trained layered regression model and fit the
        curve to your data.

        :parameter x_test: Test selection of features values
            :type x_test: :class:`ndarray`

        :returns: Array of predicted target values
            :rtype: :class:`ndarray`
        """

        input_product = np.dot(x_test, self.weights_map_["input_weights"])
        input_biased = input_product + self.weights_map_["input_bias"]
        sigmoid_applied = self.sigmoid(input_biased)
        hidden_product = np.dot(
            sigmoid_applied, self.weights_map_["hidden_weights"]
        )
        predictions = hidden_product + self.weights_map_["hidden_bias"]

        return predictions

    @staticmethod
    def sigmoid(value: ndarray) -> ndarray:
        return 1 / (1 + np.exp(-value))

    def _get_weights(self, **kwargs) -> None:
        """Generate a dictionary with random weights and bias."""
        input_size = kwargs["n_features"]
        hidden_size = (
            kwargs.get("hidden_size")
            if kwargs.get("hidden_size") is not None
            else input_size
        )

        input_weights, input_bias = np.random.randn(
            input_size, hidden_size
        ), np.random.randn(1, hidden_size)
        hidden_weights, hidden_bias = np.random.randn(
            hidden_size, 1
        ), np.random.randn(1, 1)
        self.weights_map_: WeightsMap = {
            "input_weights": input_weights,
            "input_bias": input_bias,
            "hidden_weights": hidden_weights,
            "hidden_bias": hidden_bias,
        }

    def _feed_forward(
        self, x_batch: ndarray, y_batch: ndarray
    ) -> ComputationalMetadata:
        """Make predictions and calculate loss for a linear regression 
        model.
        """
        input_product = np.dot(x_batch, self.weights_map_["input_weights"])
        input_biased = input_product + self.weights_map_["input_bias"]
        sigmoid_applied = self.sigmoid(input_biased)
        hidden_product = np.dot(
            sigmoid_applied, self.weights_map_["hidden_weights"]
        )
        predictions = hidden_product + self.weights_map_["hidden_bias"]
        loss = compute_mean_squared_error(y_batch, predictions)

        computational_meta: Dict[str, ndarray] = {
            "x_batch": x_batch,
            "y_batch": y_batch,
            "input_product": input_product,
            "input_biased": input_biased,
            "sigmoid_applied": sigmoid_applied,
            "hidden_product": hidden_product,
            "predictions": predictions,
        }
        with_loss_data: ComputationalMetadata = {
            "meta": computational_meta,
            "loss": loss,
            "weights": self.weights_map_,
        }

        return with_loss_data

    def _propagate_backwards(
        self, computational_meta: Dict[str, ndarray]
    ) -> WeightsMap:
        """Compute new gradients and evaluate them at the given in
        :class:`ComputationalMetadata` points for a linear regression model
        (all differentiation has been done manually and then hard-coded).
        """
        loss_over_predictions = -(
            computational_meta["y_batch"] - computational_meta["predictions"]
        )
        predictions_over_hidden_product = np.ones_like(
            computational_meta["hidden_product"]
        )
        loss_over_hidden_product = (
            loss_over_predictions * predictions_over_hidden_product
        )
        predictions_over_hidden_bias = np.ones_like(
            self.weights_map_["hidden_bias"]
        )
        loss_over_hidden_bias = np.sum(
            loss_over_predictions * predictions_over_hidden_bias, axis=0
        )
        hidden_product_over_hidden_weights = np.transpose(
            computational_meta["sigmoid_applied"], axes=(1, 0)
        )
        loss_over_hidden_weights = np.dot(
            hidden_product_over_hidden_weights, loss_over_predictions
        )
        hidden_product_over_sigmoid_applied = np.transpose(
            self.weights_map_["hidden_weights"]
        )
        loss_over_sigmoid_applied = np.dot(
            loss_over_hidden_product, hidden_product_over_sigmoid_applied
        )
        sigmoid_applied_over_input_biased = self.sigmoid(
            computational_meta["input_product"]
        ) * (1 - self.sigmoid(computational_meta["input_product"]))
        loss_over_input_biased = (
            loss_over_sigmoid_applied * sigmoid_applied_over_input_biased
        )
        input_biased_over_input_bias = np.ones_like(
            self.weights_map_["input_bias"]
        )
        input_biased_over_input_product = np.ones_like(
            computational_meta["input_product"]
        )
        loss_over_input_bias = np.sum(
            loss_over_input_biased * input_biased_over_input_bias, axis=0
        )
        loss_over_input_product = (
            loss_over_input_biased * input_biased_over_input_product
        )
        input_product_over_input_weights = np.transpose(
            computational_meta["x_batch"], (1, 0)
        )
        loss_over_input_weights = np.dot(
            input_product_over_input_weights, loss_over_input_product
        )

        loss_gradients: WeightsMap = {
            "input_weights": loss_over_input_weights,
            "input_bias": loss_over_input_bias,
            "hidden_weights": loss_over_hidden_weights,
            "hidden_bias": loss_over_hidden_bias,
        }

        return loss_gradients
