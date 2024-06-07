from typing import Any, Optional

import numpy as np
from numpy import ndarray
from scipy.special import logsumexp

from ._base import BaseEvaluator


class MSEEvaluator(BaseEvaluator):
    """Class for defining the Mean Squared Error loss function as an
    evaluator.
    """

    def __init__(self) -> None:
        super().__init__()

    def _apply(self) -> Any:
        """Apply the Mean Squared Error loss function and return the
        computed loss value.
        """
        applied = (
            np.sum(np.power(self.predicted - self.actual, 2))
            / self.predicted.shape[0]
        )
        return applied

    def _compute_gradient(self) -> Any:
        """Compute the gradient of the Mean Squared Error loss function
        and return the computed input gradient.
        """
        self.input_gradient = (
            2 * (self.predicted - self.actual) / self.predicted.shape[0]
        )
        return self.input_gradient


class SoftmaxCEEvaluator(BaseEvaluator):
    """Class for defining the Softmax Cross Entropy loss function as an
    evaluator.
    """

    epsilon = float(np.finfo(float).eps)

    def __init__(self, epsilon: float = epsilon) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.single_class = False
        self.softmax_predicted = None

    @staticmethod
    def normalize(array: ndarray) -> ndarray:
        return np.concatenate([array, 1 - array], axis=1)

    @staticmethod
    def denormalize(array: ndarray) -> ndarray:
        return array[np.newaxis, 0]

    @staticmethod
    def softmax(array: ndarray, axis: Optional[int] = None) -> ndarray:
        return np.exp(array - logsumexp(array, axis=axis, keepdims=True))

    def _apply(self) -> Any:
        """Apply the Softmax Cross Entropy loss function and return the
        computed loss value.
        """
        if self.actual.shape[0] == 1:
            self.single_class = True

        if self.single_class:
            self.actual, self.predicted = self.normalize(
                self.actual
            ), self.normalize(self.predicted)

        softmax_predicted = self.softmax(self.predicted, axis=0)
        self.softmax_predicted = np.clip(
            softmax_predicted, self.epsilon, 1 - self.epsilon
        )
        softmax_cross_entropy = -self.actual * np.log(
            self.softmax_predicted
        ) - (1 - self.actual) * np.log(1 - self.softmax_predicted)

        applied = np.sum(softmax_cross_entropy) / self.predicted.shape[0]
        return applied

    def _compute_gradient(self) -> Any:
        """Compute the gradient of the Softmax Cross Entropy loss function
        and return the computed input gradient.
        """
        if self.single_class:
            return self.denormalize(self.softmax_predicted - self.actual)
        return (self.softmax_predicted - self.actual) / self.predicted.shape[0]
