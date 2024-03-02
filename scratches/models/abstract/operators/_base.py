from abc import ABC, abstractmethod
from typing import Any

from numpy import ndarray


class BaseOperator(ABC):
    """Abstract base class for operators that process input data."""

    def __init__(self):
        self.input_: ndarray
        self.output_: ndarray
        self.input_gradient: ndarray

    def feed_forward(self, input_: ndarray) -> ndarray:
        """Perform the forward pass and return the output."""
        self.input_ = input_
        self.output_ = self._apply()
        return self.output_

    def propagate_backwards(self, output_gradient: ndarray) -> ndarray:
        """Perform the backward pass and return the input gradient."""
        self.input_gradient = self._compute_gradient(output_gradient)
        return self.input_gradient

    @abstractmethod
    def _apply(self) -> Any:
        """Abstract method to apply the operator."""
        message = "Any Operator must implement ``_apply()`` method"
        raise NotImplementedError(message)

    @abstractmethod
    def _compute_gradient(self, output_gradient: ndarray) -> Any:
        """Abstract method to compute the gradient of the operator."""
        message = "Any Operator must implement ``_gradient()`` method"
        raise NotImplementedError(message)


class ParameterizedOperator(BaseOperator):
    """Abstract base class for parameterized operators."""

    def __init__(self, parameter: ndarray):
        super().__init__()
        self.parameter = parameter
        self.parameterized_gradient: ndarray

    def propagate_backwards(self, output_gradient: ndarray) -> Any:
        """Perform the backward pass and return the input gradient."""
        self.input_gradient = self._compute_gradient(output_gradient)
        self.parameterized_gradient = self._compute_parameterized_gradient(
            output_gradient
        )
        return self.input_gradient

    @abstractmethod
    def _compute_gradient(self, output_gradient: ndarray) -> Any:
        """Abstract method to compute the gradient of the operator."""
        message = (
            "Any ParameterizedOperator must implement ``_apply()`` method"
        )
        raise NotImplementedError(message)

    @abstractmethod
    def _compute_parameterized_gradient(self, output_gradient: ndarray) -> Any:
        """Abstract method to compute the parameterized gradient of the operator."""
        message = (
            "Any ParameterizedOperator must implement"
            "``_parameterized_gradient()`` method"
        )
        raise NotImplementedError(message)
