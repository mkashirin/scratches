from abc import ABC, abstractmethod
from typing import Any, Tuple

from numpy import ndarray


class BasePreprocessor(ABC):
    """The BasePreprocessor class is an abstract base class for preprocessor
    implementations.
    """

    def __init__(self, copy: bool = True):
        self.copy = copy

    @abstractmethod
    def fit(self, x) -> None:
        """Fit the preprocessor to the provided features."""
        message = "Every preprocessor must implement the ``fit()`` method."
        raise NotImplementedError(message)

    @abstractmethod
    def transform(self, x) -> Any:
        """Transform the input features."""
        message = (
            "Every preprocessor must implement the ``transform()`` method."
        )
        raise NotImplementedError(message)

    @staticmethod
    def _get_values_masks(array: ndarray) -> Tuple[bool, bool]:
        non_zero_values_mask = (array != 0)
        zero_values_mask = ~non_zero_values_mask
        return non_zero_values_mask, zero_values_mask
