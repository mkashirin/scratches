from abc import ABC, abstractmethod
from logging import basicConfig, INFO
from typing import Any

from numpy import ndarray


class BaseManualModel(ABC):
    """Base machine learning model class."""

    def __init__(self) -> None:
        self.x_train: ndarray
        self.y_train: ndarray
        basicConfig(format="Model: %(message)s", level=INFO)

    @abstractmethod
    def fit(self, x_train: ndarray, y_train: ndarray, *args, **kwargs) -> None:
        """The data passed to this method would be copied and used as
        NumPy :class:`ndarray`.
        """
        self.x_train, self.y_train = x_train, y_train

    @abstractmethod
    def predict(self, x_test: ndarray) -> Any:
        message = "Every model should implement the `predict()` method"
        raise NotImplementedError(message)
