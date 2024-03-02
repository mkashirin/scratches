from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..networks.network import NeuralNetwork
from ...._typing import DecayType


class BaseOptimizer(ABC):
    """Base class for optimizers in neural network training."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        final_learning_rate: float = 0.0,
        decay_type: Optional[DecayType] = None,
    ):
        self.first = True
        self.learning_rate = learning_rate
        self.final_learning_rate = final_learning_rate
        self.decay_type = decay_type
        self.network: NeuralNetwork
        self.decay_rate: float
        self.max_epochs: int

    @abstractmethod
    def step(self) -> None:
        for parameter, parameterized_gradient in zip(
            self.network.get_parameters(),
            self.network.get_parameterized_gradients(),
        ):
            self._update_rule(
                parameter=parameter,
                parameterized_gradient=parameterized_gradient,
            )

    @abstractmethod
    def _update_rule(self, **kwargs) -> None:
        message = "Every optimizer must implement the  _update_rule() method"
        raise NotImplementedError(message)

    def _setup_decay(self) -> None:
        if not self.decay_type:
            return

        match self.decay_type:
            case "exponential":
                self.decay_rate = np.power(
                    self.final_learning_rate / self.learning_rate,
                    1 / (self.max_epochs - 1),
                )
            case "linear":
                self.decay_rate = (
                    self.learning_rate - self.final_learning_rate
                ) / (self.max_epochs - 1)

    def _decay_learning_rate(self) -> None:
        if not self.decay_rate:
            return

        match self.decay_rate:
            case "exponential":
                self.learning_rate *= self.decay_rate
            case "linear":
                self.learning_rate -= self.decay_rate
