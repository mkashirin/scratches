from typing import List, Optional

import numpy as np
from numpy import ndarray

from ._base import BaseOptimizer
from ...._typing import DecayType


class SGDOptimizer(BaseOptimizer):
    """Stochastic gradient descent optimizer for neural network training."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        final_learning_rate: float = 0.0,
        decay_type: Optional[DecayType] = None,
    ):
        super().__init__(learning_rate, final_learning_rate, decay_type)

    def step(self) -> None:
        """Performs a single step of the stochastic gradient descent
        optimization. Updates the parameters of the neural network using
        stochastic gradient descent.
        """
        super().step()

    def _update_rule(self, **kwargs) -> None:
        update = self.learning_rate * kwargs["parameterized_gradient"]
        kwargs["parameter"] -= update


class SGDMomentumOptimizer(BaseOptimizer):
    """Advanced stochastic gradient descent optimizer for network trainig,
    which allows to set the momentum and decay type of the learning rate.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        final_learning_rate: float = 0,
        decay_type: Optional[DecayType] = None,
        momentum: float = 0.9,
    ):
        super().__init__(learning_rate, final_learning_rate, decay_type)
        self.momentum = momentum
        self.velocities: List[ndarray]

    def step(self) -> None:
        """Does the same as the default SGD optimizer's ``step()`` function,
        but in a more sophicsticated way
        (regulating the learning rate value).
        """

        if self.first:
            self.velocities = [
                np.zeros_like(parameters)
                for parameters in self.network.get_parameters()
            ]
            self.first = False

        for parameter, parameterized_gradient, velocity in zip(
            self.network.get_parameters(),
            self.network.get_parameterized_gradients(),
            self.velocities,
        ):
            self._update_rule(
                parameter=parameter,
                parameterized_gradient=parameterized_gradient,
                velocity=velocity,
            )

    def _update_rule(self, **kwargs) -> None:
        kwargs["velocity"] *= self.momentum
        kwargs["velocity"] += (
            self.learning_rate * kwargs["parameterized_gradient"]
        )
        kwargs["parameter"] -= kwargs["velocity"]
