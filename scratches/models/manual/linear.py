# pyright: reportArgumentType = false

from logging import info
from time import perf_counter
from typing import Dict, Optional

import numpy as np
from numpy import ndarray

from ._base import BaseManualModel
from ..._typing import ComputationalMetadata, SamplesBatch, WeightsMap
from ...algebra import LeastSquaresSolver
from ..evaluation.metrics import compute_mean_squared_error


class LinearRegressionManualModel(BaseManualModel):
    """Linear regression model which uses gradient descent to adjust weights.
    It can also resolve the connections between x and y using algebraic
    method of the Least Squares.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(
        self,
        x_train: ndarray,
        y_train: ndarray,
        *args,
        n_iterations: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 100,
        random_seed: Optional[int] = None,
        solve_algebraically: bool = False,
        **kwargs,
    ) -> None:
        """Train a linear regression model for a specified number of
        iterations using gradient descent or algebraic resolution.

        :parameter x_train: Train selection of features values.
            :type x_train: :class:`ndarray`
        :parameter y_train: Train selection of target values.
            :type y_train: :class:`ndarray`

        :keyword n_iterations: Number of iterations for model cycling.
            :type n_iterations: :class:`int`
        :keyword learning_rate: Multiplicative coefficient of the learning
        rate, describing the speed of model movement towards minima.
            :type learning_rate: :class:`float`
        :keyword batch_size: Size of a single batch that the model will
        account for throughout one iteration.
            :type batch_size: :class:`int`
        :keyword random_seed: Random seed defined for replication.
            :type random_seed: :class:`Optional[int]`
        :keyword solve_algebraically: If True, the model will resolve the
        weights by solving the system of linear equations using the
        Least Squares method; otherwise, will continue adjusting the
        weights with gradient descent.
            :type solve_algebraically: :class:`bool`

        :raises ValueError: If x and y do not pass validation conditions,
            expressed in :method:`_validate_shapes()` method of
            the :class:`BaseManualModel` class.
        """

        super().fit(x_train, y_train)

        fitting_start_time = perf_counter()
        if solve_algebraically:
            round_to = (
                kwargs.get("round_to")
                if kwargs.get("round_to") is not None
                else 10
            )
            self._solve_algebraicaly(round_to)
            fitting_end_time = perf_counter()
            fitting_time = fitting_end_time - fitting_start_time
            info(f"Fitting complete! It took {fitting_time} seconds")
            return

        if random_seed:
            np.random.seed(random_seed)

        self.loss_values_ = list()
        batch_start = 0
        best_setting: Optional[ComputationalMetadata] = None
        # Initializing random weights map
        (
            self._get_weights(
                n_features=self.x_train.shape[1],
                hidden_size=kwargs["hidden_size"],
            )
            if "hidden_size" in kwargs
            else (self._get_weights(n_features=self.x_train.shape[1]))
        )
        self._permute_data()

        for _ in range(n_iterations):
            # Getting a batch
            if batch_start >= self.x_train.shape[0]:
                self._permute_data()
                batch_start = 0

            x_batch, y_batch = self._get_batch(
                self.x_train, self.y_train, batch_size, batch_start=batch_start
            )
            batch_start += batch_size

            # Training the model running it through with batch got
            computational_metadata = self._feed_forward(x_batch, y_batch)
            computational_meta, loss_data = (
                computational_metadata["meta"],
                computational_metadata["loss"],
            )
            # Keeping track of minimum loss
            if best_setting is None:
                best_setting = computational_metadata
            elif (
                best_setting["loss"]
                > loss_data  # pyright: ignore[reportOperatorIssue]
            ):
                best_setting = computational_metadata
            self.loss_values_.append(loss_data)

            # Adjusting weights with respect to learning rate and
            # loss gradients
            loss_gradients = self._propagate_backwards(computational_meta)
            for key in self.weights_map_.keys():
                self.weights_map_[key] -= learning_rate * loss_gradients[key]

        if best_setting is not None:
            self.weights_map_ = best_setting[
                "weights"  # pyright: ignore[reportAttributeAccessIssue]
            ]

            fitting_end_time = perf_counter()
            fitting_time = fitting_end_time - fitting_start_time
            info(f"Fitting complete! It took {fitting_time} seconds")
        return

    def predict(self, x_test: ndarray) -> ndarray:
        """Make predictions with the trained linear regression model.

        :parameter x_test: Test selection of features values.
            :type x_test: :class:`ndarray`

        :returns: Array of predicted target values.
            :rtype: :class:`ndarray`
        """

        product = np.dot(x_test, self.weights_map_["weights"])
        y_predicted = product + self.weights_map_["bias"]

        return y_predicted

    @staticmethod
    def _get_batch(
        x_train: ndarray,
        y_train: ndarray,
        batch_size: int,
        batch_start: int = 0,
    ) -> SamplesBatch:
        """Generate a batch for training starting from the 
        :parameter:`batch_start` position.
        """
        if batch_start + batch_size > x_train.shape[0]:
            batch_size = x_train.shape[0] - batch_start
        batch_end = batch_start + batch_size

        x_batch, y_batch = (
            x_train[batch_start:batch_end],
            y_train[batch_start:batch_end],
        )
        sample_batch: SamplesBatch = x_batch, y_batch

        return sample_batch

    def _permute_data(self) -> None:
        permutation = np.random.permutation(self.x_train.shape[0])
        self.x_train, self.y_train = (
            self.x_train[permutation],
            self.y_train[permutation],
        )

    def _get_weights(self, **kwargs) -> None:
        """Generate a dictionary with random weights and bias."""
        weights = np.random.randn(kwargs["n_features"], 1)
        bias = np.random.randn()
        self.weights_map_: WeightsMap = {"weights": weights, "bias": bias}

    def _feed_forward(
        self, x_batch: ndarray, y_batch: ndarray
    ) -> ComputationalMetadata:
        """Make predictions and calculate loss for a linear regression
        model.
        """
        product = np.dot(x_batch, self.weights_map_["weights"])
        predictions = product + self.weights_map_["bias"]
        loss = compute_mean_squared_error(y_batch, predictions)

        computational_meta: Dict[str, ndarray] = {
            "x_batch": x_batch,
            "y_batch": y_batch,
            "product": product,
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
        loss_over_predictions = -2 * (
            computational_meta["y_batch"] - computational_meta["predictions"]
        )
        predictions_over_product = np.ones_like(computational_meta["product"])
        predictions_over_bias = np.ones_like(self.weights_map_["bias"])
        loss_over_product = loss_over_predictions * predictions_over_product
        product_over_weights = np.transpose(
            computational_meta["x_batch"], axes=(1, 0)
        )
        loss_over_weights = np.sum(
            np.dot(product_over_weights, loss_over_product), axis=1
        ).reshape(-1, 1)
        loss_over_bias = np.sum(
            loss_over_product * predictions_over_bias, axis=0
        )[0]

        loss_gradients: WeightsMap = {
            "weights": loss_over_weights,
            "bias": loss_over_bias,
        }

        return loss_gradients

    def _solve_algebraicaly(self, round_to: int) -> None:
        solver = LeastSquaresSolver()
        solver.fit(self.x_train, self.y_train)
        if round_to is None:
            round_to = 10
        weights = solver.solve(round_to=round_to)

        self.weights_map_ = {"weights": weights, "bias": 0.0}
