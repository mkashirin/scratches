from itertools import chain
from logging import info
from time import perf_counter
from typing import Dict, Optional

import numpy as np
from numpy import ndarray
from scipy.stats import mode

from ._base import BaseManualModel


class KNNClassificationManualModel(BaseManualModel):
    """K-nearest neighbors classification model which uses
    L2 (Euclidean) distance.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weights_map: Optional[Dict[int, int]]

    def fit(self, x_train: ndarray, y_train: ndarray, *args, **kwargs) -> None:
        """Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        :parameter x_train: Train selection of features values.
            :type x_train: :class:`ndarray`
        :parameter y_train: Train selection of target values.
            :type y_train: :class:`ndarray`

        :raises ValueError: If `x` and `y` do not pass validation conditions,
        expressed in :method:`_validate_shapes()` method of the 
        :class:`BaseManualModel` class.
        """
        super().fit(x_train, y_train)

    def predict(
        self,
        x_test: ndarray,
        n_neighbors: int = 5,
        n_loops: int = 0,
        weight_map: Optional[Dict[int, int]] = None,
    ) -> ndarray:
        """Predict labels for test data using this classifier.
        The data passed to this method would be copied and used as
        NumPy :class:`ndarray`.

        :parameter x_test: Test selection of x (set x) values.
            :type x_test: :class:`ndarray`

        :keyword n_neighbors: Number of neighbors to account distance to.
            :type n_neighbors: :class:`int`
        :keyword n_loops: Number of loops used within the distances
        calculation process.
            :type n_loops: :class:`int`
        :keyword weight_map: A dictionary which maps indices of neighbors to
        their weight.
            :type weight_map: :class:`Optional[Dict[int, int]]`

        :returns: Array of predicted labels.
            :rtype: :class:`ndarray`
        """

        if weight_map is not None and len(weight_map) != n_neighbors:
            message = (
                f"The number of weights in `weight_map` and the number "
                f"`n_neighbors` must be the same"
            )
            raise ValueError(message)
        predicting_start_time = perf_counter()

        match n_loops:
            case 0:
                distances = self._compute_with_no_loops(self.x_train, x_test)
            case 1:
                distances = self._compute_with_one_loop(self.x_train, x_test)
            case 2:
                distances = self._compute_with_two_loops(self.x_train, x_test)
            case _:
                message = f"Invalid value {n_loops} for number of loops"
                raise ValueError(message)

        predicted_labels = self._predict_labels(
            distances, n_neighbors, weight_map=weight_map
        )

        predicting_end_time = perf_counter()
        predicting_time = predicting_end_time - predicting_start_time
        info(f"Predicting complete! It took {predicting_time} seconds.")
        return predicted_labels

    @staticmethod
    def _compute_with_two_loops(x_train: ndarray, x_test: ndarray) -> ndarray:
        """Compute the Euclidean distances between data points using
        two loops.
        """
        n_test_samples = x_test.shape[0]
        n_train_samples = x_train.shape[0]
        distances = np.zeros((n_test_samples, n_train_samples))

        for i in range(n_test_samples):
            for j in range(n_train_samples):
                distances[i, j] = np.sqrt(
                    np.sum(np.square(x_test[i, :] - x_train[j, :]))
                )

        return distances

    @staticmethod
    def _compute_with_one_loop(x_train: ndarray, x_test: ndarray) -> ndarray:
        """Compute the Euclidean distances between data points using 
        one loop.
        """
        n_test_samples = x_test.shape[0]
        n_train_samples = x_train.shape[0]
        distances = np.zeros((n_test_samples, n_train_samples))

        for i in range(n_test_samples):
            distances[i, :] = np.sqrt(
                np.sum(np.square(x_train - x_test[i, :]), axis=1)
            )

        return distances

    @staticmethod
    def _compute_with_no_loops(x_train: ndarray, x_test: ndarray) -> ndarray:
        """Compute the Euclidean distances between data points using
        NumPy matrix operators only.
        """
        train_sum = np.sum(np.square(x_train), axis=1, keepdims=True)
        test_sum = np.sum(np.square(x_test), axis=1, keepdims=True)
        distances = np.sqrt(
            train_sum.T + test_sum - 2 * np.dot(x_test, x_train.T)
        )

        return distances

    @staticmethod
    def _weigh_neighbors(
        closest_neighbors: list, weight_map: Dict[int, int]
    ) -> list:
        """Add more weight for a certain neighbor according to the
        :parameter:`weight_map`.
        """
        weighted_list = list()
        for i, value in enumerate(closest_neighbors):
            weighted_value = (value,) * weight_map[i]
            weighted_list.insert(i, weighted_value)
        weighted_list = list(chain.from_iterable(weighted_list))

        return weighted_list

    def _predict_labels(
        self,
        distances: ndarray,
        n_neighbors: int = 5,
        *,
        weight_map: Optional[Dict[int, int]] = None,
    ) -> ndarray:
        """Predict labels for each test point.

        :parameter distances: Array of distances between data points.
            :type distances: :class:`ndarray`
        :parameter n_neighbors: Number of neighbors to account distance to.
            :type n_neighbors: :class:`int`

        :keyword weight_map: A dictionary which maps indices of neighbors to
        their weight.
            :type weight_map: :class:`Optional[Dict[int, int]]`

        :return: Array of labels predicted based on nearest neighbors.
            :rtype: :class:`ndarray`
        """
        n_test_samples = distances.shape[0]
        y_predicted = np.zeros((n_test_samples, 1))

        for i in range(n_test_samples):
            sorted_indices = np.argsort(distances[i, :])
            farthest_neighbor = np.min([n_neighbors, len(sorted_indices)])
            closest_neighbors = self.y_train[
                sorted_indices[:farthest_neighbor]
            ]

            if weight_map is not None:
                closest_neighbors = closest_neighbors.tolist()
                closest_neighbors = self._weigh_neighbors(
                    closest_neighbors, weight_map
                )

            y_predicted[i] = mode(closest_neighbors)[0]
        return y_predicted
