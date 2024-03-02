from typing import Any

import numpy as np
from numpy import ndarray
from numpy import floating

from ._base import BasePreprocessor
from ..._typing import StrategyOption


class ImputingPreprocessor(BasePreprocessor):
    """ImputingPreprocessor class for imputing missing values in x using
    specified strategies.
    """

    def __init__(self, strategy: StrategyOption = "mean", copy: bool = True):
        super().__init__(copy)
        self.strategy = strategy
        self.fillers: Any

    def fit(self, x: ndarray, fill_with: Any = None) -> None:
        """Fit the preprocessor on the given x and compute the specified
        statistics for each feature.

        :parameter x: The input features which statistics would be used for
            imputing;
            :type x: ndarray
        :parameter fill_with: The object to fill missing values with,
            works when strategy is set to "constant".
            :type fill_with: Any
        """
        match self.strategy:
            case "constant":
                self.fillers = np.full(x.shape[1], fill_with)
            case "mean":
                self.fillers = np.nanmean(x, axis=0)
            case "median":
                self.fillers = np.nanmedian(x, axis=0)
            case _:
                message = (
                    f'Expected strategy to be one of "constant", "mean" '
                    f'or "median", but got "{self.strategy}"'
                )
                raise ValueError(message)

    def transform(self, x: ndarray) -> ndarray:
        """Transform the input features using statistics, calculated with the
        ``fit()`` method.

        :parameter x: The input x to be imputed.
            :type x: ndarray

        :returns: The x imputed.
            :rtype: ndarray
        """
        if self.copy:
            x = x.copy()

        nan_mask = np.isnan(x)
        x[nan_mask] = np.take(self.fillers, np.where(nan_mask)[1])
        return x

    def fit_transform(self, x) -> ndarray:
        """Fit and transform at the same time."""
        self.fit(x)
        transformed = self.transform(x)
        return transformed


class MMScalingPreprocessor(BasePreprocessor):
    """ScalingPreprocessor class for scaling the features using MMScaling."""

    def __init__(self, copy: bool = True):
        super().__init__(copy)
        self.min_values: ndarray
        self.max_values: ndarray

    def fit(self, x: ndarray) -> None:
        """Fit the preprocessor to the input x and computes the (min, max)
        boundaries for each feature.

        :parameter x: The features to fit the preprocessor and compute the
            boundaries;
            :type x: ndarray
        """
        self.min_values = np.nanmin(x, axis=0)
        self.max_values = np.nanmax(x, axis=0)

    def transform(self, x: ndarray) -> ndarray:
        """Transform the input features and scale the data according to the
        computed boundaries.

        :parameter x: Features to scale and transform.
            :type x: ndarray

        :return: Scaled features.
            :rtype: ndarray
        """
        if self.copy:
            x = x.copy()

        range_values = self.max_values - self.min_values
        nonzero_range_mask, zero_range_mask = self._get_values_masks(
            range_values
        )

        x[:, zero_range_mask] = 0
        x[:, nonzero_range_mask] = (
            x[:, nonzero_range_mask] - self.min_values[nonzero_range_mask]
        ) / np.where(
            range_values[nonzero_range_mask] == 0,
            1,
            range_values[nonzero_range_mask],
        )
        return x

    def fit_transform(self, x) -> ndarray:
        """Fit and transform at the same time."""
        self.fit(x)
        transformed = self.transform(x)
        return transformed


class StandardScalingPreprocessor(BasePreprocessor):
    """StandardScalingPreprocessor class for standard scaling the features."""

    def __init__(self, copy: bool = True):
        super().__init__(copy)
        self.means: ndarray
        self.stds: ndarray

    def fit(self, x: ndarray) -> None:
        """Fit the preprocessor to the input x and computes the mean and
        standard deviation for each feature.

        :parameter x: The features to fit the preprocessor and compute the
            statistics.
            :type x: ndarray
        """
        self.means = np.nanmean(x, axis=0)
        self.stds = np.nanstd(x, axis=0)

    def transform(self, x: ndarray) -> ndarray:
        """Transform the input features and standard scale the data according
        to the computed mean and standard deviation.

        :parameter x: Features to scale and transform.
            :type x: ndarray

        :return: Standard scaled features.
            :rtype: ndarray
        """
        if self.copy:
            x = x.copy()

        nonzero_std_mask, zero_std_mask = self._get_values_masks(self.stds)
        nonzero_mean_mask, _ = self._get_values_masks(self.means)

        x[:, zero_std_mask] = 0
        x[:, nonzero_std_mask] = (
            x[:, nonzero_std_mask] - self.means[nonzero_mean_mask]
        ) / self.stds[nonzero_std_mask]
        return x

    def fit_transform(self, x) -> ndarray:
        """Fit and transform at the same time."""
        self.fit(x)
        transformed = self.transform(x)
        return transformed
