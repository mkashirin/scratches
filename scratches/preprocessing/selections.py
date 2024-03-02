from typing import List, Optional

import numpy as np
from numpy import ndarray

from .._typing import Selections


def tokenize_probabilities(matrix: ndarray) -> ndarray:
    """Tokenize the probabilities of each outcome in the output layer.

    :parameter matrix: Probabilities matrix, filled with values ranging
        from 0 and 1.
        :type matrix: ndarray

    :returns: Binary matrix, where columns represent classes and rows
        represent samples.
    """
    binary_matrix = np.zeros_like(matrix)
    max_indices = np.argmax(matrix, axis=1)
    rows = np.arange(matrix.shape[0])
    binary_matrix[rows, max_indices] = 1
    return binary_matrix


def revert_matrix(matrix: ndarray) -> ndarray:
    """Revert the target binary matrix to original vector format.

    :parameter matrix: The binary matrix to revert.
        :type matrix: ndarray

    :returns: Vector formatted from the binary matrix passed.
        :rtype: ndarray
    """
    return np.argmax(matrix, axis=1).reshape(-1, 1)


def transform_binary(vector: ndarray) -> ndarray:
    """Convert the target vector to binary matrix.

    :parameter vector: Target vector, which is to be transformed.
        :type vector: ndarray

    :returns: Binary matrix, where columns represent classes and rows
        represent samples.
        :rtype: ndarray
    """
    n_values = vector.shape[0]
    max_value = np.max(vector)
    binary_matrix = np.zeros((n_values, max_value + 1))
    for i, value in enumerate(vector):
        binary_matrix[i, value] = 1

    return binary_matrix


def reshape_channel_images(
    images: ndarray, n_channels: int, *, image_height: int, image_width: int
) -> ndarray:
    """Reshape channel images stored as arrays of pixels to be properly 
    consumed by the convolutional neural nets.

    :parameter images: Array of images to be reshaped.
        :type images: ndarray
    :parameter n_channels: Number of channels of the single image.
        :type n_channels: int

    :keyword image_height: Height of the single image in pixels.
        :type image_height: int
    :keyword image_width: Width of the single image in pixels.
        :type image_width: int
    
    :returns: Array of images reshaped in a specified way.
        :rtype: ndarray
    """
    reshaped = images.reshape(-1, n_channels, image_height, image_width)
    return reshaped


def normalize_data(*, to_normalize: ndarray, std_from: ndarray) -> ndarray:
    """Normalize the data using the standard deviation from another data.
    
    :keyword to_normalize: Array of data to be normalized.
        :type to_normalize: ndarray
    :keyword std_from: Array to be taken standard deviation from.
        :type std_from: ndarray

    :returns: Normalized array.
        :rtype: ndarray
    """

    normalized = to_normalize / np.nanstd(std_from)
    return normalized


class DataSplitter:
    """Data splitting interface, which allows you to separate your data
    on train, validation and test selections.
    """

    def __init__(
        self, permute: bool = False, random_seed: Optional[int] = None
    ):
        """Set parameters for the splitting.

        :parameter permute: Defines whether data will be permuted before
            split operation or not;
            :type permute: bool
        :parameter random_seed: Random seed that will be applied during 
            the process.
            :type random_seed: Optional[int]
        """
        self.random_seed = random_seed
        self.permute = permute
        self._selections: List[ndarray]

    def split_data(
        self,
        x: ndarray,
        y: ndarray,
        *,
        test_size: float,
        valid_size: Optional[float] = None,
    ) -> Selections:
        """Split the data on train, validation and test selections.

        :parameter x: Features data, that would be split on train and
            test selections;
            :type x: ndarray
        :parameter y: Target data, that would be split on train and
            test selections;
            :type y: ndarray
        :parameter test_size: Percentage of data that will be allocated for the
            test selection.
            :type test_size: float

        :keyword valid_size: Percentage of data that will be allocated for the
            validation selection;
            :type valid_size: Optional[float]

        :returns: Tuple of selections split according to specified parameters.
            :rtype: Selections
        """
        if self.random_seed:
            np.random.seed(self.random_seed)
        if self.permute:
            permutation = np.random.permutation(x.shape[0])
            x, y = x[permutation], y[permutation]

        self._set_standard(x, y, test_size)
        if valid_size:
            test_length = self._selections[1].shape[0]
            self._add_valid(test_length, x, y, valid_size)

        selections: Selections = tuple(
            self._selections  # pyright: ignore[reportAssignmentType]
        )
        return selections

    def _set_standard(self, x: ndarray, y: ndarray, test_size: float) -> None:
        train_test_index = int(x.shape[0] * test_size)

        x_train, x_test, y_train, y_test = (
            x[train_test_index:],
            x[:train_test_index],
            y[train_test_index:],
            y[:train_test_index],
        )
        self._selections = [x_train, x_test, y_train, y_test]

    def _add_valid(
        self, test_length: int, x: ndarray, y: ndarray, valid_size: float
    ) -> None:
        test_valid_index = int(test_length * valid_size)

        self._selections[1], self._selections[3] = (
            self._selections[1][test_valid_index:],
            self._selections[3][test_valid_index:],
        )
        self._selections.insert(1, x[:test_valid_index])
        self._selections.insert(4, y[:test_valid_index])
