from typing import Any

import numpy as np
from numpy import ndarray

from ._base import BaseOperator, ParameterizedOperator


class LinearPassageOperator(BaseOperator):
    """Operator, that passes data in both forward and backward directions."""

    def _apply(self) -> ndarray:
        """Return the input data."""
        return self.input_

    def _compute_gradient(self, output_gradient: ndarray) -> ndarray:
        """Return the output gradient."""
        return output_gradient


class RelUFunctionOperator(BaseOperator):
    """Operator, that applies rectified linear unit function to the input data."""

    def _apply(self) -> Any:
        """Apply the rectified linear unit function to the input."""
        applied = np.clip(self.input_, 0, self.input_)
        return applied

    def _compute_gradient(self, output_gradient: ndarray) -> Any:
        """Compute the gradient of the rectified linear unit function."""
        mask = self.output_ >= 0
        gradient = output_gradient * mask
        return gradient


class SigmoidFunctionOperator(BaseOperator):
    """Operator, that applies a sigmoid function to the input data."""

    def _apply(self) -> ndarray:
        """Apply the sigmoid function to the input data."""
        applied = 1 / (1 + np.exp(-self.input_))
        return applied

    def _compute_gradient(self, output_gradient: ndarray) -> ndarray:
        """Compute the gradient of the sigmoid function."""
        gradient = (self.output_ * (1 - self.output_)) * output_gradient
        return gradient


class TanHFunctionOperator(BaseOperator):
    """Operator, that applies hyperbolic tangent function to the input data."""

    def __init__(self):
        super().__init__()

    def _apply(self, inference: bool = True) -> ndarray:
        """Apply the hyperbolic tangent function to the input data."""
        applied = np.tanh(self.input_)
        return applied

    def _compute_gradient(self, output_gradient: ndarray) -> ndarray:
        """Compute the gradient of the hyperbolic tangent function."""
        gradient = output_gradient * (1 - np.power(self.output_, 2))
        return gradient


class BiasAdditionOperator(ParameterizedOperator):
    """Operator, that adds a bias to the input data."""

    def __init__(self, bias: ndarray):
        super().__init__(bias)

    def _apply(self) -> ndarray:
        """Apply the bias addition operation."""
        applied = self.input_ + self.parameter
        return applied

    def _compute_gradient(self, output_gradient: ndarray) -> ndarray:
        """Compute the gradient of the bias addition operation with 
        respect to the input data.
        """
        gradient = np.ones_like(self.input_) * output_gradient
        return gradient

    def _compute_parameterized_gradient(
        self, output_gradient: ndarray
    ) -> ndarray:
        """Compute the gradient of the bias addition operation with 
        respect to the parameter.
        """
        output_gradient_matrix = np.ones_like(self.parameter) * output_gradient
        parametrized_gradient = np.sum(output_gradient_matrix, axis=0).reshape(
            1, output_gradient_matrix.shape[1]
        )
        return parametrized_gradient


class WeightedMultiplicationOperator(ParameterizedOperator):
    """Operator, that multiplies weights by the input data."""

    def __init__(self, weights: ndarray):
        super().__init__(weights)

    def _apply(self) -> Any:
        """Multiply the input data by the weights."""
        applied = np.dot(self.input_, self.parameter)
        return applied

    def _compute_gradient(self, output_gradient: ndarray) -> ndarray:
        """Compute the gradient of the product of weights and input data 
        with respect to the input.
        """
        gradient = np.dot(
            output_gradient, np.transpose(self.parameter, axes=(1, 0))
        )
        return gradient

    def _compute_parameterized_gradient(
        self, output_gradient: ndarray
    ) -> ndarray:
        """Compute the gradient of the product of weights and input data 
        with respect to the parameter.
        """
        parametrized_gradient = np.dot(
            np.transpose(self.input_, axes=(1, 0)), output_gradient
        )
        return parametrized_gradient


class DropoutOperator(BaseOperator):
    """Operator, that implements the Dropout operation. This means, that 
    is will drop off some neurons at the provided dropout rate. It 
    actually can reduce the risks of overfitting your models.
    """

    def __init__(self, dropout_rate: float = 0.75):
        super().__init__()
        self.dropout_rate = dropout_rate

    def _apply(self, inference: bool = False) -> ndarray:
        """Apply the Dropout operation."""
        if inference:
            return self.input_ * self.dropout_rate
        self.mask = np.random.binomial(
            1, self.dropout_rate, size=self.input_.shape
        )
        return self.input_ * self.mask

    def _compute_gradient(self, output_gradient: ndarray) -> ndarray:
        """Compute the gradient of the Dropout operation."""
        return output_gradient * self.mask


class ConvolutionOperator(ParameterizedOperator):
    """Operator, that implements the mathematical convolution operation."""

    def __init__(self, kernel: ndarray):
        super().__init__(kernel)
        self.parameter_size = kernel.shape[2]
        self.parameter_pad = self.parameter_size // 2

    def _pad_channel(self, input_tensor: ndarray) -> ndarray:
        """Pad the whole channel to avoid data leaks."""
        channel_padded = [
            np.pad(
                channel, self.parameter_pad, mode="constant", constant_values=0
            )
            for channel in input_tensor
        ]
        channel_padded = np.stack(channel_padded)
        return channel_padded

    def _get_image_patches(self, input_tensor: ndarray) -> ndarray:
        batch_padded = [
            self._pad_channel(observation) for observation in input_tensor
        ]
        batch_padded = np.stack(batch_padded)
        image_height = batch_padded.shape[2]
        height = image_height - self.parameter_size + 1

        patches = list()
        for i in range(height):
            for j in range(height):
                # fmt: off
                patch = batch_padded[
                    :, :, 
                    i : i + self.parameter_size, 
                    j : j + self.parameter_size,
                ]
                # fmt: on
                patches.append(patch)
        patches = np.stack(patches)
        return patches

    def _apply(self) -> Any:
        """Apply the convolution operation to the input."""

        self.batch_size = self.input_.shape[0]
        self.image_height = self.input_.shape[2]
        self.image_size = self.image_height * self.input_.shape[3]

        # fmt: off
        patch_size = (
            self.parameter.shape[0] * self.parameter.shape[2]
            * self.parameter.shape[3]
        )
        # fmt: on

        patches = self._get_image_patches(self.input_)
        patches = np.transpose(patches, axes=(1, 0, 2, 3, 4)).reshape(
            (self.batch_size, self.image_size, -1)
        )
        parameter = np.transpose(self.parameter, axes=(0, 2, 3, 1)).reshape(
            (patch_size, -1)
        )

        output_tensor = np.matmul(patches, parameter)
        output_tensor = np.transpose(
            output_tensor.reshape(
                (self.batch_size, self.image_height, self.image_height, -1)
            ),
            axes=(0, 3, 2, 1),
        )

        return output_tensor

    def _compute_gradient(self, output_gradient: ndarray) -> Any:
        """Compute the gradient of the convolution operation with 
        respect to the input.
        """

        output_patches = self._get_image_patches(output_gradient)
        output_patches = np.transpose(
            output_patches, axes=(1, 0, 2, 3, 4)
        ).reshape((self.batch_size * self.image_size, -1))
        parameter = np.transpose(
            self.parameter.reshape((self.parameter.shape[0], -1)), axes=(1, 0)
        )

        gradient = np.matmul(output_patches, parameter)
        gradient = np.transpose(
            # fmt: off
            gradient.reshape((
                self.batch_size, self.image_height,
                self.image_height, self.parameter.shape[0],
            )), 
            axes=(0, 3, 1, 2),
            # fmt: on
        )

        return gradient

    def _compute_parameterized_gradient(self, output_gradient: ndarray) -> Any:
        """Compute the gradient of the convolution operation with 
        respect to the parameter.
        """

        input_channels = self.parameter.shape[0]
        output_channels = self.parameter.shape[1]

        input_patches = self._get_image_patches(self.input_)
        input_patches = np.transpose(
            input_patches.reshape((self.batch_size * self.image_size, -1)),
            axes=(1, 0),
        )

        output_gradient = np.transpose(
            output_gradient, axes=(0, 2, 3, 1)
        ).reshape((self.batch_size * self.image_size, -1))
        parameterized_gradient = np.matmul(input_patches, output_gradient)
        parameterized_gradient = np.transpose(
            # fmt: off
            parameterized_gradient.reshape((
                input_channels, self.parameter_size,
                self.parameter_size, output_channels,
            )), 
            axes=(0, 3, 1, 2),
            # fmt: on
        )

        return parameterized_gradient


class FlattenOperator(BaseOperator):
    """Operator, that implements the Flatten operation. FlattenOperator 
    is required as the last operator applied in the convolutional neural 
    network if You want to see the human readable results coming out of 
    Your model.
    """

    def __init__(self):
        super().__init__()

    def _apply(self, inference: bool = False) -> ndarray:
        """Apply the Flatten operation."""
        return self.input_.reshape(self.input_.shape[0], -1)

    def _compute_gradient(self, output_gradient: ndarray) -> Any:
        """Compute the gradient of the Flatten operation."""
        return output_gradient.reshape(self.input_.shape)
