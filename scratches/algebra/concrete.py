import numpy as np
from numpy import ndarray

from typing import Optional

from ._base import BaseSolver


class GaussianEliminationBaseSolver(BaseSolver):
    """Solve independent systems of linear equations using Gaussian Elimination
    method with partial pivoting.
    """

    def __init__(self):
        super().__init__()

    def _eliminate(self, iteration: int, rows: int) -> None:
        """Perform Gaussian Elimination algorithm."""
        for row in range(iteration + 1, rows):
            scaling_factor = (
                self.augmented_matrix[row][iteration]
                / self.augmented_matrix[iteration][iteration]
            )
            self.augmented_matrix[row] = (
                self.augmented_matrix[row]
                - scaling_factor * self.augmented_matrix[iteration]
            )

    def _substitute_backwards(self, rows: int) -> None:
        """Substitute backwards to compute solution matrix."""
        for row in range(rows - 2, -1, -1):
            self.solution_vector[row] = self.augmented_matrix[row][rows]

            for column in range(row + 1, rows):
                self.solution_vector[row] = (
                    self.solution_vector[row]
                    - self.augmented_matrix[row][column]
                    * self.solution_vector[column]
                )
            self.solution_vector[row] = (
                self.solution_vector[row] / self.augmented_matrix[row][row]
            )

    def fit(
        self, coefficient_matrix: ndarray, scalars_vector: ndarray, **kwargs
    ) -> None:
        """Check whether given coefficient_matrix is consistent, independent
        and scalars vector has only one column before fitting.

        :parameter coefficient_matrix: Matrix of coefficients.
            :type coefficient_matrix: ndarray
        :parameter scalars_vector: column vector of scalars vector.
            :type scalars_vector: ndarray

        :returns: ``None``
            :rtype: NoneType
        """
        super().fit(coefficient_matrix, scalars_vector)

    def solve(self, *, round_to: Optional[int] = None) -> ndarray:
        """Solve the independent system of linear equations.

        :parameter round_to: Number of decimals to which solution should be
            rounded.
            :type round_to: int

        :returns: Solution matrix for the given system of linear equations.
            :rtype: ndarray
        """
        rows = len(self.scalars_vector)
        index = rows - 1
        iteration = 0
        self.solution_vector = np.zeros(rows)

        while iteration < rows:
            self._apply_partial_pivoting(iteration, rows)
            self._eliminate(iteration, rows)
            iteration += 1

        self.solution_vector[index] = (
            self.augmented_matrix[index][rows]
            / self.augmented_matrix[index][index]
        )

        self._substitute_backwards(rows)

        self.solution_vector = self.solution_vector.reshape(-1, 1)
        if round_to:
            self.solution_vector.round(round_to)

        return self.solution_vector


class LeastSquaresSolver(GaussianEliminationBaseSolver):
    """Least squares solver solves dependent systems of linear equations
    using Least Squares method.
    """

    def __init__(self):
        super().__init__()

    def fit(
        self, coefficient_matrix: ndarray, scalars_vector: ndarray, **kwargs
    ):
        """Perform computations to transform given matrices in appropriate for
        GaussianEliminationBaseSolver form.

        :parameter coefficient_matrix: Matrix of coefficients
            :type coefficient_matrix: ndarray
        :parameter scalars_vector: column vector of scalars vector
            :type scalars_vector: ndarray

        :returns: ``None``
            :rtype: NoneType
        """
        transposed = coefficient_matrix.T
        self.coefficient_matrix = transposed @ coefficient_matrix
        self.scalars_vector = transposed @ scalars_vector
        super().fit(self.coefficient_matrix, self.scalars_vector)

    def solve(self, *, round_to: Optional[int] = None):
        """Solve the transformed system of linear equations using Gaussian
        Elimination method.

        :parameter round_to: Number of decimals to which solution should be
            rounded
            :type round_to: int

        :returns: Solution matrix for the given system of linear equations
            :rtype: ndarray
        """
        self.solution_vector = super().solve(round_to=round_to)
        return self.solution_vector
