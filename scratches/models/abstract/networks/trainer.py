from copy import deepcopy
from logging import basicConfig, info, INFO
from time import perf_counter
from typing import Any, Generator, Optional, Tuple
from sys import float_info

import numpy as np
from numpy import ndarray

from .network import NeuralNetwork
from ..optimizers._base import BaseOptimizer


class Trainer:
    """A class for training neural networks."""

    def __init__(self, network: NeuralNetwork, optimizer: BaseOptimizer):
        self.network = network
        self.optimizer = optimizer
        self.best_loss = float_info.max
        self.optimizer.network = network
        self.x_train: ndarray
        self.x_valid: ndarray
        self.y_train: ndarray
        self.y_valid: ndarray
        basicConfig(format="Trainer: %(message)s", level=INFO)

    def fit(
        self,
        x_train: ndarray,
        x_valid: ndarray,
        y_train: ndarray,
        y_valid: ndarray,
        *,
        epochs: int = 100,
        evaluate_every_epochs: int = 10,
        batch_size: int = 100,
        restart: bool = True,
        stop_early: bool = True,
        evaluate_every_batches: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """Fits the neural network to the training data.

        :parameter x_train: The features training data;
            :type x_train: ndarray
        :parameter x_valid: The features validating data;
            :type x_valid: ndarray
        :parameter y_train: The target training data;
            :type y_train: ndarray
        :parameter y_valid: The target validating data.
            :type y_valid: ndarray

        :keyword epochs: The number of training epochs;
            :type epochs: int
        :keyword evaluate_every: Evaluate the model every n epochs;
            :type evaluate_every: int
        :keyword batch_size: The batch size for training;
            :type batch_size: int
        :keyword random_seed: The random seed for reproducibility;
            :type random_seed: Optional[int]
        :keyword restart: Whether to restart the training.
            :type restart: bool
        """

        # fmt: off
        self.x_train, self.x_valid, self.y_train, self.y_valid = (
            x_train, x_valid, y_train, y_valid,
        )
        # fmt: on
        self.optimizer.max_epochs = epochs
        self.optimizer._setup_decay()

        if random_seed is not None:
            np.random.seed(random_seed)
        if restart:
            for layer in self.network.layers.values():
                layer.root = True
            self.best_loss = float_info.max

        training_start_time = perf_counter()
        for epoch in range(epochs):

            if not (epoch + 1) % evaluate_every_epochs:
                last_evaluated_model = deepcopy(self.network)

            self._permute_data()
            new_batches = self._get_batches(batch_size)
            self._batch_train(new_batches, evaluate_every_batches)

            if not (epoch + 1) % evaluate_every_epochs:
                predicted = self.network.feed_forward(self.x_valid)
                valid_loss = self.network.loss_function.feed_forward(
                    self.y_valid, predicted
                )
                
                if valid_loss < self.best_loss or stop_early != True:
                    self.best_loss = valid_loss
                    info(
                        f"Validation loss after {epoch + 1} "
                        f"epochs is {valid_loss}."
                    )
                    continue
                info(
                    f"Increased loss after {epoch + 1} epochs "
                    f"{valid_loss}. Stopping training early..."
                )
                self.best_loss = valid_loss
                self.network = last_evaluated_model  # pyright: ignore[reportPossiblyUnboundVariable]
                self.optimizer.network = self.network
                break

            if self.optimizer.final_learning_rate:
                self.optimizer._decay_learning_rate()
        training_end_time = perf_counter()
        training_time = training_end_time - training_start_time
        info(f"Training complete! It took {training_time} seconds.")

    def _permute_data(self) -> None:
        permutation = np.random.permutation(self.x_train.shape[0])
        # fmt: off
        self.x_train, self.y_train = (
            self.x_train[permutation], self.y_train[permutation],
        )
        # fmt: on

    def _get_batches(
        self, batch_size: int = 100
    ) -> Generator[Any, Tuple[ndarray, ndarray], None]:
        n_samples = self.x_train.shape[0]
        for i in range(0, n_samples, batch_size):
            batch_end = i + batch_size
            # fmt: off
            x_batch, y_batch = (
                self.x_train[i:batch_end], self.y_train[i:batch_end],
            )
            # fmt: on
            yield x_batch, y_batch

    def _batch_train(
        self,
        batches: Generator[Any, Tuple[ndarray, ndarray], None],
        evaluate_every_batches: Optional[int],
    ) -> None:
        for batch_number, (x_batch, y_batch) in enumerate(batches):
            self.network.train(x_batch, y_batch)
            self.optimizer.step()

            if (
                evaluate_every_batches
                and not (batch_number + 1) % evaluate_every_batches
            ):
                batch_predicted = self.network.feed_forward(x_batch)
                batch_loss = self.network.loss_function.feed_forward(
                    y_batch, batch_predicted
                )
                info(
                    f"Loss after {batch_number + 1} batches is {batch_loss}."
                )
