"""
Module: base_model
Defines the abstract BaseModel class which provides an interface and common functionality
for models including training, hyperparameter optimization, evaluation, persistence,
and time window checking. This module is designed to be extended by concrete model implementations.
"""

from abc import ABC, abstractmethod
import optuna
from joblib import dump, load
import os
from scripts.hyperparameter.hyperparameter_optimization import HyperparameterOptimizer
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    def __init__(self, **config):
        """
        Initialize the BaseModel.

        Args:
            **config: Arbitrary keyword arguments containing configuration parameters.
                Expected keys include "models_dir" and "quantiles".
        """
        self.config = config
        self.model = None
        self.model_dir = config["models_dir"]
        self.quantiles = config["quantiles"]

    @abstractmethod
    def train(self, X_train, y_train, config=None):
        """
        Train the model without using Optuna.

        Args:
            X_train: Training features.
            y_train: Training labels.
            config (optional): Additional configuration for training.
        """
        pass

    @abstractmethod
    def get_optuna_search_space(self):
        """
        Define the Optuna search space for this model.

        Returns:
            dict: A dictionary representing the search space for hyperparameters.
        """
        pass

    @abstractmethod
    def evaluate_trial(self, trial, X_train, y_train):
        """
        Evaluate a single Optuna trial. This method is model-specific.

        Args:
            trial: The Optuna trial object.
            X_train: Training features.
            y_train: Training labels.

        Returns:
            A metric value used to assess the trial.
        """
        pass

    def optimize_hyperparameters(self, X_train, y_train):
        """
        Perform hyperparameter optimization using Optuna.

        Args:
            X_train: Training features.
            y_train: Training labels.

        Returns:
            dict or None: The best hyperparameters found, or None if Optuna is disabled.
        """
        if not self.config.get("optuna", {}).get("use_optuna"):
            logger.info("Optuna is disabled.")
            return None

        optimizer = HyperparameterOptimizer(self)
        best_params = optimizer.optimize(X_train, y_train)
        return best_params

    def save_model(self, model_path, model_data):
        """
        Save the model and associated data.

        Args:
            model_path (str): The path where the model should be saved.
            model_data: The model data to be saved.
        """
        dump(model_data, model_path)

    def load_model(self, model_path):
        """
        Load a model.

        Args:
            model_path (str): The path to the saved model.

        Returns:
            The loaded model.
        """
        return load(model_path)

    @staticmethod
    def is_in_time_window(ts):
        """
        Check if the given timestamp is within the defined time window.

        Args:
            ts (datetime): The timestamp to check.

        Returns:
            bool: True if the timestamp is within the time window, False otherwise.
        """
        if ts.weekday() == 2:
            return ts.hour >= 23
        elif ts.weekday() in (3, 4):
            return True
        elif ts.weekday() == 5:
            return ts.hour <= 23
        else:
            return False
