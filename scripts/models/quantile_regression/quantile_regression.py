"""
Module: quantile_regression_model
This module implements the QuantileRegressionModel, which trains a separate QuantileRegressor
model for each quantile. It provides functionality for training, evaluation, and hyperparameter
optimization using cross-validation.
"""

import os
import glob
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from joblib import dump, load, Parallel, delayed
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_pinball_loss
from scripts.models.base_model import BaseModel
from scripts.hyperparameter.hyperparameter_optimization import HyperparameterOptimizer
from scripts.predictor.predictor import Predictor
from scripts.utils.logger import logger
import pdb


class QuantileRegressionModel(BaseModel):
    def __init__(self, config):
        """
        Initialize the QuantileRegressionModel with configuration parameters.

        Args:
            config (dict): Configuration dictionary containing keys such as "models_dir",
                "quantiles", "params", "optuna", "optuna_search_space", and "test_size".
        """
        super().__init__(**config)
        self.config = config
        self.model_dir = config["models_dir"]
        self.quantiles = config.get("quantiles")
        self.params = config.get("params", {})
        self.optuna_cfg = config.get("optuna", {})
        self.optuna_space = config.get("optuna_search_space", {}).get("param_space", {})
        self.test_size = config.get("test_size", 0.2)
        self.fold = 0

    def train(self, X_train_input, y_train_input, config=None):
        """
        Train a separate QuantileRegressor model for each quantile.
        Models are saved as '<model>_model_<q>.joblib', and training information is collected.

        Args:
            X_train_input (DataFrame): Training features.
            y_train_input (Series/DataFrame): Training target values.
            config (dict, optional): Additional configuration to override the instance config.
        """
        if config is not None:
            self.config = config

        existing_models = glob.glob(
            os.path.join(self.config["models_dir"], f"{self.config['model']}_model*.joblib")
        )
        if len(existing_models) == len(self.quantiles):
            logger.info(
                "[QuantileRegressionModel] All %s models already exist. Skipping training.",
                len(self.quantiles),
            )
            return

        logger.info(
            "[QuantileRegressionModel] Starting training. Models will be saved in %s.",
            self.config["models_dir"],
        )
        os.makedirs(self.config["models_dir"], exist_ok=True)

        def _train_single_quantile(q, X_train_local, y_train_local):
            """
            Train a QuantileRegressor for a single quantile and collect training information.

            Args:
                q (float): The quantile to train.
                X_train_local (DataFrame): Training features.
                y_train_local (Series/DataFrame): Training target values.

            Returns:
                tuple: A tuple containing the quantile and a status string.
            """
            model_path = os.path.join(
                self.config["models_dir"], f"{self.config['model']}_model_{q}.joblib"
            )
            qr = QuantileRegressor(quantile=q, **self.params)
            start_time = time.time()
            qr.fit(X_train_local, y_train_local)
            training_duration = time.time() - start_time
            self.collect_training_info(
                q,
                model_path,
                qr,
                X_train_local,
                y_train_local,
                X_val=None,
                y_val=None,
                training_duration=training_duration,
                start_time=start_time,
            )

            return (q, "OK")

        results = Parallel(n_jobs=-2)(
            delayed(_train_single_quantile)(q, X_train_input, y_train_input) for q in self.quantiles
        )
        logger.info(
            "[QuantileRegressionModel] Training completed. Trained quantiles: %s", self.quantiles
        )

    def collect_training_info(
        self,
        q,
        model_path,
        model,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        training_duration=0,
        start_time=None,
    ):
        """
        Collect and save training information including model parameters, data summaries,
        and training metrics.

        Args:
            q (float or str): The quantile for which the model was trained.
            model_path (str): File path where the model is saved.
            model: Trained model instance.
            X_train (DataFrame): Training features.
            y_train (Series/DataFrame): Training target values.
            X_val (DataFrame, optional): Validation features.
            y_val (Series/DataFrame, optional): Validation target values.
            training_duration (float, optional): Duration of training in seconds.
            start_time (float, optional): Timestamp when training started.
        """
        training_info = {}
        training_info["quantile"] = q
        training_info["model_path"] = model_path
        training_info["feature_names"] = list(X_train.columns)
        training_info["hyperparameters"] = model.get_params()
        training_info["training_data_shape"] = X_train.shape
        training_info["training_duration_sec"] = training_duration

        # Summary statistics of the training data
        training_info["training_data_summary"] = {
            "X_train_mean": X_train.mean().to_dict(),
            "X_train_std": X_train.std().to_dict(),
            "X_train_quantiles": X_train.quantile([0.25, 0.5, 0.75]).to_dict(),
        }
        training_info["target_summary"] = {
            "y_train_mean": y_train.mean(),
            "y_train_std": y_train.std(),
            "y_train_min": y_train.min(),
            "y_train_max": y_train.max(),
        }

        # Metrics on training data
        y_train_pred = model.predict(X_train)
        training_info["train_metrics"] = {
            "mae": mean_absolute_error(y_train, y_train_pred),
            "rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "mape": mean_absolute_percentage_error(y_train, y_train_pred),
        }

        # QuantileRegressor does not provide feature importances; use absolute coefficients instead
        training_info["feature_importances"] = np.abs(model.coef_)

        # No "best_iteration" or "best_score" as in LightGBM
        training_info["best_iteration"] = None
        training_info["best_score"] = None

        # Additional metadata
        training_info["training_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if start_time:
            training_start = datetime.fromtimestamp(start_time)
            training_end = training_start + timedelta(seconds=training_duration)
            training_info["training_start_time"] = training_start.isoformat()
            training_info["training_end_time"] = training_end.isoformat()

        training_info["config"] = self.config  # For reproducibility

        # Save model and training info using BaseModel's save_model method
        self.save_model(model_path, {"model": model, "training_info": training_info})

        logger.info(
            "[QuantileRegressionModel] Model for q=%s saved: %s", q, model_path
        )

    def evaluate_trial(self, trial, X_train, y_train):
        """
        Evaluate a single trial during the Optuna hyperparameter optimization process.
        This method:
          - Constructs trial parameters (e.g., alpha, solver) from the defined search space.
          - Uses TimeSeriesSplit for cross-validation.
          - For each fold, trains a QuantileRegressor (typically for q=0.5) and computes the
            mean pinball loss on the validation set.
          - Returns the average pinball loss across all folds.

        Args:
            trial: The Optuna trial object.
            X_train (DataFrame): Training features.
            y_train (Series/DataFrame): Training target values.

        Returns:
            float: The average pinball loss across all validation folds.
        """
        trial_params = {
            "quantile": self.optuna_cfg.get("quantile_for_search", 0.5),
            "alpha": trial.suggest_float("alpha", *self.optuna_space["alpha"]),
            "solver": self.optuna_cfg.get("solver", "highs"),
            "solver_options": self.params.get("solver_options"),
        }

        model = QuantileRegressor(**trial_params)
        losses = []

        splits = self.config["optuna"].get("n_splits")
        test_size_abs = int(len(X_train) * self.config.get("test_size"))
        inner_tscv = TimeSeriesSplit(n_splits=splits, test_size=test_size_abs, gap=0)

        hyper_path = self.config["hyperparameter_dir"]
        self.fold += 1
        os.makedirs(os.path.join(hyper_path, f"fold_{self.fold}"), exist_ok=True)
        for train_index_inner, val_index_inner in inner_tscv.split(X_train):
            model_path = os.path.join(
                hyper_path, f"fold_{self.fold}", "quantile_regression_model_0.5.joblib"
            )

            X_train_inner = X_train.iloc[train_index_inner]
            X_val_inner = X_train.iloc[val_index_inner]
            y_train_inner = y_train.iloc[train_index_inner]
            y_val_inner = y_train.iloc[val_index_inner]

            model.fit(X_train_inner, y_train_inner)

            predictor = Predictor(self.config)
            predictions_inner = predictor.predict(
                x_test=X_val_inner,
                y_train=y_train_inner.to_frame("gesamt"),
                y_test=y_val_inner.to_frame("gesamt"),
                model=model,
            )
            predictions_inner_filtered = predictions_inner[
                predictions_inner.index.map(self.is_in_time_window)
            ]

            loss = mean_pinball_loss(
                predictions_inner_filtered["gesamt"],
                predictions_inner_filtered["gesamt_pred"],  # Note: column name
                alpha=0.5,
            )
            losses.append(loss)
        self.collect_training_info(
            "0.5",
            model_path,
            model,
            X_train=X_train_inner,
            y_train=y_train_inner,
            X_val=None,
            y_val=None,
            training_duration=0,
            start_time=None,
        )
        avg_loss = sum(losses) / len(losses)

        return avg_loss

    def get_optuna_search_space(self):
        """
        Retrieve the search space for Optuna hyperparameter optimization.

        Returns:
            dict: The parameter search space.
        """
        return self.config["optuna_search_space"].get("param_space", {})
