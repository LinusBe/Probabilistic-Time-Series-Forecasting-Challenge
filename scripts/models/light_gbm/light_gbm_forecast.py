"""
Module: light_gbm_forecast
This module implements the LightGBMForecastModel, which trains a separate LightGBM model for each quantile
(using objective="quantile" and alpha set to the quantile value). The model files are saved and training
information (including learning curves and metadata) is collected. Hyperparameter optimization support via Optuna
is also provided.
"""

import os
import glob
import numpy as np
import pandas as pd
from joblib import dump, Parallel, delayed
from lightgbm import LGBMRegressor, early_stopping
import pdb
from sklearn.model_selection import TimeSeriesSplit
from scripts.models.base_model import BaseModel
from sklearn.metrics import mean_pinball_loss
from scripts.utils.logger import logger
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from scripts.hyperparameter.hyperparameter_optimization import HyperparameterOptimizer
import time
from datetime import datetime
from scripts.predictor.predictor import Predictor
from datetime import datetime, timedelta


class LightGBMForecastModel(BaseModel):
    def __init__(self, config):
        """
        Initialize the LightGBMForecastModel.

        Args:
            config (dict): Configuration dictionary containing keys such as "models_dir", "quantiles",
                           "params", "optuna", "early_stopping", etc.
        """
        super().__init__(**config)
        self.config = config
        self.model_dir = config["models_dir"]
        self.quantiles = config["quantiles"]
        self.params = config.get("params")
        self.optuna_cfg = config.get("optuna", {})
        self.optuna_space = config["optuna_search_space"].get("param_space", {})
        self.test_size = self.config.get("test_size")
        self.fold = 0
        self.stopping_rounds = self.config["early_stopping"].get("rounds", None)
        self.stopping_delta = self.config["early_stopping"].get("delta", None)
        self.n_estimators = self.params.get("n_estimators")

    def train(self, X_train_input, y_train_input, config=None):
        """
        Train a separate LightGBM model for each quantile (objective="quantile", alpha=q).
        The models are saved as '<model>_model_<q>.joblib' and a plot of learning curves
        (training vs. validation) is also generated.

        Args:
            X_train_input (DataFrame): Training features.
            y_train_input (Series/DataFrame): Training target values.
            config (dict, optional): Additional configuration to override the instance config.
        """
        if config is not None:
            self.config = config
        # Check: Do all models already exist?
        existing_models = glob.glob(
            os.path.join(self.config["models_dir"], f"{self.config['model']}_model*.joblib")
        )
        if len(existing_models) == len(self.quantiles):
            logger.info(
                "[LightGBMForecastModel] All %d models already exist. Skipping training.",
                len(self.quantiles),
            )
            return

        logger.info(
            "[LightGBMForecastModel] Starting training. Models will be saved in %s.",
            self.config["models_dir"],
        )
        os.makedirs(self.config["models_dir"], exist_ok=True)

        def _train_single_quantile(q, X_train_local, y_train_local):
            """
            Train a LightGBM model for a single quantile and collect training information.

            Args:
                q (float): The quantile value.
                X_train_local (DataFrame): Local training features.
                y_train_local (Series/DataFrame): Local training target values.
            """
            model_path = os.path.join(
                self.config["models_dir"], f"{self.config['model']}_model_{q}.joblib"
            )

            # Set the number of estimators for the current quantile
            self.params["n_estimators"] = self.n_estimators[str(q)]
            lgb = LGBMRegressor(objective="quantile", alpha=q, **self.params)
            if self.config["eval_set"].get("use"):
                eval_size = self.config["eval_set"].get("size")
            else:
                eval_size = 0
            n_total = len(X_train_local)
            n_train = int(n_total * (1 - eval_size))

            X_train = X_train_local.iloc[:n_train]
            X_val = X_train_local.iloc[n_train:]
            y_train = y_train_local.iloc[:n_train]
            y_val = y_train_local.iloc[n_train:]
            start_time = time.time()
            lgb.fit(
                X_train,
                y_train,
                eval_set=[(X, y) for X, y in [(X_train, y_train), (X_val, y_val)] if not X.empty and not y.empty],
                eval_metric="quantile",
                callbacks=[early_stopping(stopping_rounds=self.stopping_rounds, min_delta=self.stopping_delta)]
                if self.stopping_rounds is not None
                else None,
            )
            training_duration = time.time() - start_time

            # Save model and metadata
            self.collect_training_info(q, model_path, lgb, X_train, y_train, X_val, y_val, training_duration, start_time)

        for q in self.quantiles:
            _train_single_quantile(q, X_train_input, y_train_input)

        logger.info(
            "[LightGBMForecastModel] Training completed. Trained quantiles: %s", self.quantiles
        )

    def collect_training_info(self, q, model_path, model, X_train, y_train, X_val=None, y_val=None, training_duration=0, start_time=None):
        """
        Collect training information and save the model along with its metadata.

        Args:
            q (float): The quantile value.
            model_path (str): Path where the model will be saved.
            model: Trained LightGBM model.
            X_train (DataFrame): Training features.
            y_train (Series/DataFrame): Training target values.
            X_val (DataFrame, optional): Validation features.
            y_val (Series/DataFrame, optional): Validation target values.
            training_duration (float, optional): Training duration in seconds.
            start_time (float, optional): Timestamp when training started.

        Returns:
            dict: The training information dictionary.
        """
        training_info = {}
        training_info["quantile"] = q
        training_info["model_path"] = model_path
        training_info["feature_names"] = list(X_train.columns)
        training_info["hyperparameters"] = model.get_params()
        training_info["training_data_shape"] = X_train.shape
        training_info["training_duration_sec"] = training_duration

        # Statistics of the training data
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
        if self.config["eval_set"].get("use"):
            training_info["eval_set_sizes"] = {
                "train": X_train.shape[0],
                "validation": X_val.shape[0],
            }
            # Also store summary of y_val (optional)
            training_info["target_summary_valid"] = {
                "y_val_mean": y_val.mean(),
                "y_val_std": y_val.std(),
                "y_val_min": y_val.min(),
                "y_val_max": y_val.max(),
            }
            y_val_pred = model.predict(X_val)
            training_info["validation_metrics"] = {
                "mae": mean_absolute_error(y_val, y_val_pred),
                "rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
                "mape": mean_absolute_percentage_error(y_val, y_val_pred),
            }
        else:
            training_info["validation_metrics"] = None

        # Metrics on training data
        y_train_pred = model.predict(X_train)
        training_info["train_metrics"] = {
            "mae": mean_absolute_error(y_train, y_train_pred),
            "rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "mape": mean_absolute_percentage_error(y_train, y_train_pred),
        }

        # Evaluation history and best iteration
        training_info["evals_result"] = model.evals_result_
        training_info["best_iteration"] = model.best_iteration_
        training_info["best_score"] = model.best_score_

        # Feature importances
        training_info["feature_importances"] = model.feature_importances_.tolist()

        # Model dump (if needed)
        training_info["model_dump"] = model.booster_.dump_model()

        # Additional metadata
        training_info["training_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if start_time:
            training_start = datetime.fromtimestamp(start_time)
            training_end = training_start + timedelta(seconds=training_duration)
            training_info["training_start_time"] = training_start.isoformat()
            training_info["training_end_time"] = training_end.isoformat()
            training_info["config"] = self.config  # for reproducibility

        self.save_model(model_path, {"model": model, "training_info": training_info})
        logger.info("LightGBM model for q=%s saved: %s", q, model_path)

        return training_info

    def evaluate_trial(self, trial, X_train, y_train):
        """
        Evaluate a single trial during the Optuna hyperparameter optimization process.

        Args:
            trial: The Optuna trial object.
            X_train (DataFrame): Training features.
            y_train (Series/DataFrame): Training target values.

        Returns:
            float: The average pinball loss across the cross-validation folds.
        """
        trial_params = {
            "objective": "quantile",
            "alpha": self.optuna_cfg.get("quantile", 0.5),
            "num_leaves": trial.suggest_int("num_leaves", *self.optuna_space["num_leaves"]),
            "max_depth": trial.suggest_int("max_depth", *self.optuna_space["max_depth"]),
            "learning_rate": trial.suggest_float("learning_rate", *self.optuna_space["learning_rate"], log=True),
            "n_estimators": trial.suggest_int("n_estimators", *self.optuna_space["n_estimators"]),
            "lambda_l1": trial.suggest_float("lambda_l1", *self.optuna_space["lambda_l1"], log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", *self.optuna_space["lambda_l2"], log=True),
            "boosting_type": trial.suggest_categorical("boosting_type", self.optuna_space["boosting_type"]),
            "feature_fraction": trial.suggest_float("feature_fraction", *self.optuna_space["feature_fraction"]),
            "bagging_fraction": trial.suggest_float("bagging_fraction", *self.optuna_space["bagging_fraction"]),
            "bagging_freq": trial.suggest_int("bagging_freq", *self.optuna_space["bagging_freq"]),
            "min_child_samples": trial.suggest_int("min_child_samples", *self.optuna_space["min_child_samples"]),
            "min_child_weight": trial.suggest_float("min_child_weight", *self.optuna_space["min_child_weight"], log=True),
            "subsample": trial.suggest_float("subsample", *self.optuna_space["subsample"]),
            "subsample_freq": trial.suggest_int("subsample_freq", *self.optuna_space["subsample_freq"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *self.optuna_space["colsample_bytree"]),
            "max_bin": trial.suggest_int("max_bin", *self.optuna_space["max_bin"]),
            "min_split_gain": trial.suggest_float("min_split_gain", *self.optuna_space["min_split_gain"]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", *self.optuna_space["min_data_in_leaf"]),
            "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", *self.optuna_space["min_sum_hessian_in_leaf"]),
            "verbosity": -1,
            "device_type": "cpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
        }

        model = LGBMRegressor(**trial_params)

        if self.config["eval_set"].get("use"):
            eval_size = self.config["eval_set"].get("size")
        else:
            eval_size = 0

        losses = []

        splits = self.config["optuna"].get("n_splits")
        test_size_abs = int(len(X_train) * self.config.get("test_size"))
        inner_tscv = TimeSeriesSplit(n_splits=splits, test_size=test_size_abs, gap=0)

        hyper_path = self.config["hyperparameter_dir"]

        self.fold += 1
        os.makedirs(os.path.join(hyper_path, "fold_%d" % self.fold), exist_ok=True)
        for train_index_inner, val_index_inner in inner_tscv.split(X_train):
            model_path = os.path.join(hyper_path, "fold_%d" % self.fold, "light_gbm_model_0.5.joblib")

            X_train_inner = X_train.iloc[train_index_inner]
            X_val_inner = X_train.iloc[val_index_inner]
            y_train_inner = y_train.iloc[train_index_inner]
            y_val_inner = y_train.iloc[val_index_inner]

            n_total = len(X_train_inner)
            n_train = int(n_total * (1 - eval_size))

            X_train_ii = X_train_inner.iloc[:n_train]
            X_val_ii = X_train_inner.iloc[n_train:]
            y_train_ii = y_train_inner.iloc[:n_train]
            y_val_ii = y_train_inner.iloc[n_train:]

            model.fit(
                X_train_ii,
                y_train_ii,
                eval_set=[(X, y) for X, y in [(X_train_ii, y_train_ii), (X_val_ii, y_val_ii)] if not X.empty and not y.empty],
                eval_metric="quantile",
                callbacks=[early_stopping(stopping_rounds=self.stopping_rounds, min_delta=self.stopping_delta)]
                if self.stopping_rounds is not None
                else None,
            )

            # Do not save the model here; this is only for optimization.
            # Use Predictor for prediction.
            predictor = Predictor(self.config)
            predictions_inner = predictor.predict(
                x_test=X_val_inner,
                y_train=y_train_inner.to_frame("gesamt"),
                y_test=y_val_inner.to_frame("gesamt"),
                model=model,
            )
            # Filter by time window.
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
            X_train=X_train_ii,
            y_train=y_train_ii,
            X_val=X_val_ii,
            y_val=y_val_ii,
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
