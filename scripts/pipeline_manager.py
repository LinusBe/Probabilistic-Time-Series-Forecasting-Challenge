"""
Pipeline management module.

This module defines the PipelineManager class, which orchestrates the end-to-end
forecasting workflow, including feature engineering, optional feature selection,
hyperparameter tuning (Optuna), model training, rolling cross-validation, and evaluation.
"""

import logging
import os
import pdb
import json
from datetime import timedelta

import pandas as pd

from scripts.feature_engineering.feature_pipeline import FeaturePipeline
from scripts.feature_engineering.feature_selection import FeatureSelectionManager
from scripts.predictor.predictor import Predictor
from scripts.evaluation.report_manager import ExtendedReportManager

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Manages the entire forecasting pipeline from feature engineering to evaluation.

    Attributes:
        model: The forecasting model instance.
        config (dict): Configuration dictionary for the current run.
        model_name (str): The name of the model (e.g., 'light_gbm', 'baseline').
        version (str): Configuration version key (e.g., 'v1.0.0').
        dataset (str): Dataset name (e.g., 'energy', 'no2').
        cfg_manager: Instance of the configuration manager.
    """

    def __init__(self, model, config, model_name, version, dataset, cfg_manager):
        """
        Initialize the PipelineManager with model, config, and other pipeline details.

        Args:
            model: Instantiated model object.
            config (dict): Configuration for the current run.
            model_name (str): Name of the model (e.g., 'light_gbm').
            version (str): Version key in the configuration file (e.g., 'v1.0.0').
            dataset (str): Name of the dataset (e.g., 'energy', 'no2').
            cfg_manager: Configuration manager instance.
        """
        self.model = model
        self.config = config
        self.fp = FeaturePipeline(config)
        self.cfg_manager = cfg_manager
        self.model_name = model_name
        self.version = version
        self.dataset = dataset
        self.path_to_cfg = os.path.join(
            self.cfg_manager.project_root,
            "configs",
            self.dataset,
            f"{self.model_name}.yml"
        )

    def run_pipeline(self):
        """
        Execute the entire pipeline: feature engineering, data splitting,
        optional feature selection, hyperparameter tuning, and model training.
        """
        self.fp.run()
        self.X_train, self.y_train = self.fp.get_train_data()
        self.X_test, self.y_test = self.fp.get_test_data()

        if self.config.get("training_mode") == "simple_split":
            self.run_simple_split()
        elif self.config["training_mode"] == "rolling_cv":
            self.run_rolling_cv()
        else:
            raise ValueError(f"Unknown training_mode: {self.config['training_mode']}")

    def run_simple_split(self):
        """
        Run a simple train/test split workflow:
        1. Optional feature selection.
        2. Optional hyperparameter optimization (Optuna).
        3. Train the model.
        4. Generate predictions.
        5. Evaluate results.
        """
        # Feature selection process
        if self.config.get("feature_selection", {}).get("run_selection"):
            self.run_feature_selection()
            # Stop after feature selection
            return

        # Optuna process
        if self.config.get("optuna", {}).get("use_optuna"):
            self.run_optuna()

        # Training process
        if self.config["model"] == "baseline":
            self.model.train()
            final_predictions = self.model.predict(
                self.X_test,
                self.y_train.to_frame("gesamt"),
                self.y_test.to_frame("gesamt")
            )
        else:
            # check if file predictions exist
            if os.path.exists(os.path.join(self.config['predictions_dir'], "final_predictions.csv")):
                
                final_predictions = pd.read_csv(os.path.join(self.config['predictions_dir'], "final_predictions.csv"), index_col=0, parse_dates=True)
            else:
        
                self.model.train(self.X_train, self.y_train)
                predictor = Predictor(self.config)
                self.X_test.to_parquet(
                    os.path.join(self.config["models_dir"], "X_test.parquet")
                )
                final_predictions = predictor.predict(
                    x_test=self.X_test,
                    x_train=self.X_train,
                    y_train=self.y_train.to_frame("gesamt"),
                    y_test=self.y_test.to_frame("gesamt")
                )

        # Evaluation process
        self.evaluate(final_predictions)

    def run_feature_selection(self):
        """
        Execute the feature selection process using a LightGBM-based approach.
        Saves the final list of selected features as 'selected_features.csv'.
        """
        fs_output_dir = os.path.join(self.config["output_dir"], "feature_selection_report")
        os.makedirs(fs_output_dir, exist_ok=True)
        fs_manager = FeatureSelectionManager(self.config, fs_output_dir, self.model)
        best_features = fs_manager.run_lgbm_feature_selection(
            self.X_train,
            self.y_train,
            mandatory_features=["hour", "weekday", "month"],
            num_models=1
        )
        logger.info("Finally selected features: %s", best_features)

        path = os.path.join(self.config["models_dir"], "selected_features.csv")
        pd.DataFrame(best_features).to_csv(path, index=False)

    def run_optuna(self):
        """
        Run hyperparameter optimization using Optuna, if not already completed.
        Loads existing best parameters if the optimization was done before.
        """
        best_params_path = os.path.join(self.config["models_dir"], "optuna_best_params.json")
        if os.path.exists(best_params_path):
            logger.info("Optuna optimization already completed. Skipping.")
            with open(best_params_path, "r", encoding="utf-8") as file_obj:
                best_params = json.load(file_obj)
            self.config["params"].update(best_params)

            # Re-instantiate the model to apply updated config
            self.model = self.model.__class__(self.config)

            self.cfg_manager.update_yaml_params(self.path_to_cfg, self.version, best_params)
        else:
            logger.info("Starting Optuna optimization.")
            best_params = self.model.optimize_hyperparameters(self.X_train, self.y_train)

            # Save X_test in compressed format in hyperparameter_dir
            [
                self.X_test.to_parquet(
                    os.path.join(self.config["hyperparameter_dir"], d, "X_test.parquet")
                )
                for d in os.listdir(self.config["hyperparameter_dir"])
                if os.path.isdir(os.path.join(self.config["hyperparameter_dir"], d))
            ]

            self.config["params"].update(best_params)
            self.cfg_manager.update_yaml_params(self.path_to_cfg, self.version, best_params)

    def evaluate(self, final_predictions):
        """
        Run the evaluation process using ExtendedReportManager.

        Args:
            final_predictions (pd.DataFrame): DataFrame containing the forecasted values.
        """
        erm = ExtendedReportManager(self.config["output_dir"], self.config)
        erm.run_all_analyses(final_predictions, self.X_test)

    def run_rolling_cv(self):
        """
        Run rolling cross-validation to iteratively expand the training window
        and move the test window forward by a specified interval.
        """

        if os.path.exists(os.path.join(self.config['predictions_dir'], "final_predictions.csv")):
            final_predictions = pd.read_csv(os.path.join(self.config['predictions_dir'], "final_predictions.csv"), index_col=0, parse_dates=True)
        else:
            logger.info("Starting rolling cross-validation.")

            X = pd.concat([self.X_train, self.X_test], axis=0)
            y = pd.concat([self.y_train, self.y_test], axis=0)
            # Original end of training dataset
            original_train_end = self.X_train.index[-1]

            # Check if original_train_end is already Wednesday 22:00
            if not (original_train_end.weekday() == 2 and original_train_end.hour == 22):
                # Search in X for the first time > original_train_end that is Wednesday 22:00
                future_times = X[X.index > original_train_end]
                target_times = future_times[
                    (future_times.index.weekday == 2) & (future_times.index.hour == 22)
                ]
                if not target_times.empty:
                    new_train_end = target_times.index[0]
                    # Determine the new initial training range: all rows up to new_train_end
                    initial_train_size = X.index.get_loc(new_train_end) + 1
                else:
                    raise ValueError("No matching Wednesday 22:00 found in the data.")
            else:
                # If it is already exactly Wednesday 22:00, it remains unchanged
                initial_train_size = len(self.X_train)

            # Further config parameters
            if len(self.y_test) > 7 * 24:
                test_window = pd.Timedelta(self.config["cv_settings"]["test_window"])
            else:
                test_window = pd.Timedelta(f"{len(self.y_test)}h")

            all_predictions = []
            fold = 0

            train_start = X.index[0]
            train_end = X.index[initial_train_size - 1]
            test_start = train_end + pd.Timedelta(hours=1)  # Gap of 1 hour
            test_end = test_start + test_window - pd.Timedelta(hours=1)
            while test_end <= X.index[-1]:
                fold += 1
                logger.info("--- Fold %s ---", fold)

                # Define the current training and test range
                X_train_fold = X[train_start:train_end]
                y_train_fold = y[train_start:train_end]
                X_test_fold = X[test_start:test_end]
                y_test_fold = y[test_start:test_end]

                # Optional: Optuna tuning, if enabled
                if self.config.get("optuna", {}).get("use_optuna") and fold <= self.config["cv_settings"]["optuna_folds"]:
                    self.run_optuna()

                # Train and save model
                model_save_path = os.path.join(
                    self.config["models_dir"],
                    f"{test_start.strftime('%Y_%m_%d')}_fold_{fold}"
                )
                os.makedirs(model_save_path, exist_ok=True)

                train_config = self.config.copy()
                train_config["models_dir"] = model_save_path

                if self.config["model"] != "baseline":
                    self.model.train(X_train_fold, y_train_fold, train_config)
                    X_test_fold.to_parquet(os.path.join(model_save_path, "X_test.parquet"))
                    predictor = Predictor(train_config)

                    fold_predictions = predictor.predict(
                        x_test=X_test_fold,
                        x_train=X_train_fold,
                        y_train=y_train_fold.to_frame("gesamt"),
                        y_test=y_test_fold.to_frame("gesamt")
                    )
                else:
                    fold_predictions = self.model.predict(
                        X_test_fold,
                        y_train_fold.to_frame("gesamt"),
                        y_test_fold.to_frame("gesamt")
                    )

                all_predictions.append(fold_predictions)

                # Extend the window (expanding training window)
                train_end = test_end
                test_start = train_end + pd.Timedelta(hours=1)
                test_end = test_start + test_window - pd.Timedelta(hours=1)

            # Combine all predictions and evaluate
            final_predictions = pd.concat(all_predictions)
        self.evaluate(final_predictions)

    def parse_timedelta(self, time_str):
        """
        Parse a simple time string (e.g., '125W', '10D', '5H') into a timedelta object.

        Args:
            time_str (str): Time string, such as '125W', '10D', or '5H'.

        Returns:
            timedelta: A timedelta object representing the parsed duration.

        Raises:
            ValueError: If the time unit is invalid.
        """
        num = int(time_str[:-1])
        unit = time_str[-1]

        if unit == "W":
            return timedelta(weeks=num)
        if unit == "D":
            return timedelta(days=num)
        if unit == "H":
            return timedelta(hours=num)

        raise ValueError(f"Invalid time unit: {unit}")
