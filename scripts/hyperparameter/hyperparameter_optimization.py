"""
Module: hyperparameter_optimizer
This module implements the HyperparameterOptimizer class which performs hyperparameter
optimization using Optuna. Detailed optimization logs are written to file and several
visualizations of the optimization process are generated.
"""

import os
import json
import datetime
import optuna
import optuna.visualization as vis
from scripts.utils.logger import logger
import time
import pdb 


class HyperparameterOptimizer:
    def __init__(self, model_instance):
        """
        Initialize the HyperparameterOptimizer.

        Args:
            model_instance: An instance of a model that contains a configuration (config) attribute.
        """
        self.model_instance = model_instance
        self.config = model_instance.config

    @staticmethod
    def is_in_time_window(ts):
        """
        Check if the given timestamp falls within the defined time window.

        Args:
            ts (datetime): The timestamp to check.

        Returns:
            bool: True if the timestamp is in the time window, False otherwise.
        """
        if ts.weekday() == 2:
            return ts.hour >= 23
        elif ts.weekday() in (3, 4):
            return True
        elif ts.weekday() == 5:
            return ts.hour <= 23
        else:
            return False

    def optimize(self, X_train, y_train):
        """
        Perform hyperparameter optimization using Optuna.

        The method creates a log file with detailed trial information and generates several plots
        of the optimization process. It adjusts the best parameters according to the model type,
        saves the best parameters to a JSON file, and returns them.

        Args:
            X_train (DataFrame): Training features.
            y_train (Series/DataFrame): Training target values.

        Returns:
            dict or None: The best hyperparameters found, or None if hyperparameter tuning is disabled.
        """
        if not self.config["optuna"].get("use_optuna", False):
            logger.info(
                "[HyperparameterOptimizer] Hyperparameter tuning is disabled for model %s",
                self.config.get("model"),
            )
            return None

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(self.model_instance.model_dir, "optuna_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"optuna_log_{timestamp}.txt")
        log_file = open(log_filename, "w")
        log_file.write("Optuna Optimization Log - %s\n" % timestamp)
        log_file.write("Config: %s\n\n" % self.config)
        log_file.flush()

        def objective(trial):
            """
            Objective function for the Optuna study.

            Args:
                trial: The current Optuna trial.

            Returns:
                float: The loss value for the trial.
            """
            try:
                loss = self.model_instance.evaluate_trial(trial, X_train, y_train)
                log_file.write("Trial %s:\n" % trial.number)
                log_file.write("  Params: %s\n" % trial.params)
                log_file.write("  Loss: %s\n" % loss)
                log_file.flush()
                return loss
            except Exception as e:
                logger.error(
                    "[HyperparameterOptimizer] Trial %s failed for model %s. Params: %s. Error: %s",
                    trial.number,
                    self.model_instance.config.get("model"),
                    trial.params,
                    str(e),
                )
                log_file.write("Trial %s: FAILED\n" % trial.number)
                log_file.write("  Params: %s\n" % trial.params)
                log_file.write("  Error: %s\n" % str(e))
                log_file.flush()
                raise

        study = optuna.create_study(direction=self.config["optuna"].get("direction", "minimize"))
        n_trials = self.config["optuna"].get("n_trials")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_value = study.best_value

        # Adjust best_params according to model type
        if self.config["model"] == "light_gbm":
            n_estimators = self.config["params"]["n_estimators"]
            best_params["n_estimators"] = {
                "0.025": n_estimators["0.025"],
                "0.25": n_estimators["0.25"],
                "0.5": best_params["n_estimators"],
                "0.75": n_estimators["0.75"],
                "0.975": n_estimators["0.975"],
            }

        if self.config["model"] == "lstm":
            if best_params["enabled"]:
                best_params["gradient_clipping"] = {
                    "enabled": best_params["enabled"],
                    "norm_type": 2.0,
                    "clip_norm": best_params["clip_norm"],
                }
                # Remove keys 'clip_norm' and 'enabled'
                best_params.pop("clip_norm")
                best_params.pop("enabled")
            else:
                best_params["gradient_clipping"] = {
                    "enabled": best_params["enabled"],
                    "norm_type": None,
                    "clip_norm": None,
                }
                best_params.pop("enabled")

        best_params_file = os.path.join(self.model_instance.model_dir, "optuna_best_params.json")
        with open(best_params_file, "w") as f:
            json.dump(best_params, f, indent=4)
        log_file.write("\n--- Best Results ---\n")
        log_file.write("Best Trial: %s\n" % study.best_trial.number)
        log_file.write("Best Params: %s\n" % best_params)
        log_file.write("Best Loss: %s\n" % best_value)
        log_file.write("\n--- All Trials ---\n")
        for trial in study.trials:
            log_file.write(
                "Trial %s: Loss=%s, Params=%s, State=%s\n" % (trial.number, trial.value, trial.params, trial.state)
            )
        log_file.flush()
        log_file.close()

        # --- Plotting code ---
        output_dir = os.path.join(self.model_instance.model_dir, "optuna_plots")
        os.makedirs(output_dir, exist_ok=True)

        try:
            plots = {
                "optimization_history": vis.plot_optimization_history(study),
                "intermediate_values": vis.plot_intermediate_values(study),
                "slice": vis.plot_slice(study),
                "parallel_coordinate": vis.plot_parallel_coordinate(study),
                "param_importances": vis.plot_param_importances(study),
                "contour": vis.plot_contour(study),
                "edf": vis.plot_edf(study),
            }
            for name, plot in plots.items():
                if plot is not None:
                    plot.write_html(os.path.join(output_dir, f"{name}.html"))
        except Exception as e:
            logger.error(
                "Error while creating or saving Optuna plots: %s", str(e)
            )
        # --- End of plotting code ---

        return best_params
