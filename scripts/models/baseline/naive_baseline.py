"""
Module: naive_baseline_model
This module implements the NaiveBaselineModel which serves as a baseline prediction model.
It does not require any training and produces naive predictions based solely on historical observations.
"""

import pandas as pd
import numpy as np  
from scripts.models.base_model import BaseModel
import pdb
from scripts.utils.logger import logger


class NaiveBaselineModel(BaseModel):
    def __init__(self, config):
        """
        Initialize the NaiveBaselineModel with the provided configuration.

        Args:
            config (dict): Configuration dictionary with keys such as "output_dir", "dataset",
                           "quantiles", "start_date", and optionally "last_t".
        """
        super().__init__(**config)
        self.output_dir = config["output_dir"]
        self.data_path = config[config["dataset"]]["data_file"]
        self.quantiles = config["quantiles"]
        self.start_date = config["start_date"]
        self.last_t = config.get("last_t")  # Example value, e.g. 4 weeks of history

    def train(self, y_test=None, y_train=None, X_test=None):
        """
        Baseline model does not require training.

        Args:
            y_test: Unused.
            y_train: Unused.
            X_test: Unused.
        """
        logger.info("[NaiveBaselineModel] train() skipped (baseline requires no training).")
        pass

    def predict(self, X_test=None, y_train=None, y_test=None):
        """
        Generate naive predictions for the provided test data.

        The method:
          1. Logs the start of prediction.
          2. Creates a copy of X_test and trims it by finding the first Wednesday at 6 PM.
          3. Combines y_train and y_test into a single DataFrame and extracts the prediction period.
          4. Assigns the observed value as the prediction ("gesamt_pred").
          5. Sets predicted values to NaN for a specified time window:
             - Wednesday from 23:00,
             - Thursday (all day),
             - Friday (all day),
             - Saturday until 20:00,
             only for indices after the prediction start.
          6. Computes rolling quantile values for each desired quantile grouped by weekday and hour,
             and assigns these to the corresponding prediction columns where predictions are NaN.
          7. Returns the resulting DataFrame starting from the first index in X_test.

        Args:
            X_test (DataFrame, optional): Test features with a datetime index.
            y_train (DataFrame or Series, optional): Training target values.
            y_test (DataFrame or Series, optional): Test target values.

        Returns:
            DataFrame: The DataFrame containing naive predictions and quantile predictions.
        """
        logger.info("[NaiveBaselineModel] Starting naive prediction.")
        # Copy of X_test for iteration
        X_test_iter = X_test.copy()
        # Find the first Wednesday at 6 PM in the index and trim X_test_iter starting from that time
        start_time = None
        for t in X_test_iter.index:
            if t.weekday() == 2 and t.hour >= 18:
                start_time = t
                break
        if start_time is not None:
            X_test_iter = X_test_iter.loc[start_time:]
        
        # Combine training and test targets and create a copy for prediction
        y_test_iter = pd.concat([y_train, y_test], axis=0)
        start_index = X_test_iter.index[0] - pd.DateOffset(hours=self.last_t * 7 * 24)
        y_test_iter = y_test_iter.loc[start_index:X_test_iter.index[-1]].copy()
        y_test_iter['gesamt_pred'] = y_test_iter['gesamt']

        # Define condition: set predictions to NaN for specific time windows
        condition = (
            (
                ((y_test_iter.index.weekday == 2) & (y_test_iter.index.hour >= 23)) |  # Wednesday from 23:00
                (y_test_iter.index.weekday == 3) |                                      # Thursday (full day)
                (y_test_iter.index.weekday == 4) |                                      # Friday (full day)
                ((y_test_iter.index.weekday == 5) & (y_test_iter.index.hour <= 20))      # Saturday until 20:00
            )
            &
            (y_test_iter.index > X_test_iter.index[0])  # And only for timestamps after the start of X_test_iter
        )

        y_test_iter['weekday'] = y_test_iter.index.weekday
        y_test_iter['hour'] = y_test_iter.index.hour

        y_test_pred = y_test_iter.copy()
        y_test_pred.loc[condition, 'gesamt_pred'] = np.nan

        # Sort index of y_test_pred
        y_test_pred = y_test_pred.sort_index()

        df_result = y_test_pred.copy()
        
        # Initialize new columns for quantile predictions
        df_result['gesamt_pred_0.025'] = df_result['gesamt_pred']
        df_result['gesamt_pred_0.25'] = df_result['gesamt_pred']
        df_result['gesamt_pred_0.5'] = df_result['gesamt_pred']
        df_result['gesamt_pred_0.75'] = df_result['gesamt_pred']
        df_result['gesamt_pred_0.975'] = df_result['gesamt_pred']
        
        # Identify rows where 'gesamt_pred' is NaN
        nan_mask = df_result['gesamt_pred'].isna()
        
        # List of quantiles to compute
        quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
        
        # For each desired quantile, compute the rolling quantile value
        for q in quantiles:
            col_name = f"gesamt_pred_{q}"
            rolling_quantiles = (
                df_result.groupby(["weekday", "hour"])["gesamt"]
                .rolling(window=self.last_t, min_periods=self.last_t)
                .quantile(q)
                .shift(1)
                .reset_index(level=[0, 1], drop=True)
            )
            # Assign the computed rolling quantile only to rows where 'gesamt_pred' is NaN
            df_result.loc[nan_mask, col_name] = rolling_quantiles.loc[nan_mask]

        df_result = df_result.loc[X_test.index[0]:] 

        return df_result

    def evaluate_trial(self, trial, X_train, y_train):
        """
        Not applicable for the baseline model.

        Args:
            trial: The Optuna trial object.
            X_train: Training features.
            y_train: Training target values.
        """
        pass

    def get_optuna_search_space(self):
        """
        Not applicable for the baseline model.

        Returns:
            None
        """
        pass
