import os
import pandas as pd
import numpy as np
from joblib import load
import logging
import pdb
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Predictor:
    """
    Predictor class for generating forecasts using either Quantile Regression models or an LSTM model which also 
    predicts Quantiles but within a single model.

    The Predictor loads pre-trained models from disk (either .joblib for quantile regressors or a PyTorch checkpoint for LSTM),
    and uses them to iteratively predict the target variable over a specified forecasting horizon.
    It supports updating lag features and handling missing values based on specified conditions.
    """

    def __init__(self, config: dict):
        """
        Initialize the Predictor with the given configuration.

        Args:
            config (dict): Configuration dictionary containing keys such as:
                - "features": A dict with a "target" key specifying lags (list of integers).
                - "models_dir": Directory path where model files are stored.
                - "quantiles": List of quantiles for probabilistic forecasts.
                - "forecast_horizon": (Optional) Forecast horizon in hours (default is 72).
                - "model": The model identifier (e.g., 'lstm' or other).
        """
        self.config = config
        features_cfg = config.get("features")
        self.lags = features_cfg["target"].get("lags")
        self.model_dir = config.get("models_dir")
        self.quantiles = config.get("quantiles")
        self.HORIZON_HOURS = config.get("forecast_horizon", 72)
        self.loaded_models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model_for_quantile(self, q):
        """
        Load the Quantile Regression model for the given quantile.

        If the model for quantile q is not already cached in self.loaded_models, it loads the model from a .joblib file.
        The file is expected to be named as '{model}_model_{q}.joblib' in self.model_dir, where 'model' is taken from the config.

        Args:
            q (float): The quantile for which to load the model (e.g., 0.5).

        Returns:
            object: The loaded model for the specified quantile.
        """
        if q not in self.loaded_models:
            model_path = os.path.join(self.model_dir, f"{self.config['model']}_model_{q}.joblib")
            self.loaded_models[q] = load(model_path)
            logger.info("[Predictor] Loaded model for quantile %s: %s", q, model_path)
        return self.loaded_models[q]

    def _load_lstm_model(self, path=None):
        """
        Load the LSTM model from a PyTorch checkpoint.

        The checkpoint is expected to contain hyperparameters under the key "hyperparams" and the model state dictionaries 
        under "state_dicts".
        If the LSTM model has already been loaded, the cached version is returned.
        The loaded model dictionary contains the LSTM, a fully connected (fc) layer, the optimizer, and hyperparameters.

        Args:
            path (str, optional): Custom path to the checkpoint file. If not provided, defaults to 'lstm_model.pth' in self.model_dir.

        Returns:
            dict: A dictionary containing:
                - "lstm": The LSTM network.
                - "fc": The fully connected layer.
                - "optimizer": The optimizer used during training.
                - "hyperparams": Hyperparameters from the checkpoint.

        Raises:
            ValueError: If the checkpoint does not contain hyperparameters.
        """
        if "lstm" in self.loaded_models:
            return self.loaded_models["lstm"]

        if not path:
            model_path = os.path.join(self.model_dir, "lstm_model.pth")
        else:
            model_path = path
        logger.info("[Predictor] Loading LSTM PyTorch model: %s", model_path)

        checkpoint = torch.load(model_path, map_location=self.device)

        # Hyperparams auslesen
        if "hyperparams" in checkpoint:
            hp = checkpoint["hyperparams"]
            bias = hp.get("bias")
            hidden_size = hp["hidden_size"]
            num_layers = hp["num_layers"]
            dropout = hp["dropout"]
            learning_rate = hp["learning_rate"]
            epochs = hp["epochs"]
            batch_size = hp["batch_size"]
            loaded_quantiles = hp["quantiles"]
            seq_length = hp["seq_length"]
            size = hp["input_size"]
        else:
            raise ValueError("Checkpoint does not contain hyperparams!")

        lstm = nn.LSTM(
            input_size=size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bias=bias,
            batch_first=True
        ).to(self.device)

        fc = nn.Linear(hidden_size, len(loaded_quantiles)).to(self.device)
        optimizer = torch.optim.Adam(list(lstm.parameters()) + list(fc.parameters()), lr=learning_rate)

        # Weights reinladen
 
        if "state_dicts" in checkpoint:
            lstm.load_state_dict(checkpoint["state_dicts"]["lstm"])
            fc.load_state_dict(checkpoint["state_dicts"]["fc"])
        else:
            logger.warning("Checkpoint does not have state_dicts!")

        loaded_model = {
            "lstm": lstm,
            "fc": fc,
            "optimizer": optimizer,
            "hyperparams": hp
        }
        self.loaded_models["lstm"] = loaded_model
        return loaded_model

    def _predict_testing_it(self, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, model=None) -> (pd.DataFrame, pd.DataFrame):
        """
        Perform iterative prediction on test data using a Quantile Regression model.

        This function:
          1. Trims X_test to start from the first timestamp on a Wednesday at or after 18:00.
          2. Concatenates y_train and y_test, sorts them, and selects a subset starting from a computed start index.
          3. Sets initial prediction values ('gesamt_pred') to the observed target ('gesamt').
          4. For specified lag columns in X_test, sets values to NaN if the lagged time is after Wednesday 23:00.
          5. Applies a condition that sets 'gesamt_pred' to NaN for timestamps corresponding to Wednesday from 23:00 onward, Thursday, Friday, and Saturday up to 23:00.
          6. Iteratively fills in missing predictions by using the quantile model (default quantile 0.5) to predict on the current row,
             and updates lag values with previous predictions if necessary.
          7. Saves intermediate X_test and prediction DataFrames to CSV files for debugging.

        Args:
            X_test (pd.DataFrame): Test features DataFrame.
            y_train (pd.DataFrame): Training target DataFrame.
            y_test (pd.DataFrame): Test target DataFrame.
            model (optional): A pre-loaded model to use instead of loading quantile 0.5 from disk.

        Returns:
            tuple: A tuple containing:
                - y_test_pred (pd.DataFrame): DataFrame with predicted target values.
                - X_test_iter (pd.DataFrame): The modified test features DataFrame used in iterative prediction.

        Raises:
            ValueError: If a current row contains NaN values when a prediction is attempted.
        """
        X_test_iter = X_test.copy()
        start_time = None
        for t in X_test_iter.index:
            if t.weekday() == 2 and t.hour >= 18:
                start_time = t
                break

        if start_time is not None:
            X_test_iter = X_test_iter.loc[start_time:]

        y_test_iter = pd.concat([y_train, y_test], axis=0).sort_index()

        shift = max(self.lags) if self.lags else 12
        start_index = X_test_iter.index[0] - pd.DateOffset(hours=shift + 24)
        y_test_iter = y_test_iter.loc[start_index:X_test_iter.index[-1]].copy()
        y_test_iter['gesamt_pred'] = y_test_iter['gesamt']

        columns_to_nan = {f'lag_{lag}': lag for lag in self.lags}

        def _should_set_nan(index, lag):
            lagged_time = index - pd.Timedelta(hours=lag)
            wednesday_23 = (index - pd.Timedelta(days=index.weekday() - 2)).replace(hour=23, minute=0, second=0)
            return lagged_time >= wednesday_23 and index >= start_time

        for col, lag in columns_to_nan.items():
            X_test_iter.loc[X_test_iter.index.map(lambda idx: _should_set_nan(idx, lag)), col] = np.nan

        condition = (
            (
                ((y_test_iter.index.weekday == 2) & (y_test_iter.index.hour == 23)) |
                (y_test_iter.index.weekday == 3) |
                (y_test_iter.index.weekday == 4) |
                ((y_test_iter.index.weekday == 5) & (y_test_iter.index.hour <= 23))
            )
            & (y_test_iter.index >= start_time)
        )

        y_test_iter['weekday'] = y_test_iter.index.weekday
        y_test_iter['hour'] = y_test_iter.index.hour

        y_test_pred = y_test_iter.copy()
        y_test_pred.loc[condition, 'gesamt_pred'] = np.nan

        if not model:
            qr = self._load_model_for_quantile(0.5)
            if isinstance(qr, dict):
                qr = qr['model']
        else:
            qr = model

        reached = False
        y = 0
        for i in range(len(y_test_pred)):
            if X_test_iter.index[0] == y_test_pred.index[i] or reached:
                reached = True
                current_row = X_test_iter.iloc[[y]]
                if self.lags:
                    if current_row[f'lag_{min(self.lags)}'].isnull().any():
                        for lag in self.lags:
                            lag_col_name = f'lag_{lag}'
                            if current_row.loc[current_row.index, lag_col_name].isna().any():
                                current_row.loc[current_row.index, lag_col_name] = y_test_pred.iloc[i - lag]['gesamt_pred']
                                X_test_iter.loc[current_row.index, lag_col_name] = y_test_pred.iloc[i - lag]['gesamt_pred']
                if y_test_pred.iloc[i:i+1, 1].isna().any():
                    if current_row.isnull().any().any():
                        raise ValueError("Current row contains NaN values")
                    y_pred = qr.predict(current_row)[0]
                    y_test_pred.loc[current_row.index, 'gesamt_pred'] = y_pred
                y += 1

        y_test_pred = y_test_pred.loc[X_test_iter.index]
        return y_test_pred, X_test_iter

    def predict_iterative_other_quantiles(self, y_test_pred: pd.DataFrame, x_test_pred: pd.DataFrame) -> pd.DataFrame:
        """
        Predict target values for quantiles other than 0.5 using the iterative test predictions.

        For each quantile in self.quantiles (excluding 0.5), this function loads the corresponding model
        from a .joblib file, makes predictions on x_test_pred, and adds the results as new columns to y_test_pred.
        Finally, the median prediction column is renamed to 'gesamt_pred_0.5'.

        Args:
            y_test_pred (pd.DataFrame): DataFrame containing the median predictions and other columns.
            x_test_pred (pd.DataFrame): Test features used for making predictions.

        Returns:
            pd.DataFrame: DataFrame containing predictions for all quantiles.
        """
        for quantile in self.quantiles:
            if quantile != 0.5:
                model_filename = os.path.join(self.model_dir, f"{self.config['model']}_model_{quantile}.joblib")
                loaded_model = load(model_filename)['model']
                qr = loaded_model
                # Vorhersagen auf den Testdaten machen
                y_pred = qr.predict(x_test_pred)
                y_test_pred[f'gesamt_pred_{quantile}'] = y_pred

        y_test_pred = y_test_pred.rename(columns={'gesamt_pred': 'gesamt_pred_0.5'})
        return y_test_pred

    def predict_lstm(self, X_test: pd.DataFrame = None, X_train: pd.DataFrame = None,
                     y_train: pd.DataFrame = None, y_test: pd.DataFrame = None, path=None) -> pd.DataFrame:
        """
        Perform iterative prediction using an LSTM model without feedback (i.e., without updating lag columns).

        The LSTM model predicts all quantiles simultaneously. This function:
          1. Loads the LSTM model (and associated fully connected layer) from a checkpoint.
          2. Finds the first Wednesday at or after 18:00 in the test set and concatenates the last seq_length rows from the training data.
          3. Prepares the test data by setting certain lag feature values to NaN based on a condition relative to Wednesday 23:00.
          4. Iteratively extracts a sliding window of features (using extract_single_lstm_window), converts it to a tensor,
             and obtains predictions from the LSTM model.
          5. Updates the prediction DataFrame with the predicted values for each quantile.
          6. Returns a DataFrame with predictions for all quantiles starting from the determined start time.

        Args:
            X_test (pd.DataFrame): Test features DataFrame.
            X_train (pd.DataFrame): Training features DataFrame.
            y_train (pd.DataFrame): Training target DataFrame.
            y_test (pd.DataFrame): Test target DataFrame.
            path (str, optional): Custom path to the LSTM checkpoint file.

        Returns:
            pd.DataFrame: DataFrame with iterative LSTM predictions.
        """
        model_dict = self._load_lstm_model(path=path)
        lstm = model_dict["lstm"]
        fc = model_dict["fc"]
        hp = model_dict["hyperparams"]
        loaded_quantiles = hp["quantiles"]
        seq_length = hp["seq_length"]
        device = self.device

        lstm.eval()

        X_test_iter = X_test.copy()
        start_time = None
        for t in X_test_iter.index:
            if t.weekday() == 2 and t.hour >= 18:
                start_time = t
                break

        start_index = start_time - pd.DateOffset(hours=1600)
        y_test_iter = pd.concat([y_train, y_test], axis=0)
        last_seq = X_train.loc[start_index:]
        X_test_iter = pd.concat([last_seq, X_test], axis=0).sort_index()
        y_test_iter = y_test_iter.loc[start_index:X_test_iter.index[-1]].copy()
        y_test_iter['gesamt_pred'] = y_test_iter['gesamt']

        columns_to_nan = {f'lag_{lag}': lag for lag in self.lags}

        def _should_set_nan(index, lag):
            lagged_time = index - pd.Timedelta(hours=lag)
            wednesday_23 = (index - pd.Timedelta(days=index.weekday() - 2)).replace(hour=23, minute=0, second=0)
            return lagged_time >= wednesday_23 and index >= start_time

        for col, lag in columns_to_nan.items():
            X_test_iter.loc[X_test_iter.index.map(lambda idx: _should_set_nan(idx, lag)), col] = np.nan

        condition = (
            (((y_test_iter.index.weekday == 2) & (y_test_iter.index.hour == 23)) |
             (y_test_iter.index.weekday == 3) |
             (y_test_iter.index.weekday == 4) |
             ((y_test_iter.index.weekday == 5) & (y_test_iter.index.hour <= 23)))
            & (y_test_iter.index >= start_time)
        )

        y_test_iter['weekday'] = y_test_iter.index.weekday
        y_test_iter['hour'] = y_test_iter.index.hour
        y_test_pred = y_test_iter.copy()
        y_test_pred.loc[condition, 'gesamt_pred'] = np.nan

        for q in loaded_quantiles:
            y_test[f'gesamt_pred_{q}'] = np.nan

        feature_cols = list(X_test_iter.columns)
        X_test_reset = X_test_iter.reset_index(drop=False)

        reached = False
        y = 0
        for i in range(len(y_test_pred)):
            if X_test_reset.loc[0, 'date_time_utc'] == y_test_pred.index[i] or reached:
                reached = True
                current_row = X_test_reset.iloc[[y]]
                if self.lags:
                    if current_row[f'lag_{min(self.lags)}'].isnull().any():
                        for lag in self.lags:
                            lag_col_name = f'lag_{lag}'
                            if current_row.loc[current_row.index, lag_col_name].isna().any():
                                current_row.loc[current_row.index, lag_col_name] = y_test_pred.iloc[i - lag]['gesamt_pred']
                                X_test_reset.loc[current_row.index, lag_col_name] = y_test_pred.iloc[i - lag]['gesamt_pred']
                
                
                if y_test_pred.iloc[i:i+1, 1].isna().any():
                    if current_row.isnull().any().any():
                        raise ValueError("Current row contains NaN values")
                    x_window = self.extract_single_lstm_window(
                        df=X_test_reset,
                        feature_cols=feature_cols,
                        t=y,
                        seq_length=seq_length
                    )
                    x_tensor = torch.tensor(x_window, dtype=torch.float).to(device)
                    with torch.no_grad():
                        out, _ = lstm(x_tensor)
                        last_hidden = out[:, -1, :]
                        pred_quantiles = fc(last_hidden)
                        pred_quantiles = pred_quantiles.squeeze().cpu().numpy()
                        # if y % 100 == 0:
                        #     pdb.set_trace()
                    y_test_pred.loc[X_test_reset.loc[y, 'date_time_utc'], 'gesamt_pred'] = pred_quantiles[2]
                    for q_idx, q in enumerate(loaded_quantiles):
                        y_test_pred.loc[X_test_reset.loc[y, 'date_time_utc'], f'gesamt_pred_{q}'] = pred_quantiles[q_idx]
                y += 1

        y_test_pred = y_test_pred.loc[start_time:]
     
        return y_test_pred

    def extract_single_lstm_window(self, df: pd.DataFrame, feature_cols: list, t: int, seq_length: int) -> np.ndarray:
        """
        Extract a single sliding window of features for LSTM prediction.

        This function extracts rows from (t - seq_length) to (t - 1) from the DataFrame 'df'
        for the specified feature columns, and reshapes the result into a NumPy array of shape
        (1, seq_length, n_features), which is suitable as input for the LSTM.

        Args:
            df (pd.DataFrame): DataFrame containing the features.
            feature_cols (list): List of column names to extract.
            t (int): The current time index (integer-based index) at which to end the window.
            seq_length (int): The number of time steps in the window.

        Returns:
            np.ndarray: An array of shape (1, seq_length, n_features) representing the extracted window.

        Raises:
            ValueError: If t is smaller than seq_length, making window extraction impossible.
        """
        start_idx = t - seq_length
        end_idx = t  # Includes rows [start_idx, ..., t-1]
        if start_idx < 0:
            raise ValueError(f"t={t} is too small; cannot extract a window of seq_length={seq_length}!")
        window_df = df.iloc[start_idx:end_idx]
        window_features = window_df[feature_cols].to_numpy(copy=True)
        x_window = np.expand_dims(window_features, axis=0)
        return x_window

    def predict(self, x_test: pd.DataFrame = None, x_train: pd.DataFrame = None,
                y_train: pd.DataFrame = None, y_test: pd.DataFrame = None, model=None) -> pd.DataFrame:
        """
        Generate forecasts using either the LSTM model or Quantile Regression models based on the configuration.

        If the configuration specifies the "lstm" model, this function calls predict_lstm to perform iterative LSTM prediction.
        Otherwise, it calls _predict_testing_it to obtain median predictions and then predict_iterative_other_quantiles
        to compute predictions for additional quantiles (unless a model is explicitly provided).

        Args:
            x_test (pd.DataFrame): Test features DataFrame.
            x_train (pd.DataFrame): Training features DataFrame.
            y_train (pd.DataFrame): Training target DataFrame.
            y_test (pd.DataFrame): Test target DataFrame.
            model (optional): A pre-loaded model to use for prediction; if not provided, the default quantile models are loaded.

        Returns:
            pd.DataFrame: A DataFrame containing forecasts for the target variable for all specified quantiles.
        """
        logger.info("[Predictor] Starting predict()")

        if self.config["model"] == "lstm":
            df_full = self.predict_lstm(x_test, x_train, y_train, y_test)
        else:
            df_median, x_test_pred = self._predict_testing_it(x_test, y_train, y_test, model=model)
            if not model:
                df_full = self.predict_iterative_other_quantiles(df_median, x_test_pred)
            else:
                df_full = df_median
        return df_full
