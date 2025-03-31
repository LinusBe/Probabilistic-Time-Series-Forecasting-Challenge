"""
Module: lstm_forecast_model
This module defines the LSTMForecastModel class which implements an LSTM-based forecasting model.
It includes methods for training, evaluation (including hyperparameter optimization using Optuna),
data preparation, loss calculation, and model persistence.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adadelta
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_pinball_loss, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from datetime import datetime, timedelta
from scripts.models.base_model import BaseModel
from scripts.predictor.predictor import Predictor
from scripts.utils.logger import logger
import pdb


class LSTMForecastModel(BaseModel):
    def __init__(self, config):
        """
        Initialize the LSTMForecastModel by reading all important hyperparameters from the config dictionary.
        The hyperparameters are stored in self.params and as instance variables.

        Args:
            config (dict): Configuration dictionary containing keys such as "models_dir", "quantiles",
                           "params", and "optuna_search_space".
        """
        super().__init__(**config)
        self.config = config
        self.model_dir = config["models_dir"]
        self.quantiles = config.get("quantiles", [0.025, 0.25, 0.5, 0.75, 0.975])
        self.params = config.get("params", {})

        # Main hyperparameters from self.params (default values)
        self.hidden_size = self.params.get("hidden_size")
        self.num_layers = self.params.get("num_layers")
        self.dropout = self.params.get("dropout")
        self.learning_rate = self.params.get("learning_rate")
        self.epochs = self.params.get("epochs")
        self.bias = self.params.get("bias")
        self.batch_size = self.params.get("batch_size")
        self.forecast_horizon = self.params.get("forecast_horizon", 1)
        self.seq_length = self.params.get("seq_length")  # e.g., 24
        self.optuna_space = config["optuna_search_space"].get("param_space", {})

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = None
        self.fc = None
        self.fold = 0

    def train(self, X_train, y_train, config=None):
        """
        Train the LSTM model using default hyperparameters defined in self.params.
        At the end, it calls self.save_model(..., local_info=...) to store the training information.

        Args:
            X_train (DataFrame): Training features.
            y_train (Series/DataFrame): Training target values.
            config (dict, optional): Additional configuration to override the instance config.
        """
        if config is not None:
            self.config = config

        existing_models = glob.glob(
            os.path.join(self.config["models_dir"], f"{self.config['model']}_model.pth")
        )
        if existing_models:
            logger.info("[LSTMForecastModel] Model file already exists. Skipping training.")
            return

        logger.info("[LSTMForecastModel] Starting training (native LSTM).")

        feature_cols = X_train.columns.tolist()
        target_col = "gesamt"

        # Evaluation set
        use_eval = self.config.get("eval_set", {}).get("use", False)
        eval_fraction = self.config.get("eval_set", {}).get("size", 0.0)
        n_total = len(X_train)
        n_train = int(n_total * (1 - eval_fraction))

        # Split the data
        X_train_ii = X_train.iloc[:n_train]
        y_train_ii = y_train.iloc[:n_train]
        X_val_ii = X_train.iloc[n_train:]
        y_val_ii = y_train.iloc[n_train:]

        # Create LSTM windows for training data
        df_train = X_train_ii.copy()
        df_train[target_col] = y_train_ii
        X_array_train, y_array_train, idx_array_train = self.create_lstm_windows(
            df=df_train,
            feature_cols=feature_cols,
            target_col=target_col,
            seq_length=self.seq_length,
            forecast_horizon=self.forecast_horizon,
        )
        X_t_train = torch.tensor(X_array_train, dtype=torch.float32)
        y_t_train = torch.tensor(y_array_train, dtype=torch.float32)

        # Prepare validation DataLoader if evaluation is used
        val_loader = None
        if use_eval and len(X_val_ii) > 0:
            df_val = X_val_ii.copy()
            df_val[target_col] = y_val_ii
            X_array_val, y_array_val, idx_array_val = self.create_lstm_windows(
                df=df_val,
                feature_cols=feature_cols,
                target_col=target_col,
                seq_length=self.seq_length,
                forecast_horizon=self.forecast_horizon,
            )
            X_t_val = torch.tensor(X_array_val, dtype=torch.float32)
            y_t_val = torch.tensor(y_array_val, dtype=torch.float32)
            val_dataset = TensorDataset(X_t_val, y_t_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize model components
        self.input_size = X_t_train.shape[2]
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bias=self.bias,
            batch_first=True,
        ).to(self.device)

        out_dim = len(self.quantiles)
        self.fc = nn.Linear(self.hidden_size, out_dim).to(self.device)

        # Setup optimizer
        optimizer_str = self.params.get("optimizer")
        learning_rate = self.params.get("learning_rate")
        weight_decay = self.params.get("weight_decay")

        if optimizer_str == "adam":
            optimizer = torch.optim.Adam(
                list(self.lstm.parameters()) + list(self.fc.parameters()),
                lr=learning_rate, weight_decay=weight_decay,
            )
        elif optimizer_str == "AdamW":
            optimizer = torch.optim.AdamW(
                list(self.lstm.parameters()) + list(self.fc.parameters()),
                lr=learning_rate, weight_decay=weight_decay,
            )
        elif optimizer_str == "sgd":
            optimizer = torch.optim.SGD(
                list(self.lstm.parameters()) + list(self.fc.parameters()),
                lr=learning_rate, weight_decay=weight_decay,
            )
        elif optimizer_str == "Adadelta":
            optimizer = torch.optim.Adadelta(
                list(self.lstm.parameters()) + list(self.fc.parameters()),
                lr=learning_rate, weight_decay=weight_decay,
            )
        else:
            raise ValueError("Unknown optimizer %s" % optimizer_str)

        # Create training DataLoader
        train_dataset = TensorDataset(X_t_train, y_t_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Early Stopping configuration
        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0
        early_stopping_rounds = self.config["early_stopping"].get("rounds", 10)

        train_loss_by_epoch = []
        val_loss_by_epoch = []

        start_time = datetime.now().timestamp()

        # Training loop
        for epoch in range(self.epochs):
            self.lstm.train()
            epoch_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()

                out, _ = self.lstm(X_batch)
                last_hidden = out[:, -1, :]
                preds = self.fc(last_hidden)
                loss = self._quantile_loss(preds, y_batch)
                loss.backward()

                # Optional gradient clipping
                clip_cfg = self.config["params"].get("gradient_clipping", {})
                if clip_cfg.get("enabled", False):
                    max_norm = clip_cfg.get("clip_norm")
                    norm_type = clip_cfg.get("norm_type")
                    nn.utils.clip_grad_norm_(
                        list(self.lstm.parameters()) + list(self.fc.parameters()),
                        max_norm=max_norm, norm_type=norm_type,
                    )

                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            train_loss_by_epoch.append(epoch_loss)

            # Validation
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader, self.lstm, self.fc)
                val_loss_by_epoch.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = {
                        "lstm": self.lstm.state_dict(),
                        "fc": self.fc.state_dict(),
                    }
                    best_epoch = epoch
                else:
                    patience_counter += 1
                if patience_counter >= early_stopping_rounds:
                    logger.info("Early stopping in epoch %d, val_loss=%.4f", epoch + 1, val_loss)
                    break
            else:
                val_loss_by_epoch.append(None)

            if val_loss is not None:
                logger.info("[Train] Epoch %d/%d - TrainLoss=%.4f, ValLoss=%.4f", epoch + 1, self.epochs, epoch_loss, val_loss)
            else:
                logger.info("[Train] Epoch %d/%d - TrainLoss=%.4f", epoch + 1, self.epochs, epoch_loss)

        # Load best weights if available
        if best_weights is not None:
            self.lstm.load_state_dict(best_weights["lstm"])
            self.fc.load_state_dict(best_weights["fc"])
            logger.info("[Train] Best weights (early stopping) loaded.")

        end_time = datetime.now().timestamp()
        training_duration = end_time - start_time

        # Final model saving (model + training info)
        model_path = os.path.join(self.config["models_dir"], "lstm_model.pth")

        local_info = {
            "X_train": X_train_ii,
            "y_train": y_train_ii,
            "X_val": X_val_ii,
            "y_val": y_val_ii,
            "training_duration": training_duration,
            "start_time": start_time,
            "train_loss_by_epoch": train_loss_by_epoch,
            "val_loss_by_epoch": val_loss_by_epoch,
            "best_epoch": best_epoch if best_weights else None,
            "train_loss": epoch_loss,
            "val_loss": best_val_loss if val_loader is not None else None,
        }


        self.save_model(
            model_path=model_path,
            lstm_model=self.lstm,
            fc_model=self.fc,
            hyperparams=None,
            local_info=local_info,
        )
        logger.info("[Train] Final model file saved at: %s", model_path)

    def evaluate_trial(self, trial, X_train, y_train):
        """
        Evaluate a single trial during the Optuna hyperparameter optimization process.
        This process includes:
          1) Obtaining hyperparameters via the trial.
          2) Cross-validation using TimeSeriesSplit where for each fold:
             - The training and validation indices are defined.
             - The training set is further split based on eval_size (similar to LightGBM).
          3) Creating the LSTM, FC, and optimizer locally, training with early stopping.
          4) Calculating the pinball loss and collecting it.
          5) Saving the best model.

        Args:
            trial: The Optuna trial object.
            X_train (DataFrame): Training features.
            y_train (Series/DataFrame): Training target values.

        Returns:
            float: The average pinball loss across the folds.
        """
        hidden_size = trial.suggest_int("hidden_size", *self.optuna_space["hidden_size"])
        num_layers = trial.suggest_int("num_layers", *self.optuna_space["num_layers"])
        dropout = trial.suggest_float("dropout", *self.optuna_space["dropout"])
        learning_rate = trial.suggest_float("learning_rate", *self.optuna_space["learning_rate"], log=True)
        weight_decay = trial.suggest_float("weight_decay", *self.optuna_space["weight_decay"], log=True)
        batch_size = trial.suggest_int("batch_size", *self.optuna_space["batch_size"])
        epochs = trial.suggest_int("epochs", *self.optuna_space["epochs"])
        optimizer_str = trial.suggest_categorical("optimizer", self.optuna_space["optimizer"])
        bias = trial.suggest_categorical("bias", self.optuna_space["bias"])
        seq_length = trial.suggest_int("seq_length", *self.optuna_space["seq_length"])
        clip_cfg = self.optuna_space.get("gradient_clipping", {})
        enabled = trial.suggest_categorical("enabled", clip_cfg["enabled"])  # Optuna chooses True or False

        if enabled:
            clip_method = clip_cfg.get("method")[0]  # Get 'norm'; using get() and [0] since it's a list.
            clip_norm = trial.suggest_float("clip_norm", *clip_cfg["clip_norm"], log=True)
            norm_type = clip_cfg.get("norm_type")  # FIX: not part of Optuna
        else:
            clip_method = None
            clip_norm = None
            norm_type = None

        # Prepare cross-validation (TimeSeriesSplit)
        n_splits = self.config["optuna"].get("n_splits")
        test_size_abs = int(len(X_train) * self.config.get("test_size"))
        inner_tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size_abs, gap=0)

        losses = []
        eval_fraction = self.config.get("eval_set", {}).get("size")
        early_stopping_rounds = self.config["early_stopping"].get("rounds")

        hyper_path = self.config["hyperparameter_dir"]
        self.fold += 1
        os.makedirs(os.path.join(hyper_path, "fold_%d" % self.fold), exist_ok=True)
        model_path = os.path.join(hyper_path, "fold_%d" % self.fold, "lstm_model.pth")

        # Loop over the folds
        for fold_idx, (train_index_inner, val_index_inner) in enumerate(inner_tscv.split(X_train)):
            # Define current fold: X_train_inner, y_train_inner and X_val_inner, y_val_inner
            X_train_inner = X_train.iloc[train_index_inner]
            X_val_inner = X_train.iloc[val_index_inner]
            y_train_inner = y_train.iloc[train_index_inner]
            y_val_inner = y_train.iloc[val_index_inner]

            n_total = len(X_train_inner)
            n_train = int(n_total * (1 - eval_fraction))

            X_train_ii = X_train_inner.iloc[:n_train]
            y_train_ii = y_train_inner.iloc[:n_train]
            X_val_ii = X_train_inner.iloc[n_train:]
            y_val_ii = y_train_inner.iloc[n_train:]

            # Create LSTM windows for training data in the fold
            feature_cols = X_train.columns.tolist()
            target_col = "gesamt"

            df_train_fold = X_train_ii.copy()
            df_train_fold[target_col] = y_train_ii
            X_array_train, y_array_train, idx_arr_tr = self.create_lstm_windows(
                df=df_train_fold,
                feature_cols=feature_cols,
                target_col=target_col,
                seq_length=seq_length,
                forecast_horizon=self.forecast_horizon,
            )
            X_t_tr = torch.tensor(X_array_train, dtype=torch.float32)
            y_t_tr = torch.tensor(y_array_train, dtype=torch.float32)
            train_dataset = TensorDataset(X_t_tr, y_t_tr)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # Prepare evaluation DataLoader if possible
            val_loader = None
            if len(X_val_ii) > 0:
                df_val_fold = X_val_ii.copy()
                df_val_fold[target_col] = y_val_ii
                if len(df_val_fold) >= seq_length + self.forecast_horizon:
                    X_array_val, y_array_val, idx_arr_val = self.create_lstm_windows(
                        df=df_val_fold,
                        feature_cols=feature_cols,
                        target_col=target_col,
                        seq_length=seq_length,
                        forecast_horizon=self.forecast_horizon,
                    )
                    if len(X_array_val) > 0:
                        X_t_val = torch.tensor(X_array_val, dtype=torch.float32)
                        y_t_val = torch.tensor(y_array_val, dtype=torch.float32)
                        val_dataset = TensorDataset(X_t_val, y_t_val)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Create local LSTM, FC, and optimizer for the fold
            input_size = X_t_tr.shape[2]
            model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bias=bias,
                batch_first=True,
            ).to(self.device)

            out_dim = len(self.quantiles)
            temp_fc = nn.Linear(hidden_size, out_dim).to(self.device)

            if optimizer_str == "adam":
                temp_optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(temp_fc.parameters()),
                    lr=learning_rate, weight_decay=weight_decay,
                )
            elif optimizer_str == "AdamW":
                temp_optimizer = torch.optim.AdamW(
                    list(model.parameters()) + list(temp_fc.parameters()),
                    lr=learning_rate, weight_decay=weight_decay,
                )
            elif optimizer_str == "sgd":
                temp_optimizer = torch.optim.SGD(
                    list(model.parameters()) + list(temp_fc.parameters()),
                    lr=learning_rate, weight_decay=weight_decay,
                )
            else:
                raise ValueError("Unknown optimizer: %s" % optimizer_str)

            # Training loop for the current fold with early stopping via val_loader
            best_val_loss = float("inf")
            best_weights = None
            patience_counter = 0

            train_loss_by_epoch = []
            val_loss_by_epoch = []
            start_time = datetime.now().timestamp()
            for epoch in range(epochs):
                model.train()
                total_train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    temp_optimizer.zero_grad()
                    out_lstm, _ = model(X_batch)
                    last_hidden = out_lstm[:, -1, :]
                    preds = temp_fc(last_hidden)
                    loss = self._quantile_loss(preds, y_batch)
                    loss.backward()

                    # Optional gradient clipping
                    if clip_method == "norm":
                        nn.utils.clip_grad_norm_(
                            list(model.parameters()) + list(temp_fc.parameters()),
                            max_norm=clip_norm, norm_type=norm_type,
                        )

                    temp_optimizer.step()
                    total_train_loss += loss.item()

                train_loss = total_train_loss / len(train_loader)
                train_loss_by_epoch.append(train_loss)

                # Validation
                val_loss = None
                if val_loader is not None:
                    val_loss = self._validate(val_loader, model=model, fc=temp_fc)
                    val_loss_by_epoch.append(val_loss)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = {
                            "lstm": model.state_dict(),
                            "fc": temp_fc.state_dict(),
                        }
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter >= early_stopping_rounds:
                        logger.info("[Fold=%d] Early stopping in epoch %d, val_loss=%.4f", fold_idx, epoch + 1, val_loss)
                        break
                else:
                    val_loss_by_epoch.append(None)
                logger.info("[Fold=%d, Epoch=%d/%d] train_loss=%.4f, val_loss=%s", fold_idx, epoch + 1, epochs, train_loss, val_loss)

            # Load best weights if available
            if best_weights is not None:
                model.load_state_dict(best_weights["lstm"])
                temp_fc.load_state_dict(best_weights["fc"])

            # Evaluate final pinball loss
            predictor = Predictor(self.config)

            local_info = {
                "X_train": X_train_ii,
                "y_train": y_train_ii,
                "X_val": X_val_ii,
                "y_val": y_val_ii,
                "train_loss": train_loss,
                "best_val_loss": best_val_loss,
                "fold_idx": fold_idx,
                "val_loss": val_loss,
                "train_loss_by_epoch": train_loss_by_epoch,
                "val_loss_by_epoch": val_loss_by_epoch,
            }

            self.save_model(
                model_path=model_path,
                lstm_model=model,
                fc_model=temp_fc,
                hyperparams={
                    "bias": bias,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "optimizer": optimizer_str,
                    "clip_method": clip_method,
                    "clip_norm": clip_norm,
                    "seq_length": seq_length,
                    "input_size": input_size,
                    "quantiles": self.quantiles,
                },
                local_info=local_info,
            )

            preds = predictor.predict_lstm(
                X_test=X_val_inner,
                y_test=y_val_inner.to_frame("gesamt"),
                y_train=y_train_inner.to_frame("gesamt"),
                X_train=X_train_inner,
                path=model_path,
            )
            predictions_filtered = preds[preds.index.map(self.is_in_time_window)]
            fold_loss = mean_pinball_loss(
                predictions_filtered["gesamt"],
                predictions_filtered["gesamt_pred"],
                alpha=0.5,
            )
            losses.append(fold_loss)

        val_pred_path = os.path.join(hyper_path, "fold_%d" % self.fold, "prediction.parquet")
        predictions_filtered.to_parquet(val_pred_path)

       
        avg_loss = float(np.mean(losses))
        logger.info("[evaluate_trial] Avg. Pinball-Loss over %d folds = %.4f", n_splits, avg_loss)
        return avg_loss

    def _convert_input_to_tensor(self, X):
        """
        Convert the input to a tensor and reshape it to (N, seq_length, self.input_size).

        Args:
            X (DataFrame or ndarray): Input data.

        Returns:
            torch.Tensor: Reshaped tensor.
        """
        if isinstance(X, pd.DataFrame):
            arr = X.values
        else:
            arr = X
        N, D = arr.shape
        if D % seq_length != 0:
            raise ValueError("Cannot reshape into (seq_length, input_size).")
        input_size = D // seq_length
        arr_reshaped = arr.reshape(N, seq_length, input_size)
        return torch.tensor(arr_reshaped, dtype=torch.float32)

    def _quantile_loss(self, preds, target):
        """
        Compute the mean pinball loss for all quantiles.

        Args:
            preds (Tensor): Predictions with shape (Batch, n_quantiles).
            target (Tensor): Target values with shape (Batch,).

        Returns:
            Tensor: Mean pinball loss.
        """
        target = target.unsqueeze(1)  # (Batch, 1)
        quantiles_t = torch.tensor(self.quantiles, device=preds.device).unsqueeze(0)  # (1, n_quantiles)
        error = target - preds  # (Batch, n_quantiles)
        loss = torch.max(quantiles_t * error, (quantiles_t - 1) * error)
        return loss.mean()

    def _validate(self, val_loader, model, fc):
        """
        Compute the average quantile loss on the validation set.

        Args:
            val_loader (DataLoader): DataLoader for validation data.
            model (nn.Module): LSTM model.
            fc (nn.Module): Fully connected layer.

        Returns:
            float: Average loss on the validation set.
        """
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                out, _ = model(X_batch)
                last_hidden = out[:, -1, :]
                preds = fc(last_hidden)
                loss = self._quantile_loss(preds, y_batch)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def save_model(self, model_path: str, lstm_model: nn.Module = None, fc_model: nn.Module = None, hyperparams: dict = None, local_info: dict = None):
        """
        Save the model along with training information to the specified model_path.
        If local_info is provided, it collects the training info and stores it in model_data["training_info"].

        Args:
            model_path (str): Path to save the model.
            lstm_model (nn.Module, optional): LSTM model to save.
            fc_model (nn.Module, optional): Fully connected model to save.
            hyperparams (dict, optional): Hyperparameters to store.
            local_info (dict, optional): Additional training information.
        """
        model_data = {}

        # (A) Save state dicts if provided
        if lstm_model is not None and fc_model is not None:
            model_data["state_dicts"] = {
                "lstm": lstm_model.state_dict(),
                "fc": fc_model.state_dict(),
            }
        else:
            model_data["state_dicts"] = {
                "lstm": self.lstm.state_dict() if self.lstm else None,
                "fc": self.fc.state_dict() if self.fc else None,
            }

        # (B) Hyperparameters
        if hyperparams is not None:
            model_data["hyperparams"] = hyperparams
        else:
            model_data["hyperparams"] = {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "bias": self.bias,
                "batch_size": self.batch_size,
                "quantiles": self.quantiles,
                "seq_length": self.seq_length,
                "input_size": getattr(self, "input_size", None),
            }

        # (C) Collect training info if local_info is provided
        if local_info is not None:
            training_info_dict = self.collect_training_info(
                model_path=model_path,
                X_train=local_info.get("X_train"),
                y_train=local_info.get("y_train"),
                X_val=local_info.get("X_val"),
                y_val=local_info.get("y_val"),
                training_duration=local_info.get("training_duration", 0.0),
                start_time=local_info.get("start_time", None),
                train_loss_by_epoch=local_info.get("train_loss_by_epoch"),
                val_loss_by_epoch=local_info.get("val_loss_by_epoch"),
                best_epoch=local_info.get("best_epoch"),
            )
            model_data["training_info"] = training_info_dict

        # (D) Final save
        torch.save(model_data, model_path)
        logger.info("[save_model] Model + data saved at: %s", model_path)

    def create_lstm_windows(self, df: pd.DataFrame, feature_cols: list, target_col: str, seq_length: int = 168, forecast_horizon: int = 1, drop_incomplete: bool = True):
        """
        Create sequence windows for LSTM training.

        Args:
            df (DataFrame): Pandas DataFrame sorted by time.
            feature_cols (list): List of columns to use as input features.
            target_col (str): Name of the target column.
            seq_length (int): Number of time steps for each input sequence.
            forecast_horizon (int): Number of future steps to forecast.
            drop_incomplete (bool, optional): Whether to drop incomplete sequences at the end.

        Returns:
            tuple: (X, y, indices) where X is an array of shape (N, seq_length, len(feature_cols)),
                   y is an array of shape (N,) for one-step prediction or (N, forecast_horizon) for multi-step,
                   and indices is a list of indices for each sample.
        """
        df = df.sort_index()
        features = df[feature_cols].to_numpy(copy=True)
        target = df[target_col].to_numpy(copy=True)

        X_list, y_list, idx_list = [], [], []
        max_i = len(df) - seq_length - forecast_horizon + 1
        if max_i < 0:
            raise ValueError("Dataframe is too short for seq_length + forecast_horizon")

        for i in range(max_i):
            x_window = features[i : i + seq_length, :]
            if forecast_horizon == 1:
                y_window = target[i + seq_length]
            else:
                y_window = target[i + seq_length : i + seq_length + forecast_horizon]

            X_list.append(x_window)
            y_list.append(y_window)
            idx_list.append(df.index[i + seq_length - 1])

        return np.array(X_list), np.array(y_list), idx_list

    def collect_training_info(self, model_path: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, training_duration: float = 0.0, start_time: float = None, train_loss_by_epoch: list = None, val_loss_by_epoch: list = None, best_epoch: int = None):
        """
        Create a dictionary containing all relevant metadata for training.
        This function does not perform any saving.

        Args:
            model_path (str): Path to the model.
            X_train (DataFrame): Training features.
            y_train (Series): Training target values.
            X_val (DataFrame, optional): Validation features.
            y_val (Series, optional): Validation target values.
            training_duration (float, optional): Duration of training in seconds.
            start_time (float, optional): Timestamp when training started.
            train_loss_by_epoch (list, optional): List of training losses per epoch.
            val_loss_by_epoch (list, optional): List of validation losses per epoch.
            best_epoch (int, optional): Best epoch index.

        Returns:
            dict: A dictionary containing training information.
        """
        training_info = {}

        # (A) Path and feature names
        training_info["model_path"] = model_path
        if isinstance(X_train, pd.DataFrame):
            training_info["feature_names"] = list(X_train.columns)
        else:
            training_info["feature_names"] = None

        # (B) Configuration for reproducibility
        training_info["config"] = self.config

        # (C) Data shapes
        if hasattr(X_train, "shape"):
            training_info["training_data_shape"] = X_train.shape
        if X_val is not None and hasattr(X_val, "shape"):
            training_info["validation_data_shape"] = X_val.shape

        # (D) Data statistics
        if isinstance(X_train, pd.DataFrame) and not X_train.empty:
            training_info["training_data_summary"] = {
                "X_train_mean": X_train.mean(numeric_only=True).to_dict(),
                "X_train_std": X_train.std(numeric_only=True).to_dict(),
                "X_train_quantiles": X_train.quantile([0.25, 0.5, 0.75]).to_dict(),
            }
        if isinstance(y_train, pd.Series) and not y_train.empty:
            training_info["target_summary"] = {
                "y_train_mean": float(y_train.mean()),
                "y_train_std": float(y_train.std()),
                "y_train_min": float(y_train.min()),
                "y_train_max": float(y_train.max()),
            }

        # (E) Timing information
        training_info["training_duration_sec"] = training_duration
        training_info["training_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if start_time is not None:
            start_dt = datetime.fromtimestamp(start_time)
            end_dt = start_dt + timedelta(seconds=training_duration)
            training_info["training_start_time"] = start_dt.isoformat()
            training_info["training_end_time"] = end_dt.isoformat()

        # (F) Loss curves
        evals_result = {}
        if train_loss_by_epoch is not None:
            evals_result["train"] = train_loss_by_epoch
        if val_loss_by_epoch is not None:
            evals_result["valid"] = val_loss_by_epoch
        training_info["evals_result"] = evals_result

        if best_epoch is not None:
            training_info["best_epoch"] = best_epoch

        return training_info

    def get_optuna_search_space(self):
        """
        Retrieve the search space for Optuna hyperparameter optimization.

        Returns:
            dict: The parameter search space.
        """
        return self.config["optuna_search_space"].get("param_space", {})
