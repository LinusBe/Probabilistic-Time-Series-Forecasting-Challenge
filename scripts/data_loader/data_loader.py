import os
import pandas as pd
import numpy as np
from joblib import load
import logging
import pdb
from sklearn.impute import KNNImputer

logger = logging.getLogger(__name__)


class DataLoader:
    """
    DataLoader class for loading and preprocessing both main and exogenous datasets.

    This class is responsible for:
      - Loading the main data (e.g., energy consumption) from a CSV file.
      - Loading exogenous data (e.g., weather and air quality) from multiple CSV files.
      - Filtering the data based on a configured start date.
      - Cleaning and imputing missing values in the exogenous data using various methods.
    """

    def __init__(self, config: dict):
        """
        Initialize the DataLoader with the provided configuration.

        The configuration should contain:
          - A 'dataset' key that indicates which dataset configuration to use.
          - Under the dataset key, a 'data_file' for the main data.
          - An 'exogenous_files' list with exactly 4 file paths (weather history,
            weather forecast, air history, air forecast).
          - A 'start_date' string.
          - A 'features' key with an 'exogenous_features' list.
          - An 'imputation_method' dict, with a key 'use' specifying the method 
            (e.g., 'time', 'knn', or 'spline') and corresponding configuration (e.g., 'time_cfg').

        Args:
            config (dict): Configuration dictionary.
        """
        self.config = config
        dataset_key = self.config['dataset']
        dataset_cfg = self.config[dataset_key]
        self.fillin = self.config.get('imputation_method')
        self.data_file = dataset_cfg['data_file']
        exog_files = dataset_cfg.get('exogenous_files', [])

        if len(exog_files) == 4:
            self.weather_hist_path = exog_files[0]
            self.weather_forecast_path = exog_files[1]
            self.air_hist_path = exog_files[2]
            self.air_forecast_path = exog_files[3]
        else:
            raise ValueError("Expected exactly 4 exogenous files (weather history/forecast, air history/forecast).")

        self.start_date = pd.to_datetime(config.get('start_date'), utc=True)
        self.exog_cols = self.config['features'].get('exogenous_features', [])
        self.missing_threshold = 0.5

    def load_main_data(self) -> pd.DataFrame:
        """
        Load and preprocess the main data file.

        The function checks for the existence of the main data file, reads it in,
        converts the index to UTC datetime, resamples the data on an hourly basis (summing values),
        and removes the 'date_time_local' column if it exists.

        Returns:
            pd.DataFrame: Preprocessed main data with an hourly time index named 'date_time_utc'.

        Raises:
            FileNotFoundError: If the main data file does not exist.
        """
        if not os.path.exists(self.data_file):
            logger.error("Main data file not found: %s", self.data_file)
            raise FileNotFoundError(f"{self.data_file} nicht gefunden.")

        df = pd.read_csv(self.data_file, index_col=0, parse_dates=True)[['gesamt']]
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('UTC')
        df = df.resample('1h').sum()
        df.index.name = 'date_time_utc'

        if 'date_time_local' in df.columns:
            df.drop(columns=['date_time_local'], inplace=True)

        logger.info("Hauptdaten geladen von %s. Shape: %s", self.data_file, df.shape)
        return df

    def load_exogenous_data(self) -> pd.DataFrame:
        """
        Load and combine exogenous data from weather and air CSV files.

        The function reads weather history, weather forecast, air quality history, and air quality forecast files.
        It ensures that all DataFrame indices are converted to UTC datetime and then resamples the air quality data
        if available. Weather and air data are merged using combine_first and concatenated along columns.
        The resulting DataFrame is sorted by time and duplicates are removed.

        Returns:
            pd.DataFrame: Combined exogenous data.

        Raises:
            ValueError: If the exogenous data starts later than the configured start_date.
        """
        weather_hist = pd.read_csv(self.weather_hist_path, index_col=0, parse_dates=True)
        weather_forecast = pd.read_csv(self.weather_forecast_path, index_col=0, parse_dates=True)
        air_hist = pd.read_csv(self.air_hist_path, index_col=0, parse_dates=True)
        air_forecast = pd.read_csv(self.air_forecast_path, index_col=0, parse_dates=True)

        for df_ in [weather_hist, weather_forecast, air_hist, air_forecast]:
            if not df_.empty:
                df_.index = pd.to_datetime(df_.index, utc=True)
                df_.index.name = 'date_time_utc'

        if not air_hist.empty:
            air_hist = air_hist.asfreq('h')
            air_forecast = air_forecast.asfreq('h')

        weather_merged = weather_hist.combine_first(weather_forecast)
        air_merged = air_hist.combine_first(air_forecast)
        all_exog = pd.concat([weather_merged, air_merged], axis=1, join="outer")
        all_exog.sort_index(inplace=True)
        all_exog = all_exog[~all_exog.index.duplicated(keep='first')]

        start_date_params = self.start_date if hasattr(self, 'start_date') else all_exog.index.min()
        if all_exog.index.min() > start_date_params:
            raise ValueError(f"Exogene Daten starten erst am {all_exog.index.min().strftime('%Y-%m-%d %H:%M')}.")

        logger.info("Alle exogenen Daten geladen und kombiniert. Shape: %s", all_exog.shape)
        return all_exog

    def _impute_time(self, df: pd.DataFrame, numeric_cols, use_cfg) -> pd.DataFrame:
        """
        Perform time-based interpolation on the specified numeric columns.

        Args:
            df (pd.DataFrame): DataFrame containing numeric columns to impute.
            numeric_cols: List or Index of numeric column names.
            use_cfg (dict): Configuration dictionary specifying:
                - 'method': The interpolation method.
                - 'limit_direction': Direction for interpolation.

        Returns:
            pd.DataFrame: Interpolated numeric data.
        """
        return df[numeric_cols].interpolate(method=use_cfg['method'], limit_direction=use_cfg['limit_direction'])

    def _impute_knn(self, df: pd.DataFrame, numeric_cols, use_cfg) -> pd.DataFrame:
        """
        Perform KNN imputation on the specified numeric columns.

        Args:
            df (pd.DataFrame): DataFrame with missing numeric values.
            numeric_cols: List or Index of numeric column names.
            use_cfg (dict): Configuration dictionary specifying:
                - 'n_neighbors': Number of neighbors for KNN.
                - 'weights': Weight function for prediction.
                - 'metric': Distance metric.

        Returns:
            pd.DataFrame: DataFrame with imputed numeric values, preserving original indices and columns.
        """
        imputer = KNNImputer(n_neighbors=use_cfg['n_neighbors'],
                             weights=use_cfg['weights'],
                             metric=use_cfg['metric'])
        imputed_array = imputer.fit_transform(df[numeric_cols])
        # Rückgabe als DataFrame, um die Indizes und Spalten beizubehalten
        return pd.DataFrame(imputed_array, index=df.index, columns=numeric_cols)

    def _impute_spline(self, df: pd.DataFrame, numeric_cols, use_cfg) -> pd.DataFrame:
        """
        Perform spline (polynomial) interpolation on the specified numeric columns.

        Args:
            df (pd.DataFrame): DataFrame with missing numeric values.
            numeric_cols: List or Index of numeric column names.
            use_cfg (dict): Configuration dictionary specifying:
                - 'method': The interpolation method ('spline').
                - 'order': Order of the polynomial.
                - 'limit_direction': Direction for interpolation.

        Returns:
            pd.DataFrame: DataFrame with imputed numeric values using spline interpolation.
        """
        return df[numeric_cols].interpolate(method=use_cfg['method'],
                                              order=use_cfg['order'],
                                              limit_direction=use_cfg['limit_direction'])

    def _clean_and_impute_exog(self, df_exog: pd.DataFrame, lag_buffer_hours) -> pd.DataFrame:
        """
        Clean and impute missing values in the exogenous data.

        The function performs the following steps:
          1. Filters the exogenous DataFrame to include data from (start_date - lag_buffer_hours) onward.
          2. Removes columns with a missing rate higher than self.missing_threshold.
          3. Applies an imputation method specified in the configuration ('time', 'knn', or 'spline').
          4. Fills any remaining gaps using forward-fill and backward-fill.
          5. Logs a warning if there are still missing values after imputation.

        Args:
            df_exog (pd.DataFrame): Raw exogenous data.
            lag_buffer_hours (int): Number of hours to subtract from start_date for filtering.

        Returns:
            pd.DataFrame: Cleaned and imputed exogenous data.
        """
        if df_exog.empty:
            logger.warning("Exogene Daten sind leer. Überspringe Imputation.")
            return df_exog

        df_exog = df_exog.loc[self.start_date - pd.Timedelta(hours=lag_buffer_hours):]

        missing_frac = df_exog.isna().mean()
        keep_cols = missing_frac[missing_frac <= self.missing_threshold].index
        drop_cols = missing_frac[missing_frac > self.missing_threshold].index

        if len(drop_cols) > 0:
            logger.info("Droppe Spalten wegen zu vieler NaNs (> %d%%): %s",
                        self.missing_threshold * 100, list(drop_cols))

        df_exog = df_exog[keep_cols]
        df_exog.sort_index(inplace=True)
        numeric_cols = df_exog.select_dtypes(include=['float', 'int']).columns

        imputation_method = self.fillin['use']
        use_cfg = self.fillin.get(f'{imputation_method}_cfg')
        logger.info("Verwende Imputation-Methode: %s", imputation_method)
        if imputation_method == "time":
            df_imputed = self._impute_time(df_exog, numeric_cols, use_cfg)
        elif imputation_method == "knn":
            df_imputed = self._impute_knn(df_exog, numeric_cols, use_cfg)
        elif imputation_method == "spline":
            df_imputed = self._impute_spline(df_exog, numeric_cols, use_cfg)
        else:
            logger.warning("Unbekannte Imputation-Methode '%s', benutze Standard (time).", imputation_method)
            df_imputed = self._impute_time(df_exog, numeric_cols, use_cfg)

        df_exog[numeric_cols] = df_imputed
        df_exog.fillna(method='ffill', inplace=True)
        df_exog.fillna(method='bfill', inplace=True)

        remain_na = df_exog.isna().sum().sum()
        if remain_na > 0:
            logger.warning("Nach Imputation sind noch %d NaNs übrig.", remain_na)

        logger.info("Exogene Daten nach Imputation: shape=%s", df_exog.shape)
        return df_exog

    def load_data(self, start_date: str = None):
        """
        Load and combine the main and exogenous data.

        This function optionally updates the start_date if provided,
        then performs the following:
          - Loads the main data using load_main_data().
          - Loads the exogenous data using load_exogenous_data().
          - Filters the main data from (start_date - lag_buffer_hours) onward.
          - Cleans and imputes missing values in the exogenous data.
        Returns a tuple of (filtered main data, cleaned exogenous data).

        Args:
            start_date (str, optional): New start date to override the configuration's start_date.

        Returns:
            tuple: (df_main_filtered, df_exog_clean)
                - df_main_filtered (pd.DataFrame): Main data filtered from (start_date - lag_buffer_hours) onward.
                - df_exog_clean (pd.DataFrame): Cleaned and imputed exogenous data.
        """
        if start_date:
            self.start_date = pd.to_datetime(start_date, utc=True)

        df_main = self.load_main_data()
        df_exog = self.load_exogenous_data()
        lag_buffer_hours = 1600  # Number of hours before start_date to include

        df_main_filtered = df_main.loc[self.start_date - pd.Timedelta(hours=lag_buffer_hours):]
        df_exog_clean = self._clean_and_impute_exog(df_exog, lag_buffer_hours)
        return df_main_filtered, df_exog_clean
