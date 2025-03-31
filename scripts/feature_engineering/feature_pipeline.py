"""
Module: feature_pipeline
This module implements the FeaturePipeline class which is responsible for feature engineering.
It loads data, creates various feature groups (time, lag, exogenous, advanced), applies normalization,
and finally splits the data into training and test sets.
"""

import os
import pandas as pd
import numpy as np
from scripts.utils.logger import logger
from scripts.data_loader.data_loader import DataLoader
import holidays
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pdb

class FeaturePipeline:
    def __init__(self, config: dict):
        """
        Initialize the FeaturePipeline.

        Args:
            config (dict): Configuration dictionary with keys such as "dataset", "results_dir",
                           "features", "train_size", "test_size", "start_date", etc.
        """
        self.config = config

        # Dataset and directory paths
        self.dataset = config["dataset"]
        self.results_dir = self.config["results_dir"]
        self.feature_file = os.path.join(self.results_dir, "engineered_features.csv")
        # self.stats_file = os.path.join(self.results_dir, "train_stats.json")  # For normalization parameters

        # Feature configuration from the config
        self.features_cfg = self.config.get("features", {})
        self.time_cfg = self.features_cfg.get("time_features", [])
        self.exog_cfg = self.features_cfg.get("exogenous", {})
        self.fourier_terms = self.features_cfg.get("fourier_terms", False)
        self.advanced_cfg = self.features_cfg.get("advanced", {})
        self.lags = self.features_cfg.get("target", {}).get("lags", [])

        # Global settings
        self.train_size = self.config.get("train_size")
        self.test_size = self.config.get("test_size")
        self.start_date = self.config.get("start_date")

        # Global list of features – starting with the target variable
        self.features = ["gesamt"]

        # Initialize DataLoader
        self.data_loader = DataLoader(config)

        # Training and test data placeholders
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

        # Normalization parameters will be stored here
        self.train_stats = {}

        # Store feature names of respective groups (important for later normalization)
        self.time_features = []
        self.lag_features = []
        self.exog_features = []
        self.advanced_features = []
        self.base_features = []

        norm_config = self.features_cfg.get("normalization", {})
        self.time_norm_cfg = norm_config.get("time", {"enabled": False})
        self.lag_norm_cfg = norm_config.get("lag", {"enabled": False})
        self.exog_norm_cfg = norm_config.get("exog", {"enabled": False})
        self.advanced_norm_cfg = norm_config.get("advanced", {"enabled": False})
        self.base_features_cfg = norm_config.get("base_features", {"enabled": False})
        logger.info("[FeaturePipeline] Initialized with dataset '%s' and results directory '%s'.", self.dataset, self.results_dir)

    def run(self):
        """Run the entire feature engineering pipeline."""
        logger.info("[FeaturePipeline] Loading data...")
        df_main, df_exog = self.data_loader.load_data(start_date=self.start_date)

        logger.info("[FeaturePipeline] Creating features...")
        df = self.create_features(df_main, df_exog)

        logger.info("[FeaturePipeline] Splitting data into train and test sets...")
        # pdb.set_trace()
        self._split_data(df)
        logger.info("[FeaturePipeline] Applying normalization to training and test data...")
        self._apply_normalization()
        logger.info("[FeaturePipeline] Feature engineering pipeline completed.")
        # logger.info("[FeaturePipeline] Saving training statistics...")
        # self._save_train_stats()

    def create_features(self, df_main: pd.DataFrame, df_exog: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features by joining main and exogenous data and generating various feature groups.

        Args:
            df_main (DataFrame): Main data.
            df_exog (DataFrame): Exogenous data.

        Returns:
            DataFrame: DataFrame with engineered features.
        """
        if os.path.exists(self.feature_file):
            logger.info("[FeaturePipeline] Engineered features file found. Loading existing features.")
            return self._load_existing_features()

        logger.info("[FeaturePipeline] Merging main and exogenous data.")
        df = df_main.copy()
        df = df.join(df_exog, how="left")

        if len(self.exog_cfg.get("base_features")) > 0:
            if self.exog_cfg.get("base_features")[0] == 'ALL':
                self.features = list(df.columns)
        else:
            missing_feats = set(self.exog_cfg.get("base_features")) - set(df.columns)
            self.features = [f for f in self.features if f not in missing_feats]
        logger.info("[FeaturePipeline] Base features set: %s", self.features)

        self.base_features = self.features.copy()
 
        if "gesamt" in self.base_features:
            self.base_features.remove("gesamt")

        # Create individual feature groups
        logger.info("[FeaturePipeline] Creating time features...")
        self._create_time_features(df)
        logger.info("[FeaturePipeline] Creating lag features...")
        self._create_lag_features(df)
        logger.info("[FeaturePipeline] Creating exogenous features...")
        self._create_exog_features(df)
        logger.info("[FeaturePipeline] Creating advanced features...")
        self._create_advanced_features(df)

        # Remove unwanted features if any
        logger.info("[FeaturePipeline] Removing unwanted features.")
        df = self._remove_features(df)

        return df

    def _create_time_features(self, df: pd.DataFrame):
        """
        Create time-based features.

        Args:
            df (DataFrame): DataFrame on which to create time features.
        """
        dt_index = df.index
        new_features = []
        if "hour" in self.time_cfg:
            df["hour"] = dt_index.hour
            new_features.append("hour")
        if "weekday" in self.time_cfg:
            df["weekday"] = dt_index.weekday
            new_features.append("weekday")
        if "month" in self.time_cfg:
            df["month"] = dt_index.month
            new_features.append("month")
        if "is_weekend" in self.time_cfg:
            df["is_weekend"] = (dt_index.weekday >= 5).astype(int)
            new_features.append("is_weekend")
        if "summer_winter_time" in self.time_cfg:
            self._summer_winter_dummies(df)
            new_features.extend(["summer_dummy", "winter_dummy"])
        if self.fourier_terms:
            self._add_fourier_terms(df)
            new_features.extend(["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"])
        self.time_features.extend(new_features)
        self.features.extend(new_features)
        logger.info("[FeaturePipeline] Time features created: %s", new_features)

    def _add_fourier_terms(self, df: pd.DataFrame):
        """
        Add Fourier transformation terms for cyclical features.

        Args:
            df (DataFrame): DataFrame on which to add Fourier terms.
        """
        if "hour" in df:
            hours = df["hour"]
            df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
            df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        if "weekday" in df:
            weekdays = df["weekday"]
            df["weekday_sin"] = np.sin(2 * np.pi * weekdays / 7)
            df["weekday_cos"] = np.cos(2 * np.pi * weekdays / 7)
        logger.info("[FeaturePipeline] Fourier terms added.")

    def _create_lag_features(self, df: pd.DataFrame):
        """
        Create lag features for the target variable.

        Args:
            df (DataFrame): DataFrame on which to create lag features.
        """
        new_features = []
        for lag in self.lags:
            colname = f"lag_{lag}"
            df[colname] = df["gesamt"].shift(lag)
            new_features.append(colname)
        self.lag_features.extend(new_features)
        self.features.extend(new_features)
        logger.info("[FeaturePipeline] Lag features created: %s", new_features)

    def _create_exog_features(self, df: pd.DataFrame):
        """
        Create exogenous features based on specified transformations.

        Args:
            df (DataFrame): DataFrame on which to create exogenous features.
        """
        new_features = []
        # Rolling statistics
        if "rolling" in self.exog_cfg.get("transformations", {}):
            cfg = self.exog_cfg["transformations"]["rolling"]
            features = cfg.get("features", [])
            for feat in features:
                for w in cfg.get("windows", []):
                    for stat in cfg.get("stats", []):
                        col_name = f"{feat}_{stat}_{w}h"
                        df[col_name] = getattr(df[feat].rolling(w, min_periods=1), stat)()
                        new_features.append(col_name)
        # Differences
        if "diff" in self.exog_cfg.get("transformations", {}):
            cfg = self.exog_cfg["transformations"]["diff"]
            features = cfg.get("features", [])
            for feat in features:
                for w in cfg.get("windows", []):
                    col_name = f"{feat}_diff_{w}h"
                    df[col_name] = df[feat].diff(w)
                    new_features.append(col_name)
        self.exog_features.extend(new_features)
        self.features.extend(new_features)
        logger.info("[FeaturePipeline] Exogenous features created: %s", new_features)

    def _create_advanced_features(self, df: pd.DataFrame):
        """
        Create advanced features including holiday features, interactions, and rolling moments.

        Args:
            df (DataFrame): DataFrame on which to create advanced features.
        """
        new_features = []
        if self.advanced_cfg.get("holiday", {}).get("enabled", False):
            self._add_holiday_features(df)
            new_features.append("is_holiday")
        if self.advanced_cfg.get("interactions"):
            for feat1, feat2 in self.advanced_cfg["interactions"]:
                if feat1 in df and feat2 in df:
                    col_name = f"{feat1}_x_{feat2}"
                    df[col_name] = df[feat1] * df[feat2]
                    new_features.append(col_name)
        if self.advanced_cfg.get("rolling_moments"):
            cols = self.advanced_cfg["rolling_moments"].get("features", [])
            for target in cols:
                for w in self.advanced_cfg["rolling_moments"].get("windows", []):
                    skew_col = f"{target}_skew_{w}h"
                    kurt_col = f"{target}_kurtosis_{w}h"
                    df[skew_col] = df[target].rolling(w).apply(skew)
                    df[kurt_col] = df[target].rolling(w).apply(kurtosis)
                    new_features.extend([skew_col, kurt_col])
        self.advanced_features.extend(new_features)
        self.features.extend(new_features)
        logger.info("[FeaturePipeline] Advanced features created: %s", new_features)

    def _add_holiday_features(self, df: pd.DataFrame):
        """
        Add holiday indicator features to the DataFrame.

        Args:
            df (DataFrame): DataFrame on which to add holiday features.
        """
        cfg = self.advanced_cfg.get("holiday", {})
        years = range(df.index.year.min() - 1, df.index.year.max() + 1)
        country_holidays = holidays.CountryHoliday(cfg.get("country", "DE"), years=years)
        df["is_holiday"] = df.index.to_series().apply(lambda x: 1 if x in country_holidays else 0)
        logger.info("[FeaturePipeline] Holiday features added: is_holiday.")
        if cfg.get("proximity", False):
            self._add_holiday_proximity(df, country_holidays)
            logger.info("[FeaturePipeline] Holiday proximity features added: days_since_holiday, days_until_holiday.")
            self.features.extend(["days_since_holiday", "days_until_holiday"])

    def _add_holiday_proximity(self, df: pd.DataFrame, country_holidays):
        """
        Add features indicating the number of days since the last holiday and until the next holiday.

        Args:
            df (DataFrame): DataFrame on which to add holiday proximity features.
            country_holidays: Holiday calendar.
        """
        holiday_dates = pd.to_datetime(list(country_holidays.keys())).normalize()
        holiday_dates = np.array(holiday_dates, dtype="datetime64[D]")
        dates = pd.to_datetime(df.index).normalize()
        dates_array = np.array(dates, dtype="datetime64[D]")
        idx_last = np.searchsorted(holiday_dates, dates_array, side="right") - 1
        last_holiday = np.where(idx_last < 0, dates_array, holiday_dates[idx_last])
        df["days_since_holiday"] = (dates_array - last_holiday).astype(int)
        idx_next = np.searchsorted(holiday_dates, dates_array, side="left")
        idx_next_clipped = np.clip(idx_next, 0, len(holiday_dates) - 1)
        next_holiday_raw = holiday_dates[idx_next_clipped]
        next_holiday = np.where(idx_next == len(holiday_dates), dates_array, next_holiday_raw)
        df["days_until_holiday"] = (next_holiday - dates_array).astype(int)

    def _summer_winter_dummies(self, df: pd.DataFrame):
        """
        Create dummy features for summer and winter.

        Args:
            df (DataFrame): DataFrame on which to create summer and winter dummy features.
        """
        df["summer_dummy"] = df.index.month.isin([4, 5, 6, 7, 8, 9]).astype(int)
        df["winter_dummy"] = (~df.index.month.isin([4, 5, 6, 7, 8, 9]).astype(bool)).astype(int)
        logger.info("[FeaturePipeline] Summer and winter dummy features created.")

    def _load_existing_features(self):
        """
        Load existing engineered features from file.

        Returns:
            DataFrame: DataFrame with engineered features.
        """
        logger.info("[FeaturePipeline] Loading existing engineered features from file: %s", self.feature_file)
        df = pd.read_csv(self.feature_file, index_col=0, parse_dates=True).asfreq("h")
        return df

    def _remove_features(self, df: pd.DataFrame):
        """
        Remove features from the DataFrame that are not in the final features list.

        Args:
            df (DataFrame): Input DataFrame with many features.

        Returns:
            DataFrame: DataFrame containing only the final selected features.
        """
        feat_to_remove = [f for f in df.columns if f not in self.features]
        logger.info("Removing features: %s", feat_to_remove)
        return df[self.features]

    def _split_data(self, df: pd.DataFrame):
        """
        Split the DataFrame into training and test sets based on configuration.

        Args:
            df (DataFrame): Input DataFrame with engineered features.
        """
        logger.info("[FeaturePipeline] Splitting data starting from %s", self.start_date)
        df = df.loc[self.start_date:]
        if df.isnull().values.any():
            raise ValueError("NaN values found in features. Please check feature creation.")

        if isinstance(self.train_size, float) and 0 < self.train_size < 1 and isinstance(self.test_size, float) and 0 < self.test_size < 1:
            n_total = len(df)
            n_train = int(n_total * self.train_size)
            logger.info("[FeaturePipeline] Total records: %d, Training records: %d, Test records: %d", n_total, n_train, n_total - n_train)
            df_train = df.iloc[:n_train]
            df_test = df.iloc[n_train:]
        else:
            train_end = pd.to_datetime(self.train_size, utc=True)
            test_start = pd.to_datetime(self.test_size, utc=True)
            test_end = test_start + pd.Timedelta(f'{self.config["forecast_horizon"]}h')
            logger.info("[FeaturePipeline] Splitting data by dates: Train end %s, Test start %s, Test end %s", train_end, test_start, test_end)
            df_train = df.loc[:train_end]
            df_test = df.loc[test_start:test_end]



        self.X_train, self.y_train = df_train.drop(columns=["gesamt"]), df_train["gesamt"]
        self.X_test, self.y_test = df_test.drop(columns=["gesamt"]), df_test["gesamt"]

    def _apply_normalization(self):
        """
        Apply normalization to different groups of features based solely on the training data.
        """
        logger.info("[FeaturePipeline] Applying normalization to feature groups...")

        def _normalize_group(X_train, X_test, features, norm_cfg, group_name):
            if not norm_cfg.get("enabled", False) or not features:
                logger.info("[FeaturePipeline] Normalization for group '%s' is disabled or no features provided.", group_name)
                return X_train, X_test

            method = norm_cfg.get("method", "standardize")
            if method == "standardize":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            elif method == "robust":
                scaler = RobustScaler()
            else:
                raise ValueError("Invalid normalization method.")

            logger.info("[FeaturePipeline] Normalizing group '%s' using %s method.", group_name, method)
            scaler.fit(X_train[features])
            X_train[features] = scaler.transform(X_train[features])
            X_test[features] = scaler.transform(X_test[features])

            # Save parameters
            self.train_stats[group_name] = {"method": method, "features": features}
            if method == "standardize":
                self.train_stats[group_name]["mean"] = scaler.mean_.tolist()
                self.train_stats[group_name]["std"] = scaler.scale_.tolist()
            elif method == "minmax":
                self.train_stats[group_name]["min"] = scaler.min_.tolist()
                self.train_stats[group_name]["max"] = scaler.data_max_.tolist()
            elif method == "robust":
                self.train_stats[group_name]["center"] = scaler.center_.tolist()
                self.train_stats[group_name]["scale"] = scaler.scale_.tolist()
            logger.info("[FeaturePipeline] Normalization parameters for group '%s': %s", group_name, self.train_stats[group_name])
            return X_train, X_test

        self.X_train, self.X_test = _normalize_group(self.X_train, self.X_test, self.time_features, self.time_norm_cfg, "time")
        self.X_train, self.X_test = _normalize_group(self.X_train, self.X_test, self.lag_features, self.lag_norm_cfg, "lag")
        self.X_train, self.X_test = _normalize_group(self.X_train, self.X_test, self.exog_features, self.exog_norm_cfg, "exog")
        self.X_train, self.X_test = _normalize_group(self.X_train, self.X_test, self.advanced_features, self.advanced_norm_cfg, "advanced")
        self.X_train, self.X_test = _normalize_group(self.X_train, self.X_test, self.base_features, self.base_features_cfg, "base")

        # Concat X_train and X_test and save to feature_file
        logger.info("[FeaturePipeline] Saving engineered features to %s", self.feature_file)
        df_exog = pd.concat([self.X_train, self.X_test], axis=0)
        df_ges = pd.concat([self.y_train, self.y_test], axis=0)
        df = pd.concat([df_ges, df_exog], axis=1)

        df.to_csv(self.feature_file)

    def get_train_data(self):
        """
        Retrieve the training data.

        Returns:
            tuple: (X_train, y_train)
        """
        return self.X_train, self.y_train

    def get_test_data(self):
        """
        Retrieve the test data.

        Returns:
            tuple: (X_test, y_test)
        """
        return self.X_test, self.y_test
