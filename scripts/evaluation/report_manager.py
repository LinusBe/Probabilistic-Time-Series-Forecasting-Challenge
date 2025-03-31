import os
import logging
import seaborn as sns
import warnings
import statsmodels.api as sm
import numpy as np
import pandas as pd
import re
import glob
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from scipy.stats import shapiro, probplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_pinball_loss, mean_absolute_error, mean_absolute_percentage_error
import joblib
import pdb
import torch
import shap
from joblib import Parallel, delayed

# Globale Stildefinition für ein einheitliches Erscheinungsbild
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "savefig.dpi": 300,           # Hohe Auflösung
    "figure.autolayout": True,    # Optimales Layout
    "lines.linewidth": 2,
    "axes.grid": True,
})

# Farbkons­tanten für ein konsistentes Farbschema
ACTUAL_COLOR = "black"
FORECAST_COLOR = "steelblue"
INTERVAL_COLOR = "steelblue"  # gleicher Farbton wie FORECAST_COLOR, aber mit alpha=0.2
RESIDUAL_COLOR = "darkgray"

logger = logging.getLogger(__name__)

# Logger-Konfiguration (bleibt unverändert)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

class ExtendedReportManager:
    """
    Performs multiple analyses for probabilistic time series forecasting.
    Analyses include:
      1) Time Series Plot (Actual vs. Forecast)
      2) Residual Plot and Histogram
      3) ACF/PACF plots & Ljung–Box test on residuals
      4) Coverage Plot comparing nominal vs. empirical coverage
      5) Shapiro–Wilk Test for normality of residuals
      6) Q–Q Plot of residuals against a normal distribution
      7) Pinball Loss (quantile loss) computation and a relative D2 metric
      8) PIT (Probability Integral Transform) histogram
      9) MAE and MAPE calculations for the median forecast (0.5 quantile)
      10) Overlaid Forecast Plot for selected days
      11) Outlier Report based on residuals
      12) Calibration Curve plotting
      13) Sharpness calculation (average prediction interval width)
      14) Residual scatter plot
      15) Aggregated feature importance trends and boxplots across folds/weeks
      16) Generation of training loss curves, SHAP summary, partial dependence plots, and overall model training summaries.
    """
    def __init__(self, output_dir: str, config: Dict[str, Any]):
        """
        Initialize the ExtendedReportManager.

        Args:
            output_dir (str): Base directory where all results (plots, stats, analysis) will be stored.
            config (dict): Configuration dictionary containing keys like "quantiles", "dataset", "model", etc.
        """
        self.output_dir = output_dir
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        self.stats_dir = os.path.join(self.output_dir, "stats")
        os.makedirs(self.stats_dir, exist_ok=True)
        self.analysis_dir = os.path.join(self.output_dir, "analysis")
        os.makedirs(self.analysis_dir, exist_ok=True)
        self.pred_path = os.path.join(self.output_dir, 'predictions')
        self.config = config
        self.quantiles = self.config.get('quantiles')
        self.test_start_date = None
        self.test_end_date = None
        self.report_lines: List[str] = []
        self.data = None  # Daten für correlation_missingness
        self.marker = False
        log_level = config.get('log_level', 'INFO').upper()
        logger.setLevel(log_level)

        fh = logging.FileHandler(os.path.join(self.output_dir, "report.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def _filter_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the predictions to retain only rows within a specific time window:
        Wednesday from 23:00, all of Thursday and Friday, and Saturday until 23:00.

        Args:
            predictions (pd.DataFrame): DataFrame with predictions indexed by time.

        Returns:
            pd.DataFrame: Filtered DataFrame containing only the rows within the target time window.
        """
        def is_in_time_window(ts):
            # Mittwoch ab 23 Uhr, Donnerstag/Freitag komplett, Samstag bis 23 Uhr
            if ts.weekday() == 2:
                return ts.hour >= 23
            elif ts.weekday() in (3, 4):
                return True
            elif ts.weekday() == 5:
                return ts.hour <= 23
            else:
                return False

        mask = predictions.index.map(is_in_time_window)
        return predictions[mask]

    def load_baseline_pre(self) -> pd.DataFrame:
        """
        Load baseline forecast predictions from a predefined file.

        Returns:
            pd.DataFrame: Baseline predictions loaded from the file.
        """
        if self.config.get('version')[-3] == '0':
            dataset = self.config['dataset']
            baseline_pred = pd.read_csv(
                rf'output/{dataset}/baseline/v1.0.0/predictions/final_predictions.csv',
                index_col=0, parse_dates=True
            ).asfreq('h')
            return baseline_pred
        elif self.config.get('version')[-3] == '1':
            dataset = self.config['dataset']
            model = self.config['model']
            # highest version with pattern v1.0.x, choose x the highest in path  rf'output/{dataset}/{model}/v1.1.x
            version = max((p for p in glob.glob(f'output/{dataset}/{model}/v1.0.*') if os.path.isdir(p) if (m := re.match(r'^v1\.0\.(\d+)$', os.path.basename(p)))), key=lambda p: int(re.match(r'^v1\.0\.(\d+)$', os.path.basename(p)).group(1)), default=None)
            baseline_pred = pd.read_csv(
                rf'{version}/predictions/final_predictions.csv',
                index_col=0, parse_dates=True
            ).asfreq('h')
            return baseline_pred
        elif self.config.get('version')[-3] == '2':
            dataset = self.config['dataset']
            model = self.config['model']
            # highest version with pattern v1.0.x, choose x the highest in path  rf'output/{dataset}/{model}/v1.1.x
            baseline_pred = pd.read_csv(
                rf'output/{dataset}/{model}/v1.1.x/final_predictions.csv',
                index_col=0, parse_dates=True
            ).asfreq('h')
            return baseline_pred
        elif self.config.get('version')[-3] == '3':
            dataset = self.config['dataset']
            model = self.config['model']
            # highest version with pattern v1.0.x, choose x the highest in path  rf'output/{dataset}/{model}/v1.1.x
            baseline_pred = pd.read_csv(
                rf'output/{dataset}/{model}/v1.1.x/final_predictions.csv',
                index_col=0, parse_dates=True
            ).asfreq('h')
            return baseline_pred

    def run_all_analyses(self, predictions: pd.DataFrame, X_train: Optional[pd.DataFrame] = None):
        """
        Run all analyses for both the overall test period and specific target time points.

        Expected prediction DataFrame columns include:
          - 'date_time_utc' (as index or column)
          - 'gesamt' (actual values)
          - 'gesamt_pred_0.5' (median forecast)
          - Optionally: additional quantile columns (e.g., 'gesamt_pred_0.025', etc.)

        Args:
            predictions (pd.DataFrame): Forecast predictions DataFrame.
            X_train (pd.DataFrame, optional): Training data for additional model-related plots.
        """
        predictions = predictions.sort_index()
        if predictions.index.duplicated().any():
            logger.warning("Duplicate timestamps found in index. Removing duplicates.")
            predictions = predictions[~predictions.index.duplicated(keep="first")]

        self.test_start_date = predictions.index[0]
        self.test_end_date = predictions.index[-1]
        predictions_filtered = self._filter_predictions(predictions)
        predictions_filtered.to_csv(os.path.join(self.pred_path, "final_predictions.csv"))

        # Run analyses for the overall test period
        self._run_analyses_for_period(predictions_filtered, "overall", X_train)
        predictions_specific_times = self._filter_by_specific_times(predictions_filtered)
        if predictions_specific_times is not None:
            self._run_analyses_for_period(predictions_specific_times, "specific", X_train)

        # LSTM-specific analyses if applicable
        if self.config.get('model') == 'lstm':
            model_files = []
            model_files.extend(glob.glob(os.path.join(self.config["models_dir"], "**", "*.pth"), recursive=True))
            model_files.extend(glob.glob(os.path.join(self.config["hyperparameter_dir"], "**", "*.pth"), recursive=True))
            if model_files:
                for model_file in model_files:
                    model_name = os.path.splitext(os.path.basename(model_file))[0]
                    self.plot_loss_curves_lstm(model_file, period_label=model_name)
                    self.plot_predictions_lstm(model_file, period_label=model_name)

        self.save_report_summary()

    def _run_analyses_for_period(self, predictions: pd.DataFrame, period_label: str, X_train: Optional[pd.DataFrame] = None):
        """
        Run all analyses for a specified period.

        Args:
            predictions (pd.DataFrame): Predictions DataFrame for the period.
            period_label (str): Label for the period ("overall" or "specific").
            X_train (pd.DataFrame, optional): Training data for model analysis.
        """
        self.plot_time_series(predictions, period_label)
        self.plot_residuals_and_histogram(predictions, period_label)
        self.acf_pacf_and_ljungbox(predictions, period_label)
        self.coverage_plot(predictions, lower_q=0.025, upper_q=0.975, period_label=period_label)
        self.shapiro_test(predictions, period_label)
        self.qq_plot_residuals(predictions, period_label)
        self.calc_pinball_losses(predictions, period_label)
        self.pit_histogram(predictions, period_label)
        self.calc_mae_mape(predictions, period_label)
        self.plot_overlaid_predictions(predictions, period_label)
        self.outlier_report(predictions, period_label)
        self.plot_calibration_diagnostics(predictions, period_label)
        self.calculate_sharpness(predictions, period_label)
        self.plot_residual_scatter(predictions, period_label)
 
        self.cusum_forecast_errors(predictions, period_label)
        self.residual_density_plot(predictions, period_label)
        self.mincer_zarnowitz_regression(predictions['gesamt'], predictions['gesamt_pred_0.5'], period_label)


        baseline_forecast = self.load_baseline_pre()['gesamt_pred_0.5']
        self.diebold_mariano_test(predictions['gesamt'], predictions['gesamt_pred_0.5'], baseline_forecast, period_label=period_label)

        if period_label == "overall" and X_train is not None:
            models_dir = os.path.join(self.output_dir, "models")
            week_model_dict = self.create_week_model_dict(models_dir)
            feature_names = list(X_train.columns)
            self.process_models_parallel(
                models_dir=models_dir,
                week_model_dict=week_model_dict,
                feature_names=feature_names,
                show_jitter=True,
                metric="quantile",
                period_label=period_label
            )

    def save_report_summary(self):
        """
        Creates a text file summary aggregating the results for the overall test period and specific target times.
        The summary includes test period dates, lists of generated plots and statistics files, and their locations.
        """
        summary_file = os.path.join(self.analysis_dir, "report_summary.txt")
        with open(summary_file, "w") as f:
            f.write("=== Summary of Analyses ===\n\n")
            f.write(f"Test Period: {self.test_start_date.strftime('%Y-%m-%d %H:%M')} to {self.test_end_date.strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("Generated Plots (see 'plots' folder):\n")
            f.write("  - time_series_plot_overall.pdf, time_series_plot_specific.pdf\n")
            f.write("  - residuals_time_overall.pdf, residuals_hist_overall.pdf, residuals_time_specific.pdf, residuals_hist_specific.pdf\n")
            f.write("  - acf_residuals_overall.pdf, pacf_residuals_overall.pdf, acf_residuals_specific.pdf, pacf_residuals_specific.pdf\n")
            f.write("  - coverage_plot_overall.pdf, coverage_plot_specific.pdf\n")
            f.write("  - qq_plot_residuals_overall.pdf, qq_plot_residuals_specific.pdf\n")
            f.write("  - pit_histogram_overall.pdf, pit_histogram_specific.pdf\n")
            f.write("  - overlaid_predictions_overall.pdf, overlaid_predictions_specific.pdf\n")
            f.write("  - calibration_curve_overall.pdf, calibration_curve_specific.pdf\n\n")
            if self.config['model'] == 'lstm':
                f.write("  - lstm_training_loss_overall.pdf\n")
            f.write("Generated Statistics (see 'stats' folder):\n")
            f.write("Overall:\n")
            f.write("  - ljung_box_test_overall.csv\n")
            f.write("  - coverage_results_overall.csv\n")
            f.write("  - shapiro_test_overall.csv\n")
            f.write("  - pinball_loss_overall.csv\n")
            f.write("  - d2_pinball_loss_overall.csv\n")
            f.write("  - pit_values_overall.csv\n")
            f.write("  - mae_mape_overall.csv\n")
            f.write("  - outlier_report_overall.csv\n")
            f.write("Specific Times (Fri/Sat at 11:00, 15:00, 19:00):\n")
            f.write("  - coverage_results_specific.csv\n")
            f.write("  - pinball_loss_specific.csv\n")
            f.write("  - mae_mape_specific.csv\n")
            f.write("\n---------------------------------------\n")
            f.write("Files are located in the 'stats' and 'plots' subdirectories.\n")
            f.write("For details, see the individual CSV and PDF files.\n\n")
            f.write("End of Report.\n")
        logger.info("Summary report saved: %s", summary_file)


        # read all csv in stats except PIT and write overall report in report summmary etx extend it
        summary_file = os.path.join(self.analysis_dir, "report_summary.txt")
        with open(summary_file, "a") as f:
            f.write("\n=== Additional CSV Info ===\n")
            stat_files = glob.glob(os.path.join(self.stats_dir, "*.csv"))
            for csv_file in stat_files:
                if 'pit' not in csv_file:
                    f.write(f"\n--- {os.path.basename(csv_file)} ---\n")
                    with open(csv_file, 'r') as cf:
                        for line in cf:
                            f.write(line)

    def plot_time_series(self, predictions: pd.DataFrame, period_label: str):
        """
        Plot the time series comparing actual values ('gesamt') to the median forecast ('gesamt_pred_0.5'),
        along with a 95% prediction interval if available.

        Args:
            predictions (pd.DataFrame): DataFrame containing actual and forecasted values.
            period_label (str): Label for the period (e.g., "overall" or "specific").
        """
        df = predictions.copy()
        if "date_time_utc" in df.columns:
            df.set_index("date_time_utc", inplace=True)
        if "gesamt" not in df.columns:
            logger.warning("Column 'gesamt' missing in predictions – time series plot may be incomplete.")
            return
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["gesamt"], label="Actual", color=ACTUAL_COLOR)
        median_col = "gesamt_pred_0.5"
        if median_col in df.columns:
            plt.plot(df.index, df[median_col], label="Forecast (Median)", color=FORECAST_COLOR)
        q_low, q_high = "gesamt_pred_0.025", "gesamt_pred_0.975"
        if q_low in df.columns and q_high in df.columns:
            plt.fill_between(df.index, df[q_low], df[q_high], color=INTERVAL_COLOR, alpha=0.2, label="95% Prediction Interval")
        plt.title(f"Time Series Plot: Actual vs. Forecast - {period_label}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        out_file = os.path.join(self.plots_dir, f"time_series_plot_{period_label}.pdf")
        plt.savefig(out_file, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info("Time series plot saved: %s", out_file)

    def plot_residuals_and_histogram(self, predictions: pd.DataFrame, period_label: str):
        """
        Plot residuals (forecast minus actual) as a time series and as a histogram.

        Args:
            predictions (pd.DataFrame): DataFrame containing forecasts and actual values.
            period_label (str): Label for the period.
        """
        df = predictions.copy()
        if "date_time_utc" in df.columns:
            df.set_index("date_time_utc", inplace=True)
        if "gesamt_pred_0.5" not in df.columns or "gesamt" not in df.columns:
            logger.warning("Columns 'gesamt_pred_0.5' or 'gesamt' missing – cannot plot residuals.")
            return
        df["residuals"] = df["gesamt_pred_0.5"] - df["gesamt"]
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df["residuals"], label="Residuals", color=RESIDUAL_COLOR)
        plt.title(f"Residuals Over Time (Forecast - Actual) - {period_label}")
        plt.xlabel("Time")
        plt.ylabel("Residual")
        plt.grid(True)
        out_line = os.path.join(self.plots_dir, f"residuals_time_{period_label}.pdf")
        plt.savefig(out_line, format="pdf", bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.hist(df["residuals"], bins=40, color=RESIDUAL_COLOR, alpha=0.7)
        plt.title(f"Histogram of Residuals - {period_label}")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.grid(True)
        out_hist = os.path.join(self.plots_dir, f"residuals_hist_{period_label}.pdf")
        plt.savefig(out_hist, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info("Residual time series and histogram saved: %s and %s", out_line, out_hist)

    def plot_residual_scatter(self, predictions: pd.DataFrame, period_label: str):
        """
        Plot a scatter plot of median forecast values against residuals.

        Args:
            predictions (pd.DataFrame): DataFrame containing actual and median forecast.
            period_label (str): Label for the period.
        """
        if "gesamt" not in predictions.columns or "gesamt_pred_0.5" not in predictions.columns:
            logger.warning("Required columns ('gesamt', 'gesamt_pred_0.5') missing.")
            return
        preds = predictions.copy()
        preds["residuals"] = preds["gesamt_pred_0.5"] - preds["gesamt"]
        plt.figure(figsize=(8, 5))
        plt.scatter(preds["gesamt_pred_0.5"], preds["residuals"], alpha=0.5, color=RESIDUAL_COLOR)
        plt.xlabel("Forecast (Median)")
        plt.ylabel("Residuals (Forecast - Actual)")
        plt.title(f"Residual Scatter Plot - {period_label}")
        plt.axhline(0, color="red", linestyle="--")  # Belassen als Warnlinie
        filename = f"residual_scatter_{period_label}.pdf"
        out_file = os.path.join(self.plots_dir, filename)
        plt.savefig(out_file, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info("Residual scatter plot saved: %s", out_file)

    def acf_pacf_and_ljungbox(self, predictions: pd.DataFrame, period_label: str):
        """
        Plot ACF and PACF of forecast residuals and perform the Ljung–Box test.

        Args:
            predictions (pd.DataFrame): DataFrame containing forecast and actual values.
            period_label (str): Label for the period.
        """
        df = predictions.copy()
        if "date_time_utc" in df.columns:
            df.set_index("date_time_utc", inplace=True)
        if "gesamt_pred_0.5" not in df.columns or "gesamt" not in df.columns:
            logger.warning("Columns 'gesamt_pred_0.5' or 'gesamt' missing – cannot plot ACF/PACF.")
            return
        df["residuals"] = df["gesamt_pred_0.5"] - df["gesamt"]
        residuals = df["residuals"].dropna()
        n = len(residuals)
        max_lag = min(48, n // 2 - 1)
        if max_lag < 1:
            logger.warning("Not enough data points to plot ACF/PACF.")
            return

        fig_acf, ax_acf = plt.subplots(figsize=(8, 4))
        plot_acf(residuals, lags=max_lag, ax=ax_acf)
        ax_acf.set_title(f"ACF of Residuals (up to Lag {max_lag})")
        out_acf = os.path.join(self.plots_dir, f"acf_residuals_{period_label}.pdf")
        fig_acf.savefig(out_acf, format="pdf", bbox_inches="tight")
        plt.close(fig_acf)

        fig_pacf, ax_pacf = plt.subplots(figsize=(8, 4))
        plot_pacf(residuals, lags=max_lag, method="yw", ax=ax_pacf)
        ax_pacf.set_title(f"PACF of Residuals (up to Lag {max_lag})")
        out_pacf = os.path.join(self.plots_dir, f"pacf_residuals_{period_label}.pdf")
        fig_pacf.savefig(out_pacf, format="pdf", bbox_inches="tight")
        plt.close(fig_pacf)
        logger.info("ACF and PACF plots saved: %s and %s", out_acf, out_pacf)

        try:
            lb_test = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
            lb_test_file = os.path.join(self.stats_dir, f"ljung_box_test_{period_label}.csv")
            lb_test.to_csv(lb_test_file, index=False)
        except Exception as e:
            logger.warning("Ljung–Box test failed (possibly too few data points): %s", e)
            return
        logger.info("Ljung–Box test results saved: %s", lb_test_file)

    def coverage_plot(self, predictions: pd.DataFrame, lower_q=0.025, upper_q=0.975, period_label: str = ""):
        """
        Create a coverage plot comparing nominal and empirical forecast interval coverages.

        Args:
            predictions (pd.DataFrame): DataFrame containing actual and forecast quantiles.
            lower_q (float): Lower quantile (default 0.025).
            upper_q (float): Upper quantile (default 0.975).
            period_label (str): Label for the period.
        """
        df = predictions.copy()
        if "date_time_utc" in df.columns:
            df.set_index("date_time_utc", inplace=True)
        low_col = f"gesamt_pred_{lower_q}"
        high_col = f"gesamt_pred_{upper_q}"
        if "gesamt" not in df.columns or low_col not in df.columns or high_col not in df.columns:
            logger.warning("Required columns for coverage plot are missing.")
            return
        in_band = (df["gesamt"] >= df[low_col]) & (df["gesamt"] <= df[high_col])
        coverage_rate = np.mean(in_band)
        nominal = upper_q - lower_q
        logger.info("Coverage for [%s, %s] = %.3f (nominal=%.3f)", low_col, high_col, coverage_rate, nominal)
        coverage_df = pd.DataFrame({
            "lower_q": [lower_q],
            "upper_q": [upper_q],
            "nominal_coverage": [nominal],
            "empirical_coverage": [coverage_rate]
        })
        filename = f"coverage_results_{period_label}.csv" if period_label else "coverage_results.csv"
        cov_file = os.path.join(self.stats_dir, filename)
        coverage_df.to_csv(cov_file, index=False)

        plt.figure(figsize=(5, 4))
        plt.bar(["Nominal", "Empirical"], [nominal, coverage_rate],
                color=["gray", "blue"], alpha=0.7)
        plt.ylim(0, 1)
        plt.title(f"Coverage Plot: Nominal vs. Empirical - {period_label}")
        plt.xlabel("Coverage Type")
        plt.ylabel("Coverage Rate")
        plt.grid(True, axis="y")
        plot_filename = f"coverage_plot_{period_label}.pdf" if period_label else "coverage_plot.pdf"
        out_cov = os.path.join(self.plots_dir, plot_filename)
        plt.savefig(out_cov, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info("Coverage plot saved: %s", out_cov)

    def shapiro_test(self, predictions: pd.DataFrame, period_label: str):
        """
        Perform the Shapiro–Wilk test for normality on the forecast residuals.

        Args:
            predictions (pd.DataFrame): DataFrame containing actual and forecast values.
            period_label (str): Label for the period.
        """
        df = predictions.copy()
        if "date_time_utc" in df.columns:
            df.set_index("date_time_utc", inplace=True)
        if "gesamt_pred_0.5" not in df.columns or "gesamt" not in df.columns:
            logger.warning("Columns 'gesamt_pred_0.5' or 'gesamt' missing – Shapiro test cannot be performed.")
            return
        df["residuals"] = df["gesamt_pred_0.5"] - df["gesamt"]
        residuals = df["residuals"].dropna()
        w_stat, p_val = shapiro(residuals)
        logger.info("Shapiro–Wilk test: W=%.4f, p-value=%.4g", w_stat, p_val)
        results = pd.DataFrame({
            "Test": ["Shapiro–Wilk"],
            "W_stat": [w_stat],
            "p_value": [p_val]
        })
        out_file = os.path.join(self.stats_dir, f"shapiro_test_{period_label}.csv")
        results.to_csv(out_file, index=False)
        logger.info("Shapiro–Wilk test results saved: %s", out_file)

    def qq_plot_residuals(self, predictions: pd.DataFrame, period_label: str):
        """
        Generate a Q–Q plot of the forecast residuals against a normal distribution.

        Args:
            predictions (pd.DataFrame): DataFrame with actual and forecast values.
            period_label (str): Label for the period.
        """
        df = predictions.copy()
        if "date_time_utc" in df.columns:
            df.set_index("date_time_utc", inplace=True)
        if "gesamt_pred_0.5" not in df.columns or "gesamt" not in df.columns:
            logger.warning("Columns 'gesamt_pred_0.5' or 'gesamt' missing – Q–Q plot cannot be generated.")
            return
        df["residuals"] = df["gesamt_pred_0.5"] - df["gesamt"]
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        probplot(df["residuals"].dropna(), dist="norm", plot=ax)
        ax.set_title(f"Q–Q Plot of Residuals (Against Normal Distribution) - {period_label}")
        qq_file = os.path.join(self.plots_dir, f"qq_plot_residuals_{period_label}.pdf")
        plt.savefig(qq_file, format="pdf", bbox_inches="tight")
        plt.close(fig)
        logger.info("Q–Q plot saved: %s", qq_file)

    def calc_pinball_losses(self, predictions: pd.DataFrame, period_label: str):
        """
        Calculate the pinball loss (quantile loss) for each forecast quantile and compute a relative metric (D2 Pinball Loss)
        by comparing the forecast to a baseline forecast.

        Args:
            predictions (pd.DataFrame): DataFrame containing forecasted and actual values.
            period_label (str): Label for the period.
        """
        df = predictions.copy()
        if "date_time_utc" in df.columns:
            df.set_index("date_time_utc", inplace=True)
        if "gesamt" not in df.columns:
            logger.warning("Column 'gesamt' missing – pinball loss cannot be calculated.")
            return
        results = []
        results_d2 = []
        y_true = df["gesamt"].dropna()
        df_base = self.load_baseline_pre()


        # cut y_true and df_base to the same index
        y_true = y_true.loc[y_true.index.intersection(df_base.index)]
        df_base = df_base.loc[df_base.index.intersection(y_true.index)]




        for q in self.quantiles:
            qcol = f"gesamt_pred_{q}"
            if qcol not in df.columns:
                logger.warning("Column %s missing – skipping pinball loss for quantile %s.", qcol, q)
                continue
            valid_idx = y_true.index.intersection(df[qcol].dropna().index)
            y_t = y_true.loc[valid_idx]
            y_p = df.loc[valid_idx, qcol]
            y_b = df_base.loc[valid_idx, qcol]
            loss = mean_pinball_loss(y_t, y_p, alpha=q)
            loss_base = mean_pinball_loss(y_t, y_b, alpha=q)
            results.append({"Quantile": q, "PinballLoss": loss, "Period": period_label})
            results_d2.append({"Quantile": q, "D2PinballLoss": 1 - (loss / loss_base) if loss_base else np.nan, "Period": period_label})
        df_pinball = pd.DataFrame(results)
        df_pinball_d2 = pd.DataFrame(results_d2)
        out_file_pbl = os.path.join(self.stats_dir, f"pinball_loss_{period_label}.csv")
        out_file_pbl_d2 = os.path.join(self.stats_dir, f"d2_pinball_loss_{period_label}.csv")
        df_pinball.to_csv(out_file_pbl, index=False)
        df_pinball_d2.to_csv(out_file_pbl_d2, index=False)
        logger.info("Pinball loss results saved for %s: %s and %s", period_label, out_file_pbl, out_file_pbl_d2)

    def pit_histogram(self, predictions: pd.DataFrame, period_label: str):
        """
        Compute Probability Integral Transform (PIT) values from forecast quantiles and plot a histogram.

        Since only 5 quantiles are available, a default of 5 bins is used to reflect the discrete nature
        of the predictive distribution.

        Args:
            predictions (pd.DataFrame): DataFrame with forecast and actual values.
            period_label (str): Label for the period.
            num_bins (int, optional): Number of bins to use in the histogram (default is 5).

        Returns:
            None
        """
        df = predictions.copy()
        if "date_time_utc" in df.columns:
            df.set_index("date_time_utc", inplace=True)
        if "gesamt" not in df.columns:
            logger.warning("Column 'gesamt' missing – PIT histogram cannot be generated.")
            return
        available_qcols = []
        for q in self.quantiles:
            qcol = f"gesamt_pred_{q}"
            if qcol in df.columns:
                available_qcols.append(q)
        if len(available_qcols) < 2:
            logger.warning("Not enough quantile columns for PIT histogram (at least 2 required).")
            return
        available_qcols = sorted(available_qcols)
        pit_values = []
        for idx, row in df.iterrows():
            y_true = row["gesamt"]
            quant_vals = []
            for q in available_qcols:
                quant_vals.append((q, row[f"gesamt_pred_{q}"]))
            quant_vals = sorted(quant_vals, key=lambda x: x[0])
            q0_level, q0_val = quant_vals[0]
            qn_level, qn_val = quant_vals[-1]
            if pd.isna(q0_val) or pd.isna(qn_val):
                continue
            if y_true <= q0_val:
                pit = 0.0
            elif y_true >= qn_val:
                pit = 1.0
            else:
                pit = None
                for i in range(len(quant_vals) - 1):
                    ql, vl = quant_vals[i]
                    qh, vh = quant_vals[i + 1]
                    if pd.isna(vl) or pd.isna(vh):
                        continue
                    v_min, v_max = min(vl, vh), max(vl, vh)
                    if (y_true >= v_min) and (y_true <= v_max):
                        if abs(vh - vl) < 1e-9:
                            pit = 0.5 * (ql + qh)
                        else:
                            alpha = (y_true - vl) / (vh - vl)
                            pit = ql + alpha * (qh - ql)
                        break
                if pit is None:
                    continue
            pit_values.append(pit)
        if len(pit_values) < 10:
            logger.warning("Only %d PIT values computed – PIT histogram may not be representative.", len(pit_values))
            return
        plt.figure(figsize=(6, 4))
        plt.hist(pit_values, bins=20, range=(0, 1), color="green", alpha=0.7)
        plt.title(f"PIT Histogram - {period_label}")
        plt.xlabel("PIT Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        out_pit = os.path.join(self.plots_dir, f"pit_histogram_{period_label}.pdf")
        plt.savefig(out_pit, format="pdf", bbox_inches="tight")
        plt.close()

        df_pit = pd.DataFrame({"pit": pit_values})
        out_pit_csv = os.path.join(self.stats_dir, f"pit_values_{period_label}.csv")
        df_pit.to_csv(out_pit_csv, index=False)
        logger.info("PIT histogram saved: %s, PIT values saved: %s", out_pit, out_pit_csv)

    def calc_mae_mape(self, predictions: pd.DataFrame, period_label: str):
        """
        Calculate MAE and MAPE for the median forecast (0.5 quantile) and save results as CSV.

        Args:
            predictions (pd.DataFrame): DataFrame containing forecasts and actual values.
            period_label (str): Label for the period.
        """
        df = predictions.copy()
        if "date_time_utc" in df.columns:
            df.set_index("date_time_utc", inplace=True)
        if "gesamt_pred_0.5" not in df.columns or "gesamt" not in df.columns:
            logger.warning("Columns 'gesamt_pred_0.5' or 'gesamt' missing – MAE/MAPE cannot be calculated.")
            return
        y_true = df["gesamt"].dropna()
        y_pred = df["gesamt_pred_0.5"].dropna()
        valid_idx = y_true.index.intersection(y_pred.index)
        y_true = y_true.loc[valid_idx]
        y_pred = y_pred.loc[valid_idx]
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        results_df = pd.DataFrame({"Metric": ["MAE", "MAPE"], "Value": [mae, mape], "Period": period_label})
        out_file = os.path.join(self.stats_dir, f"mae_mape_{period_label}.csv")
        results_df.to_csv(out_file, index=False)
        logger.info("MAE and MAPE for %s saved: %s", period_label, out_file)

    def plot_overlaid_predictions(self, predictions: pd.DataFrame, period_label: str):
        """
        Create an overlaid plot of forecasts for Thursday, Friday, and Saturday.

        The plot shows the hourly mean forecast and the 95% prediction interval.
        
        Args:
            predictions (pd.DataFrame): DataFrame with forecasts.
            period_label (str): Label for the period.
        """
        df = predictions.copy()
        if "date_time_utc" in df.columns:
            df.set_index("date_time_utc", inplace=True)
        if "gesamt_pred_0.5" not in df.columns:
            logger.warning("Column 'gesamt_pred_0.5' missing – overlaid plot cannot be generated.")
            return
        df["hour"] = df.index.hour
        df = df[df.index.weekday.isin([3, 4, 5])]  # Thursday, Friday, Saturday
        try:
            pivot_df = df.pivot_table(index="hour", columns=df.index.weekday, values="gesamt_pred_0.5")
        except pd.errors.DataError:
            logger.error("Error creating pivot table for overlaid plot.")
            return
        pivot_df["mean"] = pivot_df.mean(axis=1)
        for q in self.quantiles:
            pivot_df[f"q_{q}"] = pivot_df.quantile(q, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(pivot_df.index, pivot_df["mean"], label="Mean Forecast", color="blue")
        plt.fill_between(pivot_df.index, pivot_df["q_0.025"], pivot_df["q_0.975"],
                         color="blue", alpha=0.2, label="95% Prediction Interval")
        plt.xlabel("Hour of Day")
        plt.ylabel("Forecast (Median)")
        plt.title(f"Overlaid Forecasts for Thursday, Friday, and Saturday - {period_label}")
        plt.xticks(range(24))
        plt.grid(True)
        plt.legend()
        out_file = os.path.join(self.plots_dir, f"overlaid_predictions_{period_label}.pdf")
        plt.savefig(out_file, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info("Overlaid forecast plot saved: %s", out_file)

    def outlier_report(self, predictions: pd.DataFrame, period_label: str, threshold_factor=3.0):
        """
        Identify outliers in the residuals (forecast median minus actual) using a threshold of
        (threshold_factor * standard deviation) and save the report as a CSV file.

        Args:
            predictions (pd.DataFrame): DataFrame with actual and forecast data.
            period_label (str): Label for the period.
            threshold_factor (float, optional): Multiplier for standard deviation to define outliers (default 3.0).
        """
        df = predictions.copy()
        if "date_time_utc" in df.columns:
            df.set_index("date_time_utc", inplace=True)
        if "gesamt_pred_0.5" not in df.columns or "gesamt" not in df.columns:
            logger.warning("Columns 'gesamt_pred_0.5' or 'gesamt' missing – outlier report cannot be generated.")
            return
        df["residuals"] = df["gesamt_pred_0.5"] - df["gesamt"]
        residuals = df["residuals"].dropna()
        std_dev = residuals.std()
        threshold = threshold_factor * std_dev
        outliers = df[np.abs(df["residuals"]) > threshold]
        outlier_df = outliers[["gesamt", "gesamt_pred_0.5", "residuals"]].copy()
        outlier_df.sort_index(inplace=True)
        out_file = os.path.join(self.stats_dir, f"outlier_report_{period_label}.csv")
        outlier_df.to_csv(out_file)
        logger.info("Outlier report saved: %s", out_file)
        if len(outliers) == 0:
            logger.info("No outliers found.")
        else:
            logger.info("Number of outliers found: %d", len(outliers))


    def plot_calibration_diagnostics(self, predictions: pd.DataFrame, period_label: str, num_bins=5):
        """
        Plot calibration diagnostics: PIT Histogram and PIT eCDF.

        Args:
            predictions (pd.DataFrame): DataFrame containing actual values ('gesamt')
                                        and forecast quantiles (e.g., 'q_0.01', 'q_0.5', 'q_0.99').
            period_label (str): Label for the period (e.g., 'validation', 'test').
            num_bins (int, optional): Number of bins for the PIT histogram (default is 5).
        """
        df = predictions.copy()
        if "date_time_utc" in df.columns:
            df.set_index("date_time_utc", inplace=True)
        if "gesamt" not in df.columns:
            logger.warning("Column 'gesamt' missing – calibration plots cannot be generated.")
            return

        pit_values = self._calculate_pit_values(df)
        if pit_values is None or len(pit_values) < 10:
            logger.warning("Not enough PIT values for calibration plots.")
            return

        # --- 1. PIT Histogram ---
        bins = np.linspace(0, 1, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        observed_freqs = []
        for i in range(num_bins):
            # Ensure the last bin includes 1.0 if linspace creates it
            if i == num_bins - 1:
                 in_bin = (pit_values >= bins[i]) & (pit_values <= bins[i + 1])
            else:
                 in_bin = (pit_values >= bins[i]) & (pit_values < bins[i + 1])
            # Calculate frequency (density) within the bin
            observed_freqs.append(np.mean(in_bin) if pit_values.size > 0 else 0) # Use np.mean for frequency

        ideal_freq = 1.0 / num_bins

        plt.figure(figsize=(7, 6)) # Slightly wider figure
        plt.bar(bin_centers, observed_freqs, width=1.0/num_bins * 0.9, color="blue", label="Observed Frequency", alpha=0.7)
        # Plot ideal frequency as a horizontal line
        plt.axhline(ideal_freq, linestyle="--", color="gray", label=f"Ideal Uniform ({ideal_freq:.2f})")
        plt.xlabel("PIT Value Bins")
        plt.ylabel("Frequency (Density)")
        plt.title(f"PIT Histogram - {period_label}")
        plt.xticks(bin_centers, [f"{b:.1f}-{bins[i+1]:.1f}" for i, b in enumerate(bins[:-1])], rotation=45, ha='right') # Nicer bin labels
        plt.xlim(0, 1)
        plt.ylim(bottom=0) # Start y-axis at 0
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7) # Grid lines for y-axis
        plt.tight_layout() # Adjust layout

        out_file_hist = os.path.join(self.plots_dir, f"pit_histogram_{period_label}.pdf")
        try:
            plt.savefig(out_file_hist, format="pdf", bbox_inches="tight")
            logger.info("PIT Histogram saved: %s", out_file_hist)
        except Exception as e:
            logger.error("Failed to save PIT Histogram: %s", e)
        plt.close() # Close the figure

        # --- 2. PIT eCDF Plot ---
        sorted_pit = np.sort(pit_values)
        # Calculate y-values for eCDF: (1/n, 2/n, ..., n/n)
        y_ecdf = np.arange(1, len(sorted_pit) + 1) / len(sorted_pit)

        plt.figure(figsize=(6, 6))
        plt.plot(sorted_pit, y_ecdf, marker=".", linestyle="-", color="blue", label="Observed eCDF")
        # Plot the diagonal line for perfect calibration
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated (Uniform)")
        plt.xlabel("PIT Value")
        plt.ylabel("Cumulative Probability (eCDF)")
        plt.title(f"PIT eCDF - {period_label}")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout() # Adjust layout

        out_file_ecdf = os.path.join(self.plots_dir, f"pit_ecdf_{period_label}.pdf")
        try:
            plt.savefig(out_file_ecdf, format="pdf", bbox_inches="tight")
            logger.info("PIT eCDF plot saved: %s", out_file_ecdf)
        except Exception as e:
            logger.error("Failed to save PIT eCDF plot: %s", e)
        plt.close() # Close the figure






    def _calculate_pit_values(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Calculate the Probability Integral Transform (PIT) values from forecast quantiles.

        Args:
            df (pd.DataFrame): DataFrame containing actual ('gesamt') and forecast quantile columns.
        
        Returns:
            Optional[np.ndarray]: Array of PIT values if at least two quantile columns exist, otherwise None.
        """
        available_qcols = []
        for q in self.quantiles:
            qcol = f"gesamt_pred_{q}"
            if qcol in df.columns:
                available_qcols.append(q)
        if len(available_qcols) < 2:
            return None
        available_qcols = sorted(available_qcols)
        pit_values = []
        for idx, row in df.iterrows():
            y_true = row["gesamt"]
            quant_vals = []
            for q in available_qcols:
                quant_vals.append((q, row[f"gesamt_pred_{q}"]))
            quant_vals = sorted(quant_vals, key=lambda x: x[0])
            q0_level, q0_val = quant_vals[0]
            qn_level, qn_val = quant_vals[-1]
            if pd.isna(q0_val) or pd.isna(qn_val):
                continue
            if y_true <= q0_val:
                pit = 0.0
            elif y_true >= qn_val:
                pit = 1.0
            else:
                pit = None
                for i in range(len(quant_vals) - 1):
                    ql, vl = quant_vals[i]
                    qh, vh = quant_vals[i + 1]
                    if pd.isna(vl) or pd.isna(vh):
                        continue
                    v_min, v_max = min(vl, vh), max(vl, vh)
                    if (y_true >= v_min) and (y_true <= v_max):
                        if abs(vh - vl) < 1e-9:
                            pit = 0.5 * (ql + qh)
                        else:
                            alpha = (y_true - vl) / (vh - vl)
                            pit = ql + alpha * (qh - ql)
                        break
                if pit is None:
                    continue
            pit_values.append(pit)
        return np.array(pit_values)
    



    def calculate_sharpness(self, predictions: pd.DataFrame, period_label: str, lower_q=0.025, upper_q=0.975) -> Optional[float]:
        """
        Compute forecast sharpness defined as the average width of the prediction interval.

        Args:
            predictions (pd.DataFrame): DataFrame with forecast quantiles.
            period_label (str): Label for the period.
            lower_q (float): Lower quantile (default 0.025).
            upper_q (float): Upper quantile (default 0.975).

        Returns:
            Optional[float]: Average interval width if the necessary columns exist, otherwise None.
        """
        df = predictions.copy()
        if "date_time_utc" in df.columns:
            df.set_index("date_time_utc", inplace=True)
        low_col = f"gesamt_pred_{lower_q}"
        high_col = f"gesamt_pred_{upper_q}"
        if low_col not in df.columns or high_col not in df.columns:
            logger.warning("Columns for sharpness calculation missing – cannot compute sharpness.")
            return None
        interval_widths = df[high_col] - df[low_col]
        sharpness = interval_widths.mean()
        logger.info("Average sharpness for %s (Interval width %.3f): %.3f", period_label, upper_q - lower_q, sharpness)
        out_file = os.path.join(self.stats_dir, f"sharpness_{period_label}.csv")
     
        # sharpness to df
        sharpness = pd.DataFrame({"sharpness": [sharpness], "period": [period_label]})

        sharpness.to_csv(out_file)
        return sharpness

    def plot_single_forecast(self, predictions: pd.DataFrame, datetime_index: pd.Timestamp, period_label: str):
        """
        Plot the forecast for a single timestamp by displaying the forecast quantile values and overlaying the actual observed value.

        Args:
            predictions (pd.DataFrame): DataFrame containing forecasted and actual values.
            datetime_index (pd.Timestamp): The timestamp for which to plot the forecast.
            period_label (str): Label for the period.

        Returns:
            None
        """
        df = predictions.copy()
        if "date_time_utc" in df.columns:
            df.set_index("date_time_utc", inplace=True)
        if "gesamt" not in df.columns:
            logger.warning("Column 'gesamt' missing – single forecast cannot be plotted.")
            return
        if datetime_index not in df.index:
            logger.warning("Timestamp %s not found in index – single forecast cannot be plotted.", datetime_index)
            return
        row = df.loc[datetime_index]
        available_quantiles = []
        quantile_values = []
        for q in self.quantiles:
            qcol = f"gesamt_pred_{q}"
            if qcol in row and not pd.isna(row[qcol]):
                available_quantiles.append(q)
                quantile_values.append(row[qcol])
        if not available_quantiles:
            logger.warning("No quantile forecasts available for the selected timestamp.")
            return

        plt.figure(figsize=(8, 5))
        plt.plot(available_quantiles, quantile_values, marker="o", linestyle="-", color="blue")
        plt.axhline(row["gesamt"], color=ACTUAL_COLOR, linestyle="--", label="Actual Value")
        plt.xlabel("Quantile")
        plt.ylabel("Forecast Value")
        plt.title(f"Forecast for {datetime_index}")
        plt.grid(True)
        plt.legend()
        out_file = os.path.join(self.plots_dir, f"single_forecast_{datetime_index.strftime('%Y%m%d_%H%M')}_{period_label}.pdf")
        plt.savefig(out_file, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info("Single forecast plot saved: %s", out_file)

    def _filter_by_specific_times(self, predictions: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Filter predictions to include only specific target times: Friday and Saturday at 11:00, 15:00, and 19:00.

        Args:
            predictions (pd.DataFrame): DataFrame containing forecasts.

        Returns:
            Optional[pd.DataFrame]: Filtered DataFrame or None if no matching times are found.
        """
        df = predictions.copy()
        if "date_time_utc" in df.columns:
            df.set_index("date_time_utc", inplace=True)
        target_times = []
        for day in [4, 5]:  # Friday, Saturday
            for hour in [11, 15, 19]:
                target_times.append((day, hour))
        def is_target_time(ts):
            return (ts.weekday(), ts.hour) in target_times
        mask = df.index.map(is_target_time)
        filtered_df = df[mask]
        if len(filtered_df) == 0:
            logger.warning("No data found for specific times (Fri/Sat at 11:00, 15:00, 19:00).")
            return None
        return filtered_df

    def load_model_training_info(self, model_path: str) -> dict:
        """
        Load training information from a saved model file.
        Expects the file to contain a dictionary with a 'training_info' key.

        Args:
            model_path (str): Path to the model file.

        Returns:
            dict: Training information dictionary; empty if not found.
        """
        try:
            data = joblib.load(model_path)
            if isinstance(data, dict) and "training_info" in data:
                return data["training_info"]
            else:
                logger.warning("No 'training_info' dictionary found in file: %s", model_path)
                return {}
        except Exception as e:
            logger.error("Error loading model %s: %s", model_path, e)
            return {}

    def report_model_training_info(self, model_paths: list):
        """
        Aggregate training information from multiple models and save a summary as a CSV.

        Expected metrics include quantile, model path, training data shape, training duration,
        best iteration, best score, train MAE, train RMSE, and validation MAE.

        Args:
            model_paths (list): List of model file paths.
        """
        summaries = []
        for path in model_paths:
            info = self.load_model_training_info(path)
            if info:
                summary = {
                    "Quantile": info.get("quantile"),
                    "Model_Path": path,
                    "Training_Data_Shape": info.get("training_data_shape"),
                    "Training_Duration_sec": info.get("training_duration_sec"),
                    "Best_Iteration": info.get("best_iteration"),
                    "Best_Score": info.get("best_score"),
                    "Train_MAE": info.get("train_metrics", {}).get("mae"),
                    "Train_RMSE": info.get("train_metrics", {}).get("rmse"),
                    "Validation_MAE": info.get("validation_metrics", {}).get("mae") if info.get("validation_metrics") else None,
                }
                summaries.append(summary)
        if summaries:
            df_summary = pd.DataFrame(summaries)
            out_file = os.path.join(self.stats_dir, "model_training_summary.csv")
            df_summary.to_csv(out_file, index=False)
            logger.info("Aggregated model training info saved: %s", out_file)
        else:
            logger.warning("No training information found to generate summary.")

    def plot_training_loss_curve(self, model_path: str, metric: str = "quantile", period_label: str = "", root_dir: str = None):
        """
        Plots the training and/or validation loss curve based on 'evals_result' from the model's training information.
        Iterates over all datasets contained in 'evals_result'.

        Args:
            model_path (str): Path to the model file.
            metric (str, optional): The metric to plot (default is "quantile").
            period_label (str, optional): A label for the period.
            root_dir (str, optional): Root directory for saving plots.
        """
        info = self.load_model_training_info(model_path)
        if "evals_result" not in info:
            logger.warning("No evals_result found in %s - Loss curve cannot be generated.", model_path)
            return
        evals = info["evals_result"]
        plt.figure(figsize=(8, 5))
        plotted = False
        for dataset_name, metrics_dict in evals.items():
            if metric in metrics_dict:
                loss_vals = metrics_dict[metric]
                plt.plot(loss_vals, label=f"{dataset_name} ({metric})")
                plotted = True
            else:
                logger.debug("Dataset '%s' does not contain metric '%s'.", dataset_name, metric)
        if not plotted:
            logger.warning("No curve found for metric '%s' in evals_result.", metric)
            plt.close()
            return
        plt.xlabel("Boosting round")
        plt.ylabel(metric)
        plt.title(f"Loss curve for model {os.path.basename(model_path)}")
        plt.legend()
        root_dir = root_dir.replace("models", "plots")
        os.makedirs(os.path.join(self.plots_dir, root_dir), exist_ok=True)
        filename = f"training_loss_{os.path.basename(model_path).replace('.joblib', '')}_{period_label}.pdf"
        out_file = os.path.join(self.plots_dir, root_dir, filename)
        plt.savefig(out_file, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info("Training loss curve saved: %s", out_file)

    def plot_feature_importance_from_model(self, model_path: str, root_dir: str = None):
        """
        Plot the feature importances as stored in the model's training information.

        Args:
            model_path (str): Path to the model file.
            root_dir (str, optional): Root directory for saving plots.
        """
        info = self.load_model_training_info(model_path)
        if "feature_importances" not in info:
            logger.warning("No feature importances found in %s.", model_path)
            return
        importances = info["feature_importances"]
        feature_names = info.get("feature_names", [f"Feature {i}" for i in range(len(importances))])
        sorted_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
        sorted_importances = [importances[i] for i in sorted_idx]
        sorted_feature_names = [feature_names[i] for i in sorted_idx]
        plt.figure(figsize=(10, 0.4 * len(sorted_feature_names) + 2))
        y_pos = range(len(sorted_feature_names))
        plt.barh(y_pos, sorted_importances, align="center", color="green", alpha=0.7)
        plt.yticks(y_pos, sorted_feature_names)
        plt.xlabel("Feature Importance")
        plt.title(f"Feature Importances for Model {os.path.basename(model_path)}")
        plt.gca().invert_yaxis()
        root_dir = root_dir.replace("models", "plots")
        os.makedirs(os.path.join(self.plots_dir, root_dir), exist_ok=True)
        filename = f"feature_importance_{os.path.basename(model_path).replace('.joblib', '')}.pdf"
        out_file = os.path.join(self.plots_dir, root_dir, filename)
        plt.savefig(out_file, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info("Feature importance plot saved: %s", out_file)

    def plot_shap_summary(self, model_path: str, root_dir: str = None):
        """
        Generate a SHAP summary plot for the given model.

        Args:
            model_path (str): Path to the model file.
            root_dir (str, optional): Root directory for saving plots.
        """
        X_sample = pd.read_parquet(os.path.join(root_dir, "X_test.parquet"))
        try:
            data = joblib.load(model_path)
            model = data["model"]
        except Exception as e:
            logger.error("Error loading model %s: %s", model_path, e)
            return
        explainer = shap.TreeExplainer(model.booster_)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, show=False)
        root_dir = root_dir.replace("models", "plots")
        os.makedirs(os.path.join(self.plots_dir, root_dir), exist_ok=True)
        filename = f"shap_summary_{os.path.basename(model_path).replace('.joblib', '')}.pdf"
        out_file = os.path.join(self.plots_dir, root_dir, filename)
        plt.savefig(out_file, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary plot saved: %s", out_file)

    def plot_feature_trend(self, week_model_dict: dict, feature_names: list = None, shared_y_axis: bool = False, root_dir: str = None):
        """
        Plot aggregated feature importances over different folds/weeks.

        Args:
            week_model_dict (dict): Dictionary mapping fold/week identifiers to lists of model paths.
            feature_names (list, optional): List of feature names.
            shared_y_axis (bool, optional): Whether to share y-axis limits across plots.
            root_dir (str, optional): Root directory for saving plots.
        """
        weeks = sorted(week_model_dict.keys())
        aggregated = {}
        for week in weeks:
            imp_list = []
            for model_path in week_model_dict[week]:
                info = self.load_model_training_info(model_path)
                if info.get("feature_importances") is not None:
                    imp_list.append(info["feature_importances"])
            if imp_list:
                aggregated[week] = np.array(imp_list)
        if not aggregated:
            logger.warning("No feature importances found for trend plot.")
            return

        plt.figure(figsize=(14, 8))
        if shared_y_axis:
            all_means = np.concatenate([aggregated[w].mean(axis=0) for w in aggregated])
            all_stds = np.concatenate([aggregated[w].std(axis=0) for w in aggregated])
            y_min = np.min(all_means - all_stds)
            y_max = np.max(all_means + all_stds)
            plt.ylim(y_min * 1.1, y_max * 1.1)

        for idx in range(len(feature_names)):
            means = [aggregated[w].mean(axis=0)[idx] for w in sorted(aggregated.keys())]
            stds = [aggregated[w].std(axis=0)[idx] for w in sorted(aggregated.keys())]
            weeks_sorted = sorted(aggregated.keys())
            plt.errorbar(weeks_sorted, means, yerr=stds, marker="o", capsize=5, label=feature_names[idx])
        plt.xlabel("Fold/Week")
        plt.ylabel("Feature Importance")
        plt.title("Aggregated Feature Importance Trend over Folds/Weeks")
        plt.legend()
        plt.grid(True)
        root_dir = root_dir.replace("models", "plots")
        os.makedirs(os.path.join(self.plots_dir, root_dir), exist_ok=True)
        filename = "feature_importance_trend.pdf"
        out_file = os.path.join(self.plots_dir, root_dir, filename)
        plt.savefig(out_file, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info("Feature importance trend plot saved: %s", out_file)

    def plot_all_feature_boxplots(self, week_model_dict: dict, feature_names: list, show_jitter: bool = False, root_dir: str = None):
        """
        Create boxplots of feature importances for all features over different folds/weeks.
        Each feature is plotted in its own subplot.

        Args:
            week_model_dict (dict): Dictionary mapping fold/week identifiers to lists of model paths.
            feature_names (list): List of feature names.
            show_jitter (bool, optional): Whether to overlay jittered data points.
            root_dir (str, optional): Root directory for saving plots.
        """
        weeks = sorted(week_model_dict.keys())
        num_features = len(feature_names)
        fig, axs = plt.subplots(nrows=num_features, figsize=(12, 4 * num_features), sharex=True)
        if num_features == 1:
            axs = [axs]
        for idx, feature_name in enumerate(feature_names):
            data = []
            labels = []
            for week in weeks:
                values = []
                for model_path in week_model_dict[week]:
                    info = self.load_model_training_info(model_path)
                    if info.get("feature_importances") is not None:
                        values.append(info["feature_importances"][idx])
                if values:
                    data.append(values)
                    labels.append(week)
            ax = axs[idx]
            if data:
                bp = ax.boxplot(data, patch_artist=True, showfliers=True,
                                medianprops={'color': 'black', 'linewidth': 2},
                                flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'red'})
                overall_median = np.median(np.concatenate(data))
                ax.axhline(overall_median, color='blue', linestyle='--', linewidth=1, 
                           label=f'Overall Median: {overall_median:.3f}')
                if show_jitter:
                    for i, values in enumerate(data):
                        x = np.random.normal(i + 1, 0.04, size=len(values))
                        ax.plot(x, values, 'r.', alpha=0.3)
                ax.set_ylabel("Feature Importance")
                ax.set_title(f"Boxplots of Feature Importances for '{feature_name}'")
                ax.legend()
                ax.grid(True, axis='y')
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, rotation=45)
            else:
                ax.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center')
        plt.xlabel("Fold/Week")
        plt.tight_layout()
        root_dir = root_dir.replace("models", "plots")
        os.makedirs(os.path.join(self.plots_dir, root_dir), exist_ok=True)
        filename = "feature_boxplots_all.pdf"
        out_file = os.path.join(self.plots_dir, root_dir, filename)
        plt.savefig(out_file, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info("Boxplots for all features saved: %s", out_file)

    def create_week_model_dict(self, models_dir: str) -> Dict[str, List[str]]:
        """
        Create a dictionary mapping fold/week identifiers to lists of model file paths.

        Args:
            models_dir (str): Directory containing model files.

        Returns:
            dict: Dictionary with keys as fold/week identifiers and values as lists of model file paths.
        """
        week_model_dict: Dict[str, List[str]] = {}
        for root, _, files in os.walk(models_dir):
            for file in files:
                if file.endswith(".joblib"):
                    model_path = os.path.join(root, file)
                    week_or_fold = os.path.basename(root)
                    if week_or_fold not in week_model_dict:
                        week_model_dict[week_or_fold] = []
                    week_model_dict[week_or_fold].append(model_path)
        return week_model_dict

    def process_model(self, model_path, root, metric, period_label, week_model_dict, feature_names, show_jitter):
        """
        Process a single model: plot training loss, feature importance, SHAP summary, and (if applicable) feature trends and boxplots.

        Args:
            model_path (str): Path to the model file.
            root (str): Root directory of the model.
            metric (str): Metric name.
            period_label (str): Label for the period.
            week_model_dict (dict): Dictionary mapping weeks/folds to model paths.
            feature_names (list): List of feature names.
            show_jitter (bool): Whether to overlay jitter on boxplots.
        """
        self.plot_training_loss_curve(model_path, metric=metric, period_label=period_label, root_dir=root)
        self.plot_feature_importance_from_model(model_path, root_dir=root)
        if self.config['model'] == 'light_gbm':
            self.plot_shap_summary(model_path, root_dir=root)
        if root[-6:] != 'models':
            if week_model_dict is not None and feature_names is not None and root[-2:]== '_1' and model_path[-10:-7] == '0.5' and not self.marker:
                self.marker = True
                self.plot_feature_trend(week_model_dict, feature_names, shared_y_axis=True, root_dir=root)
            if week_model_dict is not None and root[-2:]== '_2' and model_path[-10:-7] == '0.5':
                # self.plot_all_feature_boxplots(week_model_dict, feature_names=feature_names, show_jitter=show_jitter, root_dir=root)
                print("nothing to see here")
            

    def process_models_parallel(self, models_dir, metric, period_label, week_model_dict, feature_names, show_jitter=False):
        """
        Process multiple models in parallel.

        Args:
            models_dir (str): Directory containing model files.
            metric (str): Metric name.
            period_label (str): Label for the period.
            week_model_dict (dict): Dictionary mapping weeks/folds to model paths.
            feature_names (list): List of feature names.
            show_jitter (bool, optional): Whether to overlay jitter on boxplots.
        """
        jobs = []
        for root, _, files in os.walk(models_dir):
            for file in files:
                if file.endswith(".joblib"):
                    model_path = os.path.join(root, file)
                    jobs.append(delayed(self.process_model)(
                        model_path, root, metric, period_label, week_model_dict, feature_names, show_jitter
                    ))
        if len(os.listdir(self.config['hyperparameter_dir'])) > 0:
            for root, _, files in os.walk(self.config['hyperparameter_dir']):
                for file in files:
                    if file.endswith(".joblib"):
                        models_path = os.path.join(root, file)
                        jobs.append(delayed(self.process_model)(
                            models_path, root, metric, period_label, week_model_dict, feature_names, show_jitter
                        ))
        Parallel(n_jobs=-1)(jobs)

    def plot_predictions_lstm(self, model_file: str, period_label: str = ""):
        """
        Generate forecast plots from a 'predictions.parquet' file in the same directory as the LSTM model file.
        Plots include:
          1) Time series comparing actual vs. median forecast (with 95% CI if available),
          2) Scatter plot of actual vs. predicted,
          3) Residual time series plot,
          4) Histogram of residuals.
        
        Args:
            model_file (str): Path to the LSTM model file.
            period_label (str, optional): Label for the period.
        """
        model_dir = os.path.dirname(model_file)
        parquet_file = os.path.join(model_dir, "prediction.parquet")
        if os.path.exists(parquet_file):
            try:
                preds = pd.read_parquet(parquet_file)
            except Exception as e:
                logger.error("Error loading %s: %s", parquet_file, e)
                return
        else:
            logger.warning("No predictions file found in %s.", model_dir)
            return
        if 'gesamt' not in preds.columns or 'gesamt_pred' not in preds.columns:
            logger.warning("Required columns ('gesamt' and 'gesamt_pred') missing in predictions.")
            return

        # 1) Time Series Plot
        plt.figure(figsize=(12, 6))
        if isinstance(preds.index, pd.DatetimeIndex):
            plt.plot(preds.index, preds["gesamt"], label="Actual", color=ACTUAL_COLOR)
            plt.plot(preds.index, preds["gesamt_pred"], label="Forecast (Median)", color=FORECAST_COLOR)
            if "gesamt_pred_0.25" in preds.columns and "gesamt_pred_0.975" in preds.columns:
                plt.fill_between(preds.index, preds["gesamt_pred_0.25"], preds["gesamt_pred_0.975"],
                                 color=FORECAST_COLOR, alpha=0.2, label="95% CI")
            plt.xlabel("Time")
        else:
            plt.plot(preds["gesamt"], label="Actual", color=ACTUAL_COLOR)
            plt.plot(preds["gesamt_pred"], label="Forecast (Median)", color=FORECAST_COLOR)
            if "gesamt_pred_0.25" in preds.columns and "gesamt_pred_0.975" in preds.columns:
                plt.fill_between(preds.index, preds["gesamt_pred_0.25"], preds["gesamt_pred_0.975"],
                                 color=FORECAST_COLOR, alpha=0.2, label="95% CI")
            plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title(f"Time Series: Actual vs. Forecast - {period_label}")
        plt.legend()
        plt.grid(True)
        out_file = os.path.join(model_dir, f"predictions_plot_{period_label}.pdf")
        plt.savefig(out_file, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info("Forecast time series plot saved: %s", out_file)
        # 2) Scatter Plot
        plt.figure(figsize=(8, 8))
        plt.scatter(preds["gesamt"], preds["gesamt_pred"], alpha=0.5, color=RESIDUAL_COLOR)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Scatterplot of Predictions vs. Actual - {period_label}")
        min_val = min(preds["gesamt"].min(), preds["gesamt_pred"].min())
        max_val = max(preds["gesamt"].max(), preds["gesamt_pred"].max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        plt.grid(True)
        out_file_scatter = os.path.join(model_dir, f"scatter_plot_{period_label}.pdf")
        plt.savefig(out_file_scatter, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info("Scatter plot saved: %s", out_file_scatter)

        # 3) Residual Time Series Plot
        residuals = preds["gesamt"] - preds["gesamt_pred"]
        plt.figure(figsize=(12, 6))
        if isinstance(preds.index, pd.DatetimeIndex):
            plt.plot(preds.index, residuals, label="Residuals", color=RESIDUAL_COLOR)
            plt.xlabel("Time")
        else:
            plt.plot(residuals, label="Residuals", color=RESIDUAL_COLOR)
            plt.xlabel("Index")
        plt.ylabel("Error")
        plt.title(f"Residuals Over Time - {period_label}")
        plt.grid(True)
        out_file_residuals = os.path.join(model_dir, f"residuals_{period_label}.pdf")
        plt.savefig(out_file_residuals, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info("Residual time series plot saved: %s", out_file_residuals)

        # 4) Residual Histogram
        plt.figure(figsize=(12, 6))
        plt.hist(residuals, bins=20, color=RESIDUAL_COLOR, alpha=0.7)
        plt.title(f"Histogram of Residuals - {period_label}")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.grid(True)
        out_file_hist = os.path.join(model_dir, f"residual_hist_{period_label}.pdf")
        plt.savefig(out_file_hist, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info("Residual histogram saved: %s", out_file_hist)

    def plot_loss_curves_lstm(self, model_file: str, period_label: str = ""):
        """
        Load training and validation loss data (if available) from the same directory as the model file,
        and plot the training and validation loss curves. The plot is then saved in that directory.

        Args:
            model_file (str): Path to the LSTM model file.
            period_label (str, optional): Label for the period to include in the plot title and filename.
        """
        model_dir = os.path.dirname(model_file)
        model = torch.load(model_file)

        train_loss = model['training_info']['evals_result']['train']
        val_loss = model['training_info']['evals_result']['valid']

        train_loss_df = pd.DataFrame(train_loss, index=range(len(train_loss)), columns=['train_loss'])
        val_loss_df = pd.DataFrame(val_loss, index=range(len(val_loss)), columns=['val_loss'])
        train_loss_df.index.name = 'epochs'
        val_loss_df.index.name = 'epochs'

        # For example, we use the first column as loss values for both train and validation
        train_col = train_loss_df.columns[0]
        val_col = val_loss_df.columns[0]

        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_df.index, train_loss_df[train_col], label="Train Loss", color=FORECAST_COLOR)
        plt.plot(val_loss_df.index, val_loss_df[val_col], label="Validation Loss", color="blue")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss - {period_label}")
        plt.legend()
        plt.grid(True)

        out_file = os.path.join(model_dir, f"loss_curves_{period_label}.pdf")
        plt.savefig(out_file, format="pdf", bbox_inches="tight")
        plt.close()
        logger.info("Loss curve saved: %s", out_file)

    def diebold_mariano_test(self, actual: pd.Series, forecast1: pd.Series, forecast2: pd.Series,
                            loss_func: callable = None, alternative: str = "two-sided", period_label: str = "") -> Dict[str, Any]:
        """
        Performs the Diebold–Mariano (DM) test to compare the predictive accuracy of two forecasting models.

        The DM test evaluates whether the difference in the chosen loss (forecast error) between two models is statistically significant.

        Args:
            actual (pd.Series): The actual observed values.
            forecast1 (pd.Series): Forecast values from the first model.
            forecast2 (pd.Series): Forecast values from the second model.
            loss_func (callable, optional): A function to compute the loss for a forecast error.
                If None, defaults to mean absolute error.
            alternative (str): Specifies the alternative hypothesis ('two-sided', 'less', or 'greater').
            period_label (str): Label for the period (e.g., "overall" or "specific").

        Returns:
            dict: A dictionary containing the DM test statistic and p-value.
        """
        from scipy.stats import t
        if loss_func is None:
            loss_func = lambda a, f: np.abs(a - f)

        # Compute the loss differences
        errors1 = loss_func(actual, forecast1)
        errors2 = loss_func(actual, forecast2)
        d = errors1 - errors2
        d_mean = np.mean(d)
        d_var = np.var(d, ddof=1)
        dm_stat = d_mean / np.sqrt(d_var / len(d))
        
        # Two-sided p-value based on the t-distribution with (n-1) degrees of freedom
        p_value = 2 * (1 - t.cdf(np.abs(dm_stat), df=len(d) - 1))

        self.report_lines.append(f"Diebold–Mariano test: DM Statistic = {dm_stat:.4f}, p-value = {p_value:.4f}.")
        logger.info("Diebold–Mariano test completed: DM Statistic=%.4f, p=%.4f", dm_stat, p_value)

        results = {"DM Statistic": dm_stat, "p-value": p_value}

        # Save the test results as a CSV in the stats directory
        results_df = pd.DataFrame([results])
        out_file = os.path.join(self.stats_dir, f"diebold_mariano_test_{period_label}.csv")
        results_df.to_csv(out_file, index=False)
        logger.info("Diebold–Mariano test results saved: %s", out_file)

        return results

    def mincer_zarnowitz_regression(self, actual: pd.Series, forecast: pd.Series, period_label) -> Dict[str, Any]:
        """
        Performs a Mincer–Zarnowitz regression to evaluate forecast accuracy.

        In a Mincer–Zarnowitz regression, actual values are regressed on the forecasts (plus a constant).
        Ideal forecasts yield a regression with an intercept of 0 and a slope of 1. Deviations indicate bias or inefficiency.

        Args:
            actual (pd.Series): The actual observed values.
            forecast (pd.Series): The forecast values.
            period_label (str): Label for the period (e.g., "overall" or "specific").

        Returns:
            dict: A dictionary containing regression parameters (constant and slope), R², and p-values.
        """
        df = pd.concat([actual, forecast], axis=1).dropna()
        df.columns = ["actual", "forecast"]
        X = sm.add_constant(df["forecast"])
        y = df["actual"]

        model = sm.OLS(y, X).fit()

        summary_text = model.summary().as_text()
        self.report_lines.append("Mincer–Zarnowitz Regression Summary:\n" + summary_text)
        logger.info("Mincer–Zarnowitz regression completed.")

        # Prepare results including p-values for all coefficients
        try:
            results = {
                "Intercept": model.params[0],
                "Slope": model.params[1],
                "R-squared": model.rsquared
            }
            for coef_name, p_val in model.pvalues.items():
                results[f"p-value_{coef_name}"] = p_val

            # Save the regression results as a CSV in the stats directory
            results_df = pd.DataFrame([results])
            out_file = os.path.join(self.stats_dir, f"mincer_zarnowitz_regression_{period_label}.csv")
            results_df.to_csv(out_file, index=False)
            logger.info("Mincer–Zarnowitz regression results saved: %s", out_file)

            return results
        except Exception as e:
            logger.error("Error processing Mincer–Zarnowitz regression: %s", e)
            return {}
        

    def residual_density_plot(self, predictions: pd.DataFrame, period_label: str):
        """
        Plots the density (kernel density estimate) of the forecast residuals.

        This plot visualizes the distribution of forecast errors (residuals) and can help assess
        whether the residuals are normally distributed or if they exhibit skewness/kurtosis.

        Args:
            predictions (pd.DataFrame): DataFrame containing actual ('gesamt') and forecast ('gesamt_pred_0.5') values.
            period_label (str): Label for the period (e.g., "overall" or "specific").

        Outputs:
            - Saves the density plot as "residual_density_{period_label}.pdf" in the plots directory.
            - Updates the report lines.
        """
        df = predictions.copy()
        if "date_time_utc" in df.columns:
            df.set_index("date_time_utc", inplace=True)

        if "gesamt" not in df.columns or "gesamt_pred_0.5" not in df.columns:
            logger.warning("Required columns missing – cannot generate residual density plot.")
            return

        df["residuals"] = df["gesamt_pred_0.5"] - df["gesamt"]

        plt.figure(figsize=(10, 6))
        sns.kdeplot(df["residuals"].dropna(), shade=True, color=RESIDUAL_COLOR)
        plt.title(f"Density Plot of Forecast Residuals - {period_label}")
        plt.xlabel("Residual")
        plt.ylabel("Density")
        out_file = os.path.join(self.plots_dir, f"residual_density_{period_label}.pdf")
        plt.savefig(out_file, format="pdf", bbox_inches="tight")
        plt.close()
        self.report_lines.append(f"Residual density plot saved: {out_file}.")
        logger.info("Residual density plot saved: %s", out_file)
       
    def cusum_forecast_errors(self, predictions: pd.DataFrame, period_label: str) -> Dict[str, Any]:
        """
        Plots the cumulative sum (CUSUM) of forecast errors.

        The CUSUM plot accumulates the forecast errors over time. Large deviations from zero may indicate periods
        where forecasts systematically over- or under-predict.

        Args:
            predictions (pd.DataFrame): DataFrame with actual and forecast values.
            period_label (str): Label for the period.

        Returns:
            dict: Contains the final CUSUM value and optionally some measure of deviation.

        Outputs:
            - Saves the CUSUM plot as "cusum_forecast_errors_{period_label}.pdf" in the plots directory.
            - Updates the report lines.
        """
        df = predictions.copy()
        if "date_time_utc" in df.columns:
            df.set_index("date_time_utc", inplace=True)

        if "gesamt" not in df.columns or "gesamt_pred_0.5" not in df.columns:
            logger.warning("Required columns missing – cannot generate CUSUM plot for forecast errors.")
            return {}

        df["forecast_error"] = df["gesamt_pred_0.5"] - df["gesamt"]
        df["CUSUM"] = df["forecast_error"].cumsum()

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["CUSUM"], label="CUSUM of Forecast Errors", color="blue")
        plt.axhline(0, color="red", linestyle="--")
        plt.title(f"CUSUM of Forecast Errors - {period_label}")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Sum")
        plt.legend()
        out_file = os.path.join(self.plots_dir, f"cusum_forecast_errors_{period_label}.pdf")
        plt.savefig(out_file, format="pdf", bbox_inches="tight")
        plt.close()

        final_cusum = df["CUSUM"].iloc[-1]
        self.report_lines.append(f"CUSUM plot for forecast errors saved: {out_file}. Final CUSUM: {final_cusum:.4f}.")
        logger.info("CUSUM plot saved: %s, Final CUSUM: %.4f", out_file, final_cusum)
        return {"Final CUSUM": final_cusum}
