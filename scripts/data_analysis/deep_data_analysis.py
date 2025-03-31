import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, ccf, grangercausalitytests, bds
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_goldfeldquandt, het_arch
from scipy.stats import shapiro, normaltest, jarque_bera, anderson
from sklearn.metrics import mean_pinball_loss, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import joblib
import torch
from datetime import datetime, timedelta
import statsmodels.api as sm
import pdb

# Global style settings for professional PDF output
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "savefig.dpi": 300,            # High resolution
    "figure.autolayout": True,      # Ensure proper layout
    "lines.linewidth": 2,
    "axes.grid": True,
})

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class FullTimeSeriesEDA:
    """
    FullTimeSeriesEDA conducts a comprehensive exploratory data analysis (EDA) on time series data
    intended for quantile regression modeling. It runs a battery of tests and generates visualizations
    in order to assess key properties of the series including stationarity, autocorrelation, seasonality,
    exogenous relationships, nonlinearity, outlier detection, distribution shape, heteroskedasticity,
    multicollinearity, structural breaks, cointegration, missing values, and feature importance.
    
    Each method is documented with detailed explanations on:
      - What the test or visualization is intended to show.
      - What inputs the method expects.
      - What outputs are returned or saved.
      - How to interpret the results.
    
    After running all tests, a summary report is written to an output file.
    """

    def __init__(self, data: pd.DataFrame, target: str, exog: Optional[List[str]] = None,
                 output_dir: str = "full_eda_report",target_name: str = 'Test_please fill', window: int = 24):
        """
        Initializes the FullTimeSeriesEDA instance.
        
        Args:
            data (pd.DataFrame): The complete time series dataset. It is assumed that the index contains time stamps.
            target (str): The name of the target variable (column) in the dataset to be analyzed.
            exog (List[str], optional): A list of names of exogenous variables (columns) that may have predictive power.
            output_dir (str): Directory where output plots, CSV files, and the final report will be saved.
            window (int): Rolling window size (in number of observations) for calculating rolling mean and standard deviation.
        
        Effects:
            - Makes a deep copy of the input data.
            - Ensures that the data index is a DatetimeIndex and sorts the data chronologically.
            - Creates necessary output directories (plots, stats, report).
            - Initializes an empty list to collect report lines.
        """
        self.data = data.copy()
        self.target = target
        self.target_name = target_name   
        self.exog = exog if exog is not None else []
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        self.stats_dir = os.path.join(self.output_dir, "stats")
        os.makedirs(self.stats_dir, exist_ok=True)
        self.report_file = os.path.join(self.output_dir, "eda_report.txt")
        self.window = window
        self.report_lines = []
        
        # Ensure the index is a DatetimeIndex and sort data
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        self.data.sort_index(inplace=True)

    # =============================================================================
    # I. STATIONARITY ANALYSIS
    # =============================================================================
    def adf_test(self, series: pd.Series, diff: int = 0) -> Dict[str, Any]:
        """
        Performs the Augmented Dickey–Fuller (ADF) test on a time series.
        
        The ADF test is used to check whether a time series is stationary. A low p-value (typically less than 0.05)
        indicates that the null hypothesis of a unit root can be rejected, suggesting that the series is stationary.
        Differencing may be applied to the series prior to testing if the original series is non-stationary.
        
        Args:
            series (pd.Series): The time series on which to perform the ADF test.
            diff (int): The degree of differencing applied to the series before testing (0 means the original series).
        
        Returns:
            dict: A dictionary containing:
                - "ADF Statistic": The computed test statistic.
                - "p-value": The p-value for the test.
        
        Interpretation:
            - A p-value less than 0.05 typically indicates stationarity.
            - If the test is applied after differencing, it helps determine the minimum degree of differencing required.
        """
        s = series.dropna()
        for _ in range(diff):
            s = s.diff().dropna()
        result = adfuller(s)
  
        res_dict = {"ADF Statistic": result[0], "p-value": result[1]}
        self.report_lines.append(f"ADF test (d={diff}): p-value = {result[1]:.2e}.")
        logger.info("ADF test (d=%d) completed: Statistic=%.8f, p=%.8f", diff, result[0], result[1])
        return res_dict

    def kpss_test(self, series: pd.Series, diff: int = 0) -> Dict[str, Any]:
        """
        Performs the KPSS test on a time series.
        
        The KPSS (Kwiatkowski–Phillips–Schmidt–Shin) test checks for stationarity by testing the null hypothesis that
        the series is stationary. A high p-value (typically greater than 0.05) supports stationarity, whereas a low p-value
        suggests the presence of a unit root.
        
        Args:
            series (pd.Series): The time series on which to perform the KPSS test.
            diff (int): The degree of differencing applied to the series before testing.
        
        Returns:
            dict: A dictionary containing:
                - "KPSS Statistic": The test statistic.
                - "p-value": The p-value for the test.
        
        Interpretation:
            - A p-value greater than 0.05 indicates that the series is likely stationary.
            - If differencing is applied, it can help determine the order of integration.
        """
        s = series.dropna()
        for _ in range(diff):
            s = s.diff().dropna()
        result = kpss(s, regression='c', nlags="auto")
        res_dict = {"KPSS Statistic": result[0], "p-value": result[1]}
        self.report_lines.append(f"KPSS test (d={diff}): p-value = {result[1]:.4f}.")
        logger.info("KPSS test (d=%d) completed: Statistic=%.4f, p=%.4f", diff, result[0], result[1])
        return res_dict

    def rolling_statistics(self):
        """
        Computes and plots the rolling mean and standard deviation for the target variable.
        
        This plot helps determine whether the time series has a constant mean and variance (i.e., stationarity).
        
        Outcome:
            - The original series is plotted along with its rolling mean and rolling standard deviation.
            - A stable rolling mean and variance suggest stationarity; significant changes may indicate nonstationarity.
        
        Inputs:
            - Uses the target variable from the class data.
            - Uses the rolling window size specified during initialization.
        
        Outputs:
            - Saves a plot named "rolling_statistics.tex" in the plots directory.
            - Updates the report lines with the output file path.
        """
        roll_mean = self.data[self.target].rolling(window=self.window).mean()
        roll_std = self.data[self.target].rolling(window=self.window).std()
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data[self.target], label="Original")
        plt.plot(self.data.index, roll_mean, label=f"Rolling Mean ({self.window})")
        plt.plot(self.data.index, roll_std, label=f"Rolling Std ({self.window})")
        plt.title("Rolling Mean & Standard Deviation")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        out_file_pdf = os.path.join(self.plots_dir, "rolling_statistics.pdf")
        plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
        plt.close()
        self.report_lines.append(f"Rolling statistics plot saved: {out_file_pdf}.")
        logger.info("Rolling statistics plot saved: %s", out_file_pdf)

    # =============================================================================
    # II. AUTOCORRELATION ANALYSIS
    # =============================================================================
    def acf_pacf_plots(self, lags: int = 100):
        """
        Generates Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots for the target variable.
        
        These plots are used to identify the presence of serial correlation at various lags in the time series.
        
        Args:
            lags (int): The maximum number of lags to include in the plots.
        
        Outcome:
            - The ACF plot shows the correlation between the series and its lagged values.
            - The PACF plot shows the correlation of the series with its lagged values after removing the effects of shorter lags.
            - Significant spikes outside the confidence bands suggest correlation at those lags.
        
        Outputs:
            - Saves a combined plot ("acf_pacf.tex") in the plots directory.
            - Updates the report lines with the output file path.
        """
        series = self.data[self.target].dropna()
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))
        plot_acf(series, lags=lags, ax=ax[0])
        ax[0].set_title(f"ACF of {self.target_name}")
        plot_pacf(series, lags=lags, ax=ax[1], method="ywm")
        ax[1].set_title(f"PACF of {self.target_name}")
        plt.tight_layout()
        out_file_pdf = os.path.join(self.plots_dir, "acf_pacf.pdf")
        plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
        plt.close()
        self.report_lines.append(f"ACF and PACF plots saved (up to {lags} lags): {out_file_pdf}.")
        logger.info("ACF/PACF plots saved: %s", out_file_pdf)

    def ljung_box_test(self, lags: List[int] = [10, 20]) -> pd.DataFrame:
        """
        Performs the Ljung–Box test on the target variable to test for overall randomness in the series.
        
        The Ljung–Box test checks whether any group of autocorrelations of the series is different from zero.
        
        Args:
            lags (List[int]): A list of lag values at which to test for autocorrelation.
        
        Returns:
            pd.DataFrame: A DataFrame containing the test statistic and p-values for each specified lag.
        
        Interpretation:
            - A p-value higher than 0.05 suggests that the null hypothesis of no autocorrelation cannot be rejected,
              indicating that the residuals (or series) are likely white noise.
        
        Side Effects:
            - Saves the test results as "ljung_box_test.csv" in the stats directory.
            - Updates the report lines.
        """
        series = self.data[self.target].dropna()
        lb_result = acorr_ljungbox(series, lags=lags, return_df=True)
        self.report_lines.append(f"Ljung–Box test results for lags {lags}:\n{lb_result.to_string()}")
        out_file = os.path.join(self.stats_dir, "ljung_box_test.csv")
        lb_result.to_csv(out_file, index=False)
        logger.info("Ljung–Box test results saved: %s", out_file)
        return lb_result

    def autocorrelation_heatmap(self, max_lag: int = 50):
        """
        Creates a heatmap of the autocorrelation matrix computed from the target variable's autocorrelations.
        
        This visualization helps to see how the strength of autocorrelation changes across different lags.
        
        Args:
            max_lag (int): The maximum lag for which autocorrelation is computed.
        
        Outcome:
            - A heatmap where each cell's intensity reflects the product of autocorrelations at two lags.
            - Darker or lighter areas can indicate recurring patterns.
        
        Outputs:
            - Saves the heatmap as "autocorrelation_heatmap.tex" in the plots directory.
            - Updates the report lines.
        """
        series = self.data[self.target].dropna()
        acf_vals = acf(series, nlags=max_lag)
        acf_matrix = np.outer(acf_vals, acf_vals)  # Simple approximation
        plt.figure(figsize=(10, 8))
        sns.heatmap(acf_matrix, annot=False, cmap="coolwarm")
        plt.title(f"Autocorrelation Heatmap (up to lag {max_lag})")
        out_file_pdf = os.path.join(self.plots_dir, "autocorrelation_heatmap.pdf")
        plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
        plt.close()
        self.report_lines.append(f"Autocorrelation heatmap saved: {out_file_pdf}.")
        logger.info("Autocorrelation heatmap saved: %s", out_file_pdf)

    # =============================================================================
    # III. SEASONALITY ANALYSIS
    # =============================================================================
    def seasonal_decomposition(self, period: int = 24):
        """
        Performs classical seasonal decomposition on the target variable using an additive model.
        with custom line width, marker size, and a larger figure.
        """
        series = self.data[self.target].dropna()
        try:
            decomp = seasonal_decompose(series, model='additive', period=period)
            fig = decomp.plot()
            
            # Größe der gesamten Figur anpassen (Breite, Höhe in Zoll)
            fig.set_size_inches(14, 8)  
            
            # Optional: Auflösung erhöhen (z. B. 200 dpi)
            # fig.set_dpi(200)

            # Alle Achsen durchgehen und Linienstile anpassen
            # Hier z.B. dünnere Linien und ggf. Marker für den Residualplot
            for ax in fig.axes:
                for line in ax.lines:
                    # Dünnere Linien
                    line.set_linewidth(0.8)
                    # Falls du Marker möchtest (z.B. im Residual-Plot):
                    # line.set_marker('.')
                    # line.set_markersize(2)
            
            # Wenn du ausschließlich den Residual-Plot abweichend stylen möchtest:
            # (Die Residuen liegen üblicherweise in fig.axes[3], da 0=Observed, 1=Trend, 2=Seasonal, 3=Residual)
            residual_ax = fig.axes[3]
            for line in residual_ax.lines:
                # z.B. sehr dünne Linie und kleine Punkte
                line.set_linewidth(0.6)
                line.set_marker('.')
                line.set_markersize(2)

            # Titel größer setzen
            plt.suptitle(f"Seasonal Decomposition of {self.target_name}", fontsize=16)

            # Abspeichern als PDF (oder ein anderes Format)
            out_file_pdf = os.path.join(self.plots_dir, "seasonal_decomposition.pdf")
            plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
            plt.close()

        except Exception as e:
            logger.warning("Seasonal decomposition failed: %s", e)
            out_file_pdf = "N/A"

        self.report_lines.append(f"Classical seasonal decomposition completed. Plot saved as: {out_file_pdf}.")
        logger.info("Seasonal decomposition plot saved: %s", out_file_pdf)


    def save_seasonal_decomposition(self, period: int = 24):
        """
        Führt eine klassische saisonale Zerlegung der Zielvariablen mittels eines additiven Modells durch.
        
        Diese Methode zerlegt die Zeitreihe in drei Komponenten: Trend, Saisonalität und Residuen.
        
        Args:
            period (int): Die saisonale Periode (z. B. 24 für stündliche Daten mit täglichem Zyklus).
        
        Outcome:
            - Ein Plot, der die zerlegten Komponenten zeigt.
            - Eine CSV-Datei, die die Residuen speichert.
        
        Outputs:
            - Speichert einen Plot namens "seasonal_decomposition.tex" im plots-Verzeichnis.
            - Speichert eine CSV-Datei "seasonal_decomposition_residuals.csv" mit den Residuen.
            - Aktualisiert die Reportzeilen.
        """
        series = self.data[self.target].dropna()
        try:
            decomp = seasonal_decompose(series, model='additive', period=period)
            fig = decomp.plot()
            plt.suptitle(f"Seasonal Decomposition of {self.target_name}")
            out_file_pdf = os.path.join(self.plots_dir, "seasonal_decomposition.pdf")
            plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
            plt.close()
            residuals = decomp.resid.dropna()
            resid_file = os.path.join(self.plots_dir, "seasonal_decomposition_residuals.csv")
            residuals.to_csv(resid_file, index=True)
        except Exception as e:
            logger.warning("Seasonal decomposition failed: %s", e)
            out_file_pdf = "N/A"
            resid_file = "N/A"
        self.report_lines.append(f"Classical seasonal decomposition completed. Plot saved as: {out_file_pdf}.")
        self.report_lines.append(f"Residuals saved as CSV: {resid_file}.")
        logger.info("Seasonal decomposition plot saved: %s", out_file_pdf)
        logger.info("Seasonal decomposition residuals saved as CSV: %s", resid_file)

    def stl_decomposition(self, period: int = 24):
        """
        Performs STL (Seasonal-Trend decomposition using Loess) on the target variable.
        
        STL provides a robust decomposition into trend, seasonal, and remainder components,
        and is more flexible than classical decomposition.
        
        Args:
            period (int): The seasonal period (e.g., 24 for hourly data with daily seasonality).
        
        Outcome:
            - A plot of the STL decomposition that can reveal subtle seasonal and trend patterns.
        
        Outputs:
            - Saves a plot named "stl_decomposition.tex" in the plots directory.
            - Updates the report lines.
        """
        series = self.data[self.target].dropna()
        try:
            stl = STL(series, period=period).fit()
            fig = stl.plot()
            plt.suptitle(f"STL Decomposition of {self.target_name}")
            out_file_pdf = os.path.join(self.plots_dir, "stl_decomposition.pdf")
            plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning("STL decomposition failed: %s", e)
            out_file_pdf = "N/A"
        self.report_lines.append(f"STL decomposition completed. Plot saved as: {out_file_pdf}.")
        logger.info("STL decomposition plot saved: %s", out_file_pdf)

    def boxplots_by_time(self):
        """
        Generates boxplots of the target variable grouped by month, weekday, and hour.
        Additionally, overlays mean curves for July and January on the hourly boxplot.
        """
        df = self.data.copy()
        df['month'] = df.index.month
        df['weekday'] = df.index.weekday
        df['hour'] = df.index.hour

        # -----------------------------
        # 1) Boxplot by month
        # -----------------------------
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='month', y=self.target, data=df)
        plt.title(f"{self.target_name} Distribution by Month")
        out_file_pdf1 = os.path.join(self.plots_dir, "boxplot_by_month.pdf")
        plt.savefig(out_file_pdf1, format="pdf", bbox_inches="tight")
        plt.close()

        # -----------------------------
        # 2) Boxplot by weekday
        # -----------------------------
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='weekday', y=self.target, data=df)
        plt.title(f"{self.target_name} Distribution by Weekday")
        out_file_pdf2 = os.path.join(self.plots_dir, "boxplot_by_weekday.pdf")
        plt.savefig(out_file_pdf2, format="pdf", bbox_inches="tight")
        plt.close()

        # -----------------------------
        # 3) Boxplot by hour + overlay mean curves for July & January
        # -----------------------------
        plt.figure(figsize=(12, 6))
        # Boxplot
        sns.boxplot(x='hour', y=self.target, data=df)
        plt.title(f"{self.target_name} Distribution by Hour")

        # DataFrames für Juli und Januar extrahieren
        df_july = df[df['month'] == 7]
        df_jan = df[df['month'] == 1]

        # Mean pro Stunde berechnen
        july_mean = df_july.groupby('hour')[self.target].mean()
        jan_mean = df_jan.groupby('hour')[self.target].mean()

        # Linien über dem Boxplot plotten
        # Hinweis: index = Stunden (0..23), values = Mean
       
        try:
            plt.plot(july_mean.index, july_mean.values, marker='o', color='red', label='July Mean')
            plt.plot(jan_mean.index, jan_mean.values, marker='o', color='green', label='January Mean')
            plt.legend()
        except Exception as e:
            logger.warning("Could not overlay mean curves: %s", e   )
        out_file_pdf3 = os.path.join(self.plots_dir, "boxplot_by_hour.pdf")
        plt.savefig(out_file_pdf3, format="pdf", bbox_inches="tight")
        plt.close()

        # Zusammenfassen der Ausgabe
        self.report_lines.append(
            f"Boxplots by month, weekday, and hour saved: {out_file_pdf1}, {out_file_pdf2}, {out_file_pdf3}."
        )
        logger.info("Boxplots by time saved: %s, %s, %s", out_file_pdf1, out_file_pdf2, out_file_pdf3)

    # =============================================================================
    # IV. EXOGENOUS ANALYSIS & CROSS-CORRELATION
    # =============================================================================
    def cross_correlation_plot(self):
        """
        Generates cross-correlation plots between the target variable and each exogenous variable.
        
        This method computes the cross-correlation function (CCF) between the target and each exogenous variable,
        which helps in identifying the lag at which the exogenous variable is most strongly related to the target.
        
        Outcome:
            - For each exogenous variable, a bar plot is created showing the cross-correlation values at different lags.
        
        Outputs:
            - Saves each plot as "cross_correlation_{target}_{exog}.tex" in the plots directory.
            - Updates the report lines.
        """
        if not self.exog:
            self.report_lines.append("No exogenous variables provided for cross-correlation analysis.")
            return
        for var in self.exog:
            if var in self.data.columns:
                xcorr = ccf(self.data[self.target].dropna(), self.data[var].dropna())
                plt.figure(figsize=(10, 4))
                plt.bar(range(len(xcorr)), xcorr, color='purple', alpha=0.7)
                plt.title(f"Cross-Correlation: {self.target_name} vs. {var}")
                plt.xlabel("Lag")
                plt.ylabel("Correlation")
                out_file_pdf = os.path.join(self.plots_dir, f"cross_correlation_{self.target_name}_{var}.pdf")
                plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
                plt.close()
                self.report_lines.append(f"Cross-correlation plot for {self.target_name} and {var} saved: {out_file_pdf}.")
                logger.info("Cross-correlation plot saved for %s vs. %s: %s", self.target_name, var, out_file_pdf)
    
    def cross_correlation_plot_report(self, max_lags=24):
        """
        Generates cross-correlation plots with an optimal lag indicator.
        
        Saves the plot as a PDF.
        """
        if not self.exog:
            self.report_lines.append("No exogenous variables for cross-correlation.")
            return

        for var in self.exog:
            if var in self.data.columns:
                series1 = self.data[self.target].dropna()
                series2 = self.data[var].dropna()
                min_len = min(len(series1), len(series2))
                series1 = series1.iloc[:min_len]
                series2 = series2.iloc[:min_len]
                xcorr = ccf(series1, series2, adjusted=False)
                xcorr = xcorr[:max_lags + 1]
                optimal_lag = np.argmax(np.abs(xcorr))
                optimal_corr = xcorr[optimal_lag]
                plt.figure(figsize=(10, 4))
                plt.bar(range(len(xcorr)), xcorr, color='purple', alpha=0.7)
                plt.title(f"Cross-Correlation: {self.target_name} vs. {var}")
                plt.xlabel("Lag")
                plt.ylabel("Correlation")
                plt.axvline(optimal_lag, color='red', linestyle='--', label=f'Optimal Lag: {optimal_lag}')
                plt.legend()
                out_file_pdf = os.path.join(self.plots_dir, f"cross_correlation_{self.target_name}_{var}.pdf")
                plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
                plt.close()
                self.report_lines.append(f"Cross-correlation: {self.target_name} vs. {var}. Optimal lag: {optimal_lag}, Correlation: {optimal_corr:.4f}")
                logger.info("Cross-correlation plot for %s vs. %s saved: %s. Optimal lag: %d, Correlation: %.4f", self.target_name, var, out_file_pdf, optimal_lag, optimal_corr)

    def granger_causality_test(self, maxlag: int = 24):
        """
        Performs the Granger causality test between the target variable and each exogenous variable.
        
        The Granger causality test determines whether one time series can forecast another. A low p-value 
        (typically less than 0.05) for a given lag suggests that the exogenous variable provides statistically 
        significant information about the future values of the target.
        
        Args:
            maxlag (int): The maximum number of lags to test.
        
        Returns:
            dict: A dictionary where each exogenous variable is mapped to its corresponding p-value from the test.
        
        Outputs:
            - Saves a CSV file "granger_causality_results.csv" with the p-values.
            - Updates the report lines.
        """
        results = {}
        for var in self.exog:
            if var in self.data.columns:
                test_data = self.data[[self.target, var]].dropna()
                try:
                    gc_result = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
                    p_val = gc_result[maxlag][0]['ssr_ftest'][1]
                    results[var] = p_val
                    self.report_lines.append(f"Granger causality test for {var} on {self.target_name}: p-value = {p_val:.4f}.")
                    logger.info("Granger causality test for %s: p=%.4f", var, p_val)
                except Exception as e:
                    logger.warning("Granger causality test for %s failed: %s", var, e)
        out_file = os.path.join(self.stats_dir, "granger_causality_results.csv")
        pd.DataFrame.from_dict(results, orient='index', columns=['p-value']).to_csv(out_file)
        self.report_lines.append(f"Granger causality test results saved: {out_file}.")
        logger.info("Granger causality results saved: %s", out_file)
        return results

    def scatterplot_lagged_exog(self, lag: int = 0):
        """
        Creates scatter plots between the target variable and a lagged version of each exogenous variable.
        
        This method shifts each exogenous variable by a specified lag and plots it against the target variable.
        This visualization can reveal delayed effects and help determine appropriate lag structures for modeling.
        
        Args:
            lag (int): The number of periods by which to shift the exogenous variable.
        
        Outputs:
            - For each exogenous variable, saves a plot as "scatter_{target}_{exog}_lag{lag}.tex" in the plots directory.
            - Updates the report lines.
        """
        for var in self.exog:
            if var in self.data.columns:
                plt.figure(figsize=(8, 6))
                plt.scatter(self.data[var].shift(lag), self.data[self.target])
                plt.xlabel(f"{var} (Lag {lag})")
                plt.ylabel(self.target)
                plt.title(f"Scatter Plot: {self.target_name} vs. {var} (Lag {lag})")
                out_file_pdf = os.path.join(self.plots_dir, f"scatter_{self.target_name}_{var}_lag{lag}.pdf")
                plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
                plt.close()
                self.report_lines.append(f"Scatter plot for {var} with lag {lag} saved: {out_file_pdf}.")
                logger.info("Scatter plot for %s (lag %d) saved: %s", var, lag, out_file_pdf)
   
   
    def scatterplot_lagged_exog_test(self, lag: int = 0, color_feature: str = None, cmap: str = "viridis"):
        """
        Creates scatter plots between the target variable and a lagged version of each exogenous variable,
        optionally coloring points based on another feature (e.g., daytime/nighttime).

        Args:
            lag (int): The number of periods by which to shift the exogenous variable.
            color_feature (str, optional): Name of the feature used to color data points. Default is None.
            cmap (str): Colormap to use for coloring.

        Outputs:
            - For each exogenous variable, saves a colored scatter plot in the plots directory.
            - Updates the report lines.
        """
        for var in self.exog:
            if var in self.data.columns:
                plt.figure(figsize=(8, 6))
                
                # Exogene Variable um "lag" verschieben und NaNs entfernen
                x = self.data[var].shift(lag)
                y = self.data[self.target]
                df_plot = pd.DataFrame({var: x, self.target: y})
                
                if color_feature:
                    # Falls color_feature nicht in self.data vorhanden ist, aber 'day_night' angefordert wurde,
                    # wird es temporär anhand des Stundenwerts des Index erzeugt.
                    if color_feature not in self.data.columns and color_feature == 'day_night':
                        df_plot[color_feature] = self.data.index.hour.map(lambda h: 1 if 6 <= h < 18 else 0)
                    elif color_feature in self.data.columns:
                        df_plot[color_feature] = self.data[color_feature]

                    scatter = plt.scatter(
                        df_plot[var],
                        df_plot[self.target],
                        c=df_plot[color_feature],
                        cmap=cmap,
                        alpha=0.7
                    )
                    plt.legend(handles=scatter.legend_elements()[0], labels=['Night', 'Day'], title=color_feature)
                else:
                    plt.scatter(df_plot[var], df_plot[self.target], alpha=0.7)
                
                plt.xlabel(f"{var} (Lag {lag})")
                plt.ylabel(self.target)
                title = f"Scatter Plot: {self.target_name} vs. {var}"
                plt.title(title)
                
                out_file_pdf = os.path.join(self.plots_dir, f"coloured_scatter_{self.target_name}_{var}_lag{lag}.pdf")
                plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
                plt.close()
                
                self.report_lines.append(f"Scatter plot for {var} with lag {lag} saved: {out_file_pdf}.")
                logger.info("Scatter plot for %s (lag %d) saved: %s", var, lag, out_file_pdf)

    # =============================================================================
    # V. NONLINEARITY
    # =============================================================================
    def bds_test(self):
        """
        Applies the BDS (Brock-Dechert-Scheinkman) test to the target series to detect nonlinearity.
        
        The BDS test is a statistical test that checks whether a time series is independent and identically distributed (iid).
        A p-value below 0.05 suggests the presence of nonlinearity and dependence that cannot be explained by a linear model.
        
        Outcome:
            - No output is returned, but the result (p-value) is appended to the report lines and logged.
        """
        series = self.data[self.target].dropna().values
        result = bds(series)
        p_val = result.pvalue if hasattr(result, 'pvalue') else np.nan
        self.report_lines.append(f"BDS test on {self.target_name}: p-value = {p_val:.4f}.")
        logger.info("BDS test completed for %s: p=%.4f", self.target_name, p_val)

    def scatter_target_vs_transformed(self):
        """
        Plots scatter plots of the target variable versus its squared and cubed values.
        
        This helps to visually assess whether there is evidence of quadratic or cubic nonlinearity in the data.
        
        Outcome:
            - Two plots are generated: one comparing the target with its square, and one with its cube.
        
        Outputs:
            - Saves plots "scatter_squared.tex" and "scatter_cubed.tex" in the plots directory.
            - Updates the report lines.
        """
        s = self.data[self.target].dropna()
        plt.figure(figsize=(8, 6))
        plt.scatter(s, s**2, alpha=0.5)
        plt.xlabel(self.target)
        plt.ylabel(f"{self.target_name} Squared")
        plt.title(f"Scatter Plot: {self.target_name} vs. {self.target_name} Squared")
        out_file1 = os.path.join(self.plots_dir, "scatter_squared.pdf")
        plt.savefig(out_file1, format="pdf", bbox_inches="tight")
        plt.close()
        
        plt.figure(figsize=(8, 6))
        plt.scatter(s, s**3, alpha=0.5)
        plt.xlabel(self.target)
        plt.ylabel(f"{self.target_name} Cubed")
        plt.title(f"Scatter Plot: {self.target_name} vs. {self.target_name} Cubed")
        out_file2 = os.path.join(self.plots_dir, "scatter_cubed.pdf")
        plt.savefig(out_file2, format="pdf", bbox_inches="tight")
        plt.close()
        
        self.report_lines.append(f"Scatter plots for squared and cubed transformations saved: {out_file1}, {out_file2}.")
        logger.info("Scatter plots for target transformations saved: %s and %s", out_file1, out_file2)

    def recurrence_plot(self):
        """
        Generates a recurrence plot of the target series to visualize the recurrence of similar states.
        
        A recurrence plot is a visual tool that displays when a state of a dynamical system recurs. In this 
        basic implementation, a distance matrix is computed and thresholded to form a binary recurrence matrix.
        
        Outcome:
            - A plot where darker (or lighter) regions indicate that similar values recur.
        
        Outputs:
            - Saves the recurrence plot as "recurrence_plot.tex" in the plots directory.
            - Updates the report lines.
        """
        s = self.data[self.target].dropna().values
        s = (s - np.min(s)) / (np.max(s) - np.min(s))
        eps = 0.1
        R = np.abs(np.subtract.outer(s, s)) < eps
        plt.figure(figsize=(8, 8))
        plt.imshow(R, cmap='binary', origin='lower')
        plt.title("Recurrence Plot")
        plt.xlabel("Time Index")
        plt.ylabel("Time Index")
        out_file_pdf = os.path.join(self.plots_dir, "recurrence_plot.pdf")
        plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
        plt.close()
        self.report_lines.append(f"Recurrence plot saved: {out_file_pdf}.")
        logger.info("Recurrence plot saved: %s", out_file_pdf)

    # =============================================================================
    # VI. OUTLIER DETECTION
    # =============================================================================
    def outlier_detection_zscore(self):
        """
        Detects outliers in the target series using the Z-score method.
        
        The Z-score method computes how many standard deviations each observation is from the mean.
        Observations with a Z-score greater than 3 are typically considered outliers.
        
        Outcome:
            - Returns a list of indices corresponding to outlier observations.
            - The count of detected outliers is appended to the report lines.
        """
        s = self.data[self.target].dropna()
        mean_val = s.mean()
        std_val = s.std()
        z_scores = np.abs((s - mean_val) / std_val)
        outliers = s[z_scores > 3]
        self.report_lines.append(f"Z-score outlier detection found {len(outliers)} outliers.")
        logger.info("Z-score outlier detection: %d outliers", len(outliers))
        return outliers.index.tolist()

    def outlier_detection_iqr(self):
        """
        Detects outliers using the Interquartile Range (IQR) method.
        
        The IQR method identifies outliers as observations falling below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
        
        Outcome:
            - Returns a list of indices for outlier observations.
            - The number of outliers detected is recorded in the report.
        """
        s = self.data[self.target].dropna()
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = s[(s < lower_bound) | (s > upper_bound)]
        self.report_lines.append(f"IQR outlier detection found {len(outliers)} outliers.")
        logger.info("IQR outlier detection: %d outliers", len(outliers))
        return outliers.index.tolist()

    def outlier_detection_lof(self, n_neighbors: int = 20):
        """
        Detects outliers using the Local Outlier Factor (LOF) method.
        
        LOF measures the local deviation of a given data point with respect to its neighbors.
        A negative prediction (-1) indicates an outlier.
        
        Args:
            n_neighbors (int): Number of neighbors to use for LOF computation.
        
        Outcome:
            - Returns a list of indices of detected outliers.
            - The number of outliers detected is appended to the report.
        """
        from sklearn.neighbors import LocalOutlierFactor
        s = self.data[[self.target]].dropna()
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        y_pred = lof.fit_predict(s)
        outlier_indices = s.index[y_pred == -1].tolist()
        self.report_lines.append(f"LOF outlier detection found {len(outlier_indices)} outliers.")
        logger.info("LOF outlier detection: %d outliers", len(outlier_indices))
        return outlier_indices

    # =============================================================================
    # VII. DISTRIBUTION ANALYSIS
    # =============================================================================
    def shapiro_test(self):
        """
        Performs the Shapiro–Wilk test for normality on the target variable.
        
        The Shapiro–Wilk test tests the null hypothesis that the data was drawn from a normal distribution.
        
        Outcome:
            - A p-value greater than 0.05 suggests that the target variable is normally distributed.
        
        Returns:
            dict: A dictionary containing the Shapiro test statistic and p-value.
        """
        s = self.data[self.target].dropna()
        stat, p = shapiro(s.sample(min(len(s), 5000)))
        self.report_lines.append(f"Shapiro–Wilk test: p-value = {p:.4f}.")
        logger.info("Shapiro–Wilk test: p=%.4f", p)
        return {"Shapiro Statistic": stat, "p-value": p}

    def dagostino_test(self):
        """
        Performs D'Agostino's K² test for normality on the target variable.
        
        Outcome:
            - A p-value less than 0.05 indicates significant deviation from normality.
        
        Returns:
            dict: A dictionary with the test statistic and p-value.
        """
        s = self.data[self.target].dropna()
        stat, p = normaltest(s)
        self.report_lines.append(f"D'Agostino test: p-value = {p:.4f}.")
        logger.info("D'Agostino test: p=%.4f", p)
        return {"D'Agostino Statistic": stat, "p-value": p}

    def jarque_bera_test(self):
        """
        Performs the Jarque–Bera test on the target variable.
        
        Outcome:
            - A p-value less than 0.05 indicates that the data does not follow a normal distribution.
        
        Returns:
            dict: A dictionary containing the Jarque–Bera test statistic and p-value.
        """
        s = self.data[self.target].dropna()
        stat, p = jarque_bera(s)
        self.report_lines.append(f"Jarque–Bera test: p-value = {p:.4f}.")
        logger.info("Jarque–Bera test: p=%.4f", p)
        return {"JB Statistic": stat, "p-value": p}

    def anderson_darling_test(self):
        """
        Performs the Anderson–Darling test for normality on the target variable.
        
        Outcome:
            - The test statistic is compared to a set of critical values. If the statistic exceeds a critical value
              for a given significance level, the null hypothesis of normality is rejected.
        
        Returns:
            dict: A dictionary containing the Anderson–Darling test statistic and the critical values.
        """
        s = self.data[self.target].dropna()
        result = anderson(s)
        self.report_lines.append(f"Anderson–Darling test statistic: {result.statistic:.4f}.")
        self.report_lines.append(f"Critical values: {result.critical_values}")
        logger.info("Anderson–Darling test: Statistic=%.4f, Crit. values=%s", result.statistic, result.critical_values)
        return {"AD Statistic": result.statistic, "Critical Values": result.critical_values}

    def histogram_density(self):
        """
        Plots a histogram of the target variable with an overlaid kernel density estimate.
        
        Outcome:
            - The histogram provides a visual summary of the frequency distribution of the target variable.
            - The density curve helps assess the underlying probability distribution.
        
        Outputs:
            - Saves the plot as "histogram_density.tex" in the plots directory.
            - Updates the report lines.
        """
        plt.figure(figsize=(8, 6))
        sns.histplot(self.data[self.target].dropna().values, kde=True, bins=24)
        plt.title(f"Histogram and Density of {self.target_name}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        out_file_pdf = os.path.join(self.plots_dir, "histogram_density.pdf")
        plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
        plt.close()
        self.report_lines.append(f"Histogram with density saved: {out_file_pdf}.")
        logger.info("Histogram with density saved: %s", out_file_pdf)

    # =============================================================================
    # VIII. HETEROSKEDASTICITY
    # =============================================================================
    def breusch_pagan_test(self):
        """
        Performs the Breusch–Pagan test for heteroskedasticity on a simple regression model.
        
        The Breusch–Pagan test checks whether the variance of the errors from a regression is dependent
        on the values of the independent variable(s). A p-value less than 0.05 indicates evidence of heteroskedasticity.
        
        Outcome:
            - Returns a dictionary with the Lagrange Multiplier statistic and the p-value.
        
        Interpretation:
            - A low p-value (< 0.05) suggests that the variance of the residuals is not constant.
        """
        df = self.data.dropna()
        df['time_numeric'] = np.arange(len(df))
        X = df[['time_numeric']]
        y = df[self.target]
        model = LinearRegression().fit(X, y)
        residuals = y - model.predict(X)
        X_const = sm.add_constant(X)
        bp_test = het_breuschpagan(residuals, X_const)
        p_value = bp_test[1]
        self.report_lines.append(f"Breusch–Pagan test: p-value = {p_value:.4f}.")
        logger.info("Breusch–Pagan test: p=%.4f", p_value)
        return {"Lagrange Multiplier Statistic": bp_test[0], "p-value": p_value}

    def goldfeld_quandt_test(self):
        """
        Performs the Goldfeld–Quandt test for heteroskedasticity on a simple regression model.
        
        The test compares the variances of the residuals in two different subsets of the data.
        A p-value less than 0.05 indicates that heteroskedasticity is present.
        
        Outcome:
            - Returns a dictionary with the F statistic and the p-value.
        
        Interpretation:
            - A low p-value (< 0.05) suggests that the error variance changes across the dataset.
        """
        df = self.data.dropna()
        df['time_numeric'] = np.arange(len(df))
        X = df[['time_numeric']]
        y = df[self.target]
        gq_test = het_goldfeldquandt(y, X)
        p_value = gq_test[1]
        self.report_lines.append(f"Goldfeld–Quandt test: p-value = {p_value:.4f}.")
        logger.info("Goldfeld–Quandt test: p=%.4f", p_value)
        return {"F Statistic": gq_test[0], "p-value": p_value}

    def arch_test(self):
        """
        Performs the ARCH test (Engle's Test) for autoregressive conditional heteroskedasticity on the residuals.
        
        The ARCH test checks whether the variance of the errors changes over time, indicating volatility clustering.
        A p-value below 0.05 suggests that there is significant heteroskedasticity.
        
        Outcome:
            - Returns a dictionary with the ARCH test statistic and the p-value.
        
        Interpretation:
            - A low p-value (< 0.05) indicates the presence of ARCH effects.
        """
        df = self.data.dropna()
        df['time_numeric'] = np.arange(len(df))
        X = df[['time_numeric']]
        y = df[self.target]
        model = LinearRegression().fit(X, y)
        residuals = y - model.predict(X)
        arch_result = het_arch(residuals)
        p_value = arch_result[1]
        self.report_lines.append(f"ARCH test: p-value = {p_value:.4f}.")
        logger.info("ARCH test: p=%.4f", p_value)
        return {"ARCH Statistic": arch_result[0], "p-value": p_value}

    def plot_squared_residuals(self):
        """
        Plots the squared residuals from a simple regression model fitted on the target variable.
        
        Outcome:
            - The plot of squared residuals helps to visually inspect the presence of heteroskedasticity.
            - Clusters or trends in the squared residuals indicate non-constant variance.
        
        Outputs:
            - Saves the plot as "squared_residuals.tex" in the plots directory.
            - Updates the report lines.
        """
        df = self.data.dropna().copy()
        df['time_numeric'] = np.arange(len(df))
        X = df[['time_numeric']]
        y = df[self.target]
        model = LinearRegression().fit(X, y)
        residuals = y - model.predict(X)
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, residuals**2, label="Squared Residuals", color="magenta")
        plt.title("Squared Residuals Over Time")
        plt.xlabel("Time")
        plt.ylabel("Squared Residual")
        plt.grid(True)
        out_file_pdf = os.path.join(self.plots_dir, "squared_residuals.pdf")
        plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
        plt.close()
        self.report_lines.append(f"Squared residuals plot saved: {out_file_pdf}.")
        logger.info("Squared residuals plot saved: %s", out_file_pdf)

    # =============================================================================
    # IX. MULTICOLLINEARITY
    # =============================================================================
    def correlation_matrix(self):
        """
        Generates a heatmap of the correlation matrix of all numeric features in the dataset.
        
        Outcome:
            - The heatmap visually indicates the strength and direction of linear relationships between features.
            - Strong correlations (positive or negative) may suggest multicollinearity.
        
        Outputs:
            - Saves the heatmap as "correlation_matrix.tex" in the plots directory.
            - Updates the report lines.
        """
  
        corr_matrix = self.data.corr()
        plt.figure(figsize=(20, 15))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
        plt.title("Correlation Matrix of All Features")
        out_file_pdf = os.path.join(self.plots_dir, "correlation_matrix.pdf")
        plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
        plt.close()
        self.report_lines.append(f"Correlation matrix heatmap saved: {out_file_pdf}.")
        logger.info("Correlation matrix heatmap saved: %s", out_file_pdf)

    def calculate_vif(self):
        """
        Calculates the Variance Inflation Factor (VIF) for all numeric features in the dataset.
        
        VIF quantifies the extent of multicollinearity by measuring how much the variance of an estimated regression coefficient is increased 
        due to collinearity. A VIF value greater than 10 is typically considered indicative of high multicollinearity.
        
        Returns:
            pd.DataFrame: A DataFrame containing each feature and its corresponding VIF value.
        
        Outputs:
            - Saves the VIF results as "vif_results.csv" in the stats directory.
            - Updates the report lines.
        """
        X = self.data.dropna().select_dtypes(include=[np.number])
        vif_df = pd.DataFrame()
        vif_df["feature"] = X.columns
        vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        out_file = os.path.join(self.stats_dir, "vif_results.csv")
        vif_df.to_csv(out_file, index=False)
        self.report_lines.append("VIF results:\n" + vif_df.to_string(index=False))
        logger.info("VIF results saved: %s", out_file)
        return vif_df

    # =============================================================================
    # X. STRUCTURAL BREAKS
    # =============================================================================
    def chow_test(self):
        """
        Performs the Chow test for structural breaks by splitting the dataset into two halves.
        
        The Chow test compares the fit of a regression model on the full dataset versus separate regressions on two sub-samples.
        A p-value less than 0.05 suggests that the two sub-samples differ significantly, indicating a structural break.
        
        Returns:
            dict: A dictionary containing:
                - "Chow Statistic": The computed test statistic.
                - "p-value": The p-value of the test.
        
        Interpretation:
            - A low p-value (< 0.05) indicates that the relationship between predictors and the target variable has changed.
        """
        df = self.data.dropna()
        n = len(df)
        break_point = n // 2
        y = df[self.target]
        X = df.drop(columns=[self.target]).select_dtypes(include=[np.number])
        X = pd.concat([pd.Series(1, index=X.index, name='const'), X], axis=1)
        model_full = LinearRegression().fit(X, y)
        resid_full = y - model_full.predict(X)
        ssr_full = np.sum(np.square(resid_full))
        X1, y1 = X.iloc[:break_point], y.iloc[:break_point]
        X2, y2 = X.iloc[break_point:], y.iloc[break_point:]
        model1 = LinearRegression().fit(X1, y1)
        model2 = LinearRegression().fit(X2, y2)
        ssr1 = np.sum(np.square(y1 - model1.predict(X1)))
        ssr2 = np.sum(np.square(y2 - model2.predict(X2)))
        ssr_pooled = ssr1 + ssr2
        num_params = X.shape[1]
        chow_stat = ((ssr_full - ssr_pooled) / num_params) / (ssr_pooled / (n - 2 * num_params))
        from scipy.stats import f
        p_value = 1 - f.cdf(chow_stat, num_params, n - 2 * num_params)
        self.report_lines.append(f"Chow test: statistic = {chow_stat:.3f}, p-value = {p_value:.4f}.")
        logger.info("Chow test: statistic=%.3f, p=%.4f", chow_stat, p_value)
        return {"Chow Statistic": chow_stat, "p-value": p_value}

    def plot_breakpoints(self, breakpoints: List[datetime]):
        """
        Overlays vertical lines on a time series plot to mark known or suspected structural breakpoints.
        
        Args:
            breakpoints (List[datetime]): A list of datetime objects indicating the locations of breakpoints.
        
        Outcome:
            - The plot displays the target time series with vertical dashed red lines at each breakpoint.
            - This helps visually identify periods of structural change.
        
        Outputs:
            - Saves the plot as "structural_breaks.tex" in the plots directory.
            - Updates the report lines.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data[self.target], label=self.target)
        for bp in breakpoints:
            plt.axvline(bp, color="red", linestyle="--", label="Breakpoint" if bp == breakpoints[0] else "")
        plt.title("Time Series with Structural Breakpoints")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        out_file_pdf = os.path.join(self.plots_dir, "structural_breaks.pdf")
        plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
        plt.close()
        self.report_lines.append(f"Structural breakpoints plot saved: {out_file_pdf}.")
        logger.info("Structural breakpoints plot saved: %s", out_file_pdf)

    # =============================================================================
    # XI. COINTEGRATION
    # =============================================================================
    def johansen_cointegration_test(self, det_order: int = 0, k_ar_diff: int = 1):
        """
        Performs the Johansen cointegration test on a set of time series.
        
        The Johansen test checks for the existence of cointegrating relationships (i.e., long-term equilibrium)
        among multiple time series. It provides eigenvalues that can be compared against critical values.
        
        Args:
            det_order (int): The deterministic trend assumption (0 for no trend, 1 for constant, etc.).
            k_ar_diff (int): The number of lagged differences used in the test.
        
        Returns:
            dict: A dictionary containing:
                - "Eigenvalues": The eigenvalues from the test.
                - "Critical Values": The critical values for determining significance.
        
        Interpretation:
            - If the eigenvalues exceed the corresponding critical values at a given significance level,
              it suggests cointegration among the series.
        
        Outputs:
            - Saves a CSV file "johansen_cointegration.csv" with the eigenvalues.
            - Updates the report lines.
        """
        vars_to_test = [self.target] + self.exog if self.exog else [self.target]
        df = self.data[vars_to_test].dropna()
        result = coint_johansen(df, det_order=det_order, k_ar_diff=k_ar_diff)
        self.report_lines.append(f"Johansen cointegration test eigenvalues: {result.lr1}.")
        out_file = os.path.join(self.stats_dir, "johansen_cointegration.csv")
        pd.DataFrame({"Eigenvalues": result.lr1}).to_csv(out_file, index=False)
        logger.info("Johansen cointegration test results saved: %s", out_file)
        trace_stats = result.lr1
        trace_crit_values = result.cvt
        # Maximum-Eigenwert-Statistiken und zugehörige kritische Werte
        max_eigen_stats = result.lr2
        max_eigen_crit_values = result.cvm
        return {"TraceStatistic": trace_stats,
                "TraceCriticalValues": trace_crit_values,
                "MaxEigenStatistic": max_eigen_stats,
                "MaxEigenCriticalValues": max_eigen_crit_values}

    # =============================================================================
    # XII. MISSING VALUES ANALYSIS
    # =============================================================================
    def missing_values_analysis(self):
        """
        Analyzes and visualizes the missing values in the dataset.
        
        Outcome:
            - Computes the number of missing values per column.
            - Creates a heatmap that visually shows where missing values occur over time.
        
        Outputs:
            - Saves a heatmap as "missing_values_heatmap.tex" in the plots directory.
            - Appends a summary of missing counts to the report lines.
        """
        missing_counts = self.data.isna().sum()
        self.report_lines.append("Missing values per column:\n" + missing_counts.to_string())
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.data.isna(), cbar=False)
        plt.title("Missing Values Heatmap")
        out_file_pdf = os.path.join(self.plots_dir, "missing_values_heatmap.pdf")
        plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
        plt.close()
        self.report_lines.append(f"Missing values heatmap saved: {out_file_pdf}.")
        logger.info("Missing values heatmap saved: %s", out_file_pdf)

    # =============================================================================
    # XIII. FEATURE EVALUATION (LINEAR MODELS)
    # =============================================================================
    def ols_regression(self, predictors: List[str]) -> Any:
        """
        Fits an Ordinary Least Squares (OLS) regression model using the specified predictors.
        
        The OLS regression provides estimates of model coefficients, their statistical significance (p-values),
        and overall model fit (R²). This helps in understanding the linear relationship between the predictors and the target.
        
        Args:
            predictors (List[str]): A list of column names to be used as independent variables.
        
        Returns:
            A fitted OLS regression model object (from statsmodels) containing regression results.
        
        Outputs:
            - Saves the model summary as "ols_regression_summary.txt" in the stats directory.
            - Appends the summary text to the report lines.
        """
        import statsmodels.api as sm
        df = self.data.dropna(subset=[self.target] + predictors)
        X = df[predictors]
        X = sm.add_constant(X)
        y = df[self.target]
        model = sm.OLS(y, X).fit()
        summary = model.summary().as_text()
        self.report_lines.append("OLS Regression Summary:\n" + summary)
        out_file = os.path.join(self.stats_dir, "ols_regression_summary.txt")
        with open(out_file, "w") as f:
            f.write(summary)
        logger.info("OLS regression summary saved: %s", out_file)
        return model

    def f_test_feature_importance(self, predictors: List[str]):
        """
        Computes F-test statistics for each predictor to evaluate its significance in a regression model.
        
        The F-test examines whether a predictor has a statistically significant linear relationship with the target.
        High F-values (with a p-value less than 0.05) suggest that the predictor is important.
        
        Args:
            predictors (List[str]): List of predictor variable names.
        
        Outputs:
            - Saves a CSV file "f_test_feature_importance.csv" with F statistics and p-values.
            - Updates the report lines with the test results.
        """
        from sklearn.feature_selection import f_regression
        df = self.data.dropna(subset=[self.target] + predictors)
        X = df[predictors]
        y = df[self.target]
        F, p = f_regression(X, y)
        results = pd.DataFrame({"Feature": predictors, "F": F, "p": p}).sort_values("F", ascending=False)
        out_file = os.path.join(self.stats_dir, "f_test_feature_importance.csv")
        results.to_csv(out_file, index=False)
        self.report_lines.append("F-test feature importance results:\n" + results.to_string(index=False))
        logger.info("F-test feature importance results saved: %s", out_file)
        return results

    def mutual_information(self, predictors: List[str]):
        """
        Computes the mutual information between each predictor and the target variable.
        
        Mutual information quantifies the amount of information obtained about one variable through another.
        Higher values indicate a stronger, potentially nonlinear, association.
        
        Args:
            predictors (List[str]): List of predictor variable names.
        
        Returns:
            pd.DataFrame: A DataFrame listing each predictor and its corresponding mutual information score.
        
        Outputs:
            - Saves the results as "mutual_information.csv" in the stats directory.
            - Updates the report lines.
        """
        from sklearn.feature_selection import mutual_info_regression
        df = self.data.dropna(subset=[self.target] + predictors)
        X = df[predictors].fillna(0)
        y = df[self.target].fillna(df[self.target].mean())
        mi = mutual_info_regression(X, y)
        results = pd.DataFrame({"Feature": predictors, "MI": mi}).sort_values("MI", ascending=False)
        out_file = os.path.join(self.stats_dir, "mutual_information.csv")
        results.to_csv(out_file, index=False)
        self.report_lines.append("Mutual information results:\n" + results.to_string(index=False))
        logger.info("Mutual information results saved: %s", out_file)
        return results

    def lasso_regression(self, predictors: List[str], alpha: float = 0.1):
        """
        Fits a Lasso regression model to perform feature selection via regularization.
        
        The Lasso model penalizes the absolute size of the coefficients, shrinking some to zero,
        thereby indicating less important features.
        
        Args:
            predictors (List[str]): List of predictor variable names.
            alpha (float): Regularization strength (default is 0.1). Higher values result in more coefficients shrunk to zero.
        
        Returns:
            pd.Series: A Series of Lasso regression coefficients indexed by predictor names.
        
        Outputs:
            - Appends the coefficients to the report lines.
        """
        from sklearn.linear_model import Lasso
        df = self.data.dropna(subset=[self.target] + predictors)
        X = df[predictors].fillna(0)
        y = df[self.target].fillna(df[self.target].mean())
        model = Lasso(alpha=alpha)
        model.fit(X, y)
        coef_series = pd.Series(model.coef_, index=predictors)
        self.report_lines.append("Lasso regression coefficients:\n" + coef_series.to_string())
        logger.info("Lasso regression coefficients computed.")
        return coef_series

    def random_forest_importance(self, predictors: List[str]):
        """
        Fits a Random Forest regressor to compute feature importances.
        
        Random Forests can provide an estimate of feature importance by measuring the decrease in 
        prediction error when a feature is randomly permuted.
        
        Args:
            predictors (List[str]): List of predictor variable names.
        
        Returns:
            pd.Series: A Series of feature importance scores sorted in descending order.
        
        Outputs:
            - Saves the importance results as "random_forest_importance.csv" (or similar) in the stats directory.
            - Updates the report lines.
        """
        from sklearn.ensemble import RandomForestRegressor
        df = self.data.dropna(subset=[self.target] + predictors)
        X = df[predictors].fillna(0)
        y = df[self.target].fillna(df[self.target].mean())
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=predictors).sort_values(ascending=False)
        self.report_lines.append("Random Forest feature importances:\n" + importances.to_string())
        logger.info("Random Forest feature importances computed.")
        return importances

    # =============================================================================
    # XIV. RESIDUAL ANALYSIS (AFTER MODELING)
    # =============================================================================
    def residuals_vs_fitted_plot(self, model):
        """
        Plots residuals versus fitted values for a given regression model.
        
        This plot is used to assess homoscedasticity. Ideally, residuals should be randomly scattered
        around zero without any apparent pattern.
        
        Args:
            model: A fitted regression model (e.g., from statsmodels) that has attributes 'fittedvalues' and 'resid'.
        
        Outputs:
            - Saves a plot "residuals_vs_fitted.tex" in the plots directory.
            - Updates the report lines.
        """
        fitted = model.fittedvalues
        residuals = model.resid
        plt.figure(figsize=(10, 5))
        plt.scatter(fitted, residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs. Fitted Plot")
        out_file_pdf = os.path.join(self.plots_dir, "residuals_vs_fitted.pdf")
        plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
        plt.close()
        self.report_lines.append(f"Residuals vs. fitted plot saved: {out_file_pdf}.")
        logger.info("Residuals vs. fitted plot saved: %s", out_file_pdf)

    def qq_plot_residuals(self, model):
        """
        Generates a Q–Q plot of the residuals from a given regression model.
        
        The Q–Q plot compares the quantiles of the residuals to the quantiles of a normal distribution.
        Deviations from the 45° line suggest departures from normality.
        
        Args:
            model: A fitted regression model with residuals (model.resid).
        
        Outputs:
            - Saves a plot "qq_plot_residuals.tex" in the plots directory.
            - Updates the report lines.
        """
        import statsmodels.api as sm
        residuals = model.resid
        fig = sm.qqplot(residuals, line='45', fit=True)
        plt.title("Q–Q Plot of Residuals")
        out_file_pdf = os.path.join(self.plots_dir, "qq_plot_residuals.pdf")
        plt.savefig(out_file_pdf, format="pdf", bbox_inches="tight")
        plt.close()
        self.report_lines.append(f"QQ plot of residuals saved: {out_file_pdf}.")
        logger.info("QQ plot of residuals saved: %s", out_file_pdf)

    def ljung_box_residuals(self, model, lags: List[int] = [24]) -> pd.DataFrame:
        """
        Applies the Ljung–Box test on the residuals of a regression model to test for autocorrelation.
        
        Args:
            model: A fitted regression model that provides residuals (model.resid).
            lags (List[int]): List of lag values at which to perform the test.
        
        Returns:
            pd.DataFrame: A DataFrame with test statistics and p-values for each lag.
        
        Interpretation:
            - A p-value greater than 0.05 at a given lag indicates that the residuals are approximately white noise.
        
        Outputs:
            - Saves the test results as "ljung_box_residuals.csv" in the stats directory.
            - Updates the report lines.
        """
        residuals = model.resid.dropna()
        lb_test = acorr_ljungbox(residuals, lags=lags, return_df=True)
        out_file = os.path.join(self.stats_dir, "ljung_box_residuals.csv")
        lb_test.to_csv(out_file, index=False)
        self.report_lines.append(f"Ljung–Box test on residuals results saved: {out_file}.")
        logger.info("Ljung–Box test on residuals saved: %s", out_file)
        return lb_test

    # =============================================================================
    # XV. MODEL COMPARISON
    # =============================================================================
    def anova_model_comparison(self, model1, model2):
        """
        Compares two linear regression models using ANOVA.
        
        The ANOVA test compares the fits of two models (one nested within the other) to determine if the
        more complex model provides a statistically significant improvement in fit.
        
        Args:
            model1: The simpler (reduced) regression model.
            model2: The more complex (full) regression model.
        
        Returns:
            pd.DataFrame: A DataFrame containing the ANOVA test results (F-statistic and p-value).
        
        Interpretation:
            - A p-value less than 0.05 suggests that the additional predictors in the complex model significantly improve the model.
        
        Outputs:
            - Saves the results as "anova_comparison.csv" in the stats directory.
            - Updates the report lines.
        """
        import statsmodels.api as sm
        anova_results = sm.stats.anova_lm(model1, model2, typ=1)
        out_file = os.path.join(self.stats_dir, "anova_comparison.csv")
        anova_results.to_csv(out_file)
        self.report_lines.append("ANOVA model comparison results:\n" + anova_results.to_string())
        logger.info("ANOVA model comparison results saved: %s", out_file)
        return anova_results

    # =============================================================================
    # RUN ALL TESTS & GENERATE REPORT
    # =============================================================================
    def run_all_tests(self):
        """
        Runs all the EDA tests and analyses and writes a comprehensive summary report to a file.
        
        This method sequentially executes all analyses, including stationarity tests, autocorrelation analysis,
        seasonality decomposition, exogenous analysis, nonlinearity tests, outlier detection, distribution analysis,
        heteroskedasticity tests, multicollinearity diagnostics, structural break tests, cointegration tests,
        missing values analysis, feature evaluation using linear models, residual analysis, and model comparison.
        
        Outcome:
            - A series of plots and CSV/statistical outputs are generated and saved to the output directories.
            - A text report summarizing all tests and their key results is written to "eda_report.txt".
        
        Inputs:
            - None directly; uses the data and configuration provided at initialization.
        
        Returns:
            None
        """
        self.save_seasonal_decomposition(period=24)
        self.report_lines.append("Full Time Series EDA Report")
        self.report_lines.append("=" * 60)
        self.report_lines.append(f"Target Variable: {self.target_name}")
        self.report_lines.append(f"Time Range: {self.data.index.min()} to {self.data.index.max()}")
        self.report_lines.append("")

        # # I. Stationarity
        self.report_lines.append("I. Stationarity Tests:")
        self.adf_test(self.data[self.target], diff=0)
        self.adf_test(self.data[self.target], diff=1)
        self.kpss_test(self.data[self.target], diff=0)
        self.kpss_test(self.data[self.target], diff=1)
        self.rolling_statistics()

        # II. Autocorrelation
        self.report_lines.append("\nII. Autocorrelation Analysis:")
        self.acf_pacf_plots(lags=100)
        self.ljung_box_test(lags=[10, 20])
        self.autocorrelation_heatmap(max_lag=50)

        # # III. Seasonality
        self.report_lines.append("\nIII. Seasonality Analysis:")
        self.seasonal_decomposition(period=24)
        self.stl_decomposition(period=24)
        self.boxplots_by_time()

        # IV. Exogenous Variables / Cross-Correlation
        self.report_lines.append("\nIV. Exogenous and Cross-Correlation Analysis:")
        # self.cross_correlation_plot()
        self.cross_correlation_plot_report(max_lags=24)
        self.granger_causality_test(maxlag=24)



        self.scatterplot_lagged_exog_test(lag=0, color_feature="day_night")
        self.scatterplot_lagged_exog(lag=0)

        # V. Nonlinearity
        self.report_lines.append("\nV. Nonlinearity Tests:")
        self.bds_test()
        self.scatter_target_vs_transformed()
        self.recurrence_plot()

        # VI. Outlier Detection
        self.report_lines.append("\nVI. Outlier Detection:")
        self.outlier_detection_zscore()
        self.outlier_detection_iqr()
        self.outlier_detection_lof(n_neighbors=20)

        # VII. Distribution Analysis
        self.report_lines.append("\nVII. Distribution Analysis:")
        self.shapiro_test()
        self.dagostino_test()
        self.jarque_bera_test()
        self.anderson_darling_test()
        # self.qq_plot_distribution(against="norm")
        self.histogram_density()

        # VIII. Heteroskedasticity
        self.report_lines.append("\nVIII. Heteroskedasticity Tests:")
        self.breusch_pagan_test()
        self.goldfeld_quandt_test()
        self.arch_test()
        self.plot_squared_residuals()

        # IX. Multicollinearity
        self.report_lines.append("\nIX. Multicollinearity Diagnostics:")
        self.correlation_matrix()
        self.calculate_vif()

        # X. Structural Breaks
        self.report_lines.append("\nX. Structural Break Tests:")
        self.chow_test()
        sample_breakpoints = [self.data.index[int(len(self.data)*0.33)], self.data.index[int(len(self.data)*0.66)]]
        self.plot_breakpoints(sample_breakpoints)

        # XI. Cointegration
        self.report_lines.append("\nXI. Cointegration Tests:")
        self.johansen_cointegration_test(det_order=0, k_ar_diff=1)

        # XII. Missing Values
        self.report_lines.append("\nXII. Missing Values Analysis:")
        self.missing_values_analysis()

        # XIII. Feature Evaluation (Linear Models)
        self.report_lines.append("\nXIII. Feature Evaluation:")
        predictors = list(self.data.columns.drop(self.target))
        ols_model = self.ols_regression(predictors)
        self.f_test_feature_importance(predictors)
        self.mutual_information(predictors)
        self.lasso_regression(predictors, alpha=0.1)
        self.random_forest_importance(predictors)

        # XIV. Residual Analysis (After Modeling)
        self.report_lines.append("\nXIV. Residual Analysis (Model Based):")
        self.residuals_vs_fitted_plot(ols_model)
        self.qq_plot_residuals(ols_model)
        self.ljung_box_residuals(ols_model, lags=[24])

        # XV. Model Comparison
        self.report_lines.append("\nXV. Model Comparison:")
        if len(predictors) >= 3:
            model_full = ols_model
            model_reduced = self.ols_regression(predictors[:3])
            self.anova_model_comparison(model_reduced, model_full)
        else:
            self.report_lines.append("Not enough predictors for ANOVA model comparison.")

        # Write final report
        with open(self.report_file, "w") as f:
            for line in self.report_lines:
                f.write(line + "\n")
        logger.info("Full EDA report saved: %s", self.report_file)


    # =============================================================================
    # End of Class
    # =============================================================================

# =============================================================================
# Example usage:
# =============================================================================
if __name__ == "__main__":
    # For demonstration, create a synthetic time series dataset.
   

    # import pandas as pd
    # from scripts.data_analysis.deep_data_analysis import FullTimeSeriesEDA
    # import os

  
    target_csv_path = "data/raw/no2/no2_data.csv"
    air_exog_csv_path = "data/air/history/karlsruhe/current.csv"
    exog_csv_path = "data/weather/history/karlsruhe/current.csv"
    output_dir = "scripts/data_analysis/no2/eda_report_22_25_2"
    target_name = "NO$_2$ Concentration"
  







    # target_csv_path = "data/raw/energy/energy_data.csv"
    # exog_csv_path = "data/weather/history/germany/current.csv"
    # air_exog_csv_path = "data/air/history/germany/current.csv"
    # output_dir = "scripts/data_analysis/energy/eda_report_22_25_1"
    # target_name = "Energy Demand"











    df_target = pd.read_csv(target_csv_path, index_col=0, parse_dates=[0])[['gesamt']].rename(columns={'gesamt': 'target'})
    df_exog = pd.read_csv(exog_csv_path, index_col=0, parse_dates=[0])
    df_air_exog = pd.read_csv(air_exog_csv_path, index_col=0, parse_dates=[0])

    # concat exog and air_exog  
    df_exog = pd.concat([df_exog, df_air_exog], axis=1)

    exog_cols = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
        "precipitation", "rain", "surface_pressure", "cloud_cover",
        "et0_fao_evapotranspiration", "vapour_pressure_deficit",
        "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m",
        "wet_bulb_temperature_2m", "sunshine_duration",
        "shortwave_radiation", "diffuse_radiation","pm10","pm2_5","carbon_monoxide",
        "sulphur_dioxide","ozone","dust","ammonia"
    ]

    df_exog = df_exog[exog_cols]    



    # Sicherstellen, dass die Indizes überlappen und zusammenführen
    start_date = df_target.index[-28197]
    # start_date = df_target.index[-336]
    end_date = min(df_target.index.max(), df_exog.index.max())
    data = df_target.loc[start_date:end_date].merge(df_exog.loc[start_date:end_date], left_index=True, right_index=True, how='inner')
    
    


    exog_cols = df_exog.columns.tolist()


    # EDA ausführen
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save data to csv  in output_dir
    data.to_csv(output_dir + '/data.csv')




    # rename columns target to target_name
    data.rename(columns={'target': target_name}, inplace=True)
    eda = FullTimeSeriesEDA(data, target=target_name, exog=exog_cols, output_dir=output_dir, target_name=target_name, window=24)
    eda.run_all_tests()