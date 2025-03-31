"""
Module: feature_selection
This module implements a feature selection manager that uses LightGBM to perform feature selection.
It trains multiple LightGBM models on random subsets of features and returns a final list of selected features.
"""

import os
import numpy as np
import pandas as pd
# Geändert: Importiere lgb direkt, nicht LGBMRegressor allein
import lightgbm as lgb
from sklearn.metrics import mean_pinball_loss
from scripts.utils.logger import logger
import pdb # Bleibt drin, wie vom User gepostet
from sklearn.utils import shuffle

from sklearn.utils import shuffle # Doppelter Import bleibt, wie vom User gepostet
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
# Hinzugefügt: Type Hinting
from typing import List, Tuple, Dict, Any

# Annahme: logger wird korrekt importiert oder Fallback wird verwendet
try:
    from scripts.utils.logger import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # logger.warning("Could not import logger from 'scripts.utils.logger'. Using basic logging.")


def pinball_loss_0_5(y_true, y_pred):
    """
    Calculates the pinball loss for the 0.5 quantile.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.

    Returns:
        float: The pinball loss.
    """
    return mean_pinball_loss(y_true, y_pred, alpha=0.5)


class FeatureSelectionManager:
    # __init__ bleibt exakt wie vom User gepostet
    def __init__(self, config: Dict[str, Any], output_dir: str, model: Any = None):
        """
        Initialize the FeatureSelectionManager.

        Args:
            config (dict): Configuration dictionary. Expects keys 'feature_selection' and 'params'.
            output_dir (str): Path for saving reports and plots.
            model: Not actively used; included for potential compatibility.
        """
        self.output_dir = output_dir
        self.config = config["feature_selection"]
        self.top_n = self.config.get("top_n", 15)
        self.params = config.get("params", {})
        os.makedirs(self.output_dir, exist_ok=True)

        # Zugriff auf Konfigurationsteil 'feature_selection' und 'params'
        # Setze Standardwerte falls Schlüssel fehlen
        fs_config = config.get("feature_selection", {})
        self.config = fs_config # Behalte die feature_selection config
        self.top_n = fs_config.get("top_n", 15) # top_n aus feature_selection holen

        # Hole LGBM-Parameter aus dem Haupt-Config-Bereich
        self.params = config.get("params", {})
        # Stelle sicher, dass 'objective' und 'alpha' für Quantilregression gesetzt sind
        # Überschreibe ggf. generische Params, wenn nicht explizit anders gesetzt
        if 'objective' not in self.params:
            self.params['objective'] = 'quantile'
        if 'alpha' not in self.params and self.params.get('objective') == 'quantile':
             self.params['alpha'] = 0.5
        # Setze einen default random_state, falls nicht vorhanden
        if 'random_state' not in self.params:
            self.params['random_state'] = 42

        logger.info(f"FeatureSelectionManager initialized. Output dir: {self.output_dir}")
        logger.info(f"LGBM params: {self.params}")
        logger.info(f"Top N features parameter (used in original method): {self.top_n}")


 
    def run_lgbm_feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series, mandatory_features: List[str] = ['hour', 'weekday', 'month'], num_models: int = 5) -> Tuple[List[str], pd.DataFrame]:
        """
        Train multiple LightGBM models using ALL available features (mandatory + non-mandatory)
        to assess feature importance stability and return aggregated importance metrics.
        The returned feature list is sorted by mean importance.
        MODIFIED: Calls plot_feature_importance at the end.

        Args:
            X_train (DataFrame): Training data with all features.
            y_train (Series): Target variable.
            mandatory_features (list): List of features that must be included in every run.
            num_models (int): Number of models to train (each using all features). Default: 5.

        Returns:
            tuple: A tuple containing:
                - list: All features considered, sorted by descending mean importance.
                - DataFrame: Aggregated feature importance metrics over all runs.
        """
        logger.info("Running feature importance evaluation using ALL features...")
        # (Code für Feature-Prüfung, Loop, Training, Aggregation bleibt exakt wie vom User gepostet)
        # Ensure mandatory features exist
        existing_mandatory = [f for f in mandatory_features if f in X_train.columns]
        if len(existing_mandatory) != len(mandatory_features):
            missing = set(mandatory_features) - set(existing_mandatory)
            logger.warning(f"Not all mandatory features found in X_train: {missing}")
        all_non_mandatory_features = [f for f in X_train.columns if f not in existing_mandatory]
        features_for_each_run = sorted(list(set(existing_mandatory + all_non_mandatory_features))) # Unique and sorted

        if not features_for_each_run:
            logger.error("No features available for training.")
            return [], pd.DataFrame()
        logger.info(f"Training {num_models} models, each using the same {len(features_for_each_run)} features.")
        feature_importances_collection = defaultdict(list)
        successful_runs = 0
        for i in range(num_models):
            logger.info(f"Training model {i+1}/{num_models} using all {len(features_for_each_run)} features...")
            X_full = X_train[features_for_each_run]
            current_params = self.params.copy()
            current_params['random_state'] = self.params.get('random_state', 42) + i
            model = lgb.LGBMRegressor(**current_params)
            try:
                model.fit(X_full, y_train)
                successful_runs +=1
            except Exception as e:
                logger.error(f"Error training model {i+1} on features {features_for_each_run}: {e}")
                continue
            try:
                importance = model.booster_.feature_importance(importance_type="gain")
                feat_imp_df = pd.DataFrame({"feature": features_for_each_run, "importance": importance})
                for _, row in feat_imp_df.iterrows(): feature_importances_collection[row['feature']].append(row['importance'])
            except Exception as e:
                logger.error(f"Error getting importance for model {i+1}: {e}")

        logger.info(f"Aggregating feature importance across {successful_runs} successful runs...")
        aggregated_importance = []
        all_considered_features = features_for_each_run
        for feature in all_considered_features:
            scores = feature_importances_collection.get(feature, [])
            runs_present_count = len(scores)
            if runs_present_count > 0:
                mean_importance = np.mean(scores)
                median_importance = np.median(scores)
                std_importance = np.std(scores) if runs_present_count > 1 else 0.0
            else: mean_importance=0.0; median_importance=0.0; std_importance=0.0;
            aggregated_importance.append({
                'feature': feature, 'mean_importance': mean_importance,
                'median_importance': median_importance, 'std_importance': std_importance,
                'runs_present': runs_present_count, 'num_models_trained': num_models
            })

        agg_imp_df = pd.DataFrame(aggregated_importance)
        sorted_feature_list = []
        if not agg_imp_df.empty:
             agg_imp_df = agg_imp_df.sort_values(by="mean_importance", ascending=False).reset_index(drop=True)
             sorted_feature_list = agg_imp_df['feature'].tolist()
             logger.info(f"Importance aggregation complete. Assessed {len(agg_imp_df)} features.")
        else:
             logger.warning("No feature importances could be aggregated.")
             # Still need to return empty structures
             agg_imp_df = pd.DataFrame(columns=['feature', 'mean_importance', 'median_importance', 'std_importance', 'runs_present', 'num_models_trained']) # Ensure DF schema exists even if empty


        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # HIER IST DER EINZIGE HINZUGEFÜGTE AUFRUF:
        # pdb.set_trace() # Bleibt drin, wie vom User gepostet
        if not agg_imp_df.empty:
            logger.info("Calling plot_feature_importance...")
            # Ruft die Plot-Funktion mit dem aggregierten DataFrame auf.
            # Verwendet Standardwerte für top_n, metric, filename aus der plot_feature_importance Definition.
            self.plot_feature_importance(importance_df=agg_imp_df)
        else:
             logger.warning("Skipping plot call because aggregated importance DataFrame is empty.")
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


        return sorted_feature_list, agg_imp_df # Rückgabe bleibt Tuple[List, DataFrame]


    # plot_feature_importance bleibt exakt wie vom User gepostet (inkl. pdb)
    def plot_feature_importance(self,
                                importance_df: pd.DataFrame,
                                top_n: int = 200,  # Gesamtanzahl Features über alle Plots
                                importance_metric: str = 'mean_importance',
                                filename: str = "feature_importance_plot.png",
                                chunk_size: int = 15): # Neue Variable: Max Features pro Plot
        """
        Plots the aggregated feature importance in chunks and saves multiple plots.

        Args:
            importance_df (DataFrame): DataFrame with aggregated feature importance.
            top_n (int): Total number of top features to display across all plots.
            importance_metric (str): Metric to plot ('mean_importance', 'median_importance').
            filename (str): Base name for the output plot files (e.g., "importance.png").
            chunk_size (int): Maximum number of features per plot chunk (default: 15).
        """
        if importance_df.empty or importance_metric not in importance_df.columns:
            logger.error(f"Cannot plot importance. DataFrame is empty or metric '{importance_metric}' is missing.")
            return
        if chunk_size <= 0:
            logger.error("chunk_size must be positive.")
            return

        # 1. Gesamtzahl der zu plottenden Features bestimmen
        actual_top_n = min(top_n, len(importance_df))
        if actual_top_n <= 0:
            logger.warning("No features to plot based on top_n value.")
            return

        logger.info(f"Plotting top {actual_top_n} features based on '{importance_metric}' in chunks of {chunk_size}...")

        # 2. DataFrame nach der Metrik sortieren (absteigend)
        sorted_df = importance_df.sort_values(by=importance_metric, ascending=False).reset_index(drop=True)

        # 3. Nur die Top N Features für die weitere Verarbeitung behalten
        top_df = sorted_df.head(actual_top_n)

        # 4. Anzahl der benötigten Plots berechnen
        num_plots = int(np.ceil(actual_top_n / chunk_size))

        # 5. Basis-Dateinamen und Erweiterung trennen
        base_filename, file_extension = os.path.splitext(filename)

        # 6. Loop über die Chunks/Plots
        for i in range(num_plots):
            # Start- und End-Index für den aktuellen Chunk berechnen
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, actual_top_n)

            # Daten für den aktuellen Plot auswählen
            chunk_df = top_df.iloc[start_idx:end_idx]

            if chunk_df.empty: # Sicherheitscheck
                continue

            # Für horizontalen Barplot: Daten aufsteigend nach Importance sortieren
            # damit der wichtigste Balken oben ist
            plot_chunk_df = chunk_df.iloc[::-1]
            current_chunk_feature_count = len(plot_chunk_df) # Tatsächliche Anzahl in diesem Chunk

            # Dateinamen für diesen Plot-Teil generieren
            # Führende Nullen für Sortierung, falls mehr als 9 Plots
            part_num_str = str(i + 1).zfill(len(str(num_plots)))
            chunk_filename = f"{base_filename}_part_{part_num_str}{file_extension}"
            save_path = os.path.join(self.output_dir, chunk_filename)

            logger.info(f"Generating plot part {i+1}/{num_plots} (Features {start_idx+1}-{end_idx}) -> {chunk_filename}")

            # Plot erstellen
            fig, ax = plt.subplots(figsize=(12, max(4, current_chunk_feature_count * 0.4))) # Höhe anpassen
            sns.barplot(x=importance_metric, y='feature', data=plot_chunk_df, palette='viridis', ax=ax)

            # Titel und Labels
            base_title = f'Aggregated Feature Importance by {importance_metric.replace("_", " ").title()}'
            plot_title = f'{base_title} (Features {start_idx+1}-{end_idx} of {actual_top_n})'
            ax.set_title(plot_title, fontsize=14)
            ax.set_xlabel(f'Aggregated Importance - Type: Gain', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)

            plt.tight_layout()

            # Speichern
            # pdb.set_trace() # Bleibt drin, wie vom User gefordert
            try:
                plt.savefig(save_path, dpi=150)
                logger.info(f"Feature importance plot part saved to: {save_path}")
            except Exception as e:
                logger.error(f"Failed to save plot part {i+1} to {save_path}: {e}")
            finally:
                # Plot schließen, um Speicher freizugeben
                plt.close(fig)

        logger.info(f"Finished generating {num_plots} plot parts.")
