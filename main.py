"""
Main entry point for the forecasting pipeline.

This module sets up and runs the forecasting pipeline for different datasets and model configurations.
It registers the available model classes and iterates over provided runs, merging configurations,
checking for completed pipelines, and initiating the pipeline execution.
"""

import logging
import os
import pdb

import numexpr

from scripts.utils.configs import ConfigManager
from scripts.pipeline_manager import PipelineManager
from scripts.models.quantile_regression.quantile_regression import QuantileRegressionModel
from scripts.models.baseline.naive_baseline import NaiveBaselineModel
from scripts.models.light_gbm.light_gbm_forecast import LightGBMForecastModel
from scripts.models.lstm.lstm_forecast import LSTMForecastModel
from scripts.utils.data_scraping.scraping_class import DataScraping
numexpr.set_num_threads(32)
os.environ["NUMEXPR_MAX_THREADS"] = "32"
# You can register additional model classes here:
MODEL_CLASSES = {
    "quantile_regression": QuantileRegressionModel,
    "light_gbm": LightGBMForecastModel,
    "baseline": NaiveBaselineModel,
    "feature_selection": LightGBMForecastModel,
    "lstm": LSTMForecastModel
}


def main(runs: dict = None):
    """
    Run the forecasting pipeline for specified datasets and model configurations.

    Args:
        runs (dict, optional): A dictionary mapping dataset names to a list of model names to run.
            Example: {'energy': ['baseline', 'light_gbm', 'quantile_regression', 'lstm']}
    """
    if runs is None:
        runs = {}
    
    # check if datascraping is necessary
    # DataScraping().update_target_data()
    #DataScraping().update_weather_data()


    for data_set, models in runs.items():
        for model_name in models:
            print(f"Dataset: {data_set}, Model: {model_name}")

            cfg = ConfigManager(data_set, model_name)
            # 2) Load the base configuration to access the "versions".
            #    E.g., loads configs/energy/quantile_regression.yml and configs/default/paths.yml.
            base_cfg = cfg.get_config(**{data_set: model_name, "default": "paths"})

            # 3) Check if there are any version entries in the configuration.
            if "versions" not in base_cfg or not base_cfg["versions"]:
                logging.warning("No 'versions' defined for %s in %s.", model_name, data_set)
                return

            # 4) Iterate over all version keys.
            for version_key in base_cfg["versions"].keys():
                # 5) Merge to a final configuration.
                config = cfg.merge_configs(
                    dataset=data_set,
                    model=model_name,
                    version_key=version_key
                )

                # 6) Check if a marker file already exists -> if so, skip processing.
                marker_file = os.path.join(config["output_dir"], "pipeline_finished.txt")
                if os.path.exists(marker_file):
                    logging.info("Version %s already processed. Skipping...", version_key)
                    continue

                logging.info("Starting pipeline for %s / Version=%s...", model_name, version_key)

                # 7) Retrieve the model class.
                model_cls = MODEL_CLASSES.get(model_name)
                if not model_cls:
                    raise ValueError("Unknown model: %s" % model_name)

                # 8) Instantiate the model and start the pipeline.
                model = model_cls(config)
                pipeline = PipelineManager(model, config, model_name, version_key, data_set, cfg)
                pipeline.run_pipeline()

                # 9) Create a marker file to indicate successful completion of the pipeline.
                with open(marker_file, "w", encoding="utf-8") as file_obj:
                    file_obj.write("Pipeline successfully completed.\n")

                logging.info("Pipeline completed for model=%s, Version=%s.", model_name, version_key)


if __name__ == "__main__":
    # runs = { 'energy': ['feature_selection'], 'no2': ['feature_selection']}
    runs = {
        "energy": ["lstm", "light_gbm", "quantile_regression", "baseline"],
         "no2": ["baseline", "quantile_regression", "lstm", "light_gbm"],
                  
            }
    main(runs)
