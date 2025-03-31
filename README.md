
## Table of Contents


Directory Structure:

└── ./
    ├── configs
    │   ├── default
    │   │   ├── coordinates.yml
    │   │   └── paths.yml
    │   ├── energy
    │   │   ├── baseline.yml
    │   │   ├── feature_selection.yml
    │   │   ├── light_gbm.yml
    │   │   ├── lstm.yml
    │   │   └── quantile_regression.yml
    │   └── no2
    │       ├── baseline.yml
    │       ├── light_gbm.yml
    │       ├── lstm.yml
    │       └── quantile_regression.yml
    ├── output
    │   └── energy
    │       └── baseline
    │           └── v1.0.0
    │               └── analysis
    │                   └── report_summary.txt
    ├── scripts
    │   ├── data_analysis
    │   │   ├── energy
    │   │   │   ├── eda_report_24_25
    │   │   │   │   ├── stats
    │   │   │   │   │   └── ols_regression_summary.txt
    │   │   │   │   └── eda_report.txt
    │   │   │   ├── eda_report_24_25_resids
    │   │   │   │   ├── stats
    │   │   │   │   │   └── ols_regression_summary.txt
    │   │   │   │   └── eda_report.txt
    │   │   │   └── eda_report_24_25_resids_normalizes
    │   │   │       ├── stats
    │   │   │       │   └── ols_regression_summary.txt
    │   │   │       └── eda_report.txt
    │   │   └── deep_data_analysis.py
    │   ├── data_loader
    │   │   └── data_loader.py
    │   ├── evaluation
    │   │   └── report_manager.py
    │   ├── feature_engineering
    │   │   ├── feature_pipeline.py
    │   │   └── feature_selection.py
    │   ├── hyperparameter
    │   │   └── hyperparameter_optimization.py
    │   ├── models
    │   │   ├── baseline
    │   │   │   └── naive_baseline.py
    │   │   ├── light_gbm
    │   │   │   └── light_gbm_forecast.py
    │   │   ├── lstm
    │   │   │   └── lstm_forecast.py
    │   │   ├── quantile_regression
    │   │   │   └── quantile_regression.py
    │   │   └── base_model.py
    │   ├── predictor
    │   │   └── predictor.py
    │   ├── utils
    │   │   ├── data_processing
    │   │   │   ├── config_checker.py
    │   │   │   └── data_cleaning.py
    │   │   ├── data_scraping
    │   │   │   ├── bike
    │   │   │   │   └── data_processing.py
    │   │   │   ├── energy
    │   │   │   │   └── data_processing.py
    │   │   │   ├── no2
    │   │   │   │   └── data_processing.py
    │   │   │   ├── weather_air
    │   │   │   │   ├── air_germany.py
    │   │   │   │   ├── air_karlsruhe.py
    │   │   │   │   ├── weather_germany.py
    │   │   │   │   └── weather_karlsruhe.py
    │   │   │   └── scraping_class.py
    │   │   ├── configs.py
    │   │   └── logger.py
    │   └── pipeline_manager.py
    ├── .flake8
    ├── .gitignore
    ├── main.py
    ├── README.md
    └── requirements.txt



## Table of Contents

1. **Overview**  
   - Project Goals and Use Cases  
   - Motivation

2. **Directory Structure**  
   - Overview of the Folder Layout (e.g., `scripts/`, `configs/`, `main.py`, `requirements.txt`, etc.)  
   - Detailed Tree Structure of Files and Directories

3. **Configuration**  
   - Global Paths & Environment Variables  
   - Model-Specific Configurations (YAML files in `configs/energy`, `configs/no2`, etc.)  
   - Version Management and YAML Updates (via ConfigManager)

4. **Data Acquisition & Processing**  
   - **DataLoader** (`scripts/data_loader/data_loader.py`)
   - **Data Scraping**  
     - Bike Data (`scripts/utils/data_scraping/bike/data_processing.py`)
     - Energy Data (`scripts/utils/data_scraping/energy/data_processing.py`)
     - NO₂ Data (`scripts/utils/data_scraping/no2/data_processing.py`)
     - Weather and Air Quality  
       - Air Forecast (Germany and Karlsruhe)  
         - `air_germany.py`  
         - `air_karlsruhe.py`
       - Weather Forecast (Germany and Karlsruhe)  
         - `weather_germany.py`  
         - `weather_karlsruhe.py`
       - Common Scraping Logic (`scraping_class.py`)
   - Data Cleaning & Missing Value Handling  
     - `data_cleaning.py`  
     - `config_checker.py` (checks configuration files)

5. **Feature Engineering & Selection**  
   - **FeaturePipeline** (`scripts/feature_engineering/feature_pipeline.py`)  
     - Creation of Time, Lag, Exogenous, and Advanced Features  
     - Optional Fourier Terms and Holiday Features
   - **FeatureSelectionManager** (`scripts/feature_engineering/feature_selection.py`)  
     - Feature Selection via LightGBM

6. **Modeling**  
   - **Base Model** (`scripts/models/base_model.py`)
   - **Naive Baseline Model** (`scripts/models/baseline/naive_baseline.py`)
   - **Quantile Regression Model** (`scripts/models/quantile_regression/quantile_regression.py`)
   - **LightGBM Forecast Model** (`scripts/models/light_gbm/light_gbm_forecast.py`)
   - **LSTM Forecast Model** (`scripts/models/lstm/lstm_forecast.py`)

7. **Hyperparameter Optimization**  
   - **HyperparameterOptimizer** (`scripts/hyperparameter/hyperparameter_optimization.py`)  
   - Integration of Optuna into the Models

8. **Prediction & Evaluation**  
   - **Predictor** (`scripts/predictor/predictor.py`)
   - **ExtendedReportManager** (`scripts/evaluation/report_manager.py`)
   - Generation of Plots, Statistics, and Reports  
   - Evaluation Methods (e.g., residual analyses, QQ-Plots, CUSUM, etc.)

9. **Pipeline Management**  
   - **PipelineManager** (`scripts/pipeline_manager.py`)  
   - Orchestration of the Full Workflow (Feature Engineering, Training, CV, Evaluation)

10. **Logging & Utilities**  
    - Logger Configuration and Setup (`scripts/utils/logger.py`)
    - Configuration Management (`scripts/utils/configs.py`)

11. **Usage**  
    - Setup and Installation (requirements, environment variables, etc.)  
    - Running the Pipeline (`main.py`)  
    - Command-Line Options and Environment Variables

12. **Approach**  
    - Methodological Approach and System Architecture  
    - Explanation of the Forecasting Process Steps

13. **Extensibility**  
    - How to Add New Models, Feature Modules, or Evaluation Techniques  
    - Guidelines for Extending and Adapting the Pipeline


## 1. Overview

This repository implements a comprehensive forecasting pipeline designed for time series analysis. It brings together various components—from data acquisition and preprocessing to feature engineering, model training, 
hyperparameter optimization, prediction, and evaluation. The project aims to support different forecasting models such as a naive baseline, quantile regression, LightGBM, and LSTM-based models. Its modular 
design allows for easy configuration, extension, and evaluation through a set of well-defined Python modules.

**Key Use Cases and Goals:**

- **Time Series Forecasting:** The repository is geared toward generating probabilistic forecasts using multiple model types.
- **Data Integration:** It consolidates various data sources including energy, NO₂, weather, air quality, and bike data.
- **Feature Engineering:** The pipeline creates time-based, lag, exogenous, and advanced features.
- **Model Evaluation:** A rich evaluation framework is provided to analyze model performance through diverse statistical tests and plots.
- **Hyperparameter Optimization:** Integration with Optuna enables systematic tuning of model parameters.

---

## 2. Directory Structure

The repository is organized into several key directories and modules that together implement the full forecasting workflow. Below is an overview of the structure:
- **configs/**  
  Contains YAML configuration files including global default settings (e.g., `default/paths.yml`) and dataset/model-specific configurations for energy and NO₂.

- **scripts/**  
  This directory holds all the Python modules and packages that make up the forecasting workflow:
  - **data_analysis/**  
    - `deep_data_analysis.py` – Implements comprehensive EDA for time series.
  - **data_loader/**  
    - `data_loader.py` – Loads and preprocesses both main and exogenous datasets.
  - **evaluation/**  
    - `report_manager.py` – Generates detailed evaluation reports and plots.
  - **feature_engineering/**  
    - `feature_pipeline.py` – Creates various feature groups (time, lag, exogenous, advanced).
    - `feature_selection.py` – Performs feature selection (e.g., via LightGBM).
  - **hyperparameter/**  
    - `hyperparameter_optimization.py` – Uses Optuna to optimize model hyperparameters.
  - **models/**  
    Contains model implementations:
    - **baseline/**  
      - `naive_baseline.py` – Implements a naive forecasting baseline.
    - **light_gbm/**  
      - `light_gbm_forecast.py` – Implements a LightGBM model for quantile forecasting.
    - **lstm/**  
      - `lstm_forecast.py` – Implements an LSTM-based forecasting model.
    - **quantile_regression/**  
      - `quantile_regression.py` – Implements a quantile regression model.
    - `base_model.py` – Base class for forecasting models.
  - **predictor/**  
    - `predictor.py` – Loads trained models and generates forecasts.
  - **utils/**  
    Contains utility modules for:
    - **data_processing/**  
      - `config_checker.py` – Checks configuration files for required keys.
      - `data_cleaning.py` – Contains functions for missing data handling and outlier detection.
    - **data_scraping/**  
      - **bike/**: `data_processing.py` – Scrapes bike count data.
      - **energy/**: `data_processing.py` – Scrapes energy consumption data.
      - **no2/**: `data_processing.py` – Scrapes NO₂ data.
      - **weather_air/**:  
        - `air_germany.py` & `air_karlsruhe.py` – Scrape air quality forecast data.
        - `weather_germany.py` & `weather_karlsruhe.py` – Scrape weather forecast data.
        - `scraping_class.py` – Provides common functionality for data scraping.
    - `configs.py` – Manages and loads configuration files.
    - `logger.py` – Configures the logging system.

- **main.py**  
  The entry point of the forecasting pipeline.

- **requirements.txt**  
  Lists all Python dependencies required for the project.

- **temp.py**  
  A temporary script (example or test script) for running certain functionalities.

This structure supports a modular, extensible forecasting system that covers data ingestion, feature engineering, model training, hyperparameter tuning, prediction, and evaluation.


Below is an expanded description of the configuration and its attributes as defined in the repository. The repository relies on YAML files stored in directories like `configs/energy` and 
`configs/no2` to manage settings for global paths, model-specific parameters, and version control. The configuration is loaded and merged by the **ConfigManager**.

---

## 3. Configuration

The forecasting pipeline’s behavior is governed by several YAML configuration files. These files provide settings that cover global paths, model-specific hyperparameters, 
data processing, feature engineering, and training options. Below is a detailed explanation of the key configuration attributes:

### Global Paths & Environment Variables

- **Global Paths:**  
  The default global paths are specified in files like `configs/default/paths.yml`. They define key output directories for artifacts such as models, logs, predictions, plots, and results.  
  - For example, the default configuration might include keys like `"base_output"`, and subfolder names for `"logs"`, `"results"`, `"predictions"`, `"plots"`, `"models"`, and `"hyperparameter"`.  
  - The `ConfigManager` reads these files and uses the `PROJECT_ROOT` environment variable (or the current working directory if not set) to resolve absolute paths.

### Model-Specific Configurations

Each model type and dataset has its own YAML file that details parameters specific to that forecasting method. Key attributes include:

- **`dataset` and `model`:**  
  These keys indicate which dataset the pipeline is working on (e.g., `energy` or `no2`) and which model type is used (e.g., `baseline`, `light_gbm`, `quantile_regression`, or `lstm`).

- **`quantiles`:**  
  A list of quantile levels (e.g., `[0.025, 0.25, 0.5, 0.75, 0.975]`) for probabilistic forecasting. These quantiles determine how many separate models or prediction outputs will be generated.

- **`params`:**  
  This section holds model-specific hyperparameters. For example:
  - For LightGBM, you might find parameters like learning rate, number of estimators (which may be specified per quantile), and regularization parameters.
  - For LSTM models, parameters include hidden size, number of layers, dropout rate, learning rate, number of epochs, batch size, sequence length, etc.
  - For Quantile Regression, the parameters needed by the `QuantileRegressor` (such as the solver and regularization settings) are defined here.

- **`optuna_search_space`:**  
  This key defines the hyperparameter search space for tuning with Optuna. It includes ranges or lists for parameters (e.g., the minimum and maximum values for 
  `num_leaves`, `max_depth`, `learning_rate`, etc.) that can be optimized. The `optuna` section also includes flags and options such as:
  - Whether to enable Optuna (`use_optuna`),
  - The direction of optimization (minimize or maximize),
  - The number of trials (`n_trials`) and splits (`n_splits`) to perform during hyperparameter tuning.

- **`versions`:**  
  Under the `versions` key, multiple version-specific settings are provided. These settings can override or supplement the base configuration for:
  - **Time Settings:** such as `start_date`, `train_size`, `test_size`, and evaluation set configurations.
  - **Early Stopping:** Parameters like `early_stopping_rounds` to control training.
  - **Other Data Processing Options:** Options for imputation and handling missing values are also defined here.

- **`imputation_method`:**  
  This attribute determines how missing values in exogenous or target data should be handled. It specifies:
  - Whether to use a time-based interpolation (`time_cfg`),
  - K-Nearest Neighbors imputation (`knn_cfg`),
  - or Spline interpolation (`spline_cfg`).  
  Each sub-key under `imputation_method` holds method-specific parameters (e.g., interpolation method, order, and limit direction).

- **`features`:**  
  The configuration includes a nested structure to control feature engineering:
  - **`target`:**  
    Specifies settings for target variable features, for instance, which lag values to generate.
  - **`time_features`:**  
    Lists which time-based features to create (e.g., hour, weekday, month, weekend flag, Fourier terms).
  - **`fourier_terms`:**  
    A Boolean flag indicating whether to add Fourier transformation terms to capture cyclic patterns.
  - **`exogenous`:**  
    This section covers the creation of exogenous features. It includes:
    - A list of base exogenous features.
    - Transformations for these features such as rolling statistics (windows, statistics like mean, median, etc.) and difference calculations.
  - **`advanced`:**  
    Defines more complex feature engineering steps such as:
    - **Holiday Features:** Whether to add a binary indicator for holidays and possibly compute proximity (days since or until a holiday).
    - **Interactions:** Definitions for creating interaction terms between specified feature pairs.
    - **Rolling Moments:** Options to calculate higher moments (skewness, kurtosis) over rolling windows for selected features.
  - **`normalization`:**  
    Under the `features` key, a sub-section called `normalization` is provided. It specifies whether normalization should be applied to each feature group (base, time, lag, 
	exog, advanced) and which method to use (e.g., standard scaling, min-max scaling, or robust scaling).

### Version Management and YAML Updates

- **Versioning via `versions`:**  
  The configuration files include a `versions` attribute that holds multiple version settings. This allows the pipeline to run experiments under different parameter configurations 
  while tracking changes over time.

- **ConfigManager Responsibilities:**  
  The `ConfigManager` (in `scripts/utils/configs.py`) is responsible for:
  - **Loading** the appropriate YAML files based on the dataset and model name.
  - **Merging** model-specific configurations with global default settings (such as paths from the `default` folder).
  - **Updating YAML Files:**  
    When hyperparameter tuning is performed (e.g., via Optuna), the best parameters can be merged back into the configuration file. This update is done while preserving the original YAML formatting using `ruamel.yaml`.

By centralizing configuration in this way, the repository ensures reproducibility and consistency across different forecasting experiments. Each model’s behavior can be finely controlled by 
modifying its corresponding YAML file without altering the source code.

---

Below is a detailed description of the “Data Acquisition & Processing” section, which outlines the modules responsible for loading, scraping, and cleaning the data used in the repository:

---

## 4. Data Acquisition & Processing

The repository implements a comprehensive data ingestion and preprocessing strategy. It is divided into two main parts: loading existing datasets via a dedicated **DataLoader** 
and retrieving external data using various data scraping modules. In addition, utilities for cleaning data and checking configuration consistency are provided.

### DataLoader

- **Purpose:**  
  The `DataLoader` class (located in `scripts/data_loader/data_loader.py`) is designed to load the main dataset as well as the exogenous (external) data from CSV files. It:
  - Checks for file existence and reads the main data (e.g., energy consumption data) by parsing the CSV file.
  - Converts the index into a UTC-based datetime format and resamples the data on an hourly basis.
  - Loads exogenous data from multiple sources (for example, weather and air quality) based on a list of file paths.
  - Merges the main data and the exogenous data by aligning their datetime indexes.
  - Applies preprocessing steps such as filtering the data based on a configured start date and imputation of missing values.
- **Key Attributes:**  
  The configuration passed to `DataLoader` includes:
  - **`dataset` key:** Identifies which dataset configuration to use.
  - **`data_file`:** Path to the main dataset.
  - **`exogenous_files`:** A list (usually four files) that include weather history, weather forecast, air history, and air forecast.
  - **`start_date`:** A filter to ensure that only data from a specific starting date is used.
  - **`features` & `imputation_method`:** Additional settings which control which exogenous features to include and the method to handle missing data (time-based interpolation, KNN, or spline).

### Data Scraping

In addition to loading pre-stored data, the repository also actively retrieves external data via dedicated data scraping modules. These modules are organized under 
`scripts/utils/data_scraping/` and are responsible for gathering various types of data from online APIs.

#### Bike Data

- **Location:** `scripts/utils/data_scraping/bike/data_processing.py`
- **Functionality:**  
  The module defines a function (`get_bike_data`) that retrieves bike count data from an API provided by eco-visio. It:
  - Downloads bike count data starting from a fixed date (e.g., January 1, 2013).
  - Converts date strings to datetime objects and creates an hourly counter for each day.
  - Constructs a continuous hourly time series, filling missing hours with NaN values.
  - Saves the resulting DataFrame to `data/raw/bike/bike_data.csv`.

#### Energy Data

- **Location:** `scripts/utils/data_scraping/energy/data_processing.py`
- **Functionality:**  
  This module handles data acquisition from the SMARD.de API:
  - It first retrieves a list of available Unix timestamps.
  - Checks for an existing CSV file to determine whether new data needs to be downloaded.
  - Downloads energy consumption data in chunks using the provided timestamps.
  - Converts Unix timestamps into a human-readable datetime format.
  - Adjusts the timezone from local (CET/CEST) to UTC.
  - Cleans the data by dropping missing values, renaming columns (e.g., renaming `"Netzlast_Gesamt"` to `"gesamt"`), and scaling the values.
  - Finally, it appends the new data to the existing dataset or creates a new CSV file under `data/raw/energy/energy_data.csv`.

#### NO₂ Data

- **Location:** `scripts/utils/data_scraping/no2/data_processing.py`
- **Functionality:**  
  The NO₂ module is responsible for downloading air pollution data from the Umweltbundesamt API:
  - It downloads the raw CSV data, cleans the data by removing the last row and replacing placeholder characters (like '-') with NaN.
  - Converts the measurement values to floats.
  - Extracts the hour from the `Time` column (converting 24 to 0) and creates a combined datetime column from the `Date` and `hour` columns.
  - Adjusts timestamps for late-night entries and handles daylight saving time transitions (both for the fall and spring shifts).
  - Converts the local datetime to UTC and interpolates missing measurement values.
  - The processed data is saved to `data/raw/no2/no2_data.csv`.

#### Weather and Air Quality Data

Weather and air quality data are scraped from the Open-Meteo API. The logic is separated by geography (Germany and Karlsruhe) and by type (air forecast and weather forecast):

- **Air Forecast**  
  These modules gather air quality forecast data, including measurements such as PM10, PM2.5, CO, CO₂, SO₂, ozone, dust, and ammonia.
  - **For Germany:**  
    - **`air_germany.py`:**  
      Retrieves and processes air quality forecast data for multiple stations across Germany. It sets up an API client with caching (using `requests_cache`) and retry mechanisms, 
	  tracks API call limits, and saves the raw hourly data as CSV files.
  - **For Karlsruhe:**  
    - **`air_karlsruhe.py`:**  
      Similar to the German module, but focused on stations in Karlsruhe. It uses similar API interactions and rate-limit handling to download and store data.

- **Weather Forecast**  
  These modules download meteorological forecast data including temperature, humidity, precipitation, pressure, cloud cover, radiation, wind, and soil measurements.
  - **For Germany:**  
    - **`weather_germany.py`:**  
      Collects weather forecast data across Germany. The module defines detailed parameters for the API call, processes the returned variables, and creates a DataFrame that is 
	  eventually saved as a temporary forecast file.
  - **For Karlsruhe:**  
    - **`weather_karlsruhe.py`:**  
      Handles the weather forecast for Karlsruhe in a similar manner to the German module but with a smaller set of station coordinates and possibly adjusted API call limits.
  
- **Common Scraping Logic:**  
  - **`scraping_class.py`:**  
    This module centralizes the data scraping process by invoking the appropriate classes for weather, air, bike, energy, and NO₂ data. It provides functions to update target 
	data and environmental data (both weather and air quality), ensuring that the external data sources are refreshed and stored in the expected locations.

### Data Cleaning & Missing Value Handling

Data cleaning is essential to prepare raw data for analysis and modeling. The repository provides utility modules for this purpose:

- **`data_cleaning.py`:**  
  Contains functions to check for missing values and handle them. Key functions include:
  - **`check_missing_data`:**  
    Scans the provided DataFrame and returns a report of missing values along with the percentage of missing data per column.
  - **`handle_missing_data`:**  
    Offers two methods—dropping rows with missing data or imputing missing values (for example, by filling with the mean of each column).

- **`config_checker.py`:**  
  While primarily focused on verifying YAML configuration files, this module also plays a role in ensuring that the configuration for data processing is complete and correct.
  - It defines the required schema for configuration files.
  - Checks each configuration file (located in folders such as `configs/energy` and `configs/no2`) to ensure that all necessary keys (like `dataset`, `model`, `quantiles`, etc.) are present.
  - Provides feedback (errors or confirmations) which help maintain consistency and prevent misconfiguration in data processing steps.

---

In summary, the **Data Acquisition & Processing** section of the repository is built on a modular design where:
- **DataLoader** is responsible for loading and merging local datasets.
- **Data Scraping** modules automatically retrieve up-to-date external data for bikes, energy consumption, NO₂ levels, weather forecasts, and air quality forecasts.
- Utility modules like **data_cleaning.py** and **config_checker.py** ensure that the ingested data is clean and that configuration files adhere to expected structures.

All these components work together to ensure that the forecasting pipeline starts with accurate, comprehensive, and up-to-date data.




Below is a detailed description for the “Feature Engineering & Selection” section of the repository:

---

## 5. Feature Engineering & Selection

This part of the repository is responsible for transforming raw and scraped data into a set of useful features for modeling. It is divided into two primary components:

### FeaturePipeline  
*Location: `scripts/feature_engineering/feature_pipeline.py`*

- **Purpose:**  
  The `FeaturePipeline` class handles the end‐to‐end creation of engineered features from the loaded and merged data. Its main goal is to produce a refined DataFrame that contains all the information needed for forecasting.

- **Key Responsibilities:**  
  - **Creation of Time-Based Features:**  
    Extracts features such as hour, weekday, month, and flags for weekends directly from the datetime index. These features capture periodicities and diurnal or seasonal patterns in the data.
    
  - **Lag Features:**  
    Based on a configurable list of lag intervals (defined under the target configuration in the YAML files), the pipeline creates lagged versions of the target variable. These lag features 
	help capture temporal dependencies and autocorrelation in time series data.
    
  - **Exogenous Features:**  
    Processes external variables (e.g., weather, air quality) by applying transformations. For example, it computes rolling statistics (mean, standard deviation, quantiles) and 
	differences for selected exogenous columns. This enhances the predictive power of these variables by providing additional context such as recent trends.
    
  - **Advanced Features:**  
    In addition to the basic groups, advanced features can be generated if enabled in the configuration. This includes:
    - **Holiday Features:**  
      The pipeline can automatically add a binary indicator for holidays and, optionally, features indicating proximity to a holiday (e.g., days until/after a holiday). 
	  This leverages local holiday calendars to capture effects related to special events.
    - **Interactions:**  
      It allows for the creation of interaction features by multiplying pairs of existing features.
    - **Rolling Moments:**  
      The pipeline can compute higher-order moments (such as skewness and kurtosis) over a rolling window for selected variables.
      
  - **Optional Fourier Terms:**  
    When enabled, the pipeline adds Fourier transformation terms for cyclical features (such as hour or weekday). These sine and cosine components help the model learn cyclical patterns in a continuous manner.
  
- **Configuration Integration:**  
  The behavior of the `FeaturePipeline` is driven by the YAML configuration files, which specify:
  - Which features to create (time features, lag features, exogenous transformations, advanced features).
  - The normalization settings for each feature group.
  - Optional parameters (e.g., the use of Fourier terms, definitions of holiday features, and interaction pairs).

- **Output:**  
  After processing, the pipeline outputs an engineered dataset that is split into training and testing subsets, ready for modeling. 
  It also saves a CSV file (`engineered_features.csv`) containing the final feature set.

---

### FeatureSelectionManager  
*Location: `scripts/feature_engineering/feature_selection.py`*

- **Purpose:**  
  The `FeatureSelectionManager` focuses on reducing the feature space by selecting the most important predictors. This step is critical to avoid 
  overfitting, reduce model complexity, and improve overall performance.

- **Key Responsibilities:**  
  - **Selection via LightGBM:**  
    It leverages LightGBM's ability to compute feature importance. The module trains several LightGBM models on random subsets of the available features 
	(excluding a set of mandatory ones that are always required, such as time indicators).  
  - **Parallel Model Training:**  
    The process runs multiple models (the number is configurable, for example, eight models by default) in parallel on different subsets of the non-mandatory features.
  - **Ranking and Aggregation:**  
    Each model provides an importance score (based on gain) for the features used. The manager then selects the top‑*n* features (with *n* configurable) from each 
	model and aggregates them into a final set of selected features.
  - **Output:**  
    The final set of selected features is written to a CSV file (commonly named `selected_features.csv`). These features can then be used in subsequent modeling steps.

- **Configuration Integration:**  
  The feature selection process is also controlled via the repository’s YAML configurations. The configuration defines:
  - The number of top features to select per model.
  - Which features are mandatory (e.g., core time features like hour, weekday, or month that should not be dropped).
  - Additional LightGBM parameters that guide the training of each model during the selection process.

---

In summary, the **Feature Engineering & Selection** part of the repository first enriches the raw data using the `FeaturePipeline` by creating time, lag, exogenous, 
and advanced features (with optional Fourier terms and holiday indicators). Then, the `FeatureSelectionManager` refines this feature set using a
LightGBM-based selection process to identify and retain only the most predictive features. Both components are highly configurable via YAML files in the repository 
(located in directories such as `configs/energy` and `configs/no2`), ensuring that the feature creation and selection processes are model- and dataset-specific.


Below is a comprehensive, detailed explanation of the modeling components in the repository:

---

## 6. Modeling

The repository implements several forecasting models, each designed to predict probabilistic outcomes (i.e. quantiles) for time series data. All forecasting models extend a common abstract base class that provides a standard interface and shared functionality. Below are the detailed descriptions of each modeling module:

---

### **Base Model**  
*Location: `scripts/models/base_model.py`*

- **Purpose & Design:**  
  The abstract `BaseModel` class serves as a foundation for all forecasting models in the repo. It defines the expected interface and common helper methods that concrete implementations must follow. This ensures consistency across different model types.

- **Key Methods & Attributes:**  
  - **Abstract Methods:**  
    - `train(X_train, y_train, config=None)`: Must be implemented to train the model on provided training data.  
    - `get_optuna_search_space()`: Defines the hyperparameter search space for Optuna-based optimization.  
    - `evaluate_trial(trial, X_train, y_train)`: Used during hyperparameter tuning to evaluate a given set of trial parameters.
  - **Model Persistence:**  
    - `save_model(model_path, model_data)`: Saves model data (e.g. state dictionaries or serialized object) to disk using joblib.  
    - `load_model(model_path)`: Loads and returns a saved model from disk.
  - **Utility Function:**  
    - A static method `is_in_time_window(ts)` checks if a given timestamp falls within a designated time window (e.g., to support time-based forecasting constraints).
  - **Hyperparameter Optimization Support:**  
    - The class also provides the `optimize_hyperparameters` method that leverages a separate hyperparameter optimizer (using Optuna) to fine-tune model parameters.

- **Role in the Pipeline:**  
  Every forecasting model inherits from `BaseModel`, ensuring that training, evaluation, optimization, and persistence are handled in a consistent manner.

---

### **Naive Baseline Model**  
*Location: `scripts/models/baseline/naive_baseline.py`*

- **Purpose & Approach:**  
  The `NaiveBaselineModel` implements a simple, non-parametric baseline forecast. As a model that requires no training, it relies on a straightforward strategy to generate predictions based on historical observations.

- **Key Characteristics:**  
  - **No Training Required:**  
    The `train()` method simply logs that no training is necessary.  
  - **Prediction Logic:**  
    - In the `predict()` method, the model first identifies a starting point (e.g. the first Wednesday at or after 18:00) from the test dataset.
    - It then merges the training and test targets and initially sets predictions equal to the observed values.
    - For a designated time window (for example, Wednesday from 23:00 onward, entire Thursday and Friday, and part of Saturday), predictions are explicitly set to NaN.
    - Finally, rolling quantile values are computed (based on historical data grouped by weekday and hour) to fill in the missing predictions. This mechanism enables the generation of probabilistic forecasts for different quantiles.
  - **Output:**  
    The output DataFrame includes a median forecast along with additional quantile columns (e.g. 0.025, 0.25, 0.75, 0.975).

- **Role in the Pipeline:**  
  It provides a simple benchmark against which more complex models (e.g. LightGBM, LSTM) can be compared.

---

### **Quantile Regression Model**  
*Location: `scripts/models/quantile_regression/quantile_regression.py`*

- **Purpose & Approach:**  
  The `QuantileRegressionModel` implements forecasting using classical linear quantile regression. It trains separate models for each quantile specified in the configuration.

- **Key Characteristics:**  
  - **Model Training:**  
    For every quantile (e.g. 0.025, 0.25, 0.5, etc.), the module instantiates a `QuantileRegressor` from scikit‑learn with the quantile parameter set appropriately.  
  - **Parallel Processing:**  
    It employs parallel processing (via `joblib.Parallel`) to train multiple quantile regressors concurrently, speeding up the feature selection and training process.
  - **Training Information Collection:**  
    The module collects a variety of metadata during training, including:  
    - Hyperparameters used in the regression model.  
    - Training data summary statistics (e.g., mean, standard deviation, quantiles).  
    - Performance metrics such as MAE, RMSE, and MAPE computed on the training data.
    - Since linear models do not inherently provide feature importances like tree-based models, the model uses the absolute values of the regression coefficients as a measure of feature importance.
  - **Model Persistence:**  
    Each quantile model is saved to disk (typically named with the quantile in the filename) along with its training information for later use during prediction and evaluation.

- **Role in the Pipeline:**  
  This model provides a more robust, statistically grounded approach compared to the naive baseline, offering interpretable coefficient values and serving as an alternative forecasting method.

---

### **LightGBM Forecast Model**  
*Location: `scripts/models/light_gbm/light_gbm_forecast.py`*

- **Purpose & Approach:**  
  The `LightGBMForecastModel` leverages the LightGBM framework to train gradient boosting models for quantile regression. A separate LightGBM model is trained for each quantile, using LightGBM’s built‑in support for the "quantile" objective.

- **Key Characteristics:**  
  - **Model Training:**  
    - The training method splits the data into training and validation sets based on a configurable evaluation size.
    - For each quantile, the module sets the LightGBM parameter `alpha` to the desired quantile value and adjusts the number of estimators accordingly.
    - The model is trained using early stopping callbacks based on validation loss (quantile loss is used as the metric).
  - **Training Information Collection:**  
    The module gathers comprehensive training details including:  
    - The evaluation history (learning curves) stored in `evals_result_`.  
    - The best iteration (if early stopping is triggered) and associated best score.  
    - Feature importances (gain-based) extracted from the underlying booster.
    - Additional training statistics such as training duration, and summary statistics of the training and validation data.
  - **Hyperparameter Optimization:**  
    The model supports integration with Optuna, allowing for automated hyperparameter tuning using a defined search space.
  - **Model Persistence:**  
    After training, each quantile model is saved (using joblib) along with its training metadata for later use.

- **Role in the Pipeline:**  
  As a more advanced and competitive forecasting approach, the LightGBM model typically serves as a primary model for production forecasts, benefiting from gradient boosting’s ability to capture complex patterns and interactions.

---

### **LSTM Forecast Model**  
*Location: `scripts/models/lstm/lstm_forecast.py`*

- **Purpose & Approach:**  
  The `LSTMForecastModel` implements a forecasting model based on Long Short-Term Memory (LSTM) neural networks. This model is designed to capture temporal dependencies in time series data using deep learning.

- **Key Characteristics:**  
  - **Neural Network Architecture:**  
    - The model uses PyTorch to build the LSTM network.  
    - It includes an LSTM layer (or stacked LSTM layers) for processing sequential data, followed by a fully connected (FC) layer that outputs predictions for all specified quantiles simultaneously.
  - **Data Preparation:**  
    - The module implements a sliding window (or sequence) approach to convert the raw time series data into sequences suitable for LSTM input.
    - A helper function `create_lstm_windows` extracts windows of length `seq_length` from the data and prepares corresponding target values (supporting one-step or multi-step forecasting).
  - **Training Loop:**  
    - The training method includes a typical deep learning training loop with mini-batch processing via DataLoader.
    - It computes a custom quantile loss function based on the pinball loss, which penalizes under‑ and over‑predictions according to the desired quantile.
    - Early stopping is implemented based on validation loss, with options for gradient clipping to stabilize training.
  - **Iterative Prediction:**  
    - During prediction, the model operates in an iterative fashion: starting from an initial sequence (extracted from the tail of the training data), it slides the window forward one time step at a time, updating the input sequence with the predicted values as necessary.
  - **Hyperparameter Optimization:**  
    - Similar to the LightGBM model, the LSTM model integrates with Optuna for hyperparameter tuning. The search space can include parameters such as hidden size, number of layers, dropout, learning rate, batch size, and epochs.
  - **Model Persistence:**  
    - The model’s state dictionaries (for both the LSTM and FC layers) are saved as a PyTorch checkpoint (typically a `.pth` file). Along with the model, extensive training information is stored, including learning curves, training duration, and configuration details.

- **Role in the Pipeline:**  
  The LSTM model represents the deep learning approach within the repository. It is especially suited for capturing complex, non-linear temporal dynamics in the data and provides a flexible framework that can be further fine-tuned using hyperparameter optimization.

---

Each of these modeling components is fully integrated into the pipeline and driven by the repository’s configuration files. They can be used independently or compared against one another as part of a 
comprehensive forecasting strategy. The modular design enables quick switching between methods, experimentation with different algorithms, and clear logging and persistence of model training and evaluation details.


Below is a detailed description for the Hyperparameter Optimization section based on the repository:

---

## 7. Hyperparameter Optimization

The repository integrates hyperparameter optimization using the Optuna framework. This component is designed to fine-tune model-specific parameters automatically by running multiple trials and selecting the best configuration based on a defined loss metric (typically the quantile loss). The main components of this optimization framework are described below:

### **HyperparameterOptimizer**  
*Location: `scripts/hyperparameter/hyperparameter_optimization.py`*

- **Purpose & Role:**  
  The `HyperparameterOptimizer` class serves as a centralized engine for hyperparameter tuning across various forecasting models. It encapsulates the logic needed to define a search space, run multiple trials with Optuna, log detailed trial outcomes, and generate visualizations that summarize the optimization process.

- **Key Features:**  
  - **Objective Function Definition:**  
    The optimizer defines an `objective` function that each Optuna trial will run. For each trial, the function:
    - Retrieves hyperparameter suggestions from the search space defined in the configuration.
    - Invokes a model-specific evaluation routine (via the model’s `evaluate_trial` method) using the current trial’s parameters on a training set (and optionally using cross‑validation splits).
    - Computes and returns a loss value (e.g., mean pinball loss) that Optuna uses to rank and compare trials.
  - **Logging and Tracking:**  
    Detailed trial information—including trial number, chosen hyperparameters, and loss—is written to a dedicated log file. This ensures transparency and reproducibility in the hyperparameter tuning process.
  - **Visualization:**  
    Upon completion of the study, the optimizer generates multiple plots using Optuna’s visualization tools. These plots include:
    - Optimization history (showing how loss values change over trials)
    - Intermediate value plots (tracking evaluation metrics per trial)
    - Slice and contour plots (to explore the impact of individual hyperparameters)
    - Parallel coordinate and parameter importance plots (to assess interactions among parameters)
  - **Integration with Models:**  
    The best hyperparameters found by the optimizer are saved to disk as a JSON file. In addition, the optimizer’s results are integrated back into the model configuration so that the model instances (e.g., LightGBM or LSTM) can be retrained with the optimal settings.
  - **Model-Specific Adjustments:**  
    In some cases, the optimizer makes additional adjustments to the best parameters based on the model type. For example, the LightGBM module may merge the best-found estimator value with preset quantile-specific parameters.

- **Usage in the Pipeline:**  
  Each forecasting model (e.g., LightGBM, LSTM, or Quantile Regression) implements its own version of `evaluate_trial()`—a method defined in the `BaseModel` interface. The `HyperparameterOptimizer` calls this method during each trial to determine the loss for the proposed hyperparameter configuration. Once the optimization completes:
  - The best parameters are merged into the overall configuration.
  - The YAML configuration files may be updated with these new parameters (using the ConfigManager’s update functionality).
  - The model is re-instantiated with the optimized settings and retrained for final evaluation.

### **Integration of Optuna into the Models**

- **Standard Interface via BaseModel:**  
  All forecasting models derive from the abstract `BaseModel` class, which specifies the `evaluate_trial()` and `get_optuna_search_space()` methods. This design ensures that the Optuna integration is consistent across models.

- **Model-Specific Implementations:**  
  - **LightGBM Forecast Model:**  
    In the LightGBM model module, the `evaluate_trial()` method trains the model on cross‑validated splits using the hyperparameters suggested by Optuna. The quantile loss (pinball loss) is computed for each fold, and the average loss across folds is returned as the objective value.
  - **LSTM Forecast Model:**  
    The LSTM model’s integration with Optuna is similar. It builds a neural network based on hyperparameter suggestions (such as hidden size, number of layers, dropout, learning rate, etc.) and uses an iterative training loop with early stopping. The average quantile loss from the validation sets is returned to Optuna.
  - **Quantile Regression Model:**  
    The Quantile Regression model also implements its own version of `evaluate_trial()`, where for a specified quantile (typically the median, 0.5), it performs cross-validation and computes the pinball loss.

- **Overall Workflow:**  
  When hyperparameter optimization is enabled in the configuration, the pipeline manager checks for an existing best‑parameter file (e.g. `optuna_best_params.json`). If the optimization has not been performed yet:
  - The `HyperparameterOptimizer` is invoked with the training data.
  - Multiple trials are run in parallel (as defined by `n_trials` and `n_splits` in the configuration).
  - The best hyperparameters are determined, logged, and saved.
  - The best parameters are merged into the model’s configuration, and the model is re‑instantiated for training with these optimal settings.

This tightly integrated hyperparameter optimization framework ensures that each model is tuned effectively for the forecasting task, thereby improving the overall predictive performance of the pipeline.

---

## 8. Prediction & Evaluation

This section of the repository is responsible for generating forecasts from trained models and evaluating their performance through a variety of analyses and visualizations. The two key components in this stage are the **Predictor** and the **ExtendedReportManager**.

### **Predictor**  
*Location: `scripts/predictor/predictor.py`*

- **Purpose & Role:**  
  The Predictor class acts as the interface to generate forecasts from the different model types (e.g., LSTM, LightGBM, Quantile Regression, or even the Naive Baseline). It is designed to handle both iterative prediction (especially for models that require sequential processing, like the LSTM) and direct prediction (as in the case of quantile regression models).

- **Key Functionalities:**  
  - **Model Loading:**  
    The Predictor dynamically loads the appropriate forecasting model from disk. For quantile regression, it loads a model specific to each quantile (for example, `*_model_0.5.joblib` for the median forecast). For LSTM-based forecasting, it loads the PyTorch checkpoint that contains both the LSTM network and the associated fully connected layer along with the training metadata.
  - **Iterative Forecasting:**  
    Depending on the model type, the Predictor implements iterative forecasting routines.  
    - For the LSTM model, the predictor extracts a sliding window from the test set (after concatenating with recent training data) and iteratively feeds it into the model. This approach enables the model to generate forecasts sequentially over the desired forecast horizon.
    - For quantile regression models, an iterative process is also employed to fill in missing prediction values based on the computed rolling quantiles, especially during specific time windows where predictions are purposefully set to NaN.
  - **Output Generation:**  
    After processing the input data, the Predictor outputs a unified DataFrame that includes the forecasted target values for all the specified quantiles. The DataFrame index is usually a datetime index (with an optional column for the original time) and includes columns such as `gesamt_pred_0.025`, `gesamt_pred_0.25`, `gesamt_pred_0.5`, `gesamt_pred_0.75`, and `gesamt_pred_0.975`.

### **ExtendedReportManager**  
*Location: `scripts/evaluation/report_manager.py`*

- **Purpose & Role:**  
  The ExtendedReportManager is responsible for the comprehensive evaluation of forecast performance. It consolidates various diagnostic and evaluation tests, produces visualizations, and generates a summary report for the model’s performance on the test data.

- **Key Functionalities:**  
  - **Diagnostic Plots:**  
    The manager produces a variety of plots that help to assess different aspects of the model's performance. These include:  
    - **Time Series Plots:** Comparing actual versus forecast values, often including a 95% prediction interval.
    - **Residual Analysis:** Plotting residuals over time and providing histograms of residuals to detect any patterns or biases.
    - **ACF/PACF Plots:** To analyze the autocorrelation properties of the residuals.
    - **QQ-Plots:** To assess the normality of the residual distribution by comparing quantiles against a theoretical normal distribution.
    - **CUSUM Plots:** To detect structural breaks or systematic biases over time by visualizing the cumulative sum of forecast errors.
    - **Calibration and PIT Histograms:** These help in checking the calibration of the probabilistic forecasts by comparing nominal coverage with empirical coverage.
  - **Statistical Tests and Metrics:**  
    The ExtendedReportManager runs several statistical tests and computes performance metrics. These include:
    - **Stationarity Tests:** Augmented Dickey–Fuller (ADF) and KPSS tests.
    - **Normality Tests:** Shapiro–Wilk, D’Agostino’s K², Jarque–Bera, and Anderson–Darling tests.
    - **Heteroskedasticity Tests:** Breusch–Pagan, Goldfeld–Quandt, and ARCH tests.
    - **Multicollinearity Diagnostics:** Correlation matrix heatmaps and Variance Inflation Factor (VIF) calculations.
    - **Model Comparison:** ANOVA for comparing nested models.
  - **Report Generation:**  
    After completing all evaluations, the manager aggregates all findings—including test results, plots, and diagnostic statistics—into a final text report (typically saved as `eda_report.txt` or similar) and organizes detailed CSV files and PNG files in their respective output directories. This comprehensive report facilitates quick identification of model strengths and weaknesses and assists in subsequent model tuning or refinement.

### **Overall Workflow in Prediction & Evaluation**

1. **Forecast Generation:**  
   The Predictor is invoked once the model has been trained. It processes the test dataset, handles any necessary iterative prediction steps, and produces forecasts for all desired quantiles.

2. **Evaluation:**  
   Once the forecasts are generated, the ExtendedReportManager takes over to perform an in-depth evaluation of the results.  
   - It runs various diagnostic tests and statistical evaluations on the residuals and forecasted values.
   - It generates multiple plots to visualize model performance and error characteristics.
   - It compiles all outputs into a detailed report along with accompanying CSV and image files.

3. **Integration:**  
   The evaluation outputs, including visualizations and statistics, are saved in the output directories specified in the configuration. This allows for easy review and further analysis of model performance.

Together, these components form a robust framework for making forecasts and rigorously evaluating them, ensuring that the repository not only produces predictions but also provides comprehensive insights into model behavior and forecast quality.

---



Below is a detailed description for chapters **9. Pipeline Management** and **10. Logging & Utilities**, based on the provided repository.

---

## 9. Pipeline Management

**Component:** **PipelineManager**  
*Location: `scripts/pipeline_manager.py`*

- **Purpose & Role:**  
  The PipelineManager orchestrates the entire forecasting workflow—from feature engineering through model training (including cross‑validation and optional hyperparameter tuning) to prediction and final evaluation. It is the central coordinator that brings together all the individual components in the repository.

- **Key Responsibilities:**  
  - **Workflow Coordination:**  
    The PipelineManager is responsible for executing sequential stages of the workflow:
    - **Feature Engineering:**  
      It instantiates and runs the FeaturePipeline (from `scripts/feature_engineering/feature_pipeline.py`) to load data, create engineered features (time, lag, exogenous, and advanced features), and perform data splitting into training and test sets.
    - **Optional Feature Selection:**  
      Depending on the configuration, it can invoke a feature selection process (using the FeatureSelectionManager) to narrow down the most relevant predictors before training.
    - **Training Mode:**  
      It supports different training modes such as a simple train/test split or rolling cross‑validation. In the rolling CV mode, the manager expands the training window and moves the test window forward iteratively.
    - **Model Training & Prediction:**  
      Once the data is prepared, the PipelineManager calls the appropriate model’s training method (e.g., for baseline, LightGBM, quantile regression, or LSTM models). After training, it uses the Predictor (from `scripts/predictor/predictor.py`) to generate forecasts.
    - **Evaluation:**  
      Finally, it leverages the ExtendedReportManager (from `scripts/evaluation/report_manager.py`) to run a suite of evaluation tests and generate diagnostic plots and summary reports.
  - **Configuration Integration:**  
    The PipelineManager receives the merged configuration (including model-specific parameters, version settings, and global paths) from the ConfigManager. This configuration directs every step of the pipeline.
  - **Result Organization:**  
    It ensures that outputs such as models, predictions, plots, and logs are saved in their respective directories (for example, in subdirectories like `models/`, `plots/`, `predictions/`, and `stats/` under the designated output directory).

- **How It Works:**  
  Upon invocation, the PipelineManager:
  1. Calls the FeaturePipeline to load and process data.
  2. Retrieves training and test datasets.
  3. Depending on the training mode (simple or rolling CV), it either trains a single model or iteratively trains models on different splits.
  4. If hyperparameter tuning is enabled via Optuna, it triggers the optimization process.
  5. Uses the Predictor to generate forecasts from the test data.
  6. Calls the ExtendedReportManager to perform evaluation tasks and compile a detailed report.
  7. Finally, it writes a marker file (e.g., `pipeline_finished.txt`) in the output directory to indicate successful pipeline completion.

---

## 10. Logging & Utilities

This section includes utility modules that provide critical support for the repository, such as logging configuration and configuration file management.

### **Logger Configuration and Setup**  
*Location: `scripts/utils/logger.py`*

- **Purpose & Role:**  
  The logging utility is designed to provide consistent and detailed logging throughout the entire codebase. It sets up both console logging and file logging using a rotating file handler.

- **Key Features:**  
  - **Rotating File Handler:**  
    The module configures a rotating file handler that writes logs to a file (default name: `app.log`) located in a designated `logs/` directory. The handler is set to rotate once the log file reaches a specified maximum size (e.g., 10 MB) and keeps a limited number of backup log files.
  - **Log Formatting:**  
    A consistent log format (timestamp, log level, and message) is defined so that log entries are easily traceable.
  - **Global Logger Configuration:**  
    The module initializes the root logger to include both a console output (via `basicConfig`) and the file handler. This setup ensures that all parts of the repository that use Python’s logging module automatically follow the established configuration.
  - **Module-Level Logger:**  
    It creates a module-level logger (using `__name__`) that other modules can import and use. This centralizes logging configuration and makes it easier to manage log output throughout the entire project.

### **Configuration Management**  
*Location: `scripts/utils/configs.py`*

- **Purpose & Role:**  
  The configuration management module is responsible for loading, merging, and updating YAML configuration files. It provides a consistent approach to handling configuration data across the repository.

- **Key Features:**  
  - **Loading YAML Configurations:**  
    The ConfigManager uses Python’s `ruamel.yaml` library to load YAML files. This library is chosen because it preserves formatting and comments, making the configuration files easier to maintain and update.
  - **Merging Configurations:**  
    The module is designed to merge model-specific configurations (found in directories such as `configs/energy` or `configs/no2`) with global path configurations (found in `configs/default`). This merge results in a unified configuration that covers both global settings and model-specific parameters.
  - **Version Control Integration:**  
    The configurations include a version key and a `versions` dictionary. The ConfigManager supports version management by merging version-specific settings into the base configuration. It also provides functionality to update parameters in the YAML files while preserving formatting. This is essential for iterative development and model tuning.
  - **Cache & Reusability:**  
    The ConfigManager caches loaded configurations so that subsequent requests for the same configuration do not require re‑reading from disk. This speeds up the pipeline initialization.
  - **Attribute Details:**  
    The configuration attributes in the YAML files include keys such as:
    - **Global Paths & Save Directories:**  
      These define where outputs, models, predictions, and logs should be stored.
    - **Model-Specific Parameters:**  
      Each model configuration includes hyperparameters, data splitting parameters (like train/test sizes), and evaluation settings.
    - **Optuna Settings:**  
      If hyperparameter optimization is enabled, configuration entries for Optuna (such as `n_trials`, `n_splits`, and the search space under `optuna_search_space`) are provided.
    - **Feature Engineering & Preprocessing:**  
      Attributes related to feature transformations (e.g., normalization methods, selected features, time feature options) are also specified.
  - **Updating YAML Files:**  
    In addition to loading, the ConfigManager has a utility to update YAML configuration files (preserving comments and structure) when new best parameters are obtained via hyperparameter optimization. This ensures that configurations stay in sync with the latest model settings.

---


Below is a detailed description for chapters **11. Usage**, **12. Approach**, and **13. Extensibility**. These sections explain how to set up and run the forecasting pipeline, describe the overall methodological approach of the system, and provide guidance on extending the repository with new components—all based on the provided repository structure and content.

---

## 11. Usage

**Setup and Installation**

- **Requirements:**  
  The repository includes a `requirements.txt` file that lists all the necessary Python packages. Before running the pipeline, create a virtual environment and install these dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  
- **Environment Variables:**  
  The pipeline makes use of an environment variable named `PROJECT_ROOT` to determine the base path for the project. If this variable is not set, the code defaults to the current working directory.  
  Make sure to set it if needed:
  ```bash
  export PROJECT_ROOT=/path/to/your/project
  ```
  
- **Configuration Files:**  
  Configuration is managed via YAML files stored in folders such as `configs/energy`, `configs/no2`, and `configs/default`. These files contain global paths, model-specific parameters (like hyperparameters, feature options, data splitting settings, and Optuna search spaces), and version information. The `ConfigManager` module (`scripts/utils/configs.py`) handles loading and merging these configurations.

**Running the Pipeline**

- The pipeline is executed using the `main.py` script. This is the entry point that coordinates data scraping, feature engineering, model training, prediction, and evaluation.
  
- To run the entire pipeline, simply execute:
  ```bash
  python main.py
  ```
  
- **Command-Line Options & Environment Variables:**  
  While `main.py` is the primary entry point, various aspects of the pipeline are driven by the configuration YAML files. If command‑line arguments or additional environment variables are supported, they are used to override or extend the configurations (e.g., selecting which dataset or model to run). The repository documentation and comments within `main.py` detail these options.

---

## 12. Approach

**Methodological Approach and System Architecture**

- **Modular Design:**  
  The repository is organized into modules that each address a specific aspect of the forecasting pipeline. The design separates concerns such as data acquisition, feature engineering, model training, hyperparameter tuning, prediction, and evaluation. This modularity makes the system easy to maintain and extend.

- **Data Processing Workflow:**  
  - **Data Acquisition:**  
    Data is obtained via two main channels:  
    - A **DataLoader** class (in `scripts/data_loader/data_loader.py`) loads primary datasets (such as energy consumption) from CSV files.  
    - **Data Scraping** modules (under `scripts/utils/data_scraping/`) fetch external data like bike counts, energy, NO₂, weather, and air quality for different regions. Each submodule (for example, `data_processing.py` files in bike, energy, and no2 folders; and the weather and air forecast modules) provides specific logic to process and clean raw API responses.
  
  - **Data Cleaning and Missing Values:**  
    Utilities for cleaning data and handling missing values are provided in `scripts/utils/data_processing/data_cleaning.py`. The `config_checker.py` module ensures that configuration files follow the expected schema.
  
- **Forecasting Process:**

  The forecasting process itself is broken down into these key steps:
  1. **Feature Engineering:**  
     The **FeaturePipeline** (in `scripts/feature_engineering/feature_pipeline.py`) takes raw data and creates a comprehensive set of features—including time-based (hour, weekday, month), lag features, exogenous variables, and advanced features such as Fourier terms and holiday indicators.
  
  2. **Feature Selection (Optional):**  
     The **FeatureSelectionManager** (in `scripts/feature_engineering/feature_selection.py`) uses a LightGBM‑based approach to select the most relevant features.
  
  3. **Modeling:**  
     A number of forecasting models are implemented as separate modules. Each model follows the interface defined by the **BaseModel** (in `scripts/models/base_model.py`):
     - The **Naive Baseline Model** provides a simple reference forecast.
     - The **Quantile Regression Model** trains a separate regression model for each quantile.
     - The **LightGBM Forecast Model** trains LightGBM models for probabilistic forecasting.
     - The **LSTM Forecast Model** implements a deep learning approach with an LSTM network for sequence prediction.
  
  4. **Hyperparameter Optimization:**  
     Hyperparameter tuning is integrated via the **HyperparameterOptimizer** (in `scripts/hyperparameter/hyperparameter_optimization.py`), which uses Optuna to search for optimal parameter configurations.
  
  5. **Prediction and Evaluation:**  
     After training, the **Predictor** module (in `scripts/predictor/predictor.py`) is used to generate forecasts. The **ExtendedReportManager** (in `scripts/evaluation/report_manager.py`) then produces a variety of plots and statistical evaluations (residual analysis, QQ-plots, CUSUM, etc.) to assess model performance.

- **Versioning and Reproducibility:**  
  Version information is embedded in the YAML configuration files. The ConfigManager merges version‑specific settings with global paths and parameters, ensuring that each run of the pipeline is fully reproducible.

---

## 13. Extensibility

**Adding New Models, Feature Modules, or Evaluation Techniques**

- **New Models:**  
  - **Implementing a New Model:**  
    To add a new forecasting model, create a new Python module under the `scripts/models/` directory (e.g., `new_model.py`).  
    - Inherit from the **BaseModel** class (located in `scripts/models/base_model.py`) to ensure consistency with the pipeline interface.
    - Implement required methods such as `train()`, `predict()`, `evaluate_trial()`, and `get_optuna_search_space()`.
    - Register the new model in the `MODEL_CLASSES` dictionary (in `main.py`) so that it can be selected via configuration.
  
- **New Feature Modules:**  
  - **Extending Feature Engineering:**  
    New feature types can be added by modifying or extending the **FeaturePipeline** (in `scripts/feature_engineering/feature_pipeline.py`).  
    - For example, if you want to include additional statistical moments or custom domain-specific features, add new methods (or extend the `_create_advanced_features()` method).
    - Update the corresponding YAML configuration files (found under `configs/energy` or `configs/no2`) to include options for the new feature module.
  
- **New Evaluation Techniques:**  
  - **Adding Evaluation Methods:**  
    The evaluation framework is provided by the **ExtendedReportManager** (in `scripts/evaluation/report_manager.py`).  
    - To add new evaluation metrics or diagnostic plots (e.g., additional tests for nonlinearity or custom calibration curves), create new functions in the ExtendedReportManager.
    - Update the `run_all_analyses()` method to call these new functions so that they become part of the automatic evaluation report.
  
- **General Guidelines for Extensibility:**  
  - **Follow the Existing Structure:**  
    When adding or modifying functionality, use the established folder structure and coding conventions. All configurations are maintained via YAML files, and any new components should integrate with the ConfigManager.
  - **Reusability and Modularity:**  
    New modules should be designed as standalone components that can be easily integrated into the PipelineManager. This modular approach ensures that future updates or experiments do not require rewriting the entire pipeline.
  - **Logging and Error Handling:**  
    Use the provided logger (from `scripts/utils/logger.py`) for logging information and error messages. This ensures that debugging and tracing are consistent across all new modules.
  - **Documentation and Configuration:**  
    Update the README and inline documentation to include details of any new model parameters, feature types, or evaluation methods so that future users and developers can understand and replicate your changes.

---

