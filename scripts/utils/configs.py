"""
Configuration management module.

This module defines the ConfigManager class which is responsible for loading, merging,
and updating YAML configuration files while preserving formatting using ruamel.yaml.
"""

import os
import logging
import pdb

import yaml
import ruamel.yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration files for the forecasting pipeline.

    Loads configuration files from the "configs" directory based on provided keys and filenames,
    caches the configurations, and merges them with version-specific settings.
    """

    def __init__(self, data_set, model_name):
        """
        Initialize the ConfigManager.

        Args:
            data_set (str): The dataset identifier (e.g., "energy" or "no2").
            model_name (str): The model name (e.g., "baseline", "light_gbm").
        """
        project_root = os.environ.get("PROJECT_ROOT", None)
        self.project_root = project_root if project_root else os.getcwd()
        # Check project_root *before* assignment
        if project_root is None:
            logger.error(
                "Environment variable PROJECT_ROOT not set. Using current working directory as project root: %s. "
                "This may lead to unexpected behavior!", self.project_root
            )
        self.data_set = data_set
        self.model_name = model_name
        self.yaml = ruamel.yaml.YAML()
        self._cached_configs = {}  # Instance variable for caching

    def read_cfg(self, **add_cfgs):
        """
        Loads configuration files based on provided key-value pairs.

        For example, read_cfg(energy="baseline", default="paths") searches for
        configs/energy/baseline.yml and configs/default/paths.yml.

        Args:
            **add_cfgs: Key-value pairs where the key is the config folder and the value is the config file name (without .yml).

        Returns:
            dict: Merged configuration dictionary.
        """
        cfg_dict = {}
        for key, config_name in add_cfgs.items():
            logger.info("Loading configuration for '%s' -> '%s.yml'", key, config_name)
            config_path = os.path.join(self.project_root, "configs", key, f"{config_name}.yml")
            if not os.path.isfile(config_path):
                default_config_path = os.path.join(self.project_root, "configs", "default", f"{config_name}.yml")
                if os.path.isfile(default_config_path):
                    logger.warning("File not found: %s. Using default: %s", config_path, default_config_path)
                    config_path = default_config_path
                else:
                    raise FileNotFoundError(f"Configuration file not found: {config_path}")

            try:
                with open(config_path, 'r', encoding='utf-8') as stream:
                    cfg_data = yaml.safe_load(stream)
                    if cfg_data is None:
                        cfg_data = {}
                    cfg_dict[key] = cfg_data
            except yaml.YAMLError as e:
                logger.error("Error reading YAML file %s: %s", config_path, e)
                raise  # Propagate error to allow program termination
            except Exception as e:
                logger.error("Unexpected error reading %s: %s", config_path, e)
                raise

        merged_cfg = {}
        for c in cfg_dict.values():
            merged_cfg.update(c)  # Use update() for merging dictionaries.
        return merged_cfg

    def get_config(self, **kwargs):
        """
        Retrieves the configuration from the cache or loads it if not present.

        Args:
            **kwargs: Key-value pairs specifying the configuration files to load.

        Returns:
            dict: Merged configuration dictionary.
        """
        key = tuple(sorted(kwargs.items()))
        if key not in self._cached_configs:
            self._cached_configs[key] = self.read_cfg(**kwargs)
        return self._cached_configs[key]

    def merge_configs(self, dataset, model, version_key):
        """
        Merges configurations and creates necessary directories.

        Args:
            dataset (str): The dataset identifier.
            model (str): The model name.
            version_key (str): The version key to merge into the configuration.

        Returns:
            dict: The updated configuration dictionary with merged version settings and directory paths.
        """
        config = self.get_config(**{dataset: model, 'default': 'paths'})
        self.version = version_key
        available_versions = config.get('versions', {})
        if version_key and version_key in available_versions:
            config.update(available_versions[version_key])
            config["version"] = version_key
        else:
            config["version"] = "base"

        sp = config.get("save_paths", {})
        base_output = sp.get("base_output", "output")
        model_name = config.get("model")

        if model_name != "light_gbm_feauture_selection":
            output_dir = os.path.join(self.project_root, base_output, dataset, model_name, config["version"])
        else:
            output_dir = os.path.join(self.project_root, base_output, dataset, 'feature_selection', config["version"])

        os.makedirs(output_dir, exist_ok=True)
        config["output_dir"] = output_dir

        subfolders = ["logs", "results", "predictions", "plots", "models", "hyperparameter"]
        for subfolder in subfolders:
            folder_path = os.path.join(output_dir, sp.get(f"{subfolder}_subfolder", subfolder))
            os.makedirs(folder_path, exist_ok=True)
            config[f"{subfolder}_dir"] = folder_path

        return config

    def update_yaml_params(self, yaml_path, version, updates):
        """
        Updates parameters in a YAML file while preserving formatting using ruamel.yaml.

        Args:
            yaml_path (str): The path to the YAML file.
            version (str): The version key in the YAML to update.
            updates (dict): A dictionary of parameters to update.

        Raises:
            TypeError: If updates is not a dictionary.
            ValueError: If the specified version is not found or data is missing.
            IOError: If an error occurs during writing the updated YAML.
        """
        if not isinstance(updates, dict):
            raise TypeError("'updates' must be a dictionary.")

        # Configure ruamel.yaml instance for better readability in block sequences
        self.yaml.indent(mapping=2, sequence=4, offset=2)

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = self.yaml.load(f)
        except (FileNotFoundError, ruamel.yaml.YAMLError) as e:
            raise type(e)(f"Error loading YAML file: {e}")

        if version not in config.get('versions', {}):
            raise ValueError(f"Version '{version}' not found.")

        version_config = config['versions'].get(version)
        if not version_config:
            raise ValueError(f"Version '{version}' data is missing.")

        def _update_recursive(existing, new_values):
            for key, value in new_values.items():
                if isinstance(value, dict) and key in existing and isinstance(existing[key], dict):
                    _update_recursive(existing[key], value)
                elif key not in existing:
                    existing[key] = value
                else:
                    existing[key] = value

        if 'params' not in version_config:
            version_config['params'] = {}
        _update_recursive(version_config['params'], updates)
        try:
            with open(yaml_path, 'w', encoding='utf-8') as f:
                self.yaml.dump(config, f)
        except Exception as e:
            raise IOError(f"Error writing updated YAML: {e}")
