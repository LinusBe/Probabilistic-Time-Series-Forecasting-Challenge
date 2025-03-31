import yaml
import os

# Schema-Definitionen (siehe unten) und Funktionen
REQUIRED_SCHEMA = {
    "dataset": None,
    "model": None,
    "quantiles": None,
    "optuna_search_space": {
        "param_space": {}
    },
    "versions": "VERSION_SCHEMA"
}

VERSION_SCHEMA = {
    "start_date": None,
    "train_size": None,
    "test_size": None,
    "eval_set": {
        "use": None,
        "size": None,
    "early_stopping_rounds": None,
    },
    "imputation_method": {
        "use": None,
        "time_cfg": {
            "method": None,
            "limit_direction": None,
        },
        "knn_cfg": {
            "method": None,
            "n_neighbors": None,
            "weights": None,
            "metric": None,
        },
        "spline_cfg": {
            "method": None,
            "order": None,
            "limit_direction": None,
        }
    },
    "training_mode": None,
    "cv_settings": {
        "window_type": None,
        "test_window": None,
        "optuna_folds": None,
    },
    "optuna": {
        "use_optuna": None,
        "n_trials": None,
        "n_splits": None,
        "direction": None,
        "metric": None,
    },
    "feature_selection": {
        "top_n": None,
        "run_selection": None,
    },
    "features": {
        "normalization": {
            "base_features": {
                "enabled": None,
                "method": None,
            },
            "time": {
                "enabled": None,
                "method": None,
            },
            "lag": {
                "enabled": None,
                "method": None,
            },
            "exog": {
                "enabled": None,
                "method": None,
            },
            "advanced": {
                "enabled": None,
                "method": None,
            },
        },
        "target": {
            "lags": None,
        },
        "time_features": None,
        "fourier_terms": None,
        "exogenous": {
            "base_features": None,
            "transformations": {
                "rolling": {
                    "windows": None,
                    "stats": None,
                    "features": None,
                },
                "diff": {
                    "windows": None,
                    "features": None,
                }
            }
        },
        "advanced": {
            "holiday": {
                "enabled": None,
                "proximity": None,
                "country": None,
            },
            "interactions": None,
            "rolling_moments": {
                "windows": None,
                "moments": None,
                "features": None,
            }
        }
    },
    "params": {},
    "forecast_horizon": None,
}

def check_keys(config, schema, path=""):
    """
    Rekursive Prüfung, ob in config alle in schema definierten Schlüssel vorhanden sind.
    """
    errors = []
    for key, subschema in schema.items():
        current_path = f"{path}/{key}" if path else key
        if key not in config:
            errors.append(f"Fehlender Schlüssel: {current_path}")
        else:
            if subschema is None:
                continue
            elif isinstance(subschema, dict):
                if not isinstance(config[key], dict):
                    errors.append(f"Schlüssel {current_path} sollte ein Dictionary sein.")
                else:
                    errors.extend(check_keys(config[key], subschema, current_path))
            elif isinstance(subschema, str) and subschema == "VERSION_SCHEMA":
                if not isinstance(config[key], dict):
                    errors.append(f"Schlüssel {current_path} sollte ein Dictionary mit Versionsinformationen sein.")
                else:
                    for version, version_config in config[key].items():
                        version_path = f"{current_path}/{version}"
                        if not isinstance(version_config, dict):
                            errors.append(f"Schlüssel {version_path} sollte ein Dictionary sein.")
                        else:
                            errors.extend(check_keys(version_config, VERSION_SCHEMA, version_path))
    return errors

def load_yaml(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Fehler beim Laden der Datei {file_path}: {e}")
        return None

def check_config_file(file_path):
    config = load_yaml(file_path)
    if config is None:
        return
    errors = check_keys(config, REQUIRED_SCHEMA)
    if errors:
        print(f"Fehler in {file_path}:")
        for err in errors:
            print("  -", err)
    else:
        print(f"{file_path}: Alle erforderlichen Schlüssel sind vorhanden.")


def check_all_configs_in_directory(directory):
    """
    Durchläuft alle .yml-Dateien in 'directory' und ruft check_config_file darauf auf.
    """
    for file_name in os.listdir(directory):
        if file_name.endswith(".yml"):
            file_path = os.path.join(directory, file_name)
            check_config_file(file_path)



check_all_configs_in_directory("configs/energy")
check_all_configs_in_directory("configs/no2")