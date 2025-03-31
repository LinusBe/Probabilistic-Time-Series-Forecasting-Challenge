import pandas as pd
import numpy as np

def check_missing_data(data):
    """
    Überprüft die übergebenen Daten auf fehlende Werte und gibt einen Bericht zurück.
    """
    missing_data = data.isnull().sum()
    missing_data_percent = 100 * data.isnull().sum() / len(data)
    missing_report = pd.DataFrame({'missing_values': missing_data, 'percent': missing_data_percent})
    return missing_report[missing_report['missing_values'] > 0]

def handle_missing_data(data, method='drop'):
    """
    Behandelt fehlende Daten entweder durch Entfernen oder Imputation.
    :param data: DataFrame, das behandelt werden soll.
    :param method: Methode zur Behandlung fehlender Daten ('drop' oder 'impute').
    """
    if method == 'drop':
        return data.dropna()
    elif method == 'impute':
        # Beispiel für eine einfache Imputation: Füllen mit dem Mittelwert jeder Spalte.
        for column in data.columns:
            data[column].fillna(data[column].mean(), inplace=True)
        return data
    else:
        raise ValueError("Ungültige Methode zur Behandlung fehlender Daten angegeben.")

def detect_outliers(data, column, method='IQR'):
    """
    Erkennt Ausreißer in einer spezifischen Spalte eines DataFrame.
    :param data: DataFrame, das untersucht wird.
    :param column: Spalte, in der nach Ausreißern gesucht werden soll.
    :param method: Methode zur Erkennung von Ausreißern ('IQR' oder 'Z-Score').
    """
 
    if method == 'IQR':
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = ~data[column].between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    elif method == 'Z-Score':
        from scipy.stats import zscore
        data['z_score'] = zscore(data[column])
        outlier_condition = (data['z_score'].abs() > 3)
    else:
        raise ValueError("Ungültige Methode zur Erkennung von Ausreißern angegeben.")
    return data[outlier_condition]

def remove_outliers(data, column, method='IQR'):
    """
    Entfernt Ausreißer aus einer spezifischen Spalte eines DataFrame.
    :param data: DataFrame, aus dem Ausreißer entfernt werden sollen.
    :param column: Spalte, aus der Ausreißer entfernt werden sollen.
    :param method: Methode zur Erkennung von Ausreißern ('IQR' oder 'Z-Score').
    """
    if method == 'IQR':
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        condition = data[column].between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    elif method == 'Z-Score':
        from scipy.stats import zscore
        data['z_score'] = zscore(data[column])
        condition = (data['z_score'].abs() <= 3)
        data.drop(columns='z_score', inplace=True)
    else:
        raise ValueError("Ungültige Methode zur Erkennung von Ausreißern angegeben.")
    return data[condition]
