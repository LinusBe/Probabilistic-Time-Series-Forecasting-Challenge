import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import os
from datetime import date, datetime, timedelta
from io import StringIO
import matplotlib.pyplot as plt


def get_no2_data():
    """
    Retrieve and process NO₂ data from the Umweltbundesamt API and save the results as a CSV file.

    This function downloads NO₂ measurement data in CSV format from the Umweltbundesamt API.
    It processes the data by:
      - Dropping the last row,
      - Replacing '-' with NaN and converting values to float,
      - Extracting the hour from the 'Time' column (with 24 converted to 0),
      - Creating a combined datetime column from the 'Date' and 'hour' columns,
      - Adjusting timestamps for late-night entries by adding a day,
      - Handling fall and spring daylight saving transitions,
      - Converting local datetime to UTC,
      - Interpolating missing measurement values.
    Finally, the processed DataFrame is saved to 'data/raw/no2/no2_data.csv'.

    Returns:
        None
    """
    filename = r'data/raw/no2/no2_data.csv'
    end_date = date.today().strftime('%Y-%m-%d')
    start_date = '2015-11-04'
    url = (f'https://www.umweltbundesamt.de/api/air_data/v3/measures/csv?'
           f'date_from={start_date}&time_from=24&date_to={end_date}&time_to=23&'
           f'data%5B0%5D%5Bco%5D=5&data%5B0%5D%5Bsc%5D=2&'
           f'data%5B0%5D%5Bda%5D={end_date}&data%5B0%5D%5Bti%5D=12&'
           f'data%5B0%5D%5Bst%5D=282&data%5B0%5D%5Bva%5D=27&lang=en')
    df = pd.read_csv(url, sep=';')
    df = df.drop(df.index[-1])
    df['Measure value'] = df['Measure value'].replace('-', np.nan)
    df['Measure value'] = df['Measure value'].astype(float)
    df['hour'] = df['Time'].str[1:3].astype(int)
    df['hour'] = df['hour'].replace(24, 0)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['hour'].astype(str) + ':00')
    df.loc[df['datetime'].dt.hour == 0, 'datetime'] += pd.Timedelta(days=1)
    df = df.set_index('datetime')
    df['weekday'] = df.index.weekday
    df = df.reset_index()
    df = df.rename(columns={'datetime': 'date_time_local'})
    df['date_time_local'] = pd.to_datetime(df['date_time_local'])
    # Adjust for fall transition: if duplicate and month is October at 3 AM, subtract one hour
    fall_transition = (df['date_time_local'].duplicated(keep='last') &
                       (df['date_time_local'].dt.month == 10) &
                       (df['date_time_local'].dt.hour == 3))
    df.loc[fall_transition, 'date_time_local'] -= pd.Timedelta(hours=1)
    df = df.sort_values(by='date_time_local').reset_index(drop=True)

    def get_last_sunday_of_march(year):
        """Return the last Sunday of March for the given year."""
        last_day_of_march = pd.Timestamp(year=year, month=3, day=31)
        last_sunday_of_march = last_day_of_march - pd.Timedelta(days=(last_day_of_march.weekday() + 1) % 7)
        return last_sunday_of_march

    # Adjust for spring transition: add one hour for entries on the last Sunday of March at 2 AM
    for year in df['date_time_local'].dt.year.unique():
        last_sunday_of_march = get_last_sunday_of_march(year)
        spring_transition = ((df['date_time_local'].dt.date == last_sunday_of_march.date()) &
                             (df['date_time_local'].dt.hour == 2))
        df.loc[spring_transition, 'date_time_local'] += pd.Timedelta(hours=1)
    df = df.sort_values(by='date_time_local').reset_index(drop=True)
    df['date_time_utc'] = df['date_time_local'].dt.tz_localize(
        'Europe/Berlin', ambiguous='infer', nonexistent='NaT'
    ).dt.tz_convert('UTC')
    df = df.rename(columns={'Measure value': 'gesamt'})
    df['gesamt'] = df['gesamt'].interpolate(method='linear')
    df.set_index('date_time_utc', inplace=True)
    df = df.sort_index()
    df.to_csv(filename)


if __name__ == '__main__':
    get_no2_data()
