import pandas as pd
import requests
from datetime import datetime, timedelta


def get_bike_data():
    """
    Retrieve bike data from the eco-visio API, process it, and save it as a CSV file.

    This function downloads bike count data starting from '01/01/2013'. It converts the date
    strings into datetime objects, creates an hourly time series for each day, and fills in missing
    hours with NaN. The resulting DataFrame is saved to 'data/raw/bike/bike_data.csv'.

    Returns:
        None
    """
    start_date = '01/01/2013'
    dataurl = (
        "https://www.eco-visio.net/api/aladdin/1.0.0/pbl/publicwebpageplus/data/"
        f"100126474?idOrganisme=4586&idPdc=100126474&interval=3&flowIds=100126474&debut={start_date}"
    )
    response = requests.get(dataurl)
    rawdata = response.json()

    df = pd.DataFrame(rawdata, columns=['date', 'bike_count'])
    df['bike_count'] = df['bike_count'].astype(float)

    # Convert date strings to datetime objects
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    # Sort data by date and create an hourly counter for each day
    df = df.sort_values(by='date')
    df['hour'] = df.groupby('date').cumcount()
    df['date_time_utc'] = df.apply(lambda row: row['date'] + timedelta(hours=row['hour']),
                                   axis=1)

    # Remove original date and hour columns
    df.drop(columns=['date', 'hour'], inplace=True)

    # Rename 'bike_count' column to 'gesamt'
    df = df.rename(columns={'bike_count': 'gesamt'})

    # Create a continuous hourly time series with missing hours filled with NaN
    idx = pd.date_range(start=df.date_time_utc.iloc[0], end=df.date_time_utc.iloc[-1], freq='h')
    temp = pd.DataFrame(index=idx)
    df.set_index('date_time_utc', inplace=True)
    df = temp.join(df, how='left')
    df.index.name = 'date_time_utc'

    # Save the processed data to a CSV file
    filename = r"data/raw/bike/bike_data.csv"
    df.to_csv(filename)
    print(f"Data saved to {filename}.")


if __name__ == "__main__":
    get_bike_data()
