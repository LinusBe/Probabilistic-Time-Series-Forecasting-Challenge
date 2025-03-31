import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import date, datetime, timedelta
from tqdm import tqdm
import os
import sys
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_energy_data():
    """
    Retrieve energy consumption data from the SMARD.de API, process the data, and save it to a CSV file.

    The function fetches available timestamps from SMARD.de, determines if new data needs to be downloaded 
    based on the last available date in the existing CSV file, downloads the new data for each timestamp, 
    processes the Unix timestamps into a proper datetime format, and appends the new data to the existing 
    dataset if available. Finally, the combined dataset is saved as a CSV file.

    Returns:
        None
    """
    filename = r"data/raw/energy/energy_data.csv"

    # Get all available timestamps
    stampsurl = "https://www.smard.de/app/chart_data/410/DE/index_hour.json"
    response = requests.get(stampsurl, timeout=1000)

    # Check if the file already exists
    if os.path.exists(filename):
        # Load existing data and determine the last date
        existing_data = pd.read_csv(filename, parse_dates=['date_time_utc', 'date_time_local'], index_col='date_time_utc')
        # Determine the time difference (in weeks) and add 2 weeks as a buffer
        last_date = existing_data["date_time_local"].max()
        diff = (datetime.now() - last_date).days // 7 + 2
        logger.info("Existing data found. Last date: %s. Calculated diff (weeks): %s", last_date, diff)
    else:
        # If file does not exist, load all data
        diff = 0
        logger.info("No existing file found. Loading all available data.")

    # Ignore first 6 years to speed up baseline; select only recent timestamps
    timestamps = list(response.json()["timestamps"])[-diff:]

    col_names = ['date_time_local', 'Netzlast_Gesamt']
    energydata = pd.DataFrame(columns=col_names)

    # Download new data starting from the last available date
    for stamp in tqdm(timestamps, desc="Downloading energy data"):
        dataurl = f"https://www.smard.de/app/chart_data/410/DE/410_DE_hour_{stamp}.json"
        response = requests.get(dataurl, timeout=1000)
        rawdata = response.json()["series"]

        # Convert Unix-Timestamps and format data
        for i in range(len(rawdata)):
            # Use only first 10 digits of the timestamp and convert to desired format
            rawdata[i][0] = datetime.fromtimestamp(int(str(rawdata[i][0])[:10])).strftime("%Y-%m-%d %H:%M:%S")

        energydata = pd.concat([energydata, pd.DataFrame(rawdata, columns=col_names)])

    # Parse 'date_time_local' column to datetime
    energydata['date_time_local'] = pd.to_datetime(energydata['date_time_local'])

    # Set timezone CET/CEST and convert to UTC
    energydata['date_time_utc'] = energydata['date_time_local'].dt.tz_localize(
        'Europe/Berlin', ambiguous='infer', nonexistent='NaT'
    ).dt.tz_convert('UTC')

    # Clean and adjust the data
    energydata = energydata.dropna()
    energydata.set_index("date_time_utc", inplace=True)

    # Data preprocessing: rename column and scale values
    energydata = energydata.rename(columns={"Netzlast_Gesamt": "gesamt"})
    energydata['gesamt'] = energydata['gesamt'] / 1000
    energydata = energydata.sort_index()

    if os.path.exists(filename):
        # Join existing data with new data without duplicates
        combined_data = pd.concat([existing_data, energydata])
        # Drop duplicates
        combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
        # Sort index by date
        combined_data = combined_data.sort_index()
        combined_data.to_csv(filename)
        energydata = combined_data
        logger.info("New data successfully appended to %s", filename)
        print(f"Neue Daten erfolgreich zu {filename} hinzugefügt.")
    else:
        energydata.to_csv(filename)
        logger.info("File %s created and data saved.", filename)
        print(f"{filename} erstellt und Daten gespeichert.")


if __name__ == "__main__":
    get_energy_data()
