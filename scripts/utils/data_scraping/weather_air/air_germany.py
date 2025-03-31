import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import numpy as np
import time
import sys
import os
import pdb

# Platform-specific imports for non-blocking keyboard input
if os.name == 'nt':  # Windows
    import msvcrt
else:  # Linux/macOS
    import select
    import tty
    import termios

import logging

logger = logging.getLogger(__name__)


class AirForecastGermany:
    """
    Retrieve air quality forecast data for Germany using the Open-Meteo API.

    This class sets up the API client with caching and retry mechanisms, enforces API call limits,
    and gathers hourly air quality data for multiple stations. The data is then concatenated into a single DataFrame.
    """

    def __init__(self, past_days=7, forecast_days=4, calls_per_day=20):
        """
        Initialize the AirForecastGermany instance.

        Args:
            past_days (int): Number of past days to include.
            forecast_days (int): Number of forecast days to retrieve.
            calls_per_day (int): Maximum API calls allowed per day.
        """
        # Setup the Open-Meteo API client with cache and retry on error
        self.cache_session = requests_cache.CachedSession(
            r'scripts/utils/data_scraping/weather_air/temp.cache', expire_after=-1
        )
        self.retry_session = retry(self.cache_session, retries=10, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=self.retry_session)

        # API Call Limits
        self.calls_per_day = calls_per_day
        self.minute_limit = 600  # Max API calls per minute
        self.hourly_limit = 5000 - 100  # Max API calls per hour
        self.daily_limit = 10000 - 1000  # Max API calls per day

        # API Call Counters
        self.api_calls_used = 0
        self.minute_calls_used = 0
        self.hourly_calls_used = 0
        self.daily_calls_used = 0

        # Timestamps for resetting counters
        self.start_of_minute = time.time()
        self.start_of_hour = time.time()
        self.start_of_day = time.time()

        self.past_days = past_days
        self.forecast_days = forecast_days

        # Define station coordinates (in decimal degrees)
        self.coordinates = {
            10501: (self.convert_coords(50, 47), self.convert_coords(6, 5)),
            10291: (self.convert_coords(53, 1), self.convert_coords(13, 59)),
            10091: (self.convert_coords(54, 40), self.convert_coords(13, 26)),
            10852: (self.convert_coords(48, 25), self.convert_coords(10, 56)),
            10675: (self.convert_coords(49, 52), self.convert_coords(10, 54)),
            10384: (self.convert_coords(52, 28), self.convert_coords(13, 24)),
            10224: (self.convert_coords(53, 2), self.convert_coords(8, 47)),
            10453: (self.convert_coords(51, 48), self.convert_coords(10, 37)),
            10496: (self.convert_coords(51, 47), self.convert_coords(14, 19)),
            10131: (self.convert_coords(53, 52), self.convert_coords(8, 42)),
            10488: (self.convert_coords(51, 7), self.convert_coords(13, 45)),
            10400: (self.convert_coords(51, 17), self.convert_coords(6, 46)),
            10200: (self.convert_coords(53, 23), self.convert_coords(7, 14)),
            10554: (self.convert_coords(50, 59), self.convert_coords(10, 57)),
            10578: (self.convert_coords(50, 25), self.convert_coords(12, 57)),
            10637: (self.convert_coords(50, 2), self.convert_coords(8, 35)),
            10815: (self.convert_coords(48, 27), self.convert_coords(8, 24)),
            10439: (self.convert_coords(51, 7), self.convert_coords(9, 17)),
            10895: (self.convert_coords(48, 32), self.convert_coords(13, 21)),
            10567: (self.convert_coords(50, 52), self.convert_coords(12, 7)),
            10532: (self.convert_coords(50, 36), self.convert_coords(8, 39)),
            10499: (self.convert_coords(51, 9), self.convert_coords(14, 57)),
            10184: (self.convert_coords(54, 5), self.convert_coords(13, 24)),
            10791: (self.convert_coords(49, 6), self.convert_coords(13, 8)),
            10616: (self.convert_coords(49, 56), self.convert_coords(7, 15)),
            10155: (self.convert_coords(53.33, 0), self.convert_coords(9, 58)),
            10147: (self.convert_coords(53, 38, ), self.convert_coords(9, 59)),
            10338: (self.convert_coords(52, 27), self.convert_coords(9, 40)),
            10685: (self.convert_coords(50, 18), self.convert_coords(11, 52)),
            10962: (self.convert_coords(47, 48), self.convert_coords(11, 0)),
            10427: (self.convert_coords(51, 11), self.convert_coords(8, 29)),
            10727: (self.convert_coords(49, 2), self.convert_coords(8, 21)),
            10438: (self.convert_coords(51, 17), self.convert_coords(9, 26)),
            10946: (self.convert_coords(47, 43), self.convert_coords(10, 20)),
            10046: (self.convert_coords(54, 22), self.convert_coords(10, 8)),
            10513: (self.convert_coords(50, 51), self.convert_coords(7, 9)),
            10929: (self.convert_coords(47, 40), self.convert_coords(9, 11)),
            10805: (self.convert_coords(48, 21), self.convert_coords(7, 49)),
            10469: (self.convert_coords(51, 26), self.convert_coords(12, 14)),
            10124: (self.convert_coords(53, 51), self.convert_coords(8, 7)),
            10393: (self.convert_coords(52, 12), self.convert_coords(14, 7)),
            10430: (self.convert_coords(51, 47), self.convert_coords(8, 50)),
            10361: (self.convert_coords(52, 6), self.convert_coords(11, 35)),
            10729: (self.convert_coords(49, 30), self.convert_coords(8, 33)),
            10264: (self.convert_coords(53, 19), self.convert_coords(11, 56)),
            10548: (self.convert_coords(50, 33), self.convert_coords(10, 22)),
            10870: (self.convert_coords(48, 22), self.convert_coords(11, 49)),
            10315: (self.convert_coords(52, 8), self.convert_coords(7, 42)),
            10270: (self.convert_coords(52, 54), self.convert_coords(12, 48)),
            10271: (self.convert_coords(52.94, 0), self.convert_coords(12, 85)),
            10113: (self.convert_coords(53, 42), self.convert_coords(7, 9)),
            10506: (self.convert_coords(50, 22), self.convert_coords(6, 52)),
            10763: (self.convert_coords(49, 30), self.convert_coords(11, 3)),
            10641: (self.convert_coords(50, 5), self.convert_coords(8, 47)),
            10742: (self.convert_coords(49, 12), self.convert_coords(9, 31)),
            10379: (self.convert_coords(52, 23), self.convert_coords(13, 3)),
            10776: (self.convert_coords(49, 2), self.convert_coords(12, 6)),
            10731: (self.convert_coords(48, 58), self.convert_coords(8, 20)),
            10170: (self.convert_coords(54, 1), self.convert_coords(12, 4)),
            10708: (self.convert_coords(49, 12), self.convert_coords(7, 6)),
            10035: (self.convert_coords(54, 31), self.convert_coords(9, 32)),
            10162: (self.convert_coords(53, 38), self.convert_coords(11, 23)),
            10836: (self.convert_coords(48, 40), self.convert_coords(9, 51)),
            10788: (self.convert_coords(48, 49), self.convert_coords(12, 33)),
            10738: (self.convert_coords(48, 41), self.convert_coords(9, 13)),
            10609: (self.convert_coords(49, 44), self.convert_coords(6, 39)),
            10007: (self.convert_coords(54, 10), self.convert_coords(7, 27)),
            10004: (self.convert_coords(54, 10), self.convert_coords(6, 21)),
            10544: (self.convert_coords(50, 29), self.convert_coords(9, 56)),
            10688: (self.convert_coords(49, 40), self.convert_coords(12, 11)),
            10980: (self.convert_coords(47, 42), self.convert_coords(12, 0)),
            10655: (self.convert_coords(49, 46), self.convert_coords(9, 57))
        }

        self.url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    def main(self):
        """
        Gather air quality forecast data and concatenate CSV files into a DataFrame.

        Returns:
            pd.DataFrame: Combined air quality forecast data.
        """
        self.gather()
        df = self.concat_dataframes()
        logger.info("Data gathering and concatenation complete for Germany air quality.")
        return df

    def kbhit(self):
        """
        Check if a key has been pressed (non-blocking).

        Returns:
            bool: True if a key press is detected, otherwise False.
        """
        if os.name == 'nt':
            return msvcrt.kbhit()
        else:
            return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def wait_or_continue(self):
        """
        Wait for user input or timeout if API call limits are exceeded.
        """
        current_time = time.time()
        limit_type = None
        wait_time = 0

        if self.minute_calls_used >= self.minute_limit:
            wait_time = 60
            limit_type = "minute"
            if self.hourly_calls_used >= self.hourly_limit:
                logger.info("Hourly limit exceeded")
                wait_time = 3600
                limit_type = "hourly"
                if self.daily_calls_used >= self.daily_limit:
                    wait_time = 86400
                    limit_type = "daily"
        else:
            logger.info("No limit exceeded, continuing...")
            return

        limit_used = {
            "minute": self.minute_calls_used,
            "hourly": self.hourly_calls_used,
            "daily": self.daily_calls_used
        }
        limit_max = {
            "minute": self.minute_limit,
            "hourly": self.hourly_limit,
            "daily": self.daily_limit
        }

        prompt = "{} / {} press or wait".format(limit_used[limit_type], limit_max[limit_type])
        end_time = time.time() + wait_time

        remaining_time = end_time - time.time()
        if os.name != 'nt':
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        logger.info(prompt)
        try:
            while remaining_time > 0:
                logger.info("Waiting for user input or timeout, remaining: %ds", int(remaining_time))
                sys.stdout.flush()

                if self.kbhit():
                    if os.name == 'nt':
                        key_pressed = msvcrt.getch().decode('utf-8', errors='replace')
                    else:
                        key_pressed = sys.stdin.read(1)
                    logger.info("Key pressed: %s - interrupting wait.", key_pressed)
                    break

                time.sleep(0.1)
                remaining_time = end_time - time.time()
        finally:
            if os.name != 'nt':
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        logger.info("No key pressed, continuing after timeout.")
        logger.info("Resetting limit type: %s", limit_type)
        self.reset_counters(limit_type)
        logger.info("Overall Calls Used: %s", self.daily_calls_used)

    def reset_counters(self, limit_type):
        """
        Reset API call counters based on the specified limit type.

        Args:
            limit_type (str): The limit type to reset ("minute", "hourly", or "daily").
        """
        if limit_type == "minute":
            self.minute_calls_used = 0
            self.start_of_minute = time.time()
        elif limit_type == "hourly":
            self.hourly_calls_used = 0
            self.minute_calls_used = 0
            self.start_of_hour = time.time()
            self.start_of_minute = time.time()
        elif limit_type == "daily":
            self.daily_calls_used = 0
            self.hourly_calls_used = 0
            self.minute_calls_used = 0
            self.start_of_day = time.time()
            self.start_of_hour = time.time()
            self.start_of_minute = time.time()
        logger.info("%s counters reset.", limit_type.capitalize())

    def check_and_update_call_limits(self):
        """
        Check elapsed time, update API call counters, and wait if limits are exceeded.
        """
        current_time = time.time()

        if current_time - self.start_of_minute >= 60:
            self.minute_calls_used = 0
            self.start_of_minute = current_time

        if current_time - self.start_of_hour >= 3600:
            self.hourly_calls_used = 0
            self.start_of_hour = current_time

        if current_time - self.start_of_day >= 86400:
            self.daily_calls_used = 0
            self.start_of_day = current_time

        if (self.minute_calls_used >= self.minute_limit or
                self.hourly_calls_used >= self.hourly_limit or
                self.daily_calls_used >= self.daily_limit):
            self.wait_or_continue()

    def convert_coords(self, degree, minute):
        """
        Convert coordinates from degrees and minutes to decimal format.

        Args:
            degree (int or float): The degree part.
            minute (int or float): The minute part.

        Returns:
            float: Decimal coordinate.
        """
        return float(degree) + float(minute) / 60

    def gather(self):
        """
        Gather air quality forecast data for each station and save to CSV files.
        """
        for station, values in self.coordinates.items():
            frames = []
            logger.info("Processing station: %s", station)
            latitude, longitude = values

            filename = rf"scripts/utils/data_scraping/weather_air/temp_data/air/{station}.csv"

            # If file exists, skip processing this station
            if os.path.exists(filename):
                logger.info("File %s already exists, skipping...", filename)
                continue

            required_calls = self.calls_per_day
            self.check_and_update_call_limits()
            self.api_calls_used += required_calls
            self.minute_calls_used += required_calls
            self.hourly_calls_used += required_calls
            self.daily_calls_used += required_calls

            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": [
                    "pm10", "pm2_5", "carbon_monoxide", "carbon_dioxide", "sulphur_dioxide", "ozone",
                    "dust", "ammonia"
                ],
                "past_days": self.past_days,
                "forecast_days": self.forecast_days,
                "domains": "cams_europe"
            }

            responses = self.openmeteo.weather_api(self.url, params=params)
            response = responses[0]

            # Process hourly data; the variable order must match the request.
            hourly = response.Hourly()
            hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
            hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
            hourly_carbon_monoxide = hourly.Variables(2).ValuesAsNumpy()
            hourly_carbon_dioxide = hourly.Variables(3).ValuesAsNumpy()
            hourly_sulphur_dioxide = hourly.Variables(4).ValuesAsNumpy()
            hourly_ozone = hourly.Variables(5).ValuesAsNumpy()
            hourly_dust = hourly.Variables(6).ValuesAsNumpy()
            hourly_ammonia = hourly.Variables(7).ValuesAsNumpy()

            hourly_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
            }
            hourly_data["pm10"] = hourly_pm10
            hourly_data["pm2_5"] = hourly_pm2_5
            hourly_data["carbon_monoxide"] = hourly_carbon_monoxide
            hourly_data["carbon_dioxide"] = hourly_carbon_dioxide
            hourly_data["sulphur_dioxide"] = hourly_sulphur_dioxide
            hourly_data["ozone"] = hourly_ozone
            hourly_data["dust"] = hourly_dust
            hourly_data["ammonia"] = hourly_ammonia

            hourly_dataframe = pd.DataFrame(data=hourly_data)
            frames.append(hourly_dataframe)
            combined_data_temp = pd.concat(frames, ignore_index=True)
            combined_data_temp.to_csv(filename, index=False)
            logger.info("Saved hourly air quality data for station %s to %s", station, filename)

    def concat_dataframes(self):
        """
        Concatenate CSV files from all stations into a single DataFrame, compute hourly averages,
        and remove duplicates.

        Returns:
            pd.DataFrame: Combined air quality forecast data.
        """
        frames = []
        for station, values in self.coordinates.items():
            filename = rf"scripts/utils/data_scraping/weather_air/temp_data/air/{station}.csv"
            if not os.path.exists(filename):
                logger.warning("File %s not found, skipping...", filename)
                continue
            logger.info("Reading data from %s", filename)
            data = pd.read_csv(filename)
            frames.append(data)

        if not frames:
            logger.error("No files found to concatenate.")
            return pd.DataFrame()

        combined_data = pd.concat(frames, ignore_index=True)
        numerical_columns = combined_data.select_dtypes(include=[np.number])
        hourly_avg = numerical_columns.groupby(combined_data["date"]).mean().reset_index()

        # Optional: Handle missing values if needed
        missing_values = hourly_avg.isnull().mean()
        # Example: Drop columns with more than 30% missing values:
        # columns_to_drop = missing_values[missing_values > 0.3].index
        # hourly_avg = hourly_avg.drop(columns=columns_to_drop)

        hourly_avg = hourly_avg[~hourly_avg.index.duplicated(keep='first')]
        hourly_avg = hourly_avg.sort_index()
        logger.info("Combined hourly average data:\n%s", hourly_avg)

        todays_date = pd.Timestamp.now().strftime("%Y%m%d")
        os.system('rm -f scripts/utils/data_scraping/weather_air/temp_data/air/*.csv')
        hourly_avg.to_csv(r"scripts/utils/data_scraping/weather_air/temp_data/weather/temp_forecast.csv", index=False)
        logger.info("Saved combined air forecast data to temp_forecast.csv")

        os.remove(r'scripts/utils/data_scraping/weather_air/temp.cache.sqlite')
        logger.info("Removed cache file: %s", r'scripts/utils/data_scraping/weather_air/temp.cache.sqlite')
        return hourly_avg
