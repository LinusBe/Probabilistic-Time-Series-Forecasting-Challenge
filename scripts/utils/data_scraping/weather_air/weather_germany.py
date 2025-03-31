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


class WeatherForecastGermany:
    def __init__(self, past_days=7, forecast_days=4, calls_per_day=20):
        """
        Initialize the WeatherForecastGermany instance.

        Sets up the Open-Meteo API client with caching and retry functionality, defines API call limits,
        and specifies coordinates for various stations.

        Args:
            past_days (int): Number of past days to include in the query.
            forecast_days (int): Number of forecast days to retrieve.
            calls_per_day (int): Number of API calls allowed per day.
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

        # Timestamps for resetting the counters
        self.start_of_minute = time.time()
        self.start_of_hour = time.time()
        self.start_of_day = time.time()

        self.past_days = past_days
        self.forecast_days = forecast_days

        # Define station coordinates (converted to decimal degrees)
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
            10147: (self.convert_coords(53.38, 0), self.convert_coords(9, 59)),
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

        self.url = "https://api.open-meteo.com/v1/forecast"

    def main(self):
        """
        Gather weather forecast data and concatenate CSV files into a DataFrame.

        Returns:
            pd.DataFrame: Combined weather forecast data.
        """
        self.gather()
        df = self.concat_dataframes()
        logger.info("Data gathering and concatenation complete for Germany.")
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
                print("hourly limit exceeded")
                wait_time = 3600
                limit_type = "hourly"
                if self.daily_calls_used >= self.daily_limit:
                    wait_time = 86400
                    limit_type = "daily"
        else:
            print("No limit exceeded, continuing...")
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

        prompt = f"{limit_used[limit_type]}/{limit_max[limit_type]} press or wait"
        end_time = time.time() + wait_time

        remaining_time = end_time - time.time()
        if os.name != 'nt':
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        print(prompt)
        try:
            while remaining_time > 0:
                print("Waiting for user input or timeout, remaining: %ds" % int(remaining_time), end='\r')
                sys.stdout.flush()

                if self.kbhit():
                    if os.name == 'nt':
                        key_pressed = msvcrt.getch().decode('utf-8', errors='replace')
                    else:
                        key_pressed = sys.stdin.read(1)
                    print("\nKey pressed: %s - interrupting wait." % key_pressed)
                    break

                time.sleep(0.1)
                remaining_time = end_time - time.time()
        finally:
            if os.name != 'nt':
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        print("\nNo key pressed, continuing after timeout.")
        print("Resetting limit type:", limit_type)
        self.reset_counters(limit_type)
        print("Overall Calls Used:", self.daily_calls_used)

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
        print("%s counters reset." % limit_type.capitalize())

    def check_and_update_call_limits(self):
        """
        Check elapsed time and update API call counters, and wait if limits are exceeded.
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
            degree (int or float): The degree part of the coordinate.
            minute (int or float): The minute part of the coordinate.

        Returns:
            float: Decimal coordinate.
        """
        return float(degree) + float(minute) / 60

    def gather(self):
        """
        Gather weather data for each station defined in the coordinates and save to CSV files.
        """
        for station, values in self.coordinates.items():
            frames = []
            print("Processing station: %s" % station)
            latitude, longitude = values

            filename = rf"scripts/utils/data_scraping/weather_air/temp_data/weather/{station}.csv"

            # If file exists, skip processing this station
            if os.path.exists(filename):
                print("File %s already exists, skipping..." % filename)
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
                    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
                    "precipitation", "rain", "snowfall", "snow_depth", "pressure_msl", "surface_pressure",
                    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
                    "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_100m",
                    "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m", "soil_temperature_0_to_7cm",
                    "soil_moisture_0_to_7cm", "wet_bulb_temperature_2m", "sunshine_duration", "shortwave_radiation",
                    "direct_radiation", "diffuse_radiation", "direct_normal_irradiance", "global_tilted_irradiance",
                    "terrestrial_radiation", "shortwave_radiation_instant", "direct_radiation_instant",
                    "diffuse_radiation_instant", "direct_normal_irradiance_instant",
                    "global_tilted_irradiance_instant", "terrestrial_radiation_instant"
                ],
                "timezone": "Europe/Berlin",
                "past_days": self.past_days,
                "forecast_days": self.forecast_days,
                "models": ["best_match", "ecmwf_ifs", "era5_seamless", "era5"]
            }

            responses = self.openmeteo.weather_api(self.url, params=params)
            response = responses[0]

            hourly = response.Hourly()
            hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
            hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
            hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
            hourly_apparent_temperature = hourly.Variables(3).ValuesAsNumpy()
            hourly_precipitation = hourly.Variables(4).ValuesAsNumpy()
            hourly_rain = hourly.Variables(5).ValuesAsNumpy()
            hourly_snowfall = hourly.Variables(6).ValuesAsNumpy()
            hourly_snow_depth = hourly.Variables(7).ValuesAsNumpy()
            hourly_pressure_msl = hourly.Variables(8).ValuesAsNumpy()
            hourly_surface_pressure = hourly.Variables(9).ValuesAsNumpy()
            hourly_cloud_cover = hourly.Variables(10).ValuesAsNumpy()
            hourly_cloud_cover_low = hourly.Variables(11).ValuesAsNumpy()
            hourly_cloud_cover_mid = hourly.Variables(12).ValuesAsNumpy()
            hourly_cloud_cover_high = hourly.Variables(13).ValuesAsNumpy()
            hourly_et0_fao_evapotranspiration = hourly.Variables(14).ValuesAsNumpy()
            hourly_vapour_pressure_deficit = hourly.Variables(15).ValuesAsNumpy()
            hourly_wind_speed_10m = hourly.Variables(16).ValuesAsNumpy()
            hourly_wind_speed_100m = hourly.Variables(17).ValuesAsNumpy()
            hourly_wind_direction_10m = hourly.Variables(18).ValuesAsNumpy()
            hourly_wind_direction_100m = hourly.Variables(19).ValuesAsNumpy()
            hourly_wind_gusts_10m = hourly.Variables(20).ValuesAsNumpy()
            hourly_soil_temperature_0_to_7cm = hourly.Variables(21).ValuesAsNumpy()
            hourly_soil_moisture_0_to_7cm = hourly.Variables(22).ValuesAsNumpy()
            hourly_wet_bulb_temperature_2m = hourly.Variables(23).ValuesAsNumpy()
            hourly_sunshine_duration = hourly.Variables(24).ValuesAsNumpy()
            hourly_shortwave_radiation = hourly.Variables(25).ValuesAsNumpy()
            hourly_direct_radiation = hourly.Variables(26).ValuesAsNumpy()
            hourly_diffuse_radiation = hourly.Variables(27).ValuesAsNumpy()
            hourly_direct_normal_irradiance = hourly.Variables(28).ValuesAsNumpy()
            hourly_global_tilted_irradiance = hourly.Variables(29).ValuesAsNumpy()
            hourly_terrestrial_radiation = hourly.Variables(30).ValuesAsNumpy()
            hourly_shortwave_radiation_instant = hourly.Variables(31).ValuesAsNumpy()
            hourly_direct_radiation_instant = hourly.Variables(32).ValuesAsNumpy()
            hourly_diffuse_radiation_instant = hourly.Variables(33).ValuesAsNumpy()
            hourly_direct_normal_irradiance_instant = hourly.Variables(34).ValuesAsNumpy()
            hourly_global_tilted_irradiance_instant = hourly.Variables(35).ValuesAsNumpy()
            hourly_terrestrial_radiation_instant = hourly.Variables(36).ValuesAsNumpy()

            hourly_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
            }
            hourly_data["temperature_2m"] = hourly_temperature_2m
            hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
            hourly_data["dew_point_2m"] = hourly_dew_point_2m
            hourly_data["apparent_temperature"] = hourly_apparent_temperature
            hourly_data["precipitation"] = hourly_precipitation
            hourly_data["rain"] = hourly_rain
            hourly_data["snowfall"] = hourly_snowfall
            hourly_data["snow_depth"] = hourly_snow_depth
            hourly_data["pressure_msl"] = hourly_pressure_msl
            hourly_data["surface_pressure"] = hourly_surface_pressure
            hourly_data["cloud_cover"] = hourly_cloud_cover
            hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
            hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
            hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
            hourly_data["et0_fao_evapotranspiration"] = hourly_et0_fao_evapotranspiration
            hourly_data["vapour_pressure_deficit"] = hourly_vapour_pressure_deficit
            hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
            hourly_data["wind_speed_100m"] = hourly_wind_speed_100m
            hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
            hourly_data["wind_direction_100m"] = hourly_wind_direction_100m
            hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m
            hourly_data["soil_temperature_0_to_7cm"] = hourly_soil_temperature_0_to_7cm
            hourly_data["soil_moisture_0_to_7cm"] = hourly_soil_moisture_0_to_7cm
            hourly_data["wet_bulb_temperature_2m"] = hourly_wet_bulb_temperature_2m
            hourly_data["sunshine_duration"] = hourly_sunshine_duration
            hourly_data["shortwave_radiation"] = hourly_shortwave_radiation
            hourly_data["direct_radiation"] = hourly_direct_radiation
            hourly_data["diffuse_radiation"] = hourly_diffuse_radiation
            hourly_data["direct_normal_irradiance"] = hourly_direct_normal_irradiance
            hourly_data["global_tilted_irradiance"] = hourly_global_tilted_irradiance
            hourly_data["terrestrial_radiation"] = hourly_terrestrial_radiation
            hourly_data["shortwave_radiation_instant"] = hourly_shortwave_radiation_instant
            hourly_data["direct_radiation_instant"] = hourly_direct_radiation_instant
            hourly_data["diffuse_radiation_instant"] = hourly_diffuse_radiation_instant
            hourly_data["direct_normal_irradiance_instant"] = hourly_direct_normal_irradiance_instant
            hourly_data["global_tilted_irradiance_instant"] = hourly_global_tilted_irradiance_instant
            hourly_data["terrestrial_radiation_instant"] = hourly_terrestrial_radiation_instant

            hourly_dataframe = pd.DataFrame(data=hourly_data)
            frames.append(hourly_dataframe)
            combined_data_temp = pd.concat(frames, ignore_index=True)
            combined_data_temp.to_csv(filename, index=False)
            logger.info("Saved hourly data for station %s to %s", station, filename)

    def concat_dataframes(self):
        """
        Concatenate CSV files from all stations into a single DataFrame,
        compute hourly averages, and remove duplicates.

        Returns:
            pd.DataFrame: Combined weather data.
        """
        frames = []
        for station, values in self.coordinates.items():
            filename = rf"scripts/utils/data_scraping/weather_air/temp_data/weather/{station}.csv"
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

        missing_values = hourly_avg.isnull().mean()
        # Optional: Drop columns with more than 30% missing values if needed
        # columns_to_drop = missing_values[missing_values > 0.3].index
        # hourly_avg = hourly_avg.drop(columns=columns_to_drop)

        hourly_avg = hourly_avg[~hourly_avg.index.duplicated(keep='first')]
        hourly_avg = hourly_avg.sort_index()
        logger.info("Combined hourly average data:\n%s", hourly_avg)

        todays_date = pd.Timestamp.now().strftime("%Y%m%d")
        os.system('rm -f scripts/utils/data_scraping/weather_air/temp_data/weather/*.csv')
        hourly_avg.to_csv(r"scripts/utils/data_scraping/weather_air/temp_data/weather/temp_forecast.csv", index=False)
        logger.info("Saved combined forecast data to temp_forecast.csv")

        os.remove(r'scripts/utils/data_scraping/weather_air/temp.cache.sqlite')
        logger.info("Removed cache file: %s", r'scripts/utils/data_scraping/weather_air/temp.cache.sqlite')
        return hourly_avg
