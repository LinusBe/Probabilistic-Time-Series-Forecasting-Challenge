import os
import time
import pandas as pd
import logging

from scripts.utils.data_scraping.weather_air.weather_germany import WeatherForecastGermany
from scripts.utils.data_scraping.weather_air.weather_karlsruhe import WeatherForecastKarlsruhe
from scripts.utils.data_scraping.weather_air.air_germany import AirForecastGermany
from scripts.utils.data_scraping.weather_air.air_karlsruhe import AirForecastKarlsruhe
from scripts.utils.data_scraping.bike.data_processing import get_bike_data
from scripts.utils.data_scraping.energy.data_processing import get_energy_data
from scripts.utils.data_scraping.no2.data_processing import get_no2_data

logger = logging.getLogger(__name__)


class DataScraping:
    """
    Handles data scraping for weather, air quality, and target variables.

    This class updates historical and forecast data for weather and air quality in Germany and Karlsruhe.
    It also updates target data such as bike, energy, and NO₂ measurements.
    """

    def update_weather_data(self, past_days=30, forecast_days=7):
        """
        Update weather and air quality data if the historical data is older than the threshold.

        Args:
            past_days (int): Number of past days to consider.
            forecast_days (int): Number of forecast days to fetch.
        """
        logger.info("Starting update of weather and air quality data.")

        # Load existing historical data
        old_hist_weather_germany = pd.read_csv(
            r'data/weather/history/germany/current.csv', index_col=0, parse_dates=True
        )
        hist_weather_karlsruhe = pd.read_csv(
            r'data/weather/history/karlsruhe/current.csv', index_col=0, parse_dates=True
        )
        hist_air_germany = pd.read_csv(
            r'data/air/history/germany/current.csv', index_col=0, parse_dates=True
        )
        hist_air_karlsruhe = pd.read_csv(
            r'data/air/history/karlsruhe/current.csv', index_col=0, parse_dates=True
        )

        # --- Weather Forecast Germany ---
        if (pd.Timestamp.now(tz='UTC') - old_hist_weather_germany.index[-1]) > pd.Timedelta(hours=3):
            logger.info("Old historical weather data for Germany is older than threshold, updating new data.")
            calls_per_day = 60
            logger.info(
                "Fetching new weather forecast data for Germany with past_days=%s, forecast_days=%s, calls_per_day=%s",
                past_days, forecast_days, calls_per_day
            )
            get_wea_de = WeatherForecastGermany(
                past_days=past_days, forecast_days=forecast_days, calls_per_day=calls_per_day
            )
            new_weather_germany = get_wea_de.main()
            os.system('rm -f scripts/utils/data_scraping/weather_air/temp_data/weather/*.csv')
            logger.info("Removed temporary weather forecast files for Germany.")

            new_weather_germany.set_index('date', inplace=True)
            new_weather_germany.index = pd.to_datetime(new_weather_germany.index, utc=True)

            # Split new data into historical and forecast parts
            new_hist_weather_germany = new_weather_germany.loc[new_weather_germany.index < pd.Timestamp.utcnow()]
            new_forecast_germany = new_weather_germany.loc[new_weather_germany.index >= pd.Timestamp.utcnow()]
            new_forecast_germany.to_csv(r'data/weather/forecasts/germany/current.csv')
            logger.info("Saved new forecast weather data for Germany.")

            # Concatenate and remove duplicate entries
            hist_weather_germany = pd.concat([old_hist_weather_germany, new_hist_weather_germany])
            hist_weather_germany = hist_weather_germany[~hist_weather_germany.index.duplicated(keep='first')]
            hist_weather_germany.to_csv(r'data/weather/history/germany/current.csv')
            logger.info("Updated historical weather data for Germany saved.")

            time.sleep(2)

        # --- Weather Forecast Karlsruhe ---
        if (pd.Timestamp.now(tz='UTC') - hist_weather_karlsruhe.index[-1]) > pd.Timedelta(hours=3):
            logger.info("Historical weather data for Karlsruhe is older than threshold, updating new data.")
            calls_per_day = 20
            logger.info(
                "Fetching new weather forecast data for Karlsruhe with past_days=%s, forecast_days=%s, calls_per_day=%s",
                past_days, forecast_days, calls_per_day
            )
            get_wea_ka = WeatherForecastKarlsruhe(
                past_days=past_days, forecast_days=forecast_days, calls_per_day=calls_per_day
            )
            new_weather_karlsruhe = get_wea_ka.main()
            os.system(r'del /Q scripts/utils/data_scraping/weather_air/temp_data/forecast_karlsruhe/*.csv')
            logger.info("Removed temporary weather forecast files for Karlsruhe.")

            new_weather_karlsruhe.set_index('date', inplace=True)
            new_weather_karlsruhe.index = pd.to_datetime(new_weather_karlsruhe.index)

            # Split new data into historical and forecast parts
            new_hist_weather_karlsruhe = new_weather_karlsruhe.loc[new_weather_karlsruhe.index < pd.Timestamp.utcnow()]
            new_forecast_karlsruhe = new_weather_karlsruhe.loc[new_weather_karlsruhe.index >= pd.Timestamp.utcnow()]
            new_forecast_karlsruhe.to_csv(r'data/weather/forecasts/karlsruhe/current.csv')
            logger.info("Saved new forecast weather data for Karlsruhe.")

            # Concatenate and remove duplicate entries
            hist_weather_karlsruhe = pd.concat([hist_weather_karlsruhe, new_hist_weather_karlsruhe])
            hist_weather_karlsruhe = hist_weather_karlsruhe[~hist_weather_karlsruhe.index.duplicated(keep='first')]
            hist_weather_karlsruhe.to_csv(r'data/weather/history/karlsruhe/current.csv')
            logger.info("Updated historical weather data for Karlsruhe saved.")

            time.sleep(2)

        # --- Air Forecast Germany ---
        if (pd.Timestamp.now(tz='UTC') - hist_air_germany.index[-1]) > pd.Timedelta(hours=3):
            logger.info("Historical air quality data for Germany is older than threshold, updating new data.")
            calls_per_day = 60
            logger.info(
                "Fetching new air forecast data for Germany with past_days=%s, forecast_days=%s, calls_per_day=%s",
                past_days, forecast_days, calls_per_day
            )
            get_air_de = AirForecastGermany(
                past_days=past_days, forecast_days=forecast_days, calls_per_day=calls_per_day
            )
            new_air_germany = get_air_de.main()
            os.system(r'del /Q scripts/utils/data_scraping/weather_air/temp_data/air/forecast_germany/*.csv')
            logger.info("Removed temporary air forecast files for Germany.")

            new_air_germany.set_index('date', inplace=True)
            new_air_germany.dropna(how='all', inplace=True)
            new_air_germany.index = pd.to_datetime(new_air_germany.index)

            # Split new data into historical and forecast parts
            new_hist_air_germany = new_air_germany.loc[new_air_germany.index < pd.Timestamp.utcnow()]
            new_forecast_air_germany = new_air_germany.loc[new_air_germany.index >= pd.Timestamp.utcnow()]
            new_forecast_air_germany.to_csv(r'data/air/forecasts/germany/current.csv')
            logger.info("Saved new forecast air quality data for Germany.")

            # Concatenate and remove duplicate entries
            hist_air_germany = pd.concat([hist_air_germany, new_hist_air_germany])
            hist_air_germany = hist_air_germany[~hist_air_germany.index.duplicated(keep='first')]
            hist_air_germany.to_csv(r'data/air/history/germany/current.csv')
            logger.info("Updated historical air quality data for Germany saved.")

            time.sleep(30)

        # --- Air Forecast Karlsruhe ---
        if (pd.Timestamp.now(tz='UTC') - hist_air_karlsruhe.index[-1]) > pd.Timedelta(hours=3):
            logger.info("Historical air quality data for Karlsruhe is older than threshold, updating new data.")
            calls_per_day = 20
            logger.info(
                "Fetching new air forecast data for Karlsruhe with past_days=%s, forecast_days=%s, calls_per_day=%s",
                past_days, forecast_days, calls_per_day
            )
            get_air_ka = AirForecastKarlsruhe(
                past_days=past_days, forecast_days=forecast_days, calls_per_day=calls_per_day
            )
            new_air_karlsruhe = get_air_ka.main()
            os.system(r'del /Q scripts/utils/data_scraping/weather_air/temp_data/air/forecast_karlsruhe/*.csv')
            logger.info("Removed temporary air forecast files for Karlsruhe.")

            new_air_karlsruhe.set_index('date', inplace=True)
            new_air_karlsruhe.dropna(how='all', inplace=True)
            new_air_karlsruhe.index = pd.to_datetime(new_air_karlsruhe.index)

            # Split new data into historical and forecast parts
            new_hist_air_karlsruhe = new_air_karlsruhe.loc[new_air_karlsruhe.index < pd.Timestamp.utcnow()]
            new_forecast_air_karlsruhe = new_air_karlsruhe.loc[new_air_karlsruhe.index >= pd.Timestamp.utcnow()]
            new_forecast_air_karlsruhe.to_csv(r'data/air/forecasts/karlsruhe/current.csv')
            logger.info("Saved new forecast air quality data for Karlsruhe.")

            # Concatenate and remove duplicate entries
            hist_air_karlsruhe = pd.concat([hist_air_karlsruhe, new_hist_air_karlsruhe])
            hist_air_karlsruhe = hist_air_karlsruhe[~hist_air_karlsruhe.index.duplicated(keep='first')]
            hist_air_karlsruhe.to_csv(r'data/air/history/karlsruhe/current.csv')
            logger.info("Updated historical air quality data for Karlsruhe saved.")

    def update_target_data(self):
        """
        Update target data including bike, energy, and NO₂ data.
        """
        logger.info("Updating target data for bike, energy, and NO₂.")
        get_bike_data()
        get_energy_data()
        get_no2_data()
        logger.info("Target data update completed.")
