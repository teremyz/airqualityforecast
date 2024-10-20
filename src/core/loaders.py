import datetime
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import List

import hopsworks
import numpy as np
import pandas as pd
import requests

from src.core.data import AirQalityMeasurement, AirQalityPrediction


class Loader(ABC):
    """
    Abstract base class for loading data required for air quality
    predictions.

    Args:
        None

    Returns:
        Loader: An abstract class that defines the method to load data.
    """

    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        """
        Load the air quality data.

        This method is responsible for retrieving the data necessary
        for predicting air quality. It should be implemented by
        subclasses.

        Args:
            None

        Returns:
            pd.DataFrame: A DataFrame containing the air quality data
            to be used for predictions.
        """
        pass


class MeasurementDataSource(Loader):
    """
    A data source for loading air quality measurements from a file.

    Args:
        file_path (str): The path to the CSV file containing air quality
        measurement data.

    Returns:
        MeasurementDataSource: A data source object that loads air
        quality data from the specified file.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_data(self) -> pd.DataFrame:
        """
        Load air quality measurement data from the specified CSV file.

        This method reads the CSV file and returns the data as a
        pandas DataFrame, ensuring that column names are properly
        formatted by stripping whitespace.

        Args:
            None

        Returns:
            pd.DataFrame: A DataFrame containing air quality measurement
            data from the file.
        """
        measurement_table = pd.read_csv(self.file_path)
        measurement_table.columns = [
            col.strip() for col in measurement_table.columns
        ]
        return measurement_table


class APIDataSource(Loader):
    """
    A data source for loading air quality measurements from an API.

    Args:
        token (str): The API token used for authentication.
        city_id (int, optional): The ID of the city for which to
        retrieve air quality data. Defaults to 3371.

    Returns:
        APIDataSource: A data source object that loads air quality data
        from the specified API.
    """

    def __init__(self, token: str, city_id: int = 3371):
        self.token = token
        self.city_id = city_id

    def get_data(self) -> pd.DataFrame:
        """
        Load air quality measurement data from the API.

        This method retrieves air quality data for the specified city
        from the API and returns it as a pandas DataFrame. If some
        pollutants' data is missing, it will insert NaN values for
        those fields.

        Args:
            None

        Returns:
            pd.DataFrame: A DataFrame containing the air quality
            measurements (pm25, pm10, o3, no2, so2, co) along with the
            date of the measurement.
        """
        resp = requests.get(
            f"https://api.waqi.info/feed/@{self.city_id}/?token={self.token}"
        ).json()
        resp_data = dict()

        for key in ["pm25", "pm10", "o3", "no2", "so2", "co"]:
            try:
                resp_data[key] = resp["data"]["iaqi"][key]["v"]
            except KeyError:
                resp_data[key] = np.nan

        resp_data["date"] = resp["data"]["time"]["s"]

        return pd.DataFrame(resp_data, index=[0])


class HopsworkDataLoader(Loader):
    """
    A data loader for retrieving air quality measurements from a
    Hopsworks feature store.

    Args:
        fs_api_key (str): The API key for authenticating with Hopsworks.
        fs_project_name (str): The name of the Hopsworks project.
        feature_group_name (str): The name of the feature group to
        retrieve air quality data from.
        version (int, optional): The version of the feature group.
        Defaults to 1.

    Returns:
        HopsworkDataLoader: A data loader object for accessing air
        quality data from Hopsworks.
    """

    def __init__(
        self,
        fs_api_key: str,
        fs_project_name: str,
        feature_group_name: str,
        version: int = 1,
    ):
        self.feature_group_name = feature_group_name
        self.version = version
        self.project = hopsworks.login(
            project=fs_project_name, api_key_value=fs_api_key
        )

    def get_data(self) -> pd.DataFrame:
        """
        Retrieve air quality data from the Hopsworks feature store.

        This method connects to the Hopsworks feature store and reads
        the specified version of the air quality feature group as a
        pandas DataFrame.

        Args:
            None

        Returns:
            pd.DataFrame: A DataFrame containing the air quality
            measurements from the feature group.
        """
        fs = self.project.get_feature_store()

        air_quality_fg = fs.get_feature_group(
            name=self.feature_group_name, version=self.version
        )

        return air_quality_fg.read()


class ValidationLoader(ABC):
    """
    Abstract base class for loading and validating air quality
    measurements.

    Args:
        None

    Returns:
        ValidationLoader: An abstract class that defines the method to
        load and validate air quality measurements.
    """

    @abstractmethod
    def get_measurements(self) -> list[AirQalityMeasurement]:
        """
        Retrieve a list of air quality measurements.

        This method should be implemented by subclasses to load and
        return a list of validated air quality measurements.

        Args:
            None

        Returns:
            list[AirQalityMeasurement]: A list of air quality
            measurements.
        """
        pass


class MeasurementLoader(ValidationLoader):
    """
    A loader for retrieving air quality measurements from a data
    source.

    Args:
        loader (Loader): A data source object for loading raw air
        quality data.

    Returns:
        MeasurementLoader: A loader object that retrieves and validates
        air quality measurements.
    """

    def __init__(
        self,
        loader: Loader,
    ) -> None:
        self.loader = loader

    def get_measurements(self) -> list[AirQalityMeasurement]:
        """
        Retrieve and validate air quality measurements.

        This method loads data from the specified data source, converts
        each row into an `AirQalityMeasurement` object, and calculates
        the AQI as the maximum value of the provided pollutants.

        Args:
            None

        Returns:
            list[AirQalityMeasurement]: A list of validated air quality
            measurements.
        """
        measurements = []
        for data in self.loader.get_data().itertuples():
            pm25 = pd.to_numeric(data.pm25, errors="coerce")
            pm10 = pd.to_numeric(data.pm10, errors="coerce")
            o3 = pd.to_numeric(data.o3, errors="coerce")
            no2 = pd.to_numeric(data.no2, errors="coerce")
            so2 = pd.to_numeric(data.so2, errors="coerce")
            co = pd.to_numeric(data.co, errors="coerce")

            measurement = AirQalityMeasurement(
                date=pd.to_datetime(data.date).date(),
                pm25=pm25,
                pm10=pm10,
                o3=o3,
                no2=no2,
                so2=so2,
                co=co,
                aqi=np.nanmax([pm25, pm10, o3, no2, so2, co]),
            )

            measurements.append(measurement)
        return measurements


class Inserter(ABC):
    """
    Abstract base class for inserting air quality measurement and
    prediction data into a data store.

    Args:
        None

    Returns:
        Inserter: An abstract class that defines the method for data
        insertion.
    """

    @abstractmethod
    def insert_data(
        self, data: List[AirQalityMeasurement] | List[AirQalityPrediction]
    ) -> None:
        """
        Insert air quality measurement or prediction data.

        This method should be implemented by subclasses to define how
        the data should be inserted into the data store.

        Args:
            data (list[AirQalityMeasurement] | list[AirQalityPrediction]):
            A list of air quality measurements or predictions to insert.

        Returns:
            None
        """
        pass


class HopsworkFsInserter(Inserter):
    """
    A data inserter for storing air quality measurements and
    predictions in a Hopsworks feature store.

    Args:
        fg_name (str): The name of the feature group in Hopsworks.
        fg_description (str): A description of the feature group.
        fs_project_name (str): The name of the Hopsworks project.
        fs_api_key (str): The API key for authenticating with Hopsworks.

    Returns:
        HopsworkFsInserter: An inserter object for storing air quality
        data in Hopsworks.
    """

    def __init__(
        self,
        fg_name: str,
        fg_description: str,
        fs_projet_name: str,
        fs_api_key: str,
    ) -> None:
        self.fg_name = fg_name
        self.fg_description = fg_description
        self.project = hopsworks.login(
            project=fs_projet_name, api_key_value=fs_api_key
        )

    def insert_data(
        self, data: List[AirQalityMeasurement] | List[AirQalityPrediction]
    ) -> None:
        """
        Insert air quality measurement or prediction data into the
        Hopsworks feature store.

        This method converts the provided data into a DataFrame and
        inserts it into the specified feature group.

        Args:
            data (list[AirQalityMeasurement] | list[AirQalityPrediction]):
            A list of air quality measurements or predictions to insert.

        Returns:
            None
        """
        data = pd.DataFrame([v.model_dump() for v in data])

        fs = self.project.get_feature_store()

        aqi_fg = fs.get_or_create_feature_group(
            name=self.fg_name,
            description=self.fg_description,
            version=1,
            primary_key=["date"],
            event_time="date",
        )

        # Insert data into feature group
        aqi_fg.insert(
            data,
            write_options={"wait_for_job": True},
        )


class HopsworkPredictionDataLoader(Loader):
    """
    A data loader for retrieving air quality prediction data from a
    Hopsworks feature store.

    Args:
        fs_api_key (str): The API key for authenticating with Hopsworks.
        fs_project_name (str): The name of the Hopsworks project.
        feature_group_name (str): The name of the feature group to
        retrieve air quality data from.
        prediction_date (datetime.date): The date for which predictions
        are to be made.
        lags (int): The number of days to consider for lagged features.
        version (int, optional): The version of the feature group.
        Defaults to 1.

    Returns:
        HopsworkPredictionDataLoader: A data loader object for accessing
        air quality prediction data from Hopsworks.
    """

    def __init__(
        self,
        fs_api_key: str,
        fs_project_name: str,
        feature_group_name: str,
        prediction_date: datetime.date,
        lags: int,
        version: int = 1,
    ):
        self.prediction_date = prediction_date
        self.lags = lags
        self.feature_group_name = feature_group_name
        self.version = version
        self.project = hopsworks.login(
            project=fs_project_name, api_key_value=fs_api_key
        )

    def get_data(self) -> pd.DataFrame:
        """
        Retrieve air quality prediction data from the Hopsworks feature
        store.

        This method filters the feature group to include only data from
        the specified prediction date and the specified number of lagged
        days.

        Args:
            None

        Returns:
            pd.DataFrame: A DataFrame containing the air quality data
            for the specified prediction date and lagged features.
        """
        fs = self.project.get_feature_store()

        air_quality_fg = fs.get_feature_group(
            name=self.feature_group_name, version=self.version
        )

        air_quality_fg = air_quality_fg.filter(
            (air_quality_fg.date <= self.prediction_date)
            & (
                air_quality_fg.date
                >= self.prediction_date - timedelta(days=self.lags)
            )
        )

        return air_quality_fg.read(read_options={"use_hive": True})
