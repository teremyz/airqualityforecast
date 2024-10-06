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
    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        pass


class MeasurementDataSource(Loader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_data(self) -> pd.DataFrame:
        measurement_table = pd.read_csv(self.file_path)
        measurement_table.columns = [
            col.strip() for col in measurement_table.columns
        ]
        return measurement_table


class APIDataSource(Loader):
    def __init__(self, token: str, city_id: int = 3371):
        self.token = token
        self.city_id = city_id

    def get_data(self) -> pd.DataFrame:
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
        fs = self.project.get_feature_store()

        air_quality_fg = fs.get_feature_group(
            name=self.feature_group_name, version=self.version
        )

        return air_quality_fg.read()


class ValidationLoader(ABC):
    @abstractmethod
    def get_measurements(self) -> list[AirQalityMeasurement]:
        pass


class MeasurementLoader(ValidationLoader):
    def __init__(
        self,
        loader: Loader,
    ) -> None:
        self.loader = loader

    def get_measurements(self) -> list[AirQalityMeasurement]:
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
    @abstractmethod
    def insert_data(
        self, data: List[AirQalityMeasurement] | List[AirQalityPrediction]
    ) -> None:
        # TODO: only one AirQalityMeasurement | AirQalityPrediction,
        # BaseModel not allowed for some reason
        pass


class HopsworkFsInserter(Inserter):
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
