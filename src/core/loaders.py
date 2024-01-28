import hopsworks
import numpy as np
import pandas as pd
import requests

from src.core.data import AirQalityMeasurement


class MeasurementDataSource:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_data(self) -> pd.DataFrame:
        measurement_table = pd.read_csv(self.file_path)
        measurement_table.columns = [
            col.strip() for col in measurement_table.columns
        ]
        return measurement_table


class APIDataSource:
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


class MeasurementLoader:
    def __init__(self, loader: APIDataSource | MeasurementDataSource) -> None:
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
                # date=datetime.date(
                #    *[int(num) for num in data.date.split("/")]
                # ),
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


class HopsworkFsInserter:
    def __init__(self, fs_projet_name: str, fs_api_key: str) -> None:
        self.project = hopsworks.login(
            project=fs_projet_name, api_key_value=fs_api_key
        )

    def insert_data(self, measurements: list[AirQalityMeasurement]) -> None:
        data = pd.DataFrame([v.model_dump() for v in measurements])

        fs = self.project.get_feature_store()

        aqi_fg = fs.get_or_create_feature_group(
            name="air_quality_feature_group",
            description="Budapest air quality index data",
            version=1,
            primary_key=["date"],
            event_time="date",
        )

        # Insert data into feature group
        aqi_fg.insert(
            data,
            write_options={"wait_for_job": True},
        )


class DataUpdater:
    def __init__(
        self, loader: MeasurementLoader, inserter: HopsworkFsInserter
    ):
        self.loader = loader
        self.inserter = inserter

    def run(self) -> None:
        measurements = self.loader.get_measurements()

        self.inserter.insert_data(measurements)
