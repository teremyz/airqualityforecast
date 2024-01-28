import os

import pandas as pd
from dotenv import load_dotenv

from src.core.data import AirQalityMeasurement
from src.core.loaders import (
    APIDataSource,
    MeasurementDataSource,
    MeasurementLoader,
)
from src.core.utils import load_params


def test_measurement_data_source():
    params = load_params(params_file="config.yaml")
    loader = MeasurementDataSource(file_path=params.basic.data_path)

    df = loader.get_data()

    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 7


def test_api_data_source():
    params = load_params(params_file="config.yaml")
    load_dotenv(params.basic.env_path)

    AQI_TOKEN = os.getenv("AQI_TOKEN")
    loader = APIDataSource(token=AQI_TOKEN, city_id=params.basic.city_id)

    df = loader.get_data()

    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 7


def test_measurement_loader():
    params = load_params(params_file="config.yaml")
    load_dotenv(params.basic.env_path)

    AQI_TOKEN = os.getenv("AQI_TOKEN")

    loader = MeasurementLoader(
        loader=APIDataSource(token=AQI_TOKEN, city_id=params.basic.city_id)
    )

    measurements = loader.get_measurements()
    assert isinstance(measurements, list)
    assert len(measurements) == 1
    assert isinstance(measurements[0], AirQalityMeasurement)
