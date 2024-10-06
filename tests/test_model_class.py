from datetime import date

import numpy as np
from xgboost import XGBRegressor

from src.core.data import AirQalityMeasurement
from src.core.model import AqiModel


def test_preprocess_input():
    lags = 5
    ffill_limit = 3
    input_data = [
        AirQalityMeasurement(
            date=date(2024, 2, 3),
            pm25=64,
            pm10=35,
            o3=10,
            no2=13,
            so2=4,
            co=2,
            aqi=64,
        ),
        AirQalityMeasurement(
            date=date(2024, 2, 4),
            pm25=64,
            pm10=35,
            o3=np.nan,
            no2=13,
            so2=4,
            co=2,
            aqi=64,
        ),
        AirQalityMeasurement(
            date=date(2024, 2, 5),
            pm25=64,
            pm10=35,
            o3=np.nan,
            no2=13,
            so2=4,
            co=2,
            aqi=64,
        ),
        AirQalityMeasurement(
            date=date(2024, 2, 6),
            pm25=64,
            pm10=35,
            o3=13,
            no2=13,
            so2=4,
            co=2,
            aqi=64,
        ),
        AirQalityMeasurement(
            date=date(2024, 2, 7),
            pm25=64,
            pm10=35,
            o3=np.nan,
            no2=13,
            so2=4,
            co=2,
            aqi=64,
        ),
        AirQalityMeasurement(
            date=date(2024, 2, 8),
            pm25=64,
            pm10=35,
            o3=10,
            no2=13,
            so2=4,
            co=2,
            aqi=64,
        ),
        AirQalityMeasurement(
            date=date(2024, 2, 9),
            pm25=64,
            pm10=35,
            o3=np.nan,
            no2=13,
            so2=4,
            co=2,
            aqi=64,
        ),
    ]
    model = AqiModel(
        predictor=XGBRegressor(random_state=0),
        ffill_limit=ffill_limit,
        lags=lags,
    )

    data = model.process_inputs(input_data)

    assert data.shape[0] == len(input_data)
    assert data.shape[1] == 6 * lags + 3
    assert data["o3Shifted1"].isna().sum() == 1


def test_preprocess_target():
    # TBA
    assert 1 == 1


def test_predict_output():
    # TBA
    assert 1 == 1
