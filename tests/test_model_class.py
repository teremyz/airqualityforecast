from datetime import date

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from src.core.data import AirQalityMeasurement
from src.core.model import AqiModel
from src.core.utils import reindex_dataframe


@pytest.fixture
def measurements():
    return [
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
            aqi=np.nan,
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
            date=date(2024, 2, 10),
            pm25=64,
            pm10=35,
            o3=np.nan,
            no2=13,
            so2=4,
            co=2,
            aqi=64,
        ),
    ]


def test_preprocess_input(measurements):
    lags = 5
    ffill_limit = 3
    model = AqiModel(
        predictor=XGBRegressor(random_state=0),
        ffill_limit=ffill_limit,
        lags=lags,
    )
    data = model.process_inputs(measurements=measurements)

    assert data.shape[0] == len(measurements) + 1, "Reindexing is not working"
    assert data.shape[1] == 6 * lags + 3, "Lag generation is not working"
    assert data["o3Shifted1"].isna().sum() == 1, "Ffill is not working"


def test_preprocess_target(measurements):
    lags = 5
    ffill_limit = 3
    model = AqiModel(
        predictor=XGBRegressor(random_state=0),
        ffill_limit=ffill_limit,
        lags=lags,
    )

    target = model.prerocess_target(measurements=measurements)

    assert target["aqi"].isna().sum() == 0, "fillna is not working"
    assert (
        target.shape[0] == len(measurements) + 1
    ), "Reindexing is not working"


def test_predict_output(measurements):
    lags = 5
    ffill_limit = 3
    model = AqiModel(
        predictor=XGBRegressor(random_state=0),
        ffill_limit=ffill_limit,
        lags=lags,
    )
    model.train(measurements)
    prediction = model.predict(measurements)

    assert len(prediction) == len(measurements) + 1
    assert np.sum([np.isnan(x.prediction) for x in prediction]) == 0


def test_reindex_dataframe():
    date_col_name = "date"
    df = pd.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 4)],
            "value": 1,
        }
    )
    reindexed_df = reindex_dataframe(df, date_col_name)
    assert reindexed_df.shape[0] == 4
    assert "2024-01-03" in reindexed_df.index
