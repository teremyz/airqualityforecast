import datetime

from pydantic import BaseModel


class AirQalityMeasurement(BaseModel):
    date: datetime.date
    pm25: float
    pm10: float
    o3: float
    no2: float
    so2: float
    co: float
    aqi: float


class AirQalityPrediction(BaseModel):
    date: datetime.date
    prediction: float
