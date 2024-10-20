import datetime

from pydantic import BaseModel


class AirQalityMeasurement(BaseModel):
    """
    A model to represent air quality data for a specific date.

    Args:
        date (datetime.date): The date of the air quality record.
        pm25 (float): Concentration of PM2.5 particles in µg/m³.
        pm10 (float): Concentration of PM10 particles in µg/m³.
        o3 (float): Concentration of ozone (O3) in µg/m³.
        no2 (float): Concentration of nitrogen dioxide (NO2) in µg/m³.
        so2 (float): Concentration of sulfur dioxide (SO2) in µg/m³.
        co (float): Concentration of carbon monoxide (CO) in µg/m³.
        aqi (float): Calculated Air Quality Index (AQI).

    Returns:
        AirQualityMeasurement: An object containing air quality data
        for the specified date, including the AQI.
    """

    date: datetime.date
    pm25: float
    pm10: float
    o3: float
    no2: float
    so2: float
    co: float
    aqi: float


class AirQalityPrediction(BaseModel):
    """
    A model to represent the predicted air quality for a specific date.

    Args:
        date (datetime.date): The date for which the air quality is
        predicted.
        prediction (float): The predicted Air Quality Index (AQI)
        for the given date.

    Returns:
        AirQualityPrediction: An object containing the predicted AQI
        for the specified date.
    """

    date: datetime.date
    prediction: float
