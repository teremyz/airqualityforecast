"""
Main execution script for running the feature pipeline.

This script loads environment variables and configuration parameters,
initializes a feature pipeline, and processes air quality data from
an external API. It inserts the processed data into Hopsworks Feature Store.

Args:
    config (str): Path to the configuration file containing
    parameters for the feature pipeline.
    aqi_token (str, optional): Air Quality Index (AQI) API token
    used for accessing data.
    fs_api_key (str, optional): Hopsworks Feature Store API key for
    authentication.
    fs_project_name (str, optional): Name of the Hopsworks Feature
    Store project.

Returns:
    None
"""

import logging
import os

import typer
from dotenv import load_dotenv

from src.core.loaders import (
    APIDataSource,
    HopsworkFsInserter,
    MeasurementLoader,
)
from src.core.pipelines import FeaturePipeline
from src.core.utils import load_params

logging.basicConfig(level=logging.INFO)
app = typer.Typer()


@app.command()
def main(
    config: str,
    aqi_token: str = "",
    fs_api_key: str = "",
    fs_project_name: str = "",
) -> None:
    """
    Main function to run the feature pipeline.

    This function loads environment variables and configuration
    parameters, sets up data loading from the AQI API, and inserts
    processed data into the Hopsworks Feature Store.

    Args:
        config (str): Path to the configuration file containing
        parameters for the feature pipeline.
        aqi_token (str, optional): Air Quality Index (AQI) API token
        used for accessing data.
        fs_api_key (str, optional): Hopsworks Feature Store API key
        for authentication.
        fs_project_name (str, optional): Name of the Hopsworks Feature
        Store project.

    Returns:
        None
    """
    logging.info("Load params..")
    params = load_params(params_file=config)
    logging.info("Get env variables")
    load_dotenv(params.basic.env_path)

    AQI_TOKEN = os.getenv("AQI_TOKEN", aqi_token)
    FS_API_KEY = os.getenv("FS_API_KEY", fs_api_key)
    FS_PROJECT_NAME = os.getenv("FS_PROJECT_NAME", fs_project_name)

    logging.info("Feature pipeline has started..")
    feature_pipeline = FeaturePipeline(
        loader=MeasurementLoader(
            loader=APIDataSource(token=AQI_TOKEN, city_id=params.basic.city_id)
        ),
        inserter=HopsworkFsInserter(
            fg_name=params.basic.feature_group_name,
            fg_description=params.basic.feature_group_description,
            fs_projet_name=FS_PROJECT_NAME,
            fs_api_key=FS_API_KEY,
        ),
    )
    feature_pipeline.run()
    logging.info("Feature pipeline is ready!")


if __name__ == "__main__":
    app()
