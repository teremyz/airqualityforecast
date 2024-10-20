"""
Main execution script for running the inference pipeline.

This script loads environment variables and configuration parameters,
initializes the inference pipeline for predicting air quality data
using a pre-trained model, and inserts predictions into the
Hopsworks Feature Store.

Args:
    config (str): Path to the configuration file containing
    parameters for the inference pipeline.
    comet_project_name (str, optional): CometML project name for model
    management.
    comet_workspace (str, optional): CometML workspace name for model
    management.
    fs_api_key (str, optional): Hopsworks Feature Store API key for
    authentication.
    fs_project_name (str, optional): Name of the Hopsworks Feature
    Store project.
    comet_api_key (str, optional): CometML API key for accessing the
    CometML platform.

Returns:
    None
"""

import datetime
import logging
import os

import typer
from dotenv import load_dotenv

from src.core.loaders import (
    HopsworkFsInserter,
    HopsworkPredictionDataLoader,
    MeasurementLoader,
)
from src.core.model import CometModelDownloader
from src.core.pipelines import InferencePipeline
from src.core.utils import load_params

logging.basicConfig(level=logging.INFO)
app = typer.Typer()


@app.command()
def main(
    config: str,
    comet_project_name: str = "",
    comet_workspace: str = "",
    fs_api_key: str = "",
    fs_project_name: str = "",
    comet_api_key: str = "",
) -> None:
    """
    Main function to run the inference pipeline.

    This function loads environment variables and configuration
    parameters, sets up the data loader, model downloader, and prediction
    writer, and executes the inference pipeline.

    Args:
        config (str): Path to the configuration file containing
        parameters for the inference pipeline.
        comet_project_name (str, optional): CometML project name for
        model management.
        comet_workspace (str, optional): CometML workspace name for
        model management.
        fs_api_key (str, optional): Hopsworks Feature Store API key for
        authentication.
        fs_project_name (str, optional): Name of the Hopsworks Feature
        Store project.
        comet_api_key (str, optional): CometML API key for accessing the
        CometML platform.

    Returns:
        None
    """
    logging.info(f"Config file path: {config}")
    params = load_params(params_file=config)

    logging.info(f"Get env variables: {config}")
    load_dotenv(params.basic.env_path)
    project_name = os.getenv("COMETML_PROJECT_NAME", comet_project_name)
    workspace = os.getenv("COMETML_WORKSPACE_NAME", comet_workspace)

    logging.info("Inference pipeline has started...")
    inference_pipeline = InferencePipeline(
        loader=MeasurementLoader(
            loader=HopsworkPredictionDataLoader(
                # prediction_date = datetime.date.today(),
                prediction_date=datetime.date(
                    2024, 5, 1
                ),  # TODO: change for today
                lags=params.train.lags,
                fs_api_key=os.getenv("FS_API_KEY", fs_api_key),
                fs_project_name=os.getenv("FS_PROJECT_NAME", fs_project_name),
                feature_group_name=params.basic.feature_group_name,
                version=params.basic.feature_group_version,
            )
        ),
        model_downloader=CometModelDownloader(
            api_key=os.getenv("COMETML_API_KEY", comet_api_key),
            workspace=workspace,
            project_name=project_name,
            model_dir=params.inference.model_dir,
            model_name=params.train.model_name,
        ),
        prediction_writer=HopsworkFsInserter(
            fg_name=params.basic.feature_group_name,  # TODO: config
            fg_description=params.basic.feature_group_description,
            fs_projet_name=os.getenv("FS_PROJECT_NAME", fs_project_name),
            fs_api_key=os.getenv("FS_API_KEY", fs_api_key),
        ),
    )

    inference_pipeline.run()
    logging.info("Inference pipeline is ready!")


if __name__ == "__main__":
    app()
