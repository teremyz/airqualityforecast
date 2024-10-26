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
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
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
    credential = DefaultAzureCredential()
    secret_client = SecretClient(
        vault_url=params.azure.vault_url,
        credential=credential,
    )

    project_name = os.getenv(
        "COMETML_PROJECT_NAME",
        secret_client.get_secret("COMETML-PROJECT-NAME").value,
    )
    workspace = os.getenv(
        "COMETML_WORKSPACE_NAME",
        secret_client.get_secret("COMETML-WORKSPACE-NAME").value,
    )

    logging.info("Inference pipeline has started...")
    inference_pipeline = InferencePipeline(
        loader=MeasurementLoader(
            loader=HopsworkPredictionDataLoader(
                # prediction_date = datetime.date.today(),
                prediction_date=datetime.date(
                    2024, 5, 1
                ),  # TODO: change for today
                lags=params.train.lags,
                fs_api_key=os.getenv(
                    "FS_API_KEY", secret_client.get_secret("FS-API-KEY").value
                ),
                fs_project_name=os.getenv(
                    "FS_PROJECT_NAME",
                    secret_client.get_secret("FS-PROJECT-NAME").value,
                ),
                feature_group_name=params.basic.feature_group_name,
                version=params.basic.feature_group_version,
            )
        ),
        model_downloader=CometModelDownloader(
            api_key=os.getenv(
                "COMETML_API_KEY",
                secret_client.get_secret("COMETML-API-KEY").value,
            ),
            workspace=workspace,
            project_name=project_name,
            model_dir=params.inference.model_dir,
            model_name=params.train.model_name,
        ),
        prediction_writer=HopsworkFsInserter(
            fg_name=params.basic.prediction_group_name,
            fg_description=params.basic.prediction_group_description,
            fs_projet_name=os.getenv(
                "FS_PROJECT_NAME",
                secret_client.get_secret("FS-PROJECT-NAME").value,
            ),
            fs_api_key=os.getenv(
                "FS_API_KEY", secret_client.get_secret("FS-API-KEY").value
            ),
        ),
    )

    inference_pipeline.run()
    logging.info("Inference pipeline is ready!")


if __name__ == "__main__":
    app()
