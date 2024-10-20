"""
Main execution script for running an Azure feature pipeline.

This script loads environment variables, retrieves configuration parameters,
and executes an Azure Machine Learning pipeline command using the Azure CLI.

Args:
    config (str): Path to the configuration file containing parameters
    for the pipeline.

Returns:
    None
"""

import os

import typer
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from src.core.utils import load_params, run_command_on_azure


def main(config: str) -> None:
    """
    Main function that loads environment variables, configuration
    parameters, and runs the feature pipeline command on Azure.

    Args:
        config (str): Path to the configuration file containing parameters.

    Returns:
        None
    """

    load_dotenv()
    params = load_params(params_file=config)

    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    FS_API_KEY = os.getenv("FS_API_KEY", "")
    FS_PROJECT_NAME = os.getenv("FS_PROJECT_NAME", "")
    AQI_TOKEN = os.getenv("AQI_TOKEN", "")

    aml_url = run_command_on_azure(
        config=config,
        cli_command=f"""feature_pipeline config.yaml  \
            --aqi-token {AQI_TOKEN} \
            --fs-api-key {FS_API_KEY} \
            --fs-project-name {FS_PROJECT_NAME}""",
        params=params,
        ml_client=MLClient(
            DefaultAzureCredential(),
            subscription_id,
            params.azure.resource_group,
            params.azure.workspace,
        ),
    )

    print(f"Monitor your job at {aml_url}")


if __name__ == "__main__":
    typer.run(main)
