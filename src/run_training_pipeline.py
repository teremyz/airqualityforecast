"""
Main execution script for running the training pipeline on Azre ML.

This script loads environment variables, retrieves configuration
parameters, and executes an Azure Machine Learning training
pipeline command.

Args:
    config (str): Path to the configuration file containing
    parameters for the training pipeline.

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
    load_dotenv()
    params = load_params(params_file=config)

    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    FS_API_KEY = os.getenv("FS_API_KEY", "")
    FS_PROJECT_NAME = os.getenv("FS_PROJECT_NAME", "")
    COMETML_API_KEY = os.getenv("COMETML_API_KEY", "")
    COMETML_PROJECT_NAME = os.getenv("COMETML_PROJECT_NAME", "")
    COMETML_WORKSPACE_NAME = os.getenv("COMETML_WORKSPACE_NAME", "")

    aml_url = run_command_on_azure(
        config=config,
        cli_command=f"""training_pipeline config.yaml  \
            --comet-project-name {COMETML_PROJECT_NAME} \
            --comet-workspace {COMETML_WORKSPACE_NAME} \
            --fs-api-key {FS_API_KEY} \
            --fs-project-name {FS_PROJECT_NAME} \
            --comet-api-key {COMETML_API_KEY}""",
        display_name="training-pipeline",
        environment=params.azure.environment,
        compute=params.azure.compute,
        ml_client=MLClient(
            DefaultAzureCredential(),
            subscription_id,
            params.azure.resource_group,
            params.azure.workspace,
        ),
    )
    print("Monitor your job at", aml_url)


if __name__ == "__main__":
    typer.run(main)
