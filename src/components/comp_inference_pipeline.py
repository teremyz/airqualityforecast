import datetime
import logging

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
def main(config: str) -> None:
    logging.info(f"Config file path: {config}")
    params = load_params(params_file=config)

    logging.info(f"Get env variables: {config}")
    load_dotenv(params.basic.env_path)

    # Get credentials
    credential = DefaultAzureCredential()
    secret_client = SecretClient(
        vault_url="https://mlairquakeyvaultfeeda681.vault.azure.net/",
        credential=credential,
    )

    project_name = secret_client.get_secret("COMETML-PROJECT-NAME").value
    workspace = secret_client.get_secret("COMETML-WORKSPACE-NAME").value
    fs_api_key = secret_client.get_secret("FS-API-KEY").value
    fs_project_name = secret_client.get_secret("FS-PROJECT-NAME").value
    comet_api_key = secret_client.get_secret("COMETML-API-KEY").value

    logging.info("Inference pipeline has started...")
    inference_pipeline = InferencePipeline(
        loader=MeasurementLoader(
            loader=HopsworkPredictionDataLoader(
                # prediction_date = datetime.date.today(),
                prediction_date=datetime.date(
                    2024, 5, 1
                ),  # TODO: change for today
                lags=params.train.lags,
                fs_api_key=fs_api_key,
                fs_project_name=fs_project_name,
                feature_group_name=params.basic.feature_group_name,
                version=params.basic.feature_group_version,
            )
        ),
        model_downloader=CometModelDownloader(
            api_key=comet_api_key,
            workspace=workspace,
            project_name=project_name,
            model_dir=params.inference.model_dir,
            model_name=params.train.model_name,
        ),
        prediction_writer=HopsworkFsInserter(
            fg_name=params.basic.feature_group_name,  # TODO: config
            fg_description=params.basic.feature_group_description,
            fs_projet_name=fs_project_name,
            fs_api_key=fs_project_name,
        ),
    )

    inference_pipeline.run()
    logging.info("Inference pipeline is ready!")


if __name__ == "__main__":
    app()
