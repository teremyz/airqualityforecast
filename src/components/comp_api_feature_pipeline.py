import logging

import typer
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
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
def main(config: str, config_path: str) -> None:

    credential = DefaultAzureCredential()
    secret_client = SecretClient(
        vault_url="https://mlairquakeyvaultfeeda681.vault.azure.net/",
        credential=credential,
    )

    AQI_TOKEN = secret_client.get_secret("AQI-TOKEN").value
    FS_API_KEY = secret_client.get_secret("FS-API-KEY").value
    FS_PROJECT_NAME = secret_client.get_secret("FS-PROJECT-NAME").value

    logging.info("Load params..")
    params = load_params(params_file=config)
    logging.info("Get env variables")
    load_dotenv(params.basic.env_path)

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
    logging.info(f"MOCK param: {config_path}")


if __name__ == "__main__":
    app()
