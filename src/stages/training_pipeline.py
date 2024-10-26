import os

import typer
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv
from xgboost import XGBRegressor

from src.core.loaders import HopsworkDataLoader, MeasurementLoader
from src.core.model import (
    AqiExperimentLogger,
    AqiModel,
    AqiSplitter,
    ModelRegistry,
)
from src.core.pipelines import TrainingPipeline
from src.core.utils import load_params

app = typer.Typer()


@app.command()
def main(
    config: str,
) -> None:
    params = load_params(params_file=config)
    load_dotenv(params.basic.env_path)

    credential = DefaultAzureCredential()
    secret_client = SecretClient(
        vault_url=params.azure.vault_url,
        credential=credential,
    )

    training_pipeline = TrainingPipeline(
        loader=MeasurementLoader(
            loader=HopsworkDataLoader(
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
        model=AqiModel(
            predictor=XGBRegressor(random_state=0),
            ffill_limit=params.train.ffill_limit,
            lags=params.train.lags,
        ),
        logger=AqiExperimentLogger(
            api_key=os.getenv(
                "COMETML_API_KEY",
                secret_client.get_secret("COMETML-API-KEY").value,
            ),
            project_name=os.getenv(
                "COMETML_PROJECT_NAME",
                secret_client.get_secret("COMETML-PROJECT-NAME").value,
            ),
            workspace=os.getenv(
                "COMETML_WORKSPACE_NAME",
                secret_client.get_secret("COMETML-WORKSPACE-NAME").value,
            ),
            artifact_dir=params.train.artifact_dir,
            model_name=params.train.model_name,
            src_dir=params.train.src_dir,
        ),
        splitter=AqiSplitter(),
        model_registry=ModelRegistry(
            api_key=os.getenv(
                "COMETML_API_KEY",
                secret_client.get_secret("COMETML-API-KEY").value,
            ),
            # TODO: api_key, workspace_name and project_name
            # are in ModelRegistry and AqiExperimentLogger,
            # very crowded, Create ModelRegistryConnectionData
            # class, pass that one to them
            workspace_name=os.getenv(
                "COMETML_WORKSPACE_NAME",
                secret_client.get_secret("COMETML-WORKSPACE-NAME").value,
            ),
            project_name=os.getenv(
                "COMETML_PROJECT_NAME",
                secret_client.get_secret("COMETML-PROJECT-NAME").value,
            ),
            model_name=params.train.model_name,
            status="Production",
        ),
    )
    training_pipeline.run()


if __name__ == "__main__":
    app()
