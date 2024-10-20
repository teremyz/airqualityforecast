import os

import typer
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
    comet_project_name: str = "",
    comet_workspace: str = "",
    fs_api_key: str = "",
    fs_project_name: str = "",
    comet_api_key: str = "",
) -> None:
    params = load_params(params_file=config)
    load_dotenv(params.basic.env_path)

    training_pipeline = TrainingPipeline(
        loader=MeasurementLoader(
            loader=HopsworkDataLoader(
                fs_api_key=os.getenv("FS_API_KEY", fs_api_key),
                fs_project_name=os.getenv("FS_PROJECT_NAME", fs_project_name),
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
            api_key=os.getenv("COMETML_API_KEY", comet_api_key),
            project_name=os.getenv("COMETML_PROJECT_NAME", comet_project_name),
            workspace=os.getenv("COMETML_WORKSPACE_NAME", comet_workspace),
            artifact_dir=params.train.artifact_dir,
            model_name=params.train.model_name,
            src_dir=params.train.src_dir,
        ),
        splitter=AqiSplitter(),
        model_registry=ModelRegistry(
            api_key=os.getenv("COMETML_API_KEY", comet_api_key),
            # TODO: api_key, workspace_name and project_name
            # are in ModelRegistry and AqiExperimentLogger,
            # very crowded, Create ModelRegistryConnectionData
            # class, pass that one to them
            workspace_name=os.getenv(
                "COMETML_WORKSPACE_NAME", comet_project_name
            ),
            project_name=os.getenv("COMETML_PROJECT_NAME", comet_workspace),
            model_name=params.train.model_name,
            status="Production",
        ),
    )
    training_pipeline.run()


if __name__ == "__main__":
    app()
