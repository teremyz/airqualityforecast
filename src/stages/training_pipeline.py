import os

import typer
from dotenv import load_dotenv
from xgboost import XGBRegressor

from src.core.loaders import HopsworkDataLoader, MeasurementLoader
from src.core.model import AqiExperimentLogger, AqiModel, AqiSplitter
from src.core.pipelines import TrainingPipeline
from src.core.utils import load_params


def main(config: str = "config.yaml") -> None:
    params = load_params(params_file=config)
    load_dotenv(params.basic.env_path)

    training_pipeline = TrainingPipeline(
        loader=MeasurementLoader(
            loader=HopsworkDataLoader(
                fs_api_key=os.getenv("FS_API_KEY", ""),
                fs_project_name=os.getenv("FS_PROJECT_NAME", ""),
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
            api_key=os.getenv("COMETML_API_KEY", ""),
            project_name=os.getenv("COMETML_PROJECT_NAME", ""),
            workspace=os.getenv("COMETML_WORKSPACE_NAME", ""),
            artifact_dir=params.train.artifact_dir,
            model_name=params.train.model_name,
            src_dir=params.train.src_dir,
        ),
        splitter=AqiSplitter(),
    )
    training_pipeline.run()


if __name__ == "__main__":
    typer.run(main)
