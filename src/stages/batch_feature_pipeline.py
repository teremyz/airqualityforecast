import os

import typer
from dotenv import load_dotenv

from src.core.loaders import (
    HopsworkFsInserter,
    MeasurementDataSource,
    MeasurementLoader,
)
from src.core.pipelines import FeaturePipeline
from src.core.utils import load_params


def main(config: str = "config.yaml") -> None:
    params = load_params(params_file=config)
    load_dotenv(params.basic.env_path)

    FS_API_KEY = os.getenv("FS_API_KEY", "no api key")
    FS_PROJECT_NAME = os.getenv("FS_PROJECT_NAME", "no project name")

    feature_pipeline = FeaturePipeline(
        loader=MeasurementLoader(
            loader=MeasurementDataSource(file_path=params.basic.data_path)
        ),
        inserter=HopsworkFsInserter(
            fg_name=params.basic.feature_group_name,
            fg_description=params.basic.feature_group_description,
            fs_projet_name=FS_PROJECT_NAME,
            fs_api_key=FS_API_KEY,
        ),
    )
    feature_pipeline.run()


if __name__ == "__main__":
    typer.run(main)
