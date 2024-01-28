import os

import typer
from dotenv import load_dotenv

from src.core.loaders import (
    APIDataSource,
    DataUpdater,
    HopsworkFsInserter,
    MeasurementLoader,
)
from src.core.utils import load_params


def main(config: str = "config.yaml") -> None:
    params = load_params(params_file=config)
    load_dotenv(params.basic.env_path)

    AQI_TOKEN = os.getenv("AQI_TOKEN", "no token")
    FS_API_KEY = os.getenv("FS_API_KEY", "no api key")
    FS_PROJECT_NAME = os.getenv("FS_PROJECT_NAME", "no project name")

    data_updater = DataUpdater(
        loader=MeasurementLoader(
            loader=APIDataSource(token=AQI_TOKEN, city_id=params.basic.city_id)
        ),
        inserter=HopsworkFsInserter(
            fs_projet_name=FS_PROJECT_NAME, fs_api_key=FS_API_KEY
        ),
    )
    data_updater.run()


if __name__ == "__main__":
    typer.run(main)
