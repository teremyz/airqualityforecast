import os
import sys
from pathlib import Path

# isort: off
src_path = Path(os.path.abspath("")).resolve()  # noqa: E731
sys.path.append(str(src_path))  # noqa: E731
# isort: on

import typer  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from core.loaders import (  # noqa: E402
    APIDataSource,
    DataUpdater,
    HopsworkFsInserter,
    MeasurementLoader,
)
from core.utils import load_params  # noqa: E402


def main(config: str = "../config.yaml") -> None:
    print(__file__)
    params = load_params(params_file=config)
    load_dotenv(params.basic.env_path)

    AQI_TOKEN = os.getenv("AQI_TOKEN")
    FS_API_KEY = os.getenv("FS_API_KEY")
    FS_PROJECT_NAME = os.getenv("FS_PROJECT_NAME")

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
