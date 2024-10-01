import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import yaml
from azure.ai.ml import MLClient, command
from box import ConfigBox
from comet_ml import API, Experiment


def load_params(params_file: str) -> ConfigBox:
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    return ConfigBox(params)


def reindex_dataframe(df: pd.DataFrame, date_col_name: str) -> pd.DataFrame:
    df = df.sort_values(date_col_name, ascending=True)
    df = df.set_index(date_col_name)
    df.index = pd.DatetimeIndex(df.index)

    df = df.reindex(
        pd.date_range(df.index.min(), df.index.max()), fill_value=np.nan
    )

    return df


def get_best_experiment_today(
    api_key: str,
    workspace_name: str,
    project_name: str,
    metric_name: str = "test_rmse",
) -> Experiment:
    api = API(api_key=api_key)
    min_metric_value: float = 2000

    for experiment in api.get(os.path.join(workspace_name, project_name)):
        experiment_date = datetime.utcfromtimestamp(
            experiment.get_metadata()["endTimeMillis"] / 1000
        ).strftime("%Y-%m-%d")

        if experiment_date == date.today().strftime("%Y-%m-%d"):
            metrics = experiment.get_metrics()
            print(metrics)
            metric = float(
                [x for x in metrics if x["metricName"] == metric_name][0][
                    "metricValue"
                ]
            )
            if min_metric_value > metric:
                min_metric_value = metric
                best_experiment = experiment
    return best_experiment


def set_prod_status_to_none(
    api_key: str, workspace: str, registry_name: str
) -> None:
    api = API(api_key=api_key)
    registry_assets = api.get_model_registry_version_assets(
        workspace=workspace, registry_name=registry_name
    )
    model = api.get_model(workspace=workspace, model_name=registry_name)

    for registered_model in registry_assets["experimentModel"][
        "registryRecords"
    ]:
        if registered_model["status"] == "Production":
            current_prod_version = registered_model["version"]

    model.set_status(version=current_prod_version, status="None")


def run_command_on_azure(
    config: str, cli_command: str, params: ConfigBox, ml_client: MLClient
) -> str:
    # configure job
    job = command(
        code=config,
        command=cli_command,
        environment=params.azure.environment,
        compute=params.azure.compute,
        display_name=params.azure.feature_display_name,
        experiment_name=params.azure.feature_experiment_name,
    )

    # submit job
    returned_job = ml_client.create_or_update(job)
    return returned_job.studio_url
