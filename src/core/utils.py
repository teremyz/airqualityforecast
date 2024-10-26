import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import yaml
from azure.ai.ml import MLClient, command
from box import ConfigBox
from comet_ml import API, Experiment


def load_params(params_file: str) -> ConfigBox:
    """
    Load parameters from a YAML file and return them as a ConfigBox object.

    This function reads a YAML file, loads its contents into a dictionary, and
    wraps it into a ConfigBox object for easier access to the configuration
    parameters.

    Args:
        params_file (str): The path to the YAML file containing configuration
            parameters.

    Returns:
        ConfigBox: A ConfigBox object containing the parameters loaded from the
        YAML file.

    Raises:
        FileNotFoundError: If the specified `params_file` does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.

    Example:
        >>> config = load_params("config.yaml")
        This will load the parameters from the 'config.yaml' file and return a
        ConfigBox object.
    """
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    return ConfigBox(params)


def reindex_dataframe(df: pd.DataFrame, date_col_name: str) -> pd.DataFrame:
    """
    Reindex a DataFrame based on a specified date column.

    This function sorts the DataFrame by the specified date column, sets the
    date column as the index, and reindexes the DataFrame to ensure that all
    dates in the range from the minimum to the maximum date are included,
    filling missing dates with NaN.

    Args:
        df (pd.DataFrame): The DataFrame to reindex.
        date_col_name (str): The name of the column containing date values.

    Returns:
        pd.DataFrame: A reindexed DataFrame with a DatetimeIndex.

    Raises:
        KeyError: If `date_col_name` does not exist in the DataFrame.
        ValueError: If the DataFrame is empty or the date column has invalid
            values.

    Example:
        >>> df = pd.DataFrame({
        ...     'date': ['2024-01-01', '2024-01-03'],
        ...     'value': [10, 20]
        ... })
        >>> df['date'] = pd.to_datetime(df['date'])
        >>> reindexed_df = reindex_dataframe(df, 'date')
    """
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
    """
    Retrieve the best experiment for today based on a specified metric.

    This function connects to the API using the provided API key, retrieves
    all experiments for the specified workspace and project, and identifies
    the experiment with the lowest value for the specified metric that was
    completed today.

    Args:
        api_key (str): The API key for authentication.
        workspace_name (str): The name of the workspace to query.
        project_name (str): The name of the project to query.
        metric_name (str, optional): The name of the metric to evaluate.
            Defaults to "test_rmse".

    Returns:
        Experiment: The best experiment object for today based on the specified
        metric.

    Raises:
        ValueError: If no experiments are found for today.
    """
    api = API(api_key=api_key)
    min_metric_value: float = 2000

    for experiment in api.get(os.path.join(workspace_name, project_name)):
        experiment_date = datetime.utcfromtimestamp(
            experiment.get_metadata()["endTimeMillis"] / 1000
        ).strftime("%Y-%m-%d")

        if experiment_date == date.today().strftime("%Y-%m-%d"):
            metrics = experiment.get_metrics()

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
    """
    Set the production status of the specified model registry to None.

    This function retrieves the model registry assets and sets the
    production status of the currently active version of the model to
    None.

    Args:
        api_key (str): The API key for authenticating with the model
        registry.
        workspace (str): The name of the workspace containing the model
        registry.
        registry_name (str): The name of the model registry.

    Returns:
        None
    """
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
    config: str,
    cli_command: str,
    ml_client: MLClient,
    display_name: str,
    environment: str,
    compute: str,
) -> str | None:
    """
    Run a specified command on Azure using the provided configuration.

    This function configures and submits a job to Azure Machine
    Learning based on the provided command and parameters.

    Args:
        config (str): The path to the configuration file.
        cli_command (str): The command to execute on Azure.
        ml_client (MLClient): The MLClient for interacting with Azure.
        display_name (str): Name of the job in Azure ML
        environment (str): environment name and version in AzureML
        compute (str): compute type


    Returns:
        str | None: The URL of the submitted job in the Azure Studio or
        None if the job submission fails.
    """
    # configure job
    job = command(
        code=config,
        command=cli_command,
        environment=environment,
        compute=compute,
        display_name=display_name,
        experiment_name=display_name,
    )

    # submit job
    returned_job = ml_client.create_or_update(job)
    return returned_job.studio_url
