import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_true_vs_pred_line_plot(
    df: pd.DataFrame,
    saving_dir: str,
    predictions: List[float],
) -> None:
    """
    Creates a line plot comparing true vs. predicted values over time and
    saves it as a PNG file.

    This function generates a time series plot where the x-axis represents
    the dates and the y-axis represents the actual (true) and predicted
    Air Quality Index (AQI) values.
    The plot is saved to the specified directory.

    Args:
        df (pd.DataFrame):
            A DataFrame containing at least two columns: 'date' and 'aqi'.
            The 'date' column should contain date values that can
            be converted to `datetime`.
        saving_dir (str): Directory path where the plot
            image will be saved.
        predictions (np.typing.NDArray[np.float64]): Array of
            predicted AQI values. It must have the same length
            as the number of rows in `df`.

    Returns:
        None

    Raises:
        FileNotFoundError: If the provided saving directory does not exist.
        ValueError: If the length of `predictions` does not match the number
        of rows in `df`.

    Example:
        >>> create_true_vs_pred_line_plot(df, "/path/to/save", predictions)
        This will generate and save a plot 'true_vs_predicted_ts.png' in
        the specified directory.
    """
    plt.figure(figsize=(6.4, 4.8))

    df["date"] = pd.to_datetime(df["date"])
    df["predictions"] = predictions
    df = df.sort_values("date", ascending=False)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(df.date, df.aqi, label="aqi")
    plt.plot(df.date, df.predictions, label="predictions")
    plt.xticks(rotation=90)

    plt.xlabel("date")
    plt.legend()
    plt.savefig(os.path.join(saving_dir, "true_vs_predicted_ts.png"))


def create_true_vs_predicted_scatter(
    df: pd.DataFrame,
    saving_dir: str,
    predictions: List[float],
) -> None:
    """
    Create and save a scatter plot comparing true vs. predicted AQI values.

    This function generates a scatter plot where actual AQI values
    are plotted on the x-axis and predicted AQI values are plotted on the
    y-axis. A diagonal red line is added to the plot to indicate perfect
    predictions. The plot is saved as 'true_vs_predicted.png' in the
    specified directory.

    Args:
        df (pd.DataFrame): DataFrame containing actual AQI values
            in the 'aqi' column. saving_dir (str): Directory where
            the plot image will be saved.
        predictions (np.typing.NDArray[np.float64]): Array of predicted
            AQI values corresponding to the values in `df`.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified saving directory
        does not exist.
        ValueError: If the number of predictions does not match
        the length of `df`.

    Example:
        >>> create_true_vs_predicted_scatter(
                df, "/path/to/save", predictions)
        This will generate a scatter plot and save it in the
        specified directory.
    """
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(df.aqi, predictions)
    plt.xlabel("aqi")
    plt.ylabel("predictions")
    plt.axline(
        (0, 0),
        (
            np.max([df.predictions.max(), df.aqi.max()]),
            np.max([df.predictions.max(), df.aqi.max()]),
        ),
        color="red",
    )
    plt.savefig(os.path.join(saving_dir, "true_vs_predicted.png"))
