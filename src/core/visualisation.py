import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_true_vs_pred_line_plot(
    df: pd.DataFrame,
    saving_dir: str,
    predictions: np.typing.NDArray[np.float64],
) -> None:
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
    predictions: np.typing.NDArray[np.float64],
) -> None:
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
