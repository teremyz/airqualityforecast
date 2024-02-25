import numpy as np
import pandas as pd
import yaml
from box import ConfigBox


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
