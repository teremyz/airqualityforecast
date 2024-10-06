import datetime
import gc
import os
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import pandas as pd
from comet_ml import Experiment
from comet_ml.api import API
from optuna.trial import Trial
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

from src.core.data import AirQalityMeasurement, AirQalityPrediction
from src.core.utils import (
    get_best_experiment_today,
    reindex_dataframe,
    set_prod_status_to_none,
)
from src.core.visualisation import (
    create_true_vs_pred_line_plot,
    create_true_vs_predicted_scatter,
)


class Model(ABC):
    @abstractmethod
    def process_inputs(
        self, measurements: List[AirQalityMeasurement]
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def prerocess_target(
        self, measurements: List[AirQalityMeasurement]
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def train(self, measurements: List[AirQalityMeasurement]) -> None:
        pass

    @abstractmethod
    def predict(
        self, measurements: List[AirQalityMeasurement]
    ) -> List[AirQalityPrediction]:
        pass


class AqiModel(Model):
    def __init__(self, predictor: XGBRegressor, ffill_limit: int, lags: int):
        self.predictor = predictor
        self.trained = False
        self.ffill_limit = ffill_limit
        self.lags = lags

    def process_inputs(
        self, measurements: List[AirQalityMeasurement]
    ) -> pd.DataFrame:
        past_covariates = ["pm25", "pm10", "o3", "no2", "so2", "co"]
        future_covariates = ["year", "month", "day"]
        df = pd.DataFrame([v.model_dump() for v in measurements])

        df = reindex_dataframe(df=df, date_col_name="date")

        df = df.ffill(limit=self.ffill_limit, axis="index")
        df["year"] = df.index.year
        df["month"] = df.index.month
        df["day"] = df.index.day

        past_predictors = []
        for lag in range(1, self.lags + 1):
            shifted_names = [x + f"Shifted{lag}" for x in past_covariates]
            df[shifted_names] = df[past_covariates].shift(lag)
            past_predictors.extend(shifted_names)

        return df[past_predictors + future_covariates]

    def prerocess_target(
        self, measurements: List[AirQalityMeasurement]
    ) -> pd.DataFrame:
        target = pd.DataFrame([v.model_dump() for v in measurements])[
            ["aqi", "date"]
        ]
        target = reindex_dataframe(df=target, date_col_name="date")
        target = target.ffill(limit=self.ffill_limit, axis="index")

        return target

    def train(self, measurements: List[AirQalityMeasurement]) -> None:
        inputs = self.process_inputs(measurements)
        targets = self.prerocess_target(measurements)
        self.predictor.fit(inputs, targets)
        self.trained = True

    def predict(
        self, measurements: List[AirQalityMeasurement]
    ) -> List[AirQalityPrediction]:
        inputs = self.process_inputs(measurements)
        prediction = self.predictor.predict(inputs)

        return [
            AirQalityPrediction(
                date=date_prediction[0], prediction=date_prediction[1]
            )
            for date_prediction in zip(inputs.index, prediction)
        ]


class AqiExperimentLogger:
    def __init__(
        self,
        api_key: str,
        project_name: str,
        workspace: str,
        artifact_dir: str,
        model_name: str,
        src_dir: str,
    ):
        self.artifact_dir = artifact_dir
        self.model_name = model_name
        self.src_dir = src_dir
        self.api_key = api_key
        self.project_name = project_name
        self.workspace = workspace

        for filename in os.listdir(artifact_dir):
            os.remove(os.path.join(artifact_dir, filename))

    def create_evaluation_logs(
        self,
        test_data: List[AirQalityMeasurement],
        prediction: List[AirQalityPrediction],
    ) -> Dict[str, float]:
        labels = [x.aqi for x in test_data]
        predictions = [x.prediction for x in prediction]

        metrics = dict(
            mae=mean_absolute_error(labels, predictions),
            mse=mean_squared_error(labels, predictions),
            rmse=mean_squared_error(labels, predictions, squared=False),
            r_squared=r2_score(labels, predictions),
            mape=mean_absolute_percentage_error(labels, predictions),
        )

        df = pd.DataFrame([v.model_dump() for v in test_data])
        create_true_vs_pred_line_plot(
            df=df, saving_dir=self.artifact_dir, predictions=predictions
        )

        create_true_vs_predicted_scatter(
            df=df, saving_dir=self.artifact_dir, predictions=predictions
        )

        return metrics

    def log(self, metrics: Dict[str, float], model: AqiModel) -> None:
        experiment = Experiment(
            api_key=self.api_key,
            project_name=self.project_name,
            workspace=self.workspace,
            log_code=True,
        )

        with experiment.test():
            experiment.log_metrics(metrics, step=1)

            experiment.log_parameters(
                model.predictor.get_params()
            )  # TODO: add get_params to AqiModel class
            # and log both parameters custom + xgb

        pickle.dump(
            model, open(f"{self.artifact_dir}/{self.model_name}.pkl", "wb")
        )
        experiment.log_model(
            name=self.model_name,
            file_or_folder=f"{self.artifact_dir}/{self.model_name}.pkl",
        )

        experiment.log_asset_folder(self.artifact_dir)

        experiment.log_code(folder=self.src_dir)

        experiment.end()


class AqiSplitter:
    def split(
        self, measurements: List[AirQalityMeasurement]
    ) -> Tuple[List[AirQalityMeasurement], List[AirQalityMeasurement]]:
        train = [x for x in measurements if x.date < datetime.date(2024, 1, 1)]
        test = [x for x in measurements if x.date >= datetime.date(2024, 1, 1)]
        return train, test


def objective(
    trial: Trial,
    model: AqiModel,
    train_data: List[AirQalityMeasurement],
    test_data: List[AirQalityMeasurement],
    logger: AqiExperimentLogger,
) -> float:
    params: Dict[str, float | int | str] = {
        "verbosity": 0,
        "silent": True,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_estimators": trial.suggest_int("n_estimators", 10, 200),
        # "early_stopping_rounds": 10,
        "feature_importances_": "gain",
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.0001, 0.2, log=True
        ),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "lambda": trial.suggest_float("lambda", 0, 2),
        "alpha": trial.suggest_float("alpha", 0, 2),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 3),
        "subsample": trial.suggest_float("subsample", 0.2, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
        "max_bin": trial.suggest_int("max_bin", 12, 512),
    }

    model.predictor = XGBRegressor(**params)

    model.train(train_data)

    predictions = model.predict(test_data)

    metrics = logger.create_evaluation_logs(test_data, predictions)

    logger.log(metrics=metrics, model=model)

    gc.collect()

    return metrics["rmse"]


class ModelRegistry:
    def __init__(
        self,
        api_key: str,
        workspace_name: str,
        project_name: str,
        model_name: str,
        status: str,
    ):
        self.api_key = api_key
        self.workspace_name = workspace_name
        self.project_name = project_name
        self.model_name = model_name
        self.status = status

    def register_model_(self) -> None:
        best_experiment = get_best_experiment_today(
            api_key=self.api_key,
            workspace_name=self.workspace_name,
            project_name=self.project_name,
            metric_name="test_rmse",
        )

        set_prod_status_to_none(
            api_key=self.api_key,
            workspace=self.workspace_name,
            registry_name=self.project_name,
        )

        best_experiment.register_model(
            model_name=self.model_name,
            workspace=self.workspace_name,
            registry_name=self.project_name,
            status=self.status,
        )


class ModelDownloader(ABC):
    @abstractmethod
    def get_model(self) -> Model:
        pass


class CometModelDownloader(ModelDownloader):
    def __init__(
        self,
        api_key: str,
        workspace: str,
        project_name: str,
        model_dir: str,
        model_name: str,
    ) -> None:
        self.workspace = workspace
        self.project_name = project_name
        self.model_dir = model_dir
        self.api_key = api_key
        self.model_name = model_name

    def get_model(self) -> Model:
        api = API(api_key=self.api_key)

        model_api = api.get_model(
            workspace=self.workspace, model_name=self.project_name
        )
        latest_version = model_api.find_versions(
            version_prefix="", status="Production", tag=None
        )[0]
        model_api.download(
            version=latest_version, output_folder=self.model_dir
        )

        model = pickle.load(
            open(os.path.join(self.model_dir, f"{self.model_name}.pkl"), "rb")
        )

        return model
