from functools import partial

import optuna

from src.core.loaders import HopsworkFsInserter, MeasurementLoader
from src.core.model import (
    AqiExperimentLogger,
    AqiModel,
    AqiSplitter,
    objective,
)


class FeaturePipeline:
    def __init__(
        self, loader: MeasurementLoader, inserter: HopsworkFsInserter
    ):
        self.loader = loader
        self.inserter = inserter

    def run(self) -> None:
        measurements = self.loader.get_measurements()

        self.inserter.insert_data(measurements)


class TrainingPipeline:
    def __init__(
        self,
        loader: MeasurementLoader,
        splitter: AqiSplitter,
        model: AqiModel,
        logger: AqiExperimentLogger,
    ):
        self.loader = loader
        self.model = model
        self.logger = logger
        self.splitter = splitter

    def run(self) -> None:
        measurements = self.loader.get_measurements()

        train_data, test_data = self.splitter.split(measurements)

        self.model.train(train_data)

        predictions = self.model.predict(test_data)

        self.metrics = self.logger.create_evaluation_logs(
            test_data, predictions
        )

        self.logger.log(metrics=self.metrics, model=self.model.predictor)


class HyperparameterOptimizationPipeline:
    def __init__(
        self,
        loader: MeasurementLoader,
        model: AqiModel,
        logger: AqiExperimentLogger,
        train_test_splitter: AqiSplitter,
    ) -> None:
        self.loader = loader
        self.model = model
        self.logger = logger
        self.train_test_splitter = train_test_splitter

    def run(self, n_trials: int, direction: str, study_name: str) -> None:
        measurements = self.loader.get_measurements()

        train_data, test_data = self.train_test_splitter.split(measurements)

        objective_with_params = partial(
            objective,
            model=self.model,
            logger=self.logger,
            train_data=train_data,
            test_data=test_data,
        )

        study = optuna.create_study(study_name=study_name, direction=direction)
        study.optimize(objective_with_params, n_trials=n_trials)
