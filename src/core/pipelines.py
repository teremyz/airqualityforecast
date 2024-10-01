from abc import ABC, abstractmethod
from functools import partial

import optuna

from src.core.loaders import Inserter, MeasurementLoader, ValidationLoader
from src.core.model import (
    AqiExperimentLogger,
    AqiModel,
    AqiSplitter,
    ModelDownloader,
    ModelRegistry,
    objective,
)


class Pipeline(ABC):
    @abstractmethod
    def run(self) -> None:
        pass


class FeaturePipeline(Pipeline):
    def __init__(self, loader: MeasurementLoader, inserter: Inserter):
        self.loader = loader
        self.inserter = inserter

    def run(self) -> None:
        measurements = self.loader.get_measurements()

        self.inserter.insert_data(measurements)


class TrainingPipeline(Pipeline):
    def __init__(
        self,
        loader: MeasurementLoader,
        splitter: AqiSplitter,
        model: AqiModel,
        logger: AqiExperimentLogger,
        model_registry: ModelRegistry,
    ):
        self.loader = loader
        self.model = model
        self.logger = logger
        self.splitter = splitter
        self.model_registry = model_registry

    def run(self) -> None:
        measurements = self.loader.get_measurements()

        train_data, test_data = self.splitter.split(measurements)

        self.model.train(train_data)

        predictions = self.model.predict(test_data)

        self.metrics = self.logger.create_evaluation_logs(
            test_data, predictions
        )

        self.logger.log(metrics=self.metrics, model=self.model)

        self.model_registry.register_model_()


class HyperparameterOptimizationPipeline:  # TODO: no parent class
    def __init__(
        self,
        loader: MeasurementLoader,
        model: AqiModel,
        logger: AqiExperimentLogger,
        train_test_splitter: AqiSplitter,
        model_registry: ModelRegistry,
    ) -> None:
        self.loader = loader
        self.model = model
        self.logger = logger
        self.train_test_splitter = train_test_splitter
        self.model_registry = model_registry

    def run(
        self, n_trials: int, direction: str, study_name: str
    ) -> None:  # TODO: Is it a pipeline child? IF yes delete parameters
        # TODO: put them in init or config
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

        self.model_registry.register_model_()


class InferencePipeline(Pipeline):
    def __init__(
        self,
        loader: ValidationLoader,
        model_downloader: ModelDownloader,
        prediction_writer: Inserter,
    ):
        self.loader = loader
        self.model_downloader = model_downloader
        self.prediction_writer = prediction_writer

    def run(self) -> None:
        data = self.loader.get_measurements()

        model = self.model_downloader.get_model()

        prediction = model.predict(data)[-1]

        self.prediction_writer.insert_data([prediction])
