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
    """
    Abstract base class for defining a pipeline.

    Args:
        None

    Returns:
        None
    """

    @abstractmethod
    def run(self) -> None:
        """
        Run the pipeline.

        Args:
            None

        Returns:
            None
        """
        pass


class FeaturePipeline(Pipeline):
    """
    A pipeline for loading measurements and inserting them into a data store.

    Args:
        loader (MeasurementLoader): The loader to retrieve measurements.
        inserter (Inserter): The inserter to store the data.
    """

    def __init__(self, loader: MeasurementLoader, inserter: Inserter):
        self.loader = loader
        self.inserter = inserter

    def run(self) -> None:
        """
        Execute the feature pipeline.

        This method retrieves measurements using the loader and
        inserts them into a data store.

        Args:
            None

        Returns:
            None
        """
        measurements = self.loader.get_measurements()

        self.inserter.insert_data(measurements)


class TrainingPipeline(Pipeline):
    """
    A pipeline for training a machine learning model.

    Args:
        loader (MeasurementLoader): The loader to retrieve measurements.
        splitter (AqiSplitter): The splitter to divide data into train/test.
        model (AqiModel): The model to be trained.
        logger (AqiExperimentLogger): The logger to record metrics.
        model_registry (ModelRegistry): The registry for managing models.
    """

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
        """
        Execute the training pipeline.

        This method retrieves measurements, splits them into train and
        test sets, trains the model, generates predictions, logs
        evaluation metrics, and registers the model.

        Args:
            None

        Returns:
            None
        """
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
    """
    A pipeline for optimizing hyperparameters of a machine learning model.

    Args:
        loader (MeasurementLoader): The loader to retrieve measurements.
        model (AqiModel): The model to optimize.
        logger (AqiExperimentLogger): The logger to record metrics.
        train_test_splitter (AqiSplitter): The splitter for train/test data.
        model_registry (ModelRegistry): The registry for managing models.
    """

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
        # TODO: put them in init or config. Consider to change this to function
        """
        Execute hyperparameter optimization.

        This method retrieves measurements, splits them into train
        and test sets, and performs optimization using Optuna.

        Args:
            n_trials (int): The number of trials for optimization.
            direction (str): The direction of optimization ("minimize" or
                "maximize").
            study_name (str): The name of the Optuna study.

        Returns:
            None
        """
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
    """
    A pipeline for performing inference using a trained model.

    Args:
        loader (ValidationLoader): The loader to retrieve validation data.
        model_downloader (ModelDownloader): The downloader for the model.
        prediction_writer (Inserter): The inserter to store predictions.
    """

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
        """
        Execute the inference pipeline.

        This method retrieves validation measurements, downloads the
        model, makes a prediction, and stores the prediction.

        Args:
            None

        Returns:
            None
        """
        data = self.loader.get_measurements()

        model = self.model_downloader.get_model()
        print(f"{data=}")
        prediction = model.predict(data)[-1]
        print(f"{prediction=}")
        self.prediction_writer.insert_data([prediction])
