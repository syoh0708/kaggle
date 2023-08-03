from numpy import ndarray
from sklearn.metrics import fbeta_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from datetime import datetime
from typing import Any, Callable, List
from pandas import DataFrame, Index
import mlflow
import numpy as np


class Dataset:
    def __init__(self, df: DataFrame, label_col: int = -1, scaler: Any = None):
        self.scaler = scaler
        self.__ready = False
        self.X = df.drop(df.columns[label_col], axis=1)
        self.y = df.iloc[:, label_col]
        if self.scaler is not None:
            self.X = self.scaler.fit_transform(self.X)

    def split(self, test_size: float = 0.3, random_state: int = 0):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                            random_state=random_state, stratify=self.y)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.__ready = True

    def split_by_index(self, train_index: Index, test_index: Index):
        self.X_train = self.X.loc[train_index, :]
        self.X_test = self.X.loc[test_index, :]
        self.y_train = self.y[train_index]
        self.y_test = self.y[test_index]
        self.__ready = True

    def is_ready(self):
        return self.__ready


class Model:
    def __init__(self, model: Any):
        self.model: Any = model

    def get_algorithm_name(self):
        return self.model.__class__.__name__

    def save_model(self, path: str = 'model'):
        model_type = type(self.model)
        if model_type.__module__.startswith('sklearn.'):
            mlflow.sklearn.save_model(self.model, path)
        elif model_type.__module__.startswith('lightgbm.'):
            return mlflow.lightgbm.save_model(self.model, path)
        else:
            raise ValueError(f'Unsupported model type: {model_type}')

    def fit(self, X_train: ndarray, y_train: ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: ndarray) -> ndarray:
        return self.model.predict(X_test)

    def predict_proba(self, X_test: ndarray) -> ndarray:
        return self.model.predict_proba(X_test)


class EvaluationResult:
    def __init__(self, y_test: ndarray, pred: ndarray, pred_proba: ndarray):
        self.y_test: ndarray = y_test
        self.pred: ndarray = pred
        self.pred_proba: ndarray = pred_proba


class Metric:
    def __init__(self, name: str, func: Callable[[Any, Any, Any], float]):
        self.name: str = name
        self.func: Callable[[Any, Any, Any], float] = func
        self.value: float = 0.0

    def calc(self, result: EvaluationResult) -> float:
        self.value = self.func(result.y_test, result.pred, result.pred_proba)

        return self.value

    def calc_multiple(self, results: List[EvaluationResult]) -> ndarray:
        self.value = np.mean([self.calc(result) for result in results])

        return self.value


class ModelEvaluator:
    def __init__(self, model: Any, dataset: Dataset):
        self.model: Any = model
        self.dataset: Dataset = dataset

    def log_metrics(self, metrics: List[Metric], tag: str = 'default', experiment_name: str = 'default'):
        now = datetime.now()
        algorithm_name = self.model.get_algorithm_name()
        run_name = f'{algorithm_name}_{tag}_{now.strftime("%H:%M:%S")}'
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = mlflow.create_experiment(experiment_name) if experiment is None else experiment.experiment_id

        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
            for metric in metrics:
                mlflow.log_metric(metric.name, metric.value)
                print(f'{metric.name}: {metric.value}')

    def _eval(self, test_size: float = 0.3, random_state: int = 0, train_index: Index = None, test_index: Index = None) \
            -> EvaluationResult:
        if train_index is not None and test_index is not None:
            self.dataset.split_by_index(train_index, test_index)
        else:
            self.dataset.split(test_size=test_size, random_state=random_state)

        self.model.fit(self.dataset.X_train, self.dataset.y_train)
        pred = self.model.predict(self.dataset.X_test)
        pred_proba = self.model.predict_proba(self.dataset.X_test)[:, 1]

        return EvaluationResult(self.dataset.y_test, pred, pred_proba)

    def eval(self, save_model: bool = False, test_size: float = 0.3, random_state: int = 0,
             tag: str = 'default', experiment_name: str = 'default'):
        result = self._eval(test_size=test_size, random_state=random_state)

        for metric in metrics:
            metric.calc(result)

        self.log_metrics(metrics, tag, experiment_name)

        if save_model:
            self.model.save_model()

    def cross_val_eval(self, metrics: List[Metric], tag: str = 'default', experiment_name: str = 'default',
                       k: int = 5, save_model: bool = False):
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        metric_scores = {metric.name: [] for metric in metrics}

        for train_index, test_index in kfold.split(self.dataset.X, self.dataset.y):
            result = self._eval(train_index=train_index, test_index=test_index)

            for metric in metrics:
                metric_scores[metric.name].append(result)

        for metric in metrics:
            metric.calc_multiple(metric_scores[metric.name])
            print(f'{metric.name}: {metric.value}')

        self.log_metrics(metrics, tag, experiment_name)

        if save_model:
            self.model.fit(self.dataset.X, self.dataset.y)
            self.model.save_model()


metrics = [
    Metric('f2_score', lambda y_test, pred, pred_proba: fbeta_score(y_test, pred, beta=2)),
    Metric('roc_auc', lambda y_test, pred, pred_proba: roc_auc_score(y_test, pred_proba)),
    Metric('precision', lambda y_test, pred, pred_proba: precision_score(y_test, pred)),
    Metric('recall', lambda y_test, pred, pred_proba: recall_score(y_test, pred))
]
