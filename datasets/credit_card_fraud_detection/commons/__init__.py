from numpy import ndarray
from imblearn.over_sampling import SMOTE
from sklearn.metrics import fbeta_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from datetime import datetime
from typing import Any, Callable, List
from pandas import DataFrame, Index
import mlflow
import lightgbm as lgb
import numpy as np


class LogParameter:
    def __init__(self, tag: str = 'default', experiment_name: str = 'default'):
        self.tag: str = tag
        self.experiment_name: str = experiment_name


class SplitParameter:
    def __init__(self, test_size: float = 0.3, train_index: Index = None, test_index: Index = None,
                 oversampling_strategy: Any = None):
        self.test_size: float = test_size
        self.train_index: Index = train_index
        self.test_index: Index = test_index
        self.oversampling_strategy: Any = oversampling_strategy


class Dataset:
    def __init__(self, df: DataFrame, label_col: int = -1, scaler: Any = None):
        self.X: ndarray = df.drop(df.columns[label_col], axis=1).to_numpy()
        self.y: ndarray = df.iloc[:, label_col].to_numpy()
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.y_train = None
        if scaler is not None:
            self.X = scaler.fit_transform(self.X)

    def split(self, split_params: SplitParameter):
        if split_params.train_index is not None and split_params.test_index is not None:
            self.X_train = self.X[split_params.train_index, :]
            self.X_test = self.X[split_params.test_index, :]
            self.y_train = self.y[split_params.train_index]
            self.y_test = self.y[split_params.test_index]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=split_params.test_size, random_state=random_state, stratify=self.y)

        if split_params.oversampling_strategy is not None:
            smote = SMOTE(sampling_strategy=split_params.oversampling_strategy, random_state=random_state)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

    def split_by_index(self, train_index: Index, test_index: Index, oversampling_strategy: Any = None):
        self.X_train = self.X[train_index, :]
        self.X_test = self.X[test_index, :]
        self.y_train = self.y[train_index]
        self.y_test = self.y[test_index]

        if oversampling_strategy is not None:
            smote = SMOTE(sampling_strategy=oversampling_strategy, random_state=random_state)
            X_train, y_train = smote.fit_resample(self.X_train, self.y_train)
            self.X_train = X_train
            self.y_train = y_train


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
        if type(self.model) == lgb.LGBMClassifier:
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1,
                                                                random_state=random_state, stratify=y_train)

            self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(50)])
        else:
            self.model.fit(X_train, y_train)

    def predict(self, dataset: Dataset) -> ndarray:
        return self.model.predict(dataset.X_test)

    def predict_proba(self, dataset: Dataset) -> ndarray:
        return self.model.predict_proba(dataset.X_test)


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
    def __init__(self, model: Model, dataset: Dataset):
        self.model: Model = model
        self.dataset: Dataset = dataset

    def log_metrics(self, metrics: List[Metric], log_params: LogParameter = LogParameter()):
        now = datetime.now()
        algorithm_name = self.model.get_algorithm_name()
        experiment_name = log_params.experiment_name
        run_name = f'{algorithm_name}_{log_params.tag}_{now.strftime("%H:%M:%S")}'
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = mlflow.create_experiment(experiment_name) if experiment is None else experiment.experiment_id

        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
            for metric in metrics:
                mlflow.log_metric(metric.name, metric.value)
                print(f'{metric.name}: {metric.value}')

    def _eval(self, split_params: SplitParameter) -> EvaluationResult:
        self.dataset.split(split_params)
        self.model.fit(self.dataset.X_train, self.dataset.y_train)
        pred = self.model.predict(self.dataset)
        pred_proba = self.model.predict_proba(self.dataset)[:, 1]

        return EvaluationResult(self.dataset.y_test, pred, pred_proba)

    def eval(self, split_params: SplitParameter, log_params: LogParameter = LogParameter(), save_model: bool = False):
        result = self._eval(split_params=split_params)

        for metric in metrics:
            metric.calc(result)

        self.log_metrics(metrics, log_params)

        if save_model:
            self.model.save_model()

    def cross_val_eval(self, metrics: List[Metric], oversampling_strategy: Any = None,
                       log_params: LogParameter = LogParameter(), k: int = 5, save_model: bool = False):
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        metric_scores = {metric.name: [] for metric in metrics}

        for train_index, test_index in kfold.split(self.dataset.X, self.dataset.y):
            split_params = SplitParameter(train_index=train_index, test_index=test_index,
                                          oversampling_strategy=oversampling_strategy)
            result = self._eval(split_params=split_params)

            for metric in metrics:
                metric_scores[metric.name].append(result)

        for metric in metrics:
            metric.calc_multiple(metric_scores[metric.name])
            print(f'{metric.name}: {metric.value}')

        self.log_metrics(metrics, log_params)

        if save_model:
            self.model.fit(self.dataset.X, self.dataset.y)
            self.model.save_model()


metrics = [
    Metric('f2_score', lambda y_test, pred, pred_proba: fbeta_score(y_test, pred, beta=2)),
    Metric('roc_auc', lambda y_test, pred, pred_proba: roc_auc_score(y_test, pred_proba)),
    Metric('precision', lambda y_test, pred, pred_proba: precision_score(y_test, pred)),
    Metric('recall', lambda y_test, pred, pred_proba: recall_score(y_test, pred))
]
random_state = 0
