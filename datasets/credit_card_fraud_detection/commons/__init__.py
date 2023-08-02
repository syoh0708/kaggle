from sklearn.metrics import confusion_matrix, fbeta_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from datetime import datetime
from typing import Any, Dict, Callable
from pandas import DataFrame
import mlflow


class Dataset:
    def __init__(self, df: DataFrame, label_col: int = -1, scaler: Any = None):
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.df = df
        self.scaler = scaler
        self.label_col = label_col
        self.__ready = False

        self.X = df.drop(df.columns[label_col], axis=1)
        self.y = df.iloc[:, label_col]

    def scale(self):
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

    def scale_and_split(self, test_size: float = 0.3, random_state: int = 0):
        self.scale()
        self.split(test_size=test_size, random_state=random_state)

    def is_ready(self):
        return self.__ready


class Model:
    def __init__(self, model: Any, dataset: Dataset):
        self.model: Any = model
        self.dataset: Dataset = dataset
        self.is_trained = False
        self.pred = None
        self.pred_proba = None

    def log_metric(self, metrics: Dict[str, Callable[[Any, Any, Any], float]], tag: str = 'default',
                   experiment_name: str = 'default'):
        if self.is_trained is False:
            raise ValueError('Model is not trained. Call eval() first.')

        now = datetime.now()
        algorithm_name = self.model.__class__.__name__
        run_name = f'{algorithm_name}_{tag}_{now.strftime("%H:%M:%S")}'
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = mlflow.create_experiment(experiment_name) if experiment is None else experiment.experiment_id

        confusion = confusion_matrix(self.dataset.y_test, self.pred)
        print('confusion_matrix')
        print(confusion)

        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
            for name, func in metrics.items():
                val = func(self.dataset.y_test, self.pred, self.pred_proba)
                mlflow.log_metric(name, val)
                print(f'{name}: {val}')

    def save_model(self):
        model_type = type(self.model)
        if model_type.__module__.startswith('sklearn.'):
            mlflow.sklearn.save_model(self.model, 'model')
        elif model_type.__module__.startswith('lightgbm.'):
            return mlflow.lightgbm.save_model(self.model, 'model')
        else:
            raise ValueError(f'Unsupported model type: {model_type}')

    def eval(self, save_model: bool = False):
        if self.dataset.is_ready() is False:
            raise ValueError('Dataset is not ready. Call scale_and_split() first.')

        self.model.fit(self.dataset.X_train, self.dataset.y_train)
        self.pred = self.model.predict(self.dataset.X_test)
        self.pred_proba = self.model.predict_proba(self.dataset.X_test)[:, 1]
        self.is_trained = True

        if save_model:
            self.save_model()


metrics = {
    'f2_score': lambda y_test, pred, pred_proba: fbeta_score(y_test, pred, beta=2),
    'roc_auc': lambda y_test, pred, pred_proba: roc_auc_score(y_test, pred_proba),
    'precision': lambda y_test, pred, pred_proba: precision_score(y_test, pred),
    'recall': lambda y_test, pred, pred_proba: recall_score(y_test, pred)
}
