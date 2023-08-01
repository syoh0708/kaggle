from sklearn.metrics import confusion_matrix, fbeta_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from datetime import datetime
from typing import Any, Dict
from pandas import DataFrame
import mlflow


class Dataset:
    def __init__(self, df: DataFrame, label_col: int = -1, scaler: Any = None):
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.df = df
        self.metrics = metrics
        self.scaler = scaler
        self.label_col = label_col

        self.X = df.drop(df.columns[label_col], axis=1)
        self.y = df.iloc[:, label_col]

    def scale(self):
        if self.scaler is not None:
            self.X = self.scaler.fit_transform(self.X)

    def split(self, test_size=0.3, random_state=0):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                            random_state=random_state, stratify=self.y)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def log_metric(self, pred, pred_proba):
        confusion = confusion_matrix(self.y_test, pred)
        print('confusion_matrix')
        print(confusion)

        for name, func in self.metrics.items():
            val = func(self.y_test, pred, pred_proba)
            mlflow.log_metric(name, val)
            print(f'{name}: {val}')

    def eval(self, model, tag='default', experiment_name='default'):
        now = datetime.now()
        algorithm_name = model.__class__.__name__
        run_name = f'{algorithm_name}_{tag}_{now.strftime("%H:%M:%S")}'
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = mlflow.create_experiment(experiment_name) if experiment is None else experiment.experiment_id

        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
            model.fit(self.X_train, self.y_train)
            pred = model.predict(self.X_test)
            pred_proba = model.predict_proba(self.X_test)[:, 1]
            self.log_metric(pred, pred_proba)


metrics = {
    'f2_score': lambda y_test, pred, pred_proba: fbeta_score(y_test, pred, beta=2),
    'roc_auc': lambda y_test, pred, pred_proba: roc_auc_score(y_test, pred_proba),
    'precision': lambda y_test, pred, pred_proba: precision_score(y_test, pred),
    'recall': lambda y_test, pred, pred_proba: recall_score(y_test, pred)
}
