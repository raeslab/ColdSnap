from typing import Optional
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from .data import Data
from .serializable import Serializable
from .mixins import ConfusionMatrixMixin, ROCMixin, SHAPMixin
import pickle
import hashlib


class Model(Serializable, ConfusionMatrixMixin, ROCMixin, SHAPMixin):
    def __init__(
        self,
        data: Optional[Data] = None,
        clf: Optional[BaseEstimator] = None,
        description: Optional[str] = None,
        short_description: Optional[str] = None,
    ):
        self._data = data
        self._clf = clf
        self._description = description
        self._short_description = short_description

    @property
    def data(self) -> Optional[Data]:
        return self._data

    @data.setter
    def data(self, data: Data) -> None:
        self._data = data

    @property
    def clf(self) -> Optional[BaseEstimator]:
        return self._clf

    @clf.setter
    def clf(self, clf: BaseEstimator) -> None:
        self._clf = clf

    @property
    def description(self) -> Optional[str]:
        return self._description

    @description.setter
    def description(self, description: str) -> None:
        self._description = description

    @property
    def short_description(self) -> Optional[str]:
        return self._short_description

    @short_description.setter
    def short_description(self, short_description: str) -> None:
        self._short_description = short_description

    @property
    def hash(self) -> str:
        # Serialize the classifier to a byte stream
        clf_bytes = pickle.dumps(self._clf)

        # Compute and return the hex digest of the serialized classifier
        return hashlib.sha256(clf_bytes).hexdigest()

    def fit(self) -> None:
        if self._data is None:
            raise ValueError("No data provided to fit the classifier.")
        if self._clf is None:
            raise ValueError("No classifier provided to fit the data.")

        self._clf.fit(self._data.X_train, self._data.y_train)

    def predict(self, data):
        return self._clf.predict(data)

    def predict_proba(self, data):
        if hasattr(self._clf, "predict_proba"):
            return self._clf.predict_proba(data)
        else:
            raise NotImplementedError(
                "The classifier does not support probability predictions."
            )

    def evaluate(self, X_test: Optional = None, y_test: Optional = None) -> dict:
        if X_test is None and y_test is None and not self._data:
            raise ValueError("No data provided to evaluate the classifier.")
        if not self._clf:
            raise ValueError("No classifier provided to evaluate.")

        if X_test is None and y_test is None:
            X_test = self._data.X_test
            y_test = self._data.y_test
        elif bool(X_test is None) ^ bool(y_test is None):
            raise ValueError(
                "If external validation data is used, both X_test and y_test need to be provided."
            )

        y_pred = self._clf.predict(X_test)
        y_proba = (
            self._clf.predict_proba(X_test)
            if hasattr(self._clf, "predict_proba")
            else None
        )

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
        }

        if y_proba is not None:
            if y_proba.shape[1] == 2:  # Binary classification
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
            else:  # Multi-class classification
                metrics["roc_auc"] = roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="weighted"
                )
        else:
            metrics["roc_auc"] = None

        return metrics

    def purge(self) -> None:
        """Purges the associated data instance if it exists."""
        if self._data is not None:
            self._data.purge()

    def summary(self) -> dict:
        if self._data is None:
            raise ValueError("No data available for summary.")

        num_features = self._data.X_train.shape[
            1
        ]  # Number of features in training data
        feature_list = ", ".join(
            self._data.features
        )  # Get features from the Data class
        num_classes = len(self._data.classes)  # Get number of unique classes
        class_list = ", ".join(
            map(str, self._data.classes)
        )  # Unique classes as a string

        return {
            "model_code": self._short_description,
            "model_description": self._description,
            "model_hash": self.hash,
            "data_code": self._data.short_description,
            "data_description": self._data.description,
            "data_hash": self._data.hash,
            "num_features": num_features,
            "features": feature_list,
            "num_classes": num_classes,
            "classes": class_list,
        }
