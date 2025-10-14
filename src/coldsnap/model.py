from typing import Optional, Literal, List
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
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
        estimator: Optional[BaseEstimator] = None,
        description: Optional[str] = None,
        short_description: Optional[str] = None,
    ):
        # Accept either clf or estimator parameter, not both
        if clf is not None and estimator is not None:
            raise ValueError("Provide either 'clf' or 'estimator' parameter, not both.")

        self._data = data
        # Store in _clf for backward compatibility with existing pickles
        self._clf = estimator if estimator is not None else clf
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
        """Classifier/estimator (backward compatible property)."""
        return self._clf

    @clf.setter
    def clf(self, clf: BaseEstimator) -> None:
        self._clf = clf

    @property
    def estimator(self) -> Optional[BaseEstimator]:
        """Generic estimator property (supports classifiers, regressors, and transformers)."""
        return self._clf

    @estimator.setter
    def estimator(self, estimator: BaseEstimator) -> None:
        self._clf = estimator

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

    def _get_estimator_type(
        self,
    ) -> Optional[Literal["classifier", "regressor", "transformer"]]:
        """Determine the type of the estimator."""
        if self._clf is None:
            return None
        if is_classifier(self._clf):
            return "classifier"
        elif is_regressor(self._clf):
            return "regressor"
        else:
            return "transformer"

    @property
    def hash(self) -> str:
        # Serialize the classifier to a byte stream
        clf_bytes = pickle.dumps(self._clf)

        # Compute and return the hex digest of the serialized classifier
        return hashlib.sha256(clf_bytes).hexdigest()

    def fit(self) -> None:
        if self._data is None:
            raise ValueError("No data provided to fit the estimator.")
        if self._clf is None:
            raise ValueError("No estimator provided to fit the data.")

        estimator_type = self._get_estimator_type()

        # Transformers only need X data, no labels
        if estimator_type == "transformer":
            self._clf.fit(self._data.X_train)
        else:
            # Classifiers and regressors need both X and y
            self._clf.fit(self._data.X_train, self._data.y_train)

    def predict(self, data):
        if self._clf is None:
            raise ValueError("No estimator provided.")

        estimator_type = self._get_estimator_type()
        if estimator_type == "transformer":
            raise TypeError(
                "Cannot call predict() on a transformer. Use transform() instead."
            )

        return self._clf.predict(data)

    def transform(self, data):
        """Transform data using a fitted transformer.

        When the input is a pandas DataFrame, the output will preserve the DataFrame
        structure including index and column names. This is achieved using scikit-learn's
        set_output API (available in scikit-learn >= 1.2).

        Args:
            data: Data to transform (pandas DataFrame or array-like)

        Returns:
            Transformed data (pandas DataFrame if input is DataFrame, otherwise array)

        Raises:
            ValueError: If no estimator is provided or estimator is not fitted
            TypeError: If estimator is not a transformer
        """
        if self._clf is None:
            raise ValueError("No estimator provided.")

        estimator_type = self._get_estimator_type()
        if estimator_type != "transformer":
            raise TypeError(
                f"Cannot call transform() on a {estimator_type}. "
                "This method is only available for transformers."
            )

        if not hasattr(self._clf, "transform"):
            raise AttributeError("The estimator does not have a transform method.")

        # Preserve DataFrame structure if input is a DataFrame
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            # Configure transformer to output pandas DataFrames
            self._clf.set_output(transform="pandas")

        return self._clf.transform(data)

    def predict_proba(self, data):
        if self._clf is None:
            raise ValueError("No estimator provided.")

        estimator_type = self._get_estimator_type()
        if estimator_type == "transformer":
            raise TypeError("Cannot call predict_proba() on a transformer.")
        elif estimator_type == "regressor":
            raise TypeError(
                "Cannot call predict_proba() on a regressor. "
                "Probability predictions are only available for classifiers."
            )

        if hasattr(self._clf, "predict_proba"):
            return self._clf.predict_proba(data)
        else:
            raise NotImplementedError(
                "The classifier does not support probability predictions."
            )

    def evaluate(self, X_test: Optional = None, y_test: Optional = None) -> dict:
        if X_test is None and y_test is None and not self._data:
            raise ValueError("No data provided to evaluate the estimator.")
        if not self._clf:
            raise ValueError("No estimator provided to evaluate.")

        estimator_type = self._get_estimator_type()
        if estimator_type == "transformer":
            raise TypeError(
                "Cannot evaluate() a transformer. "
                "Evaluation metrics are only available for classifiers and regressors."
            )

        if X_test is None and y_test is None:
            X_test = self._data.X_test
            y_test = self._data.y_test
        elif bool(X_test is None) ^ bool(y_test is None):
            raise ValueError(
                "If external validation data is used, both X_test and y_test need to be provided."
            )

        y_pred = self._clf.predict(X_test)

        # Handle regression metrics
        if estimator_type == "regressor":
            metrics = {
                "rmse": root_mean_squared_error(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "mse": mean_squared_error(y_test, y_pred),
            }
            return metrics

        # Handle classification metrics
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

        estimator_type = self._get_estimator_type()

        summary_dict = {
            "model_code": self._short_description,
            "model_description": self._description,
            "model_hash": self.hash,
            "estimator_type": estimator_type,
            "data_code": self._data.short_description,
            "data_description": self._data.description,
            "data_hash": self._data.hash,
            "num_features": num_features,
            "features": feature_list,
        }

        # Only include class information for classifiers
        if estimator_type == "classifier" and self._data.classes is not None:
            num_classes = len(self._data.classes)  # Get number of unique classes
            class_list = ", ".join(
                map(str, self._data.classes)
            )  # Unique classes as a string
            summary_dict["num_classes"] = num_classes
            summary_dict["classes"] = class_list

        return summary_dict

    @property
    def features(self) -> Optional[List[str]]:
        """Get feature names, preferring sklearn's feature_names_in_ over Data object."""
        # First try to get from fitted estimator (most reliable)
        if self._clf is not None and hasattr(self._clf, "feature_names_in_"):
            return list(self._clf.feature_names_in_)

        # Fall back to Data object if available
        if self._data is not None and hasattr(self._data, "features"):
            return self._data.features

        return None
