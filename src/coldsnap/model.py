"""Machine learning model wrapper with serialization and evaluation capabilities.

This module provides the Model class which wraps scikit-learn estimators (classifiers,
regressors, or transformers) with their associated Data. It includes automatic type
detection, evaluation metrics, visualization capabilities, and serialization.
"""

import hashlib
import pickle
from typing import Literal, Optional

from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from .data import Data
from .mixins import ConfusionMatrixMixin, ROCMixin, SHAPMixin
from .serializable import Serializable


class Model(Serializable, ConfusionMatrixMixin, ROCMixin, SHAPMixin):
    """ML model wrapper supporting classifiers, regressors, and transformers.

    Model combines a scikit-learn estimator with Data to create a complete snapshot
    of a machine learning workflow. It provides:
    - Automatic estimator type detection (classifier/regressor/transformer)
    - Type-appropriate methods (fit, predict, transform, predict_proba, evaluate)
    - Evaluation metrics (classification or regression specific)
    - Visualization capabilities (confusion matrix, ROC curves, SHAP)
    - Serialization with data integrity checking

    The class maintains backward compatibility by accepting both 'clf' and 'estimator'
    parameters, though 'estimator' is preferred for new code.

    Attributes:
        data: Data object containing train/test splits.
        estimator: The scikit-learn estimator (classifier, regressor, or transformer).
        clf: Backward-compatible alias for estimator.
        description: Optional detailed description of the model.
        short_description: Optional brief description for summaries.

    Examples:
        Classifier workflow:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = Model(data=data, estimator=RandomForestClassifier())
        >>> model.fit()
        >>> predictions = model.predict(data.X_test)
        >>> metrics = model.evaluate()
        >>> model.display_confusion_matrix()

        Regressor workflow:
        >>> from sklearn.linear_model import LinearRegression
        >>> model = Model(data=data, estimator=LinearRegression())
        >>> model.fit()
        >>> predictions = model.predict(data.X_test)
        >>> metrics = model.evaluate()  # Returns RMSE, MAE, R2, MSE

        Transformer workflow:
        >>> from sklearn.preprocessing import StandardScaler
        >>> model = Model(data=data, estimator=StandardScaler())
        >>> model.fit()
        >>> X_scaled = model.transform(data.X_train)  # Preserves DataFrame structure
    """

    def __init__(
        self,
        data: Data | None = None,
        clf: BaseEstimator | None = None,
        estimator: BaseEstimator | None = None,
        description: str | None = None,
        short_description: str | None = None,
    ):
        """Initialize Model with data and estimator.

        Args:
            data: Data object containing train/test splits. Optional, can be set later.
            clf: Scikit-learn estimator (backward-compatible parameter). Use 'estimator'
                for new code.
            estimator: Scikit-learn estimator (classifier, regressor, or transformer).
                Preferred parameter name.
            description: Optional detailed description of the model.
            short_description: Optional brief description for summaries and tables.

        Raises:
            ValueError: If both 'clf' and 'estimator' parameters are provided.

        Examples:
            >>> model = Model(data=data, estimator=RandomForestClassifier())
            >>> model = Model(data=data, clf=RandomForestClassifier())  # Legacy syntax
        """
        # Accept either clf or estimator parameter, not both
        if clf is not None and estimator is not None:
            raise ValueError("Provide either 'clf' or 'estimator' parameter, not both.")

        self._data = data
        # Store in _clf for backward compatibility with existing pickles
        self._clf = estimator if estimator is not None else clf
        self._description = description
        self._short_description = short_description

    @property
    def data(self) -> Data | None:
        return self._data

    @data.setter
    def data(self, data: Data) -> None:
        self._data = data

    @property
    def clf(self) -> BaseEstimator | None:
        """Classifier/estimator (backward compatible property)."""
        return self._clf

    @clf.setter
    def clf(self, clf: BaseEstimator) -> None:
        self._clf = clf

    @property
    def estimator(self) -> BaseEstimator | None:
        """Generic estimator property (supports classifiers, regressors, and transformers)."""
        return self._clf

    @estimator.setter
    def estimator(self, estimator: BaseEstimator) -> None:
        self._clf = estimator

    @property
    def description(self) -> str | None:
        return self._description

    @description.setter
    def description(self, description: str) -> None:
        self._description = description

    @property
    def short_description(self) -> str | None:
        return self._short_description

    @short_description.setter
    def short_description(self, short_description: str) -> None:
        self._short_description = short_description

    def _get_estimator_type(
        self,
    ) -> Literal["classifier", "regressor", "transformer"] | None:
        """Determine the type of the estimator using sklearn's type checking.

        Returns:
            One of "classifier", "regressor", "transformer", or None if no estimator set.

        Note:
            This uses sklearn's is_classifier() and is_regressor() functions. Anything
            that doesn't match these is assumed to be a transformer.
        """
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
        """Generate SHA256 hash of the serialized estimator.

        This provides a unique fingerprint of the model's state (parameters, structure)
        that can be used to identify identical models or detect changes.

        Returns:
            64-character hexadecimal SHA256 hash string.

        Examples:
            >>> model.hash
            'a1b2c3d4e5f6...'
        """
        # Serialize the classifier to a byte stream
        clf_bytes = pickle.dumps(self._clf)

        # Compute and return the hex digest of the serialized classifier
        return hashlib.sha256(clf_bytes).hexdigest()

    def fit(self) -> None:
        """Fit the estimator on training data.

        Automatically handles different estimator types:
        - Transformers: Fit on X_train only (no labels needed)
        - Classifiers/Regressors: Fit on X_train and y_train

        Raises:
            ValueError: If no data or estimator is provided.

        Examples:
            >>> model = Model(data=data, estimator=RandomForestClassifier())
            >>> model.fit()  # Trains on data.X_train, data.y_train
        """
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
        """Generate predictions using the fitted estimator.

        Args:
            data: Input features (array-like or pandas DataFrame).

        Returns:
            Predicted values (classification labels or regression targets).

        Raises:
            ValueError: If no estimator is provided.
            TypeError: If called on a transformer (use transform() instead).

        Examples:
            >>> predictions = model.predict(data.X_test)
        """
        if self._clf is None:
            raise ValueError("No estimator provided.")

        estimator_type = self._get_estimator_type()
        if estimator_type == "transformer":
            raise TypeError("Cannot call predict() on a transformer. Use transform() instead.")

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
        """Generate probability predictions for classification.

        Only available for classifiers that support probability predictions.

        Args:
            data: Input features (array-like or pandas DataFrame).

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities.

        Raises:
            ValueError: If no estimator is provided.
            TypeError: If called on a transformer or regressor.
            NotImplementedError: If classifier doesn't support probability predictions.

        Examples:
            >>> probas = model.predict_proba(data.X_test)
            >>> probas.shape
            (100, 3)  # 100 samples, 3 classes
        """
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
            raise NotImplementedError("The classifier does not support probability predictions.")

    def evaluate(self, X_test: Optional = None, y_test: Optional = None) -> dict:
        """Evaluate the fitted estimator and return performance metrics.

        Automatically computes appropriate metrics based on estimator type:
        - Classifiers: accuracy, precision, recall, f1, roc_auc
        - Regressors: rmse, mae, r2, mse

        Args:
            X_test: Optional test features. If not provided, uses self.data.X_test.
            y_test: Optional test labels. If not provided, uses self.data.y_test.

        Returns:
            Dictionary of metric names to values.

        Raises:
            ValueError: If no data is provided and model has no data, or if only one
                of X_test/y_test is provided.
            TypeError: If called on a transformer.

        Examples:
            Classifier evaluation:
            >>> metrics = model.evaluate()
            >>> metrics
            {'accuracy': 0.95, 'precision': 0.94, 'recall': 0.95, 'f1': 0.94, 'roc_auc': 0.98}

            Regressor evaluation:
            >>> metrics = model.evaluate()
            >>> metrics
            {'rmse': 2.34, 'mae': 1.87, 'r2': 0.89, 'mse': 5.48}

            External data evaluation:
            >>> metrics = model.evaluate(X_external, y_external)
        """
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
        y_proba = self._clf.predict_proba(X_test) if hasattr(self._clf, "predict_proba") else None

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
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
        """Remove all data rows while preserving structure.

        Calls purge() on the associated Data object if one exists. This is useful for
        sharing model snapshots without exposing training data.

        Examples:
            >>> model.purge()  # Removes all data rows, keeps column structure
        """
        if self._data is not None:
            self._data.purge()

    def summary(self) -> dict:
        """Generate comprehensive summary of model and data metadata.

        Returns:
            Dictionary containing model and data information:
            - model_code: Short description of the model
            - model_description: Detailed description of the model
            - model_hash: SHA256 hash of the estimator
            - estimator_type: One of "classifier", "regressor", "transformer"
            - data_code: Short description of the data
            - data_description: Detailed description of the data
            - data_hash: SHA256 hash of the data
            - num_features: Number of features
            - features: Comma-separated list of feature names
            - num_classes: Number of classes (classifiers only)
            - classes: Comma-separated list of class values (classifiers only)

        Raises:
            ValueError: If no data is available.

        Examples:
            >>> summary = model.summary()
            >>> summary['estimator_type']
            'classifier'
            >>> summary['num_features']
            4
        """
        if self._data is None:
            raise ValueError("No data available for summary.")

        num_features = self._data.X_train.shape[1]  # Number of features in training data
        feature_list = ", ".join(self._data.features)  # Get features from the Data class

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
            class_list = ", ".join(map(str, self._data.classes))  # Unique classes as a string
            summary_dict["num_classes"] = num_classes
            summary_dict["classes"] = class_list

        return summary_dict

    @property
    def features(self) -> list[str] | None:
        """Get feature names, preferring sklearn's feature_names_in_ over Data object."""
        # First try to get from fitted estimator (most reliable)
        if self._clf is not None and hasattr(self._clf, "feature_names_in_"):
            return list(self._clf.feature_names_in_)

        # Fall back to Data object if available
        if self._data is not None and hasattr(self._data, "features"):
            return self._data.features

        return None
