"""Mixins providing evaluation and visualization capabilities for Model class.

This module contains mixins that add classifier-specific visualization methods to the
Model class. These mixins are only applicable to classifiers and will raise TypeErrors
when used with regressors or transformers.
"""

import matplotlib.pyplot as plt
import shap
from shap import TreeExplainer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
)


class ConfusionMatrixMixin:
    """Mixin providing confusion matrix generation and visualization.

    This mixin adds methods for computing and displaying confusion matrices for
    classification models. Methods are only available for classifiers and will raise
    TypeError for other estimator types.
    """

    def _predict_from_data(self):
        """Helper method to get true labels and predictions from test data.

        Internal method used by confusion matrix methods to validate the model is a
        classifier and generate predictions.

        Returns:
            Tuple of (y_test, y_pred) arrays.

        Raises:
            ValueError: If no data or estimator is provided.
            TypeError: If estimator is not a classifier.
        """
        if not self._data:
            raise ValueError("No data provided to perform predictions.")
        if not self._clf:
            raise ValueError("No estimator provided to perform predictions.")

        # Check that estimator is a classifier
        estimator_type = self._get_estimator_type()
        if estimator_type != "classifier":
            raise TypeError(
                f"Confusion matrix requires a classifier, but estimator is a {estimator_type}."
            )

        X_test, y_test = self._data.X_test, self._data.y_test
        y_pred = self._clf.predict(X_test)

        return y_test, y_pred

    def confusion_matrix(self):
        """Compute confusion matrix for the classifier on test data.

        Returns:
            numpy array of shape (n_classes, n_classes) with confusion matrix values.

        Raises:
            ValueError: If no data or estimator is provided.
            TypeError: If estimator is not a classifier.

        Examples:
            >>> cm = model.confusion_matrix()
            >>> cm
            array([[50,  0,  0],
                   [ 0, 47,  3],
                   [ 0,  2, 48]])
        """
        y_test, y_pred = self._predict_from_data()
        return confusion_matrix(y_test, y_pred)

    def display_confusion_matrix(self, **kwargs):
        """Display confusion matrix visualization.

        Creates and displays a confusion matrix plot using matplotlib. The plot can be
        customized using any keyword arguments accepted by
        sklearn.metrics.ConfusionMatrixDisplay.from_predictions().

        Args:
            **kwargs: Additional arguments passed to ConfusionMatrixDisplay.from_predictions().
                Common options: cmap, normalize, values_format.

        Returns:
            ConfusionMatrixDisplay object.

        Raises:
            ValueError: If no data or estimator is provided.
            TypeError: If estimator is not a classifier.

        Examples:
            >>> display = model.display_confusion_matrix()
            >>> display = model.display_confusion_matrix(normalize='true', cmap='Blues')
        """
        y_test, y_pred = self._predict_from_data()
        cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, **kwargs)
        return cm_display


class ROCMixin:
    """Mixin providing ROC curve visualization for classifiers.

    This mixin adds methods for generating ROC (Receiver Operating Characteristic) curves
    for classification models. Supports both binary and multi-class classification with
    one-vs-rest curves. Only available for classifiers with probability predictions.
    """

    def display_roc_curve(self, **kwargs):
        """Display ROC curves for classifier predictions.

        For multi-class problems, creates one-vs-rest ROC curves for each class. For
        binary classification, creates a single ROC curve. Only works with classifiers
        that support probability predictions (predict_proba).

        Args:
            **kwargs: Additional arguments passed to RocCurveDisplay.from_predictions().
                Common options: ax, name, color.

        Returns:
            RocCurveDisplay object for the last class plotted (useful for styling).

        Raises:
            ValueError: If no data or estimator is provided.
            TypeError: If estimator is not a classifier.
            NotImplementedError: If classifier doesn't support probability predictions.

        Examples:
            >>> display = model.display_roc_curve()
            >>> display = model.display_roc_curve(color='red')

        Note:
            For multi-class problems, all class curves are plotted on the same axes,
            with a chance level line shown after the last class.
        """
        if not self._data:
            raise ValueError("No data provided to perform predictions.")
        if not self._clf:
            raise ValueError("No estimator provided to perform predictions.")

        # Check that estimator is a classifier
        estimator_type = self._get_estimator_type()
        if estimator_type != "classifier":
            raise TypeError(
                f"ROC curve requires a classifier, but estimator is a {estimator_type}."
            )

        if not hasattr(self._clf, "predict_proba"):
            raise NotImplementedError("The classifier does not support probability predictions.")

        y_predicted = self._clf.predict_proba(self.data.X_test)

        rfc_disp = None

        for idx, classlabel in enumerate(self._clf.classes_):
            y_test_roc = [1 if y == classlabel else 0 for y in self.data.y_test]
            y_pred_roc = [p[idx] for p in y_predicted]

            rfc_disp = RocCurveDisplay.from_predictions(
                y_test_roc,
                y_pred_roc,
                name=f"{classlabel} vs Other",
                plot_chance_level=(idx == len(self._clf.classes_) - 1),
                **kwargs,
            )

        return rfc_disp


class SHAPMixin:
    """Mixin providing SHAP (SHapley Additive exPlanations) visualization.

    This mixin adds methods for generating SHAP feature importance visualizations for
    tree-based classifiers. SHAP values help explain individual predictions and overall
    feature importance. Only available for tree-based classifiers (RandomForest,
    GradientBoosting, XGBoost, etc.).
    """

    def display_shap_beeswarm(self, ref_class_idx=0, **kwargs):
        """Display SHAP beeswarm plot showing feature importance.

        Creates a SHAP beeswarm plot that shows how each feature impacts predictions
        for the specified class. Features are ordered by importance, with each dot
        representing a sample. Color indicates feature value (red=high, blue=low).

        Args:
            ref_class_idx: Index of the target class to explain (default: 0).
                For binary classification, use 0 or 1. For multi-class, select the
                class of interest.
            **kwargs: Additional arguments passed to shap.plots.beeswarm().
                Common options: max_display, plot_size.

        Returns:
            matplotlib Axes object containing the plot.

        Raises:
            ValueError: If no data or estimator is provided.
            TypeError: If estimator is not a classifier.

        Examples:
            >>> # Show SHAP values for first class (default)
            >>> ax = model.display_shap_beeswarm()

            >>> # Show SHAP values for second class
            >>> ax = model.display_shap_beeswarm(ref_class_idx=1)

            >>> # Limit to top 10 features
            >>> ax = model.display_shap_beeswarm(max_display=10)

        Note:
            Only works with tree-based estimators (RandomForest, GradientBoosting, etc.)
            that are compatible with SHAP's TreeExplainer.
        """
        if not self._data:
            raise ValueError("No data provided to perform predictions.")
        if not self._clf:
            raise ValueError("No estimator provided to perform predictions.")

        # Check that estimator is a classifier
        estimator_type = self._get_estimator_type()
        if estimator_type != "classifier":
            raise TypeError(
                f"SHAP visualization requires a classifier, but estimator is a {estimator_type}."
            )

        explainer = TreeExplainer(self._clf)
        explanation = explainer(self.data.X_train)

        shap.plots.beeswarm(explanation[:, :, ref_class_idx], **kwargs)

        return plt.gca()
