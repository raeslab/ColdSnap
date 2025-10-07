from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
import shap
from shap import TreeExplainer
import matplotlib.pyplot as plt


class ConfusionMatrixMixin:
    def _predict_from_data(self):
        """Helper method to get true labels and predictions."""
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
        y_test, y_pred = self._predict_from_data()
        return confusion_matrix(y_test, y_pred)

    def display_confusion_matrix(self, **kwargs):
        y_test, y_pred = self._predict_from_data()
        cm_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, **kwargs)
        return cm_display


class ROCMixin:
    def display_roc_curve(self, **kwargs):
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
            raise NotImplementedError(
                "The classifier does not support probability predictions."
            )

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
    def display_shap_beeswarm(self, ref_class_idx=0, **kwargs):
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
