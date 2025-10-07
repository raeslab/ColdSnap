import pytest
from sklearn.ensemble import RandomForestClassifier
from coldsnap import Data, Model
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
import matplotlib

matplotlib.use("Agg")  # Include this line to suppress windows popping up during tests


def test_confusion_matrix(model_instance):
    # Fit the model and test confusion_matrix output
    model_instance.fit()
    cm = model_instance.confusion_matrix()

    assert cm.shape[0] == cm.shape[1], "Confusion matrix should be square."
    assert cm.sum() == len(model_instance.data.y_test), (
        "Sum of confusion matrix elements should match the number of test samples."
    )


def test_confusion_matrix_no_data():
    # Initialize Model without data
    model_without_data = Model(clf=RandomForestClassifier())

    # Attempt to call confusion_matrix without data
    with pytest.raises(ValueError, match="No data provided to perform predictions."):
        model_without_data.confusion_matrix()


def test_confusion_matrix_no_classifier(sample_dataframe):
    # Initialize Model with data but no classifier
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    model_without_clf = Model(data=data_instance)

    # Attempt to call confusion_matrix without classifier
    with pytest.raises(
        ValueError, match="No estimator provided to perform predictions."
    ):
        model_without_clf.confusion_matrix()


def test_display_confusion_matrix(model_instance):
    # Fit the model and test display_confusion_matrix output
    model_instance.fit()
    cm_display = model_instance.display_confusion_matrix()

    assert isinstance(cm_display, ConfusionMatrixDisplay), (
        "display_confusion_matrix should return an instance of ConfusionMatrixDisplay."
    )


def test_display_confusion_matrix_no_data():
    # Initialize Model without data
    model_without_data = Model(clf=RandomForestClassifier())

    # Attempt to call display_confusion_matrix without data
    with pytest.raises(ValueError, match="No data provided to perform predictions."):
        model_without_data.display_confusion_matrix()


def test_display_confusion_matrix_no_classifier(sample_dataframe):
    # Initialize Model with data but no classifier
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    model_without_clf = Model(data=data_instance)

    # Attempt to call display_confusion_matrix without classifier
    with pytest.raises(
        ValueError, match="No estimator provided to perform predictions."
    ):
        model_without_clf.display_confusion_matrix()


def test_display_roc_curve_no_data():
    # Initialize Model without data
    model_without_data = Model(clf=RandomForestClassifier())

    with pytest.raises(ValueError, match="No data provided to perform predictions."):
        model_without_data.display_roc_curve()


def test_display_roc_curve_no_classifier(sample_dataframe):
    # Initialize Model with data but no classifier
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    model_without_clf = Model(data=data_instance)

    with pytest.raises(
        ValueError, match="No estimator provided to perform predictions."
    ):
        model_without_clf.display_roc_curve()


def test_display_roc_curve_no_predict_proba(model_instance_no_proba):
    # Test with a classifier that doesn't support predict_proba
    model_instance_no_proba.fit()

    with pytest.raises(
        NotImplementedError,
        match="The classifier does not support probability predictions.",
    ):
        model_instance_no_proba.display_roc_curve()


def test_display_roc_curve_with_proba(model_instance_with_proba):
    # Test with a valid classifier that supports predict_proba
    model_instance_with_proba.fit()

    # Capture the ROC display output
    roc_display = model_instance_with_proba.display_roc_curve()

    assert isinstance(roc_display, RocCurveDisplay), (
        "display_roc_curve did not return a RocCurveDisplay object."
    )


def test_display_shap_beeswarm(model_instance):
    model_instance.fit()

    ax = model_instance.display_shap_beeswarm()

    assert isinstance(ax, matplotlib.axes._axes.Axes), (
        "display_shap_beeswarm did not return a Axes object."
    )


def test_display_shap_beeswarm_no_data_clf(model_instance):
    model_instance.clf = None

    with pytest.raises(
        ValueError, match="No estimator provided to perform predictions."
    ):
        model_instance.display_shap_beeswarm()

    model_instance.data = None

    with pytest.raises(ValueError, match="No data provided to perform predictions."):
        model_instance.display_shap_beeswarm()
