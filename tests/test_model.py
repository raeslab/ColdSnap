import pytest
from sklearn.ensemble import RandomForestClassifier
from coldsnap import Data, Model


def test_model_initialization(model_instance):
    assert isinstance(model_instance, Model)
    assert model_instance.data is not None
    assert model_instance.clf is not None


def test_fit(model_instance):
    model_instance.fit()
    assert hasattr(
        model_instance.clf, "estimators_"
    )  # Check if the model has been fitted (specific to ensemble models)


def test_evaluate(model_instance):
    model_instance.fit()
    metrics = model_instance.evaluate()

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics

    metrics = model_instance.evaluate(
        X_test=model_instance.data.X_test, y_test=model_instance.data.y_test
    )

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics


def test_evaluate_binary(binary_model_instance):
    binary_model_instance.fit()
    metrics = binary_model_instance.evaluate()

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics


def test_evaluate_no_proba(model_instance_svc):
    model_instance_svc.fit()
    metrics = model_instance_svc.evaluate()

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] is None


def test_serialization(model_instance, tmp_path):
    # Serialize
    path = tmp_path / "model_instance.pkl.gz"
    model_instance.to_pickle(path)

    # Deserialize
    loaded_instance = Model.from_pickle(path)

    assert isinstance(loaded_instance, Model)
    assert loaded_instance.data is not None
    assert loaded_instance.clf is not None
    assert loaded_instance.description == model_instance.description


def test_setters(model_instance, sample_dataframe):
    new_description = "New model description"
    model_instance.description = new_description
    model_instance.short_description = new_description
    assert model_instance.short_description == new_description
    assert model_instance.description == new_description

    new_clf = RandomForestClassifier(random_state=43)
    model_instance.clf = new_clf
    assert model_instance.clf is new_clf

    new_data = Data.from_df(sample_dataframe, "label", test_size=0.2, random_state=42)
    model_instance.data = new_data
    assert model_instance.data is new_data


def test_fit_without_data():
    # Initialize Model without data
    model_without_data = Model(clf=RandomForestClassifier())

    # Attempt to fit without data
    with pytest.raises(ValueError, match="No data provided to fit the classifier."):
        model_without_data.fit()


def test_fit_without_classifier(sample_dataframe):
    # Initialize Model with data but no classifier
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    model_without_clf = Model(data=data_instance)

    # Attempt to fit without a classifier
    with pytest.raises(ValueError, match="No classifier provided to fit the data."):
        model_without_clf.fit()


def test_evaluate_without_external_data(model_instance):
    model_instance.fit()

    # Attempt to evaluate with partial data
    with pytest.raises(
        ValueError,
        match="If external validation data is used, both X_test and y_test need to be provided.",
    ):
        model_instance.evaluate(X_test=model_instance.data.X_test)
    with pytest.raises(
        ValueError,
        match="If external validation data is used, both X_test and y_test need to be provided.",
    ):
        model_instance.evaluate(y_test=model_instance.data.y_test)


def test_evaluate_without_data():
    # Initialize Model with a classifier but no data
    model_without_data = Model(clf=RandomForestClassifier())

    # Attempt to evaluate without data
    with pytest.raises(
        ValueError, match="No data provided to evaluate the classifier."
    ):
        model_without_data.evaluate()


def test_evaluate_without_classifier(sample_dataframe):
    # Initialize Model with data but no classifier
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    model_without_clf = Model(data=data_instance)

    # Attempt to evaluate without a classifier
    with pytest.raises(ValueError, match="No classifier provided to evaluate."):
        model_without_clf.evaluate()


def test_predict(model_instance):
    # Test predict method
    model_instance.fit()
    X_test = model_instance.data.X_test
    predictions = model_instance.predict(X_test)
    assert len(predictions) == len(X_test), (
        "Prediction length does not match test data length."
    )
    assert set(predictions).issubset(set(model_instance.data.y_train.unique())), (
        "Predictions contain unexpected classes."
    )


def test_predict_proba_supported(model_instance):
    # Test predict_proba when the classifier supports it
    model_instance.fit()
    X_test = model_instance.data.X_test
    probabilities = model_instance.predict_proba(X_test)

    assert probabilities.shape == (
        len(X_test),
        len(model_instance.data.classes),
    ), "Shape of predict_proba output is incorrect."
    assert (probabilities >= 0).all() and (probabilities <= 1).all(), (
        "Probabilities should be between 0 and 1."
    )


def test_predict_proba_not_supported(model_instance_svc):
    # Test predict_proba when the classifier does not support it
    model_instance_svc.fit()
    X_test = model_instance_svc.data.X_test
    with pytest.raises(
        NotImplementedError,
        match="The classifier does not support probability predictions.",
    ):
        model_instance_svc.predict_proba(X_test)


def test_hash(model_instance):
    model_instance.fit()

    assert isinstance(model_instance.hash, str)


def test_summary(model_instance):
    # Test the summary method
    model_instance.fit()
    summary = model_instance.summary()

    assert isinstance(summary, dict)
    assert "model_description" in summary
    assert "model_hash" in summary
    assert "data_description" in summary
    assert "data_hash" in summary
    assert "num_features" in summary
    assert "features" in summary
    assert "num_classes" in summary
    assert "classes" in summary

    assert summary["num_features"] == len(model_instance.data.features), (
        "Number of features in summary does not match."
    )
    assert summary["num_classes"] == len(model_instance.data.classes), (
        "Number of classes in summary does not match."
    )
    assert set(summary["features"].split(", ")) == set(model_instance.data.features), (
        "Features in summary do not match."
    )
    assert set(summary["classes"].split(", ")) == set(
        map(str, model_instance.data.classes)
    ), "Classes in summary do not match."

    model_instance._data = None
    with pytest.raises(ValueError, match="No data available for summary."):
        model_instance.summary()


def test_purge(model_instance):
    model_instance.purge()
    assert model_instance.data is not None  # Ensure data instance still exists
    assert model_instance.data.X_train.empty
    assert model_instance.data.y_train.empty
    assert model_instance.data.X_test.empty
    assert model_instance.data.y_test.empty
