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

    # Test estimator setter
    another_clf = RandomForestClassifier(random_state=44)
    model_instance.estimator = another_clf
    assert model_instance.estimator is another_clf
    assert model_instance.clf is another_clf  # Should update both properties

    new_data = Data.from_df(sample_dataframe, "label", test_size=0.2, random_state=42)
    model_instance.data = new_data
    assert model_instance.data is new_data


def test_fit_without_data():
    # Initialize Model without data
    model_without_data = Model(clf=RandomForestClassifier())

    # Attempt to fit without data
    with pytest.raises(ValueError, match="No data provided to fit the estimator."):
        model_without_data.fit()


def test_fit_without_classifier(sample_dataframe):
    # Initialize Model with data but no classifier
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    model_without_clf = Model(data=data_instance)

    # Attempt to fit without a classifier
    with pytest.raises(ValueError, match="No estimator provided to fit the data."):
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
    with pytest.raises(ValueError, match="No data provided to evaluate the estimator."):
        model_without_data.evaluate()


def test_evaluate_without_classifier(sample_dataframe):
    # Initialize Model with data but no classifier
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    model_without_clf = Model(data=data_instance)

    # Attempt to evaluate without a classifier
    with pytest.raises(ValueError, match="No estimator provided to evaluate."):
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


def test_predict_without_estimator(sample_dataframe):
    # Test predict raises error when no estimator is set
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    model = Model(data=data_instance)

    with pytest.raises(ValueError, match="No estimator provided."):
        model.predict(data_instance.X_test)


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


def test_predict_proba_without_estimator(sample_dataframe):
    # Test predict_proba raises error when no estimator is set
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    model = Model(data=data_instance)

    with pytest.raises(ValueError, match="No estimator provided."):
        model.predict_proba(data_instance.X_test)


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


# Tests for transformer support
def test_transformer_workflow(sample_dataframe):
    """Test that transformers work correctly with fit and transform."""
    from sklearn.preprocessing import StandardScaler

    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    model = Model(data=data_instance, estimator=scaler)

    # Fit the transformer
    model.fit()

    # Transform the data
    X_transformed = model.transform(model.data.X_train)

    assert X_transformed.shape == model.data.X_train.shape
    # StandardScaler should produce mean ~0 and std ~1
    assert abs(X_transformed.mean()) < 0.1


def test_transform_without_estimator(sample_dataframe):
    """Test that transform raises error when no estimator is set."""
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    model = Model(data=data_instance)

    with pytest.raises(ValueError, match="No estimator provided."):
        model.transform(data_instance.X_test)


def test_transform_estimator_without_transform_method(sample_dataframe):
    """Test that transform raises error when estimator lacks transform method."""
    from sklearn.base import BaseEstimator

    # Create a mock transformer without transform method
    class MockTransformer(BaseEstimator):
        def fit(self, X, y=None):
            return self

    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    mock_transformer = MockTransformer()
    model = Model(data=data_instance, estimator=mock_transformer)
    model.fit()

    with pytest.raises(AttributeError, match="does not have a transform method"):
        model.transform(data_instance.X_test)


def test_transformer_cannot_predict(sample_dataframe):
    """Test that transformers cannot use predict()."""
    from sklearn.preprocessing import StandardScaler

    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    model = Model(data=data_instance, estimator=scaler)
    model.fit()

    with pytest.raises(TypeError, match="Cannot call predict.*transformer"):
        model.predict(model.data.X_test)


def test_transformer_cannot_predict_proba(sample_dataframe):
    """Test that transformers cannot use predict_proba()."""
    from sklearn.preprocessing import StandardScaler

    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    model = Model(data=data_instance, estimator=scaler)
    model.fit()

    with pytest.raises(TypeError, match="Cannot call predict_proba.*transformer"):
        model.predict_proba(model.data.X_test)


def test_transformer_cannot_evaluate(sample_dataframe):
    """Test that transformers cannot use evaluate()."""
    from sklearn.preprocessing import StandardScaler

    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    model = Model(data=data_instance, estimator=scaler)
    model.fit()

    with pytest.raises(TypeError, match="Cannot evaluate.*transformer"):
        model.evaluate()


def test_classifier_cannot_transform(sample_dataframe):
    """Test that classifiers cannot use transform()."""
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(random_state=42)
    model = Model(data=data_instance, estimator=clf)
    model.fit()

    with pytest.raises(TypeError, match="Cannot call transform.*classifier"):
        model.transform(model.data.X_test)


# Tests for regressor support
def test_regressor_workflow(sample_dataframe):
    """Test that regressors work correctly with fit and predict."""
    from sklearn.linear_model import LinearRegression

    # Create a regression dataset
    df = sample_dataframe.copy()
    df["target"] = df["feature1"] * 2 + df["feature2"] * 3
    # Drop the 'label' column as it's not needed for regression
    df = df.drop(columns=["label"])

    data_instance = Data.from_df(df, "target", test_size=0.2, random_state=42)
    regressor = LinearRegression()
    model = Model(data=data_instance, estimator=regressor)

    # Fit and predict should work
    model.fit()
    predictions = model.predict(model.data.X_test)

    assert len(predictions) == len(model.data.X_test)


def test_regressor_cannot_predict_proba(sample_dataframe):
    """Test that regressors cannot use predict_proba()."""
    from sklearn.linear_model import LinearRegression

    df = sample_dataframe.copy()
    df["target"] = df["feature1"] * 2 + df["feature2"] * 3
    df = df.drop(columns=["label"])

    data_instance = Data.from_df(df, "target", test_size=0.2, random_state=42)
    regressor = LinearRegression()
    model = Model(data=data_instance, estimator=regressor)
    model.fit()

    with pytest.raises(TypeError, match="Cannot call predict_proba.*regressor"):
        model.predict_proba(model.data.X_test)


def test_regressor_evaluate_not_implemented(sample_dataframe):
    """Test that regressor evaluate() raises NotImplementedError."""
    from sklearn.linear_model import LinearRegression

    df = sample_dataframe.copy()
    df["target"] = df["feature1"] * 2 + df["feature2"] * 3
    df = df.drop(columns=["label"])

    data_instance = Data.from_df(df, "target", test_size=0.2, random_state=42)
    regressor = LinearRegression()
    model = Model(data=data_instance, estimator=regressor)
    model.fit()

    with pytest.raises(NotImplementedError, match="Evaluation for regressors"):
        model.evaluate()


# Tests for backward compatibility
def test_clf_parameter_still_works(sample_dataframe):
    """Test that the original clf parameter still works."""
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(random_state=42)
    model = Model(data=data_instance, clf=clf)

    model.fit()
    predictions = model.predict(model.data.X_test)

    assert len(predictions) == len(model.data.X_test)


def test_estimator_parameter_works(sample_dataframe):
    """Test that the new estimator parameter works."""
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(random_state=42)
    model = Model(data=data_instance, estimator=clf)

    model.fit()
    predictions = model.predict(model.data.X_test)

    assert len(predictions) == len(model.data.X_test)


def test_cannot_provide_both_clf_and_estimator(sample_dataframe):
    """Test that providing both clf and estimator raises an error."""
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(random_state=42)

    with pytest.raises(ValueError, match="either 'clf' or 'estimator'"):
        Model(data=data_instance, clf=clf, estimator=clf)


def test_clf_property_works(sample_dataframe):
    """Test that the clf property still works for backward compatibility."""
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(random_state=42)
    model = Model(data=data_instance, estimator=clf)

    # Should be able to access via .clf
    assert model.clf is clf
    assert isinstance(model.clf, RandomForestClassifier)


def test_estimator_property_works(sample_dataframe):
    """Test that the estimator property works."""
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(random_state=42)
    model = Model(data=data_instance, clf=clf)

    # Should be able to access via .estimator
    assert model.estimator is clf
    assert isinstance(model.estimator, RandomForestClassifier)


def test_get_estimator_type(sample_dataframe):
    """Test that _get_estimator_type correctly identifies estimator types."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression

    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )

    # Test classifier
    clf_model = Model(data=data_instance, estimator=RandomForestClassifier())
    assert clf_model._get_estimator_type() == "classifier"

    # Test regressor
    reg_model = Model(data=data_instance, estimator=LinearRegression())
    assert reg_model._get_estimator_type() == "regressor"

    # Test transformer
    trans_model = Model(data=data_instance, estimator=StandardScaler())
    assert trans_model._get_estimator_type() == "transformer"

    # Test None
    none_model = Model(data=data_instance)
    assert none_model._get_estimator_type() is None


def test_transformer_summary(sample_dataframe):
    """Test that summary works for transformers and doesn't include class info."""
    from sklearn.preprocessing import StandardScaler

    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    model = Model(data=data_instance, estimator=scaler)
    model.fit()

    summary = model.summary()

    assert "estimator_type" in summary
    assert summary["estimator_type"] == "transformer"
    # Transformers shouldn't have class information
    assert "num_classes" not in summary or summary.get("num_classes") is None
    assert "classes" not in summary or summary.get("classes") is None


def test_classifier_summary_includes_classes(sample_dataframe):
    """Test that summary for classifiers still includes class info."""
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(random_state=42)
    model = Model(data=data_instance, estimator=clf)
    model.fit()

    summary = model.summary()

    assert "estimator_type" in summary
    assert summary["estimator_type"] == "classifier"
    assert "num_classes" in summary
    assert "classes" in summary
    assert summary["num_classes"] == 3  # From sample_dataframe fixture
