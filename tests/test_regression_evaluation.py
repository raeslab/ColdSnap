import pytest
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from coldsnap import Data, Model


def test_regressor_evaluate_basic(regressor_model_instance):
    """Test that regressor evaluate() returns correct metrics structure."""
    regressor_model_instance.fit()
    metrics = regressor_model_instance.evaluate()

    assert isinstance(metrics, dict)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "mse" in metrics

    # Verify metric values are reasonable
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert metrics["mse"] > 0
    assert -1 <= metrics["r2"] <= 1  # R2 can be negative for poor models

    # For a good linear model on linear data, R2 should be high
    assert metrics["r2"] > 0.8


def test_regressor_evaluate_with_external_data(regressor_model_instance):
    """Test that regressor evaluate() works with external X_test and y_test."""
    regressor_model_instance.fit()

    # Test with external data
    metrics = regressor_model_instance.evaluate(
        X_test=regressor_model_instance.data.X_test,
        y_test=regressor_model_instance.data.y_test,
    )

    assert isinstance(metrics, dict)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "mse" in metrics


def test_regressor_evaluate_partial_external_data_raises_error(
    regressor_model_instance,
):
    """Test that providing only X_test or y_test raises ValueError."""
    regressor_model_instance.fit()

    with pytest.raises(
        ValueError,
        match="If external validation data is used, both X_test and y_test need to be provided.",
    ):
        regressor_model_instance.evaluate(X_test=regressor_model_instance.data.X_test)

    with pytest.raises(
        ValueError,
        match="If external validation data is used, both X_test and y_test need to be provided.",
    ):
        regressor_model_instance.evaluate(y_test=regressor_model_instance.data.y_test)


def test_regressor_evaluate_multiple_algorithms(sample_regression_dataframe):
    """Test evaluation with different regression algorithms."""
    regressors = [
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge(random_state=42)),
        ("Lasso", Lasso(random_state=42)),
        ("RandomForest", RandomForestRegressor(random_state=42, n_estimators=10)),
    ]

    for name, regressor in regressors:
        data_instance = Data.from_df(
            sample_regression_dataframe, "target", test_size=0.2, random_state=42
        )
        model = Model(data=data_instance, estimator=regressor)
        model.fit()
        metrics = model.evaluate()

        assert isinstance(metrics, dict), f"{name} evaluate() should return dict"
        assert "rmse" in metrics, f"{name} should have rmse metric"
        assert "mae" in metrics, f"{name} should have mae metric"
        assert "r2" in metrics, f"{name} should have r2 metric"
        assert "mse" in metrics, f"{name} should have mse metric"


def test_regressor_serialization_with_evaluation(regressor_model_instance, tmp_path):
    """Test that regressor can be saved, loaded, and evaluated."""
    regressor_model_instance.fit()

    # Get metrics before saving
    metrics_before = regressor_model_instance.evaluate()

    # Save the model
    path = tmp_path / "regressor_model.pkl.gz"
    regressor_model_instance.to_pickle(path)

    # Load the model
    loaded_model = Model.from_pickle(path)

    # Get metrics after loading
    metrics_after = loaded_model.evaluate()

    # Metrics should be identical
    assert metrics_before.keys() == metrics_after.keys()
    for key in metrics_before.keys():
        assert np.isclose(metrics_before[key], metrics_after[key]), (
            f"{key} should be the same before and after loading"
        )


def test_regressor_summary(regressor_model_instance):
    """Test that summary works correctly for regressors."""
    regressor_model_instance.fit()
    summary = regressor_model_instance.summary()

    assert isinstance(summary, dict)
    assert "estimator_type" in summary
    assert summary["estimator_type"] == "regressor"
    assert "num_features" in summary
    assert "features" in summary

    # Regressors should not have class information
    assert "num_classes" not in summary or summary.get("num_classes") is None
    assert "classes" not in summary or summary.get("classes") is None


def test_regressor_predict_accuracy(regressor_model_instance):
    """Sanity check that predictions are reasonable."""
    regressor_model_instance.fit()

    X_test = regressor_model_instance.data.X_test
    y_test = regressor_model_instance.data.y_test

    predictions = regressor_model_instance.predict(X_test)

    assert len(predictions) == len(y_test)
    # Predictions should be in a reasonable range (not NaN or Inf)
    assert np.all(np.isfinite(predictions))

    # For our linear data with linear model, predictions should be close to actual
    mean_absolute_error = np.mean(np.abs(predictions - y_test))
    assert mean_absolute_error < 0.5  # Should be reasonably accurate


def test_regressor_evaluate_without_data():
    """Test that evaluate() raises error when no data is provided."""
    from sklearn.linear_model import LinearRegression

    model_without_data = Model(estimator=LinearRegression())

    with pytest.raises(ValueError, match="No data provided to evaluate the estimator."):
        model_without_data.evaluate()


def test_regressor_evaluate_without_estimator(sample_regression_dataframe):
    """Test that evaluate() raises error when no estimator is provided."""
    data_instance = Data.from_df(
        sample_regression_dataframe, "target", test_size=0.2, random_state=42
    )
    model_without_estimator = Model(data=data_instance)

    with pytest.raises(ValueError, match="No estimator provided to evaluate."):
        model_without_estimator.evaluate()


def test_regressor_mse_rmse_relationship(regressor_model_instance):
    """Test that RMSE is the square root of MSE."""
    regressor_model_instance.fit()
    metrics = regressor_model_instance.evaluate()

    # RMSE should be sqrt(MSE)
    assert np.isclose(metrics["rmse"], np.sqrt(metrics["mse"]))


def test_regressor_metrics_non_negative(regressor_model_instance):
    """Test that RMSE, MAE, and MSE are non-negative."""
    regressor_model_instance.fit()
    metrics = regressor_model_instance.evaluate()

    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
    assert metrics["mse"] >= 0
