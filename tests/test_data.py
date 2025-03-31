import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from coldsnap.data import Data


def test_data_initialization(sample_dataframe):
    X_train, X_test, y_train, y_test = train_test_split(
        sample_dataframe.drop(columns="label"),
        sample_dataframe["label"],
        test_size=0.2,
        random_state=42,
    )
    data_instance = Data(X_train, y_train, X_test, y_test)

    assert isinstance(data_instance, Data)
    assert data_instance.X_train.shape == X_train.shape
    assert data_instance.y_train.shape == y_train.shape
    assert data_instance.X_test.shape == X_test.shape
    assert data_instance.y_test.shape == y_test.shape
    assert data_instance.description is None
    assert data_instance.short_description is None

    new_description = "New description"
    data_instance.set_description(new_description)
    data_instance.set_short_description(new_description)
    assert data_instance.description == new_description
    assert data_instance.short_description == new_description


def test_from_df(sample_dataframe):
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )

    assert isinstance(data_instance, Data)
    assert (
        data_instance.X_train.shape[0] + data_instance.X_test.shape[0]
        == sample_dataframe.shape[0]
    )
    assert set(data_instance.y_train.unique()).union(
        data_instance.y_test.unique()
    ) == set(sample_dataframe["label"])

    assert isinstance(data_instance.hash, str)


def test_serialization(sample_dataframe, tmp_path):
    X_train, X_test, y_train, y_test = train_test_split(
        sample_dataframe.drop(columns="label"),
        sample_dataframe["label"],
        test_size=0.2,
        random_state=42,
    )
    data_instance = Data(X_train, y_train, X_test, y_test)

    # Serialize
    path = tmp_path / "data_instance.pkl.gz"
    data_instance.to_pickle(path)

    # Deserialize
    loaded_instance = Data.from_pickle(path)

    assert isinstance(loaded_instance, Data)
    pd.testing.assert_frame_equal(loaded_instance.X_train, data_instance.X_train)
    pd.testing.assert_series_equal(loaded_instance.y_train, data_instance.y_train)
    pd.testing.assert_frame_equal(loaded_instance.X_test, data_instance.X_test)
    pd.testing.assert_series_equal(loaded_instance.y_test, data_instance.y_test)


def test_features_property(sample_dataframe):
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    expected_features = list(data_instance.X_train.columns)

    assert data_instance.features == expected_features


def test_classes_property(sample_dataframe):
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    expected_classes = set(
        data_instance.y_train.tolist() + data_instance.y_test.tolist()
    )

    assert all(cls in expected_classes for cls in data_instance.classes)


def test_data_init_non_dataframe_series():
    # Prepare invalid data: using lists instead of DataFrames/Series
    X_train = [[1, 2], [3, 4]]
    y_train = [0, 1]
    X_test = [[5, 6]]
    y_test = [1]

    with pytest.raises(
        ValueError,
        match="X_train, y_train, X_test, and y_test must be pandas DataFrames or Series.",
    ):
        Data(X_train, y_train, X_test, y_test)


def test_data_init_inconsistent_lengths_train():
    # Prepare data with inconsistent lengths between X_train and y_train
    X_train = pd.DataFrame([[1, 2], [3, 4]])
    y_train = pd.Series([0])  # Incorrect length
    X_test = pd.DataFrame([[5, 6]])
    y_test = pd.Series([1])

    with pytest.raises(
        ValueError, match="Inconsistent data lengths between X and y splits."
    ):
        Data(X_train, y_train, X_test, y_test)


def test_data_init_inconsistent_lengths_test():
    # Prepare data with inconsistent lengths between X_test and y_test
    X_train = pd.DataFrame([[1, 2], [3, 4]])
    y_train = pd.Series([0, 1])
    X_test = pd.DataFrame([[5, 6]])
    y_test = pd.Series([1, 0])  # Incorrect length

    with pytest.raises(
        ValueError, match="Inconsistent data lengths between X and y splits."
    ):
        Data(X_train, y_train, X_test, y_test)


def test_purge(sample_dataframe):
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )

    # Ensure initial data is not empty
    assert not data_instance.X_train.empty
    assert not data_instance.y_train.empty
    assert not data_instance.X_test.empty
    assert not data_instance.y_test.empty

    # Purge data
    data_instance.purge()

    # Check that all data is removed but columns remain
    assert data_instance.X_train.empty and list(data_instance.X_train.columns) == list(
        sample_dataframe.drop(columns="label").columns
    )
    assert data_instance.y_train.empty and data_instance.y_train.name == "label"
    assert data_instance.X_test.empty and list(data_instance.X_test.columns) == list(
        sample_dataframe.drop(columns="label").columns
    )
    assert data_instance.y_test.empty and data_instance.y_test.name == "label"
