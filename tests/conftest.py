import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from coldsnap import Data, Model


@pytest.fixture
def sample_dataframe():
    data = {
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
        "label": np.random.choice(["class1", "class2", "class3"], size=100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_binary_dataframe():
    data = {
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
        "label": np.random.choice(["class1", "class2"], size=100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def model_instance(sample_dataframe):
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(random_state=42)
    return Model(data=data_instance, clf=clf)


@pytest.fixture
def binary_model_instance(sample_binary_dataframe):
    data_instance = Data.from_df(
        sample_binary_dataframe, "label", test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(random_state=42)
    return Model(data=data_instance, clf=clf)


@pytest.fixture
def model_instance_svc(sample_dataframe):
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    clf = SVC(random_state=42)
    return Model(data=data_instance, clf=clf)


@pytest.fixture
def model_instance_with_proba(sample_dataframe):
    """Creates a Model instance with a classifier that supports predict_proba."""
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(random_state=42)
    return Model(data=data_instance, clf=clf)


@pytest.fixture
def model_instance_no_proba(sample_dataframe):
    """Creates a Model instance with a classifier that does not support predict_proba."""
    data_instance = Data.from_df(
        sample_dataframe, "label", test_size=0.2, random_state=42
    )
    clf = SVC(random_state=42)  # SVC without probability support
    return Model(data=data_instance, clf=clf)
