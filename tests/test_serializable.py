import pytest
import pickle
import gzip
from coldsnap.serializable import (
    Serializable,
)  # Replace `your_module` with the actual module name


# Mock class inheriting from Serializable for testing
class MockSerializable(Serializable):
    def __init__(self, data):
        self.data = data


def test_serialization(tmp_path):
    # Create an instance of the mock class
    obj = MockSerializable(data="Test data")

    # Define the path for serialization
    path = tmp_path / "mock_serializable.pkl.gz"

    # Test to_pickle
    obj.to_pickle(path)
    assert path.is_file(), "Serialized file was not created."

    # Test overwrite=False (should raise FileExistsError)
    with pytest.raises(FileExistsError, match="File '.*' already exists"):
        obj.to_pickle(path, overwrite=False)

    # Test overwrite=True (should allow overwriting)
    obj.to_pickle(path, overwrite=True)
    assert path.is_file(), "File was not overwritten when overwrite=True."

    # Test from_pickle
    loaded_obj = MockSerializable.from_pickle(path)
    assert isinstance(loaded_obj, MockSerializable), (
        "Loaded object is not of type MockSerializable."
    )
    assert loaded_obj.data == obj.data, "Loaded data does not match original data."


def test_invalid_path_serialization(tmp_path):
    # Create an instance of the mock class
    obj = MockSerializable(data="Test data")

    # Test error handling on invalid path
    with pytest.raises(OSError, match="Failed to write MockSerializable object"):
        obj.to_pickle("/invalid_path/mock_serializable.pkl.gz")


def test_invalid_path_deserialization():
    # Test error handling on invalid path during deserialization
    with pytest.raises(OSError, match="Failed to load MockSerializable object"):
        MockSerializable.from_pickle("/invalid_path/non_existent_file.pkl.gz")


def test_type_mismatch_deserialization(tmp_path):
    # Serialize an object of a different type to a file
    incorrect_obj = {"key": "value"}
    path = tmp_path / "incorrect_type.pkl.gz"

    # Write the incorrect object directly using gzip + pickle
    with gzip.open(path, "wb") as f:
        pickle.dump(incorrect_obj, f)

    # Test deserialization with type mismatch
    with pytest.raises(
        IOError, match="The loaded object is not an instance of MockSerializable"
    ):
        MockSerializable.from_pickle(path)
