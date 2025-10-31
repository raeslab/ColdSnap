"""Base class for serializing objects to compressed pickle files.

This module provides the Serializable base class that enables any subclass to save
and load itself using gzip-compressed pickle files. This is used throughout ColdSnap
for persisting Data and Model objects.
"""

import gzip
import os
import pickle


class Serializable:
    """Base class that provides pickle-based serialization with gzip compression.

    This class provides methods for saving objects to compressed pickle files and loading
    them back. It includes safety features like overwrite protection and type validation.

    Subclasses automatically inherit serialization capabilities by inheriting from this class.

    Examples:
        >>> class MyClass(Serializable):
        ...     def __init__(self, value):
        ...         self.value = value
        >>> obj = MyClass(42)
        >>> obj.to_pickle("my_object.pkl.gz")
        >>> loaded = MyClass.from_pickle("my_object.pkl.gz")
        >>> loaded.value
        42
    """

    def to_pickle(self, path: str, overwrite: bool = False) -> None:
        """Save the object to a gzip-compressed pickle file.

        Args:
            path: Path where the pickle file should be saved. Convention is to use
                .pkl.gz extension.
            overwrite: If True, overwrite existing file. If False (default), raise
                FileExistsError if file exists.

        Raises:
            FileExistsError: If file exists and overwrite=False.
            IOError: If writing to file fails for any reason.

        Examples:
            >>> obj = MyClass(42)
            >>> obj.to_pickle("my_object.pkl.gz")
            >>> obj.to_pickle("my_object.pkl.gz", overwrite=True)  # Overwrite existing
        """
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(
                f"File '{path}' already exists. Use overwrite=True to overwrite it."
            )

        try:
            with gzip.open(path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            raise OSError(f"Failed to write {self.__class__.__name__} object to {path}: {e}") from e

    @classmethod
    def from_pickle(cls, path: str):
        """Load an object from a gzip-compressed pickle file.

        Args:
            path: Path to the pickle file to load.

        Returns:
            The deserialized object, guaranteed to be an instance of the calling class.

        Raises:
            TypeError: If the loaded object is not an instance of the expected class.
            IOError: If reading from file fails or file doesn't exist.

        Examples:
            >>> loaded = MyClass.from_pickle("my_object.pkl.gz")
            >>> isinstance(loaded, MyClass)
            True
        """
        try:
            with gzip.open(path, "rb") as f:
                obj = pickle.load(f)
            if not isinstance(obj, cls):
                raise TypeError(f"The loaded object is not an instance of {cls.__name__}.")
            return obj
        except Exception as e:
            raise OSError(f"Failed to load {cls.__name__} object from {path}: {e}") from e
