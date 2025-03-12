import pickle
import gzip
import os


class Serializable:
    def to_pickle(self, path: str, overwrite: bool = False) -> None:
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(
                f"File '{path}' already exists. Use overwrite=True to overwrite it."
            )

        try:
            with gzip.open(path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            raise IOError(
                f"Failed to write {self.__class__.__name__} object to {path}: {e}"
            )

    @classmethod
    def from_pickle(cls, path: str):
        try:
            with gzip.open(path, "rb") as f:
                obj = pickle.load(f)
            if not isinstance(obj, cls):
                raise TypeError(
                    f"The loaded object is not an instance of {cls.__name__}."
                )
            return obj
        except Exception as e:
            raise IOError(f"Failed to load {cls.__name__} object from {path}: {e}")
