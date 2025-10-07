from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Optional, List
from .serializable import Serializable
import hashlib


class Data(Serializable):
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        description: Optional[str] = None,
        short_description: Optional[str] = None,
    ):
        if not all(
            isinstance(arg, (pd.DataFrame, pd.Series))
            for arg in [X_train, y_train, X_test, y_test]
        ):
            raise ValueError(
                "X_train, y_train, X_test, and y_test must be pandas DataFrames or Series."
            )
        if len(X_train) != len(y_train) or len(X_test) != len(y_test):
            raise ValueError("Inconsistent data lengths between X and y splits.")

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.description = description
        self.short_description = short_description

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        label_col: str,
        description: Optional[str] = None,
        short_description: Optional[str] = None,
        **train_test_split_kwargs,
    ) -> "Data":
        X = df.drop(columns=[label_col])
        y = df[label_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, **train_test_split_kwargs
        )
        return cls(
            X_train,
            y_train,
            X_test,
            y_test,
            description=description,
            short_description=short_description,
        )

    def set_description(self, description: str) -> None:
        self.description = description

    def set_short_description(self, short_description: str) -> None:
        self.short_description = short_description

    def purge(self) -> None:
        """Removes all data while keeping the dataframe headers intact."""
        self.X_train = self.X_train.iloc[0:0]
        self.y_train = self.y_train.iloc[0:0]
        self.X_test = self.X_test.iloc[0:0]
        self.y_test = self.y_test.iloc[0:0]

    @property
    def features(self) -> List[str]:
        return list(self.X_train.columns)

    @property
    def classes(self) -> List:
        return sorted(set(self.y_train.tolist() + self.y_test.tolist()))

    @property
    def hash(self) -> str:
        # Concatenate all data into a single byte stream
        data_bytes = pd.util.hash_pandas_object(self.X_train).values.tobytes()
        data_bytes += pd.util.hash_pandas_object(self.y_train).values.tobytes()
        data_bytes += pd.util.hash_pandas_object(self.X_test).values.tobytes()
        data_bytes += pd.util.hash_pandas_object(self.y_test).values.tobytes()

        # Compute and return the hex digest of the combined data
        return hashlib.sha256(data_bytes).hexdigest()
