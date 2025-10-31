"""Data handling for train/test splits with serialization.

This module provides the Data class which manages train/test data splits for machine
learning workflows. It wraps pandas DataFrames and Series, providing utilities for
data integrity checking, serialization, and metadata management.
"""

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from .serializable import Serializable


class Data(Serializable):
    """Container for train/test splits with serialization and integrity checking.

    The Data class wraps train/test data splits (X_train, y_train, X_test, y_test)
    along with optional descriptions. It provides methods for data integrity checking
    via SHA256 hashing, feature and class information extraction, and data purging
    for sharing model snapshots without exposing training data.

    Attributes:
        X_train: Training features as a pandas DataFrame.
        y_train: Training labels as a pandas Series.
        X_test: Test features as a pandas DataFrame.
        y_test: Test labels as a pandas Series.
        description: Optional long description of the dataset.
        short_description: Optional brief description of the dataset.

    Examples:
        Create Data from pre-split data:
        >>> data = Data(X_train, y_train, X_test, y_test, description="Iris dataset")

        Create Data from a single DataFrame:
        >>> data = Data.from_df(df, label_col="species", test_size=0.2, random_state=42)

        Remove data while keeping structure:
        >>> data.purge()  # Keeps column names and types, removes all rows
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        description: str | None = None,
        short_description: str | None = None,
    ):
        """Initialize Data with train/test splits.

        Args:
            X_train: Training features as pandas DataFrame.
            y_train: Training labels as pandas Series.
            X_test: Test features as pandas DataFrame.
            y_test: Test labels as pandas Series.
            description: Optional detailed description of the dataset.
            short_description: Optional brief description for summaries.

        Raises:
            ValueError: If any argument is not a DataFrame/Series, or if X and y
                splits have inconsistent lengths.
        """
        if not all(
            isinstance(arg, (pd.DataFrame, pd.Series)) for arg in [X_train, y_train, X_test, y_test]
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
        description: str | None = None,
        short_description: str | None = None,
        **train_test_split_kwargs,
    ) -> "Data":
        """Create Data object from a single DataFrame by performing train/test split.

        This is the most common way to create a Data object. It automatically splits
        the DataFrame into train and test sets using sklearn's train_test_split.

        Args:
            df: Complete dataset as a pandas DataFrame.
            label_col: Name of the column containing labels/targets.
            description: Optional detailed description of the dataset.
            short_description: Optional brief description for summaries.
            **train_test_split_kwargs: Additional arguments passed to sklearn's
                train_test_split (e.g., test_size=0.2, random_state=42, stratify=None).

        Returns:
            Data object with train/test splits created from the input DataFrame.

        Examples:
            >>> data = Data.from_df(df, label_col="target", test_size=0.2, random_state=42)
            >>> data = Data.from_df(df, "species", stratify=df["species"], test_size=0.3)
        """
        X = df.drop(columns=[label_col])
        y = df[label_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, **train_test_split_kwargs)
        return cls(
            X_train,
            y_train,
            X_test,
            y_test,
            description=description,
            short_description=short_description,
        )

    def set_description(self, description: str) -> None:
        """Set or update the detailed description of the dataset.

        Args:
            description: Detailed description text.
        """
        self.description = description

    def set_short_description(self, short_description: str) -> None:
        """Set or update the brief description of the dataset.

        Args:
            short_description: Brief description text for summaries.
        """
        self.short_description = short_description

    def purge(self) -> None:
        """Remove all data rows while keeping DataFrame structure intact.

        This method is useful for sharing model snapshots with their data structure
        (column names, types) without exposing the actual training data. After purging,
        all DataFrames will be empty but retain their column information.

        Examples:
            >>> data.X_train.shape
            (100, 5)
            >>> data.purge()
            >>> data.X_train.shape
            (0, 5)  # Rows removed, columns preserved
        """
        self.X_train = self.X_train.iloc[0:0]
        self.y_train = self.y_train.iloc[0:0]
        self.X_test = self.X_test.iloc[0:0]
        self.y_test = self.y_test.iloc[0:0]

    @property
    def features(self) -> list[str]:
        """Get list of feature names from the training data.

        Returns:
            List of column names from X_train.

        Examples:
            >>> data.features
            ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        """
        return list(self.X_train.columns)

    @property
    def classes(self) -> list | None:
        """Get sorted list of unique classes from labels, or None if no labels exist.

        Returns None for empty datasets (e.g., after purge()) or for transformer-only
        workflows where labels may not be meaningful.

        Returns:
            Sorted list of unique class values, or None if labels are empty.

        Examples:
            >>> data.classes
            ['setosa', 'versicolor', 'virginica']
            >>> data.purge()
            >>> data.classes
            None
        """
        # Check if y_train and y_test are empty (e.g., after purge or for transformers)
        if self.y_train.empty and self.y_test.empty:
            return None
        return sorted(set(self.y_train.tolist() + self.y_test.tolist()))

    @property
    def hash(self) -> str:
        """Generate SHA256 hash of all data for integrity checking.

        Combines hash values from all four data components (X_train, y_train, X_test,
        y_test) to create a unique fingerprint of the dataset. This can be used to
        verify data hasn't changed or to identify duplicate datasets.

        Returns:
            64-character hexadecimal SHA256 hash string.

        Examples:
            >>> data.hash
            '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8'
        """
        # Concatenate all data into a single byte stream
        data_bytes = pd.util.hash_pandas_object(self.X_train).values.tobytes()
        data_bytes += pd.util.hash_pandas_object(self.y_train).values.tobytes()
        data_bytes += pd.util.hash_pandas_object(self.X_test).values.tobytes()
        data_bytes += pd.util.hash_pandas_object(self.y_test).values.tobytes()

        # Compute and return the hex digest of the combined data
        return hashlib.sha256(data_bytes).hexdigest()
