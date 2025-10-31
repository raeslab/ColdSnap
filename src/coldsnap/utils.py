"""Utility functions for model analysis and comparison.

This module provides helper functions for working with multiple ColdSnap models,
including creating comparison tables and generating overview reports.
"""

import pandas as pd

from .model import Model


def create_overview(paths):
    """Create a comparison DataFrame from multiple saved models.

    Loads each model from the provided paths, extracts summary and evaluation metrics,
    and combines them into a single DataFrame for easy comparison. Failed loads are
    printed to console and skipped.

    Args:
        paths: List of file paths to saved Model pickle files.

    Returns:
        pandas DataFrame with one row per successfully loaded model, containing:
        - path: The file path to the model
        - All fields from model.summary() (model_code, model_description, etc.)
        - All fields from model.evaluate() (accuracy, precision, rmse, etc.)

    Examples:
        >>> paths = ["model1.pkl.gz", "model2.pkl.gz", "model3.pkl.gz"]
        >>> overview = create_overview(paths)
        >>> overview[['model_code', 'accuracy', 'f1']]
           model_code  accuracy    f1
        0  model_rf    0.95        0.94
        1  model_gb    0.96        0.95
        2  model_lr    0.92        0.91

        >>> # Use for model comparison and selection
        >>> best_model_path = overview.loc[overview['f1'].idxmax(), 'path']

    Note:
        Models that fail to load will print an error message but won't stop processing
        of remaining models.
    """
    # List to store the information for each model
    model_data = []

    for path in paths:
        try:
            # Load model from pickle
            model = Model.from_pickle(path)

            # Get summary and evaluation metrics
            summary = model.summary()
            evaluation = model.evaluate()

            # Combine path, summary, and evaluation into a single dictionary
            model_info = {"path": path}
            model_info.update(summary)
            model_info.update(evaluation)

            # Append to list
            model_data.append(model_info)

        except Exception as e:
            print(f"Failed to load model from {path}: {e}")

    # Convert the list of dictionaries to a DataFrame
    return pd.DataFrame(model_data)
