from .model import Model

import pandas as pd


def create_overview(paths):
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
