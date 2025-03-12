from coldsnap.utils import create_overview
import pandas as pd


def test_create_overview():
    # Define the test data path
    test_paths = ["./tests/data/test_model.pkl.gz"]

    # Run the function with the test path
    result_df = create_overview(test_paths)

    # Check that the result is a DataFrame
    assert isinstance(result_df, pd.DataFrame), "Output should be a DataFrame."

    # Check that the DataFrame has exactly one row, as we passed only one model path
    assert (
        len(result_df) == 1
    ), "DataFrame should have exactly one row for one input path."

    # Check required columns in the DataFrame
    expected_columns = {
        "path",
        "model_description",
        "data_description",
        "num_features",
        "features",
        "num_classes",
        "classes",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }
    assert expected_columns.issubset(
        result_df.columns
    ), "DataFrame columns are not as expected."

    # Additional checks based on test data expectations (e.g., non-null values for metrics)
    assert (
        result_df["path"].iloc[0] == test_paths[0]
    ), "Path should match the input path."
    assert result_df["accuracy"].iloc[0] is not None, "Accuracy should not be None."
    assert (
        result_df["model_description"].iloc[0] is not None
    ), "Model description should not be None."
    assert (
        result_df["num_features"].iloc[0] > 0
    ), "Number of features should be greater than zero."


def test_load_non_existing_file(capfd):
    # Path to a non-existent file
    non_existing_path = "./tests/data/non_existent_model.pkl.gz"

    # Call the function with the non-existing path
    result_df = create_overview([non_existing_path])

    # Capture the printed output
    captured = capfd.readouterr()

    # Check that the function handled the missing file and didn't raise an exception
    assert (
        result_df is not None
    ), "Function should return a DataFrame even if the file is missing."
    assert len(result_df) == 0, "DataFrame should be empty when file is missing."

    # Check that an appropriate error message was printed
    assert (
        "Failed to load model from" in captured.out
    ), "Function should print an error message for missing file."
