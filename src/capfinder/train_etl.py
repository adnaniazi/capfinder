import hashlib
import os
from typing import Any, List, Tuple

import numpy as np
import polars as pl
from dask.distributed import LocalCluster
from polars import DataFrame
from prefect import flow, task
from prefect.engine import PrefectFuture, TaskRunContext
from prefect.tasks import task_input_hash
from prefect_dask.task_runners import DaskTaskRunner
from typing_extensions import Literal


def custom_hash(context: TaskRunContext, parameters: dict[str, Any]) -> str:
    folder_path = parameters.get("folder_path")
    files = os.listdir(folder_path)
    # Sort files to ensure consistent hash
    files.sort()
    # Generate hash based on file names, sizes, and modification times
    file_metadata = [
        (
            file,
            os.path.getsize(os.path.join(folder_path, file)),  # type: ignore
            os.path.getmtime(os.path.join(folder_path, file)),  # type: ignore
        )
        for file in files
    ]
    hash_value = hashlib.sha256(str(file_metadata).encode()).hexdigest()

    return hash_value


@task(name="list-csv-files")
def list_csv_files(folder_path: str) -> list:
    """List all CSV files in a folder.

    Parameters
    ----------
    folder_path: str
        Path to the folder containing CSV files.

    Returns
    -------
    list
        List of CSV files in the folder.

    """
    return [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".csv") and not file.startswith(".")
    ]


@task(name="load-csv", cache_key_fn=task_input_hash)
def load_csv(file_path: str) -> pl.DataFrame:
    """Load a CSV file into a DataFrame.

    Parameters
    ----------
    file_path: str
        Path to the CSV file.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the CSV data.
    """
    return pl.read_csv(file_path)


@task(name="concatenate-dataframes", cache_key_fn=task_input_hash)
def concatenate_dataframes(
    dataframes: List[PrefectFuture[DataFrame, Literal[False]]]
) -> pl.DataFrame:
    """Concatenate a list of DataFrames vertically.

    Parameters
    ----------
    dataframes: List[PrefectFuture[DataFrame, Literal[False]]]
        List of DataFrames to concatenate.

    Returns
    -------
    pl.DataFrame
        Concatenated DataFrame. Returns an empty DataFrame if the input list is empty.
    """
    if not dataframes:
        return pl.DataFrame()
    return pl.concat(dataframes, how="vertical")  # type: ignore


@task(name="make-train-val-test-split", cache_key_fn=task_input_hash)
def make_train_val_test_split(
    df: pl.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    random_seed: int,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split a Polars DataFrame into train, validation, and test sets.

    Parameters
    ----------
        df (pl.DataFrame): The input Polars DataFrame.
        train_frac (float): The fraction of the data to use for the training set.
        val_frac (float): The fraction of the data to use for the validation set.
        test_frac (float): The fraction of the data to use for the test set.
        random_seed (int): The random seed for repeatable splits.

    Returns
    -------
        Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: A tuple containing the train, validation, and test sets.
    """

    # Verify that the sum of fractions is equal to 1.0
    if train_frac + val_frac + test_frac != 1.0:
        raise ValueError(
            "The sum of train_frac, val_frac, and test_frac must be equal to 1.0"
        )

    # Shuffle the DataFrame rows based on the random seed
    df = df.sample(fraction=1.0, with_replacement=False, seed=random_seed)

    # Calculate the number of rows for each split
    total_rows = df.height
    train_rows = int(total_rows * train_frac)
    val_rows = int(total_rows * val_frac)

    # Split the DataFrame
    train_df = df[:train_rows]
    val_df = df[train_rows : train_rows + val_rows]
    test_df = df[train_rows + val_rows :]

    return train_df, val_df, test_df


@task(name="zero-pad-and-reshape", cache_key_fn=task_input_hash)
def zero_pad_and_reshape(
    df: pl.DataFrame, column_name: str, target_length: int
) -> np.ndarray:
    """
    Zero pads the time series column in a Polars DataFrame to a specified length and reshapes the data.

    Parameters
    ----------
        df (pl.DataFrame): The input Polars DataFrame.
        column_name (str): Name of the time series column to zero pad.
        target_length (int): The desired length of each time series.

    Returns
    -------
        np.ndarray: The zero-padded and reshaped data with dimensions (num_examples x num_timesteps x num_features).
    """

    # Function to parse the time series from string to numpy array, and zero pad or truncate
    def parse_and_pad(series: str) -> np.ndarray:
        # Remove brackets and replace newline characters with spaces
        series_cleaned = series.replace("\n", " ").strip("[]")
        # Convert to a numpy array of floats
        series_array = np.fromstring(series_cleaned, sep=" ")

        # Calculate the length of the series
        series_length = len(series_array)

        # Handle the case where the length is greater than the target length
        if series_length > target_length:
            # Truncate the series to the target length
            series_array = series_array[:target_length]
        # Handle the case where the length is less than the target length
        elif series_length < target_length:
            # Pad the series with zeros to reach the target length
            series_array = np.pad(
                series_array,
                (0, target_length - series_length),
                "constant",
                constant_values=0,
            )

        return series_array

    # Apply the function to parse and pad each time series in the column
    zero_padded_data = (
        df[column_name]
        .map_elements(parse_and_pad, return_dtype=pl.List(pl.Float64))
        .to_numpy()
    )

    # Convert the array to shape (num_examples, num_timesteps, num_features)
    zero_padded_data_2d = np.array(list(zero_padded_data))

    reshaped_data = zero_padded_data_2d.reshape(len(zero_padded_data), target_length, 1)

    return reshaped_data


@task(name="make-x-y-read-id-sets", cache_key_fn=task_input_hash)
def make_x_y_read_id_sets(
    dataset: pl.DataFrame, target_length: int = 500
) -> Tuple[np.ndarray, np.ndarray, pl.Series]:
    """
    Extract the features, labels, and read IDs from a Polars DataFrame.

    Parameters
    ----------
        dataset (pl.DataFrame): The input Polars DataFrame containing the data.
        target_length (int): The desired length of each time series.

    Returns
    -------
        Tuple[np.ndarray, np.ndarray, pl.Series]: A tuple containing the features, labels, and read IDs.
    """
    x = zero_pad_and_reshape(dataset, "timeseries", target_length)
    y = dataset["cap_class"].to_numpy()
    read_id = dataset["read_id"]
    return x, y, read_id


# Top level task because Prefect does not allow flow level
# results to be cached to memory. We create a flow as a task
# to save the result to memory.
# https://github.com/PrefectHQ/prefect/issues/7288
@task(name="pipeline-task", cache_key_fn=custom_hash)
def pipeline_task(
    folder_path: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    pl.Series,
    np.ndarray,
    np.ndarray,
    pl.Series,
    np.ndarray,
    np.ndarray,
    pl.Series,
]:
    """Pipeline task to load and concatenate CSV files.

    Parameters
    ----------
    folder_path: str
        Path to the folder containing CSV files.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, pl.Series, np.ndarray, np.ndarray, pl.Series, np.ndarray, np.ndarray, pl.Series]
        x_train, y_train, read_id_train, x_val, y_val, read_id_val, x_test, y_test, read_id_test

    """
    csv_files = list_csv_files(folder_path)
    loaded_dataframes = [load_csv.submit(file_path) for file_path in csv_files]
    concatenated_df = concatenate_dataframes(loaded_dataframes)
    train, val, test = make_train_val_test_split(
        concatenated_df, train_frac=0.6, val_frac=0.2, test_frac=0.2, random_seed=42
    )
    target_length = 1000
    x_train, y_train, read_id_train = make_x_y_read_id_sets(train, target_length)
    x_val, y_val, read_id_val = make_x_y_read_id_sets(val, target_length)
    x_test, y_test, read_id_test = make_x_y_read_id_sets(test, target_length)
    return (
        x_train,
        y_train,
        read_id_train,
        x_val,
        y_val,
        read_id_val,
        x_test,
        y_test,
        read_id_test,
    )


# Calling the pipeline in flow such that we can choose the
# number of workers to use in the Dask cluster.
def training_data_pipeline(folder_path: str, n_workers: int) -> Tuple[
    np.ndarray,
    np.ndarray,
    pl.Series,
    np.ndarray,
    np.ndarray,
    pl.Series,
    np.ndarray,
    np.ndarray,
    pl.Series,
]:
    """Create a Prefect flow for loading and concatenating CSV files.

    Parameters
    ----------
    folder_path: str
        Path to the folder containing CSV files.

    n_workers: int
        Number of workers to use in the Dask cluster.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, pl.Series, np.ndarray, np.ndarray, pl.Series, np.ndarray, np.ndarray, pl.Series]
        x_train, y_train, read_id_train, x_val, y_val, read_id_val, x_test, y_test, read_id_test
    """

    @flow(
        name="training-data-pipeline",
        task_runner=DaskTaskRunner(
            cluster_class=LocalCluster,
            cluster_kwargs={"n_workers": n_workers, "threads_per_worker": 2},
        ),
    )
    def create_df(
        folder_path: str,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        pl.Series,
        np.ndarray,
        np.ndarray,
        pl.Series,
        np.ndarray,
        np.ndarray,
        pl.Series,
    ]:
        return pipeline_task(folder_path)

    return create_df(folder_path)


if __name__ == "__main__":
    folder_path = (
        "/export/valenfs/data/processed_data/MinION/9_madcap/dummy_data/real_data/"
    )
    n_workers = 10
    (
        x_train,
        y_train,
        read_id_train,
        x_val,
        y_val,
        read_id_val,
        x_test,
        y_test,
        read_id_test,
    ) = training_data_pipeline(folder_path, n_workers)
    print(x_train.shape, x_test.shape)
