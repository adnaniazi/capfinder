import hashlib
import os
from typing import Any, List

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
    return pl.concat([df.result() for df in dataframes], how="vertical")


# Top level task because Prefect does not allow flow level
# results to be cached to memory. We create a flow as a task
# to save the result to memory.
# https://github.com/PrefectHQ/prefect/issues/7288
@task(name="pipeline-task", cache_key_fn=custom_hash)
def pipeline_task(folder_path: str) -> pl.DataFrame:
    """Pipeline task to load and concatenate CSV files.

    Parameters
    ----------
    folder_path: str
        Path to the folder containing CSV files.

    Returns
    -------
    pl.DataFrame
        Concatenated DataFrame.
    """
    csv_files = list_csv_files(folder_path)
    loaded_dataframes = [load_csv.submit(file_path) for file_path in csv_files]
    concatenated_df = concatenate_dataframes(loaded_dataframes)
    return concatenated_df


# Calling the pipeline in flow such that we can choose the
# number of workers to use in the Dask cluster.
def training_data_pipeline(folder_path: str, n_workers: int) -> pl.DataFrame:
    """Create a Prefect flow for loading and concatenating CSV files.

    Parameters
    ----------
    folder_path: str
        Path to the folder containing CSV files.

    n_workers: int
        Number of workers to use in the Dask cluster.

    Returns
    -------
    pl.DataFrame
        Concatenated DataFrame.
    """

    @flow(
        name="training-data-pipeline",
        task_runner=DaskTaskRunner(
            cluster_class=LocalCluster,
            cluster_kwargs={"n_workers": n_workers, "threads_per_worker": 2},
        ),
    )
    def create_df(folder_path: str) -> pl.DataFrame:
        return pipeline_task(folder_path)

    return create_df(folder_path)


if __name__ == "__main__":
    folder_path = (
        "/export/valenfs/data/processed_data/MinION/9_madcap/dummy_data/real_data/"
    )
    n_workers = 2
    combined_df = training_data_pipeline(folder_path, n_workers)
    print(combined_df)
