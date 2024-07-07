import hashlib
import os
from typing import Any, List, Optional, Tuple, Type, Union

import comet_ml
import numpy as np
import polars as pl
from imblearn.under_sampling import RandomUnderSampler
from loguru import logger
from polars import DataFrame
from prefect import flow, task
from prefect.engine import TaskRunContext
from prefect.tasks import task_input_hash
from typing_extensions import Literal

from capfinder.utils import get_dtype

# Define custom types
TrainData = Tuple[
    np.ndarray,  # x_train
    np.ndarray,  # y_train
    pl.Series,  # read_id_train
    np.ndarray,  # x_test
    np.ndarray,  # y_test
    pl.Series,  # read_id_test
    dict,  # dataset_info
]
DtypeLiteral = Literal["float16", "float32", "float64"]
DtypeNumpy = Union[np.float16, np.float32, np.float64]


def custom_hash(context: TaskRunContext, parameters: dict[str, Any]) -> str:
    data_dir = parameters.get("data_dir")
    dtype = parameters.get("dtype")

    files = os.listdir(data_dir)
    # Sort files to ensure consistent hash
    files.sort()
    # Generate hash based on file names, sizes, and modification times
    file_metadata = [
        (
            file,
            dtype,
            os.path.getsize(os.path.join(data_dir, file)),  # type: ignore
            os.path.getmtime(os.path.join(data_dir, file)),  # type: ignore
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
def load_csv(file_path: str, max_records: int = 1_300_000) -> pl.DataFrame:
    """Load a CSV file into a single DataFrame with a maximum number of records.

    Parameters
    ----------
    file_path: str
        Path to the CSV file.
    max_records: int, optional
        Maximum number of records to read, defaults to 20.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the CSV data.
    """
    result_df = pl.DataFrame()
    records_read = 0
    chunk_size = 50000

    # Read the first chunk with header
    first_chunk = pl.read_csv(file_path, n_rows=chunk_size, has_header=True)
    result_df = pl.concat([result_df, first_chunk])
    records_read += len(first_chunk)

    # Get the column names from the first chunk
    column_names = first_chunk.columns

    # Read subsequent chunks with skip_rows
    while records_read < max_records:
        try:
            chunk = pl.read_csv(
                file_path,
                n_rows=chunk_size,
                skip_rows=records_read,
                has_header=False,
                new_columns=column_names,
            )
            if chunk.is_empty():
                break
        except Exception:
            break
        result_df = pl.concat([result_df, chunk])
        records_read += len(chunk)
        logger.info(f"Records read: {records_read}")

    # Limit the result to max_records
    return result_df.head(max_records)


@task(name="concatenate-dataframes", cache_key_fn=task_input_hash)
def concatenate_dataframes(dataframes: List[DataFrame]) -> pl.DataFrame:
    """Concatenate a list of DataFrames vertically.

    Parameters
    ----------
    dataframes: List[DataFrame]
        List of DataFrames to concatenate.

    Returns
    -------
    pl.DataFrame
        Concatenated DataFrame. Returns an empty DataFrame if the input list is empty.
    """
    # List[PrefectFuture[DataFrame, Literal[False]]
    if not dataframes:
        return pl.DataFrame()
    logger.info("Concatenating DataFrames")
    return pl.concat(dataframes, how="vertical")


@task(name="make-train-test-split", cache_key_fn=task_input_hash)
def make_train_test_split(
    df: pl.DataFrame,
    train_frac: float,
    random_seed: int,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split a Polars DataFrame into balanced train and test sets.

    Parameters
    ----------
        df (pl.DataFrame): The input Polars DataFrame.
        train_frac (float): The fraction of the data to use for the training set.
        random_seed (int): The random seed for repeatable splits.

    Returns
    -------
        Tuple[pl.DataFrame, pl.DataFrame]: A tuple containing the balanced train and test sets.
    """

    # Verify that the train fraction is valid
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")

    def balance_classes(df: pl.DataFrame, random_seed: int) -> pl.DataFrame:
        y = df.drop_in_place("cap_class")  # Target
        x = df  # Features

        # Convert Polars DataFrame to numpy arrays
        x_array = x.to_numpy()
        y_array = y.to_numpy()

        # Instantiate the RandomUnderSampler
        under_sampler = RandomUnderSampler(random_state=random_seed)

        # Resample the dataset
        x_resampled, y_resampled = under_sampler.fit_resample(x_array, y_array)

        # Get the column names from the DataFrame's schema
        column_names = df.columns

        # Combine features and target into a Polars DataFrame
        balanced_df = pl.DataFrame(
            {
                "cap_class": pl.Series(y_resampled),
                **{col: x_resampled[:, i] for i, col in enumerate(column_names)},
            }
        )
        return balanced_df

    balanced_df = balance_classes(df=df, random_seed=random_seed)

    # Shuffle the DataFrame using numpy
    def shuffle_df(df: pl.DataFrame, random_seed: int) -> pl.DataFrame:
        np.random.seed(random_seed)
        shuffled_indices = np.random.permutation(len(df))
        return df[shuffled_indices]

    balanced_df = shuffle_df(df=balanced_df, random_seed=random_seed)

    # Split the balanced DataFrame
    train_size = int(len(balanced_df) * train_frac)
    train_df = balanced_df[:train_size]
    test_df = balanced_df[train_size:]

    balanced_train_df = balance_classes(train_df, random_seed)
    balanced_test_df = balance_classes(test_df, random_seed)

    # Final shuffle to ensure randomness
    balanced_train_df = shuffle_df(balanced_train_df, random_seed)
    balanced_test_df = shuffle_df(balanced_test_df, random_seed)

    return balanced_train_df, balanced_test_df


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
    def parse_and_pad(series: str) -> pl.Series:

        # Convert to a numpy array of floats
        series_array = np.fromstring(series, sep=",")

        # Calculate the length of the series
        series_length = len(series_array)

        # Handle the case where the length is greater than the target length
        if series_length > target_length:
            # Calculate the excess length (how many elements to remove)
            excess_length = series_length - target_length

            # Calculate removal from each end
            remove_from_each_end = excess_length / 2

            # Handle floating-point division
            if remove_from_each_end != int(
                remove_from_each_end
            ):  # Check if it's a float
                # Remove one extra element (either left or right)
                remove_from_left = int(
                    np.ceil(remove_from_each_end)
                )  # Round up for left side
                remove_from_right = excess_length - remove_from_left
            else:
                # Integer division, remove equally from both sides
                remove_from_left = int(remove_from_each_end)
                remove_from_right = remove_from_left

            # Truncate the series equally at both ends
            series_array = series_array[remove_from_left:-remove_from_right]

        # Handle the case where the length is less than the target length
        elif series_length < target_length:
            # Pad the series with zeros to reach the target length
            series_array = np.pad(
                series_array,
                (0, target_length - series_length),
                "constant",
                constant_values=0,
            )

        return pl.Series(series_array)

    # Apply the function to parse and pad each time series in the column
    zero_padded_data = df[column_name].map_elements(
        parse_and_pad, return_dtype=pl.List(pl.Float64)
    )

    zero_padded_data_np: list[np.ndarray] = zero_padded_data.to_numpy()  # type: ignore
    zero_padded_data_2d = np.stack(zero_padded_data_np)

    # Convert the array to shape (num_examples, num_timesteps, num_features)
    reshaped_data: np.ndarray = zero_padded_data_2d.reshape(
        len(zero_padded_data), target_length, 1
    )
    return reshaped_data


@task(name="make-x-y-read-id-sets", cache_key_fn=task_input_hash)
def make_x_y_read_id_sets(
    dataset: pl.DataFrame, target_length: int, dtype_n: Type[np.floating]
) -> Tuple[np.ndarray, np.ndarray, pl.Series]:
    """
    Extract the features, labels, and read IDs from a Polars DataFrame.

    Parameters
    ----------
        dataset (pl.DataFrame): The input Polars DataFrame containing the data.
        target_length (int): The desired length of each time series.
        dtype_n (Type[np.floating]): The data type to use for the features.

    Returns
    -------
        Tuple[np.ndarray, np.ndarray, pl.Series]: A tuple containing the features, labels, and read IDs.
    """

    x = zero_pad_and_reshape(dataset, "timeseries", target_length)
    x = x.astype(dtype_n)
    y = (
        dataset["cap_class"].to_numpy().astype(np.int8)
    )  # Casting to np.int8 for fast training
    read_id = dataset["read_id"]
    return x, y, read_id


class CometArtifactManager:
    """
    A class to manage CometML artifacts and experiments.

    Attributes:
        project_name (str): The name of the CometML project.
        save_dir (str): The local directory to save the processed data.
        experiment (Optional[comet_ml.Experiment]): The current CometML experiment.
        artifact (Optional[comet_ml.Artifact]): The current CometML artifact.
    """

    def __init__(self, project_name: str, save_dir: str):
        """
        Initialize the CometArtifactManager with the given project name.

        Args:
            project_name (str): The name of the CometML project.
            save_dir (str): The local directory to save the processed data.
        """
        self.project_name = project_name
        self.save_dir = save_dir
        self.experiment: Optional[comet_ml.Experiment] = None
        self.artifact: Optional[comet_ml.Artifact] = None
        self.create_artifact()

    def create_artifact(self) -> None:
        """
        Create a CometML artifact and initialize the experiment.
        """
        self.initialize_comet_ml_experiment()
        if self.experiment is not None:
            self.artifact = comet_ml.Artifact(
                name="cap_data",
                artifact_type="dataset",
                aliases=["processed"],
                metadata={"task": "RNA caps classification"},
            )
        return None

    def end_experiment(self) -> dict:
        """
        End the CometML experiment and log the artifact.
        """
        info = {
            "version": None,
            "source_experiment_key": None,
        }
        if self.experiment is not None:
            self.experiment.add_tag("upload")
            if self.artifact is not None:
                art_info = self.experiment.log_artifact(self.artifact)
                info["version"] = art_info.version
                info["source_experiment_key"] = art_info.source_experiment_key
            self.experiment.end()
        return info

    def initialize_comet_ml_experiment(self) -> None:
        """
        Initialize a CometML experiment.

        Args:
            project_name (str): The name of the CometML project.

        Returns:
            Optional[comet_ml.Experiment]: The initialized CometML experiment, or None if initialization fails.
        """
        try:
            logger.info(
                f"Initializing CometML experiment for project: {self.project_name}"
            )
            comet_api_key = os.getenv("COMET_API_KEY")
            self.experiment = comet_ml.Experiment(
                api_key=comet_api_key, project_name=self.project_name
            )
        except Exception as e:
            logger.warning(f"Failed to initialize CometML experiment: {e}")
            self.experiment = None


@task(name="upload-to-comet-artifacts", cache_key_fn=task_input_hash)
def upload_to_comet_artifacts(
    obj: CometArtifactManager,
    x: np.ndarray,
    y: np.ndarray,
    read_id: pl.Series,
    split_name: str,
) -> None:
    """
    Upload the features, labels, and read IDs to a CometML artifact.

    Parameters
    ----------
    obj (CometArtifactManager): The CometArtifactManager object.
    x (np.ndarray): The features to upload.
    y (np.ndarray): The labels to upload.
    read_id (pl.Series): The read IDs to upload.
    split_name (str): The name of the split (e.g., "train", "test").
    save_dir (str): The directory to save the processed data.

    Returns
    -------
    None
    """
    if not os.path.exists(obj.save_dir):
        os.makedirs(obj.save_dir, exist_ok=True)

    dfp_x = pl.from_numpy(data=x.reshape(x.shape[:2]), orient="row")
    dfp_y = pl.from_numpy(data=y.reshape(y.shape[:2]), orient="row")
    dfp_id = pl.DataFrame(read_id)

    x_path = os.path.join(obj.save_dir, f"{split_name}_x.csv")
    dfp_x.write_csv(file=x_path)
    y_path = os.path.join(obj.save_dir, f"{split_name}_y.csv")
    dfp_y.write_csv(file=y_path)
    id_path = os.path.join(obj.save_dir, f"{split_name}_read_id.csv")
    dfp_id.write_csv(file=id_path)

    # Add files to the artifact with logical paths including the split name
    if obj.artifact:
        obj.artifact.add(
            local_path_or_data=x_path,
            logical_path=f"{split_name}_x.csv",
            metadata={"dataset_stage": "processed", "dataset_split": split_name},
        )
        obj.artifact.add(
            local_path_or_data=y_path,
            logical_path=f"{split_name}_y.csv",
            metadata={"dataset_stage": "processed", "dataset_split": split_name},
        )
        obj.artifact.add(
            local_path_or_data=id_path,
            logical_path=f"{split_name}_read_id.csv",
            metadata={"dataset_stage": "processed", "dataset_split": split_name},
        )


# Top level task because Prefect does not allow flow level
# results to be cached to memory. We create a flow as a task
# to save the result to memory.
# https://github.com/PrefectHQ/prefect/issues/7288
@task(name="pipeline-task", cache_key_fn=custom_hash)
def pipeline_task(
    data_dir: str, save_dir: str, target_length: int, dtype_n: Type[np.floating]
) -> TrainData:
    """Pipeline task to load and concatenate CSV files.

    Parameters
    ----------
    data_dir: str
        Path to the directory containing raw CSV files.

    save_dir: str
        Path to the directory where the processed data will be saved.

    target_length: int
        The desired length of each time series.

    dtype_n: Type[np.floating]
        The data type to use for the features.

    Returns
    -------
    TrainData
        A tuple containing the training and test data sets:
            - x_train: Training features (numpy array)
            - y_train: Training target labels (numpy array)
            - read_id_train: Training data IDs (polars Series)
            - x_test: Test features (numpy array)
            - y_test: Test target labels (numpy array)
            - read_id_test: Test data IDs (polars Series)
            - dataset_info: Information about the uploaded dataset (dict)
    """
    comet_obj = CometArtifactManager(
        project_name="capfinder-datasets", save_dir=save_dir
    )
    csv_files = list_csv_files(data_dir)
    # loaded_dataframes = [load_csv.submit(file_path) for file_path in csv_files]
    loaded_dataframes = [load_csv(file_path) for file_path in csv_files]

    concatenated_df = concatenate_dataframes(loaded_dataframes)
    train, test = make_train_test_split(concatenated_df, train_frac=0.8, random_seed=42)
    x_train, y_train, read_id_train = make_x_y_read_id_sets(
        train, target_length, dtype_n
    )
    x_test, y_test, read_id_test = make_x_y_read_id_sets(test, target_length, dtype_n)

    upload_to_comet_artifacts(
        comet_obj, x=x_train, y=y_train, read_id=read_id_train, split_name="train"
    )

    upload_to_comet_artifacts(
        comet_obj, x=x_test, y=y_test, read_id=read_id_test, split_name="test"
    )
    dataset_info = {}
    if comet_obj.experiment is not None:
        dataset_info["etl_experiment_url"] = comet_obj.experiment.url
    else:
        dataset_info["etl_experiment_url"] = None

    di = comet_obj.end_experiment()
    dataset_info.update(di)

    return (x_train, y_train, read_id_train, x_test, y_test, read_id_test, dataset_info)


# Calling the pipeline in flow such that we can choose the
# number of workers to use in the Dask cluster.
def train_etl(
    data_dir: str,
    save_dir: str,
    target_length: int,
    dtype: DtypeLiteral,
    n_workers: int,
) -> TrainData:
    """Create a Prefect flow for loading and concatenating CSV files.

    Parameters
    ----------
    data_dir: str
        Path to the directory containing raw CSV files.

    save_dir: str
        Path to the directory where the processed data will be saved.

    target_length: int
        The desired length of each time series.

    dtype: Literal["float16", "float32", "float64"]
        The data type to use for the features.

    n_workers: int
        Number of workers to use in the Dask cluster.

    Returns
    -------
    TrainData
        A tuple containing the training and test data sets:
            - x_train: Training features (numpy array)
            - y_train: Training target labels (numpy array)
            - read_id_train: Training data IDs (polars Series)
            - x_test: Test features (numpy array)
            - y_test: Test target labels (numpy array)
            - read_id_test: Test data IDs (polars Series)
            - dataset_info: Information about the uploaded dataset (dict)
    """

    # @flow(
    #     name="training-data-pipeline",
    #     task_runner=DaskTaskRunner(
    #         cluster_class=LocalCluster,
    #         cluster_kwargs={
    #             "n_workers": n_workers,
    #             "threads_per_worker": 1,
    #             "heartbeat_interval": 1000000
    #             },
    #     ),
    # )
    @flow(
        name="training-data-pipeline",
    )
    def create_datasets(
        data_dir: str,
        save_dir: str,
        target_length: int,
        dtype: DtypeLiteral,
    ) -> TrainData:

        return pipeline_task(
            data_dir, save_dir, target_length, dtype_n=get_dtype(dtype)
        )

    # -----------------------------------------
    return create_datasets(data_dir, save_dir, target_length, dtype)


if __name__ == "__main__":
    data_dir = (
        "/export/valenfs/data/processed_data/MinION/9_madcap/dummy_data/real_data3/"
    )
    save_dir = (
        "/export/valenfs/data/processed_data/MinION/9_madcap/dummy_data/saved_data/"
    )

    target_length = 500
    dtype: DtypeLiteral = "float16"
    n_workers = 1

    (
        x_train,
        y_train,
        read_id_train,
        x_test,
        y_test,
        read_id_test,
        dataset_info,
    ) = train_etl(data_dir, save_dir, target_length, dtype, n_workers)
    print(x_train.shape, x_test.shape)
    print(x_train.dtype)
    print(dataset_info)
