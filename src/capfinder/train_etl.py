import csv
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Tuple

import comet_ml
from loguru import logger
from tqdm import tqdm

# Assuming these are imported from your existing module
from capfinder.inference_data_loader import DtypeLiteral, get_dtype
from capfinder.ml_libs import tf
from capfinder.utils import map_cap_int_to_name

csv.field_size_limit(4096 * 4096)  # Set a higher field size limit (e.g., 1MB)


def read_dataset_version_info(dataset_dir: str) -> Optional[str]:
    version_file = os.path.join(dataset_dir, "artifact_version.txt")
    if os.path.exists(version_file):
        with open(version_file) as f:
            return f.read().strip()
    return None


def write_dataset_version_info(dataset_dir: str, version: str) -> None:
    version_file = os.path.join(dataset_dir, "artifact_version.txt")
    with open(version_file, "w") as f:
        f.write(version)


class CometArtifactManager:
    """
    A class to manage CometML artifacts and experiments.

    Attributes:
        project_name (str): The name of the CometML project.
        experiment (Optional[comet_ml.Experiment]): The current CometML experiment.
        artifact (Optional[comet_ml.Artifact]): The current CometML artifact.
    """

    def __init__(self, project_name: str, dataset_dir: str):
        """
        Initialize the CometArtifactManager with the given project name.

        Args:
            project_name (str): The name of the CometML project.
        """
        self.project_name = project_name
        self.dataset_dir = dataset_dir
        self.artifact_name = "cap_data"
        self.experiment = self.initialize_comet_ml_experiment()
        self.artifact = self.create_artifact()
        self.info: Dict[str, Optional[str]] = {
            "version": None,
            "source_experiment_key": None,
        }
        self.create_artifact()

    def create_artifact(self) -> comet_ml.Artifact:
        """
        Create a CometML artifact and initialize the experiment.
        """
        # self.initialize_comet_ml_experiment()
        return comet_ml.Artifact(
            name=self.artifact_name,
            artifact_type="dataset",
            aliases=["processed"],
            metadata={"task": "RNA caps classification"},
        )

    def initialize_comet_ml_experiment(self) -> comet_ml.Experiment:
        """
        Initialize a CometML experiment.

        Returns:
            comet_ml.Experiment: The initialized CometML experiment or None if initialization fails.

        Raises:
            SystemExit: If the COMET_API_KEY is not set or if there's an error in initialization.
        """
        logger.info(f"Initializing CometML experiment for project: {self.project_name}")

        comet_api_key = os.getenv("COMET_API_KEY")
        if not comet_api_key:
            logger.error("COMET_API_KEY environment variable is not set.")
            logger.info(
                "Please set the COMET_API_KEY environment variable to your CometML API key."
            )
            sys.exit(1)

        try:
            experiment = comet_ml.Experiment(
                api_key=comet_api_key, project_name=self.project_name
            )
            logger.success("CometML experiment initialized successfully.")
            return experiment
        except Exception as e:
            logger.error(f"Failed to initialize CometML experiment: {e}")
            logger.info("Please check your CometML configuration and API key.")
            sys.exit(1)

    def latest_artifact_version(self) -> str:
        """Get the latest version of the CometML artifact."""
        try:
            # Get the latest version of the artifact
            art = self.experiment.get_artifact(
                artifact_name=self.artifact_name,
                version_or_alias="latest",
            )
            current_art_version = (
                f"{art.version.major}.{art.version.minor}.{art.version.patch}"
            )
            return current_art_version
        except Exception:
            logger.info(
                f"No existing dataset with name {self.artifact_name} found in CometML."
            )
            sys.exit(1)

    def make_comet_artifacts(
        self,
        split_name: str,
    ) -> None:
        """
        Upload the features, labels, and read IDs to a CometML artifact.

        Parameters
        ----------
        dataset_dir (str): The directory where the processed dataset reside.
        split_name (str): The name of the split (e.g., "train", "test").

        Returns
        -------
        dict: Information about the uploaded artifact.
        """
        x_path = os.path.join(self.dataset_dir, f"{split_name}_x.csv")
        y_path = os.path.join(self.dataset_dir, f"{split_name}_y.csv")
        id_path = os.path.join(self.dataset_dir, f"{split_name}_read_id.csv")
        # Add files to the artifact with logical paths including the split name
        if self.artifact:
            self.artifact.add(
                local_path_or_data=x_path,
                logical_path=f"{split_name}_x.csv",
                metadata={"dataset_stage": "processed", "dataset_split": split_name},
            )
            self.artifact.add(
                local_path_or_data=y_path,
                logical_path=f"{split_name}_y.csv",
                metadata={"dataset_stage": "processed", "dataset_split": split_name},
            )
            self.artifact.add(
                local_path_or_data=id_path,
                logical_path=f"{split_name}_read_id.csv",
                metadata={"dataset_stage": "processed", "dataset_split": split_name},
            )

    def log_artifacts_to_comet(self) -> Dict[str, Any]:
        """
        Log the local dataset artifacts to CometML artifacts.
        """

        if self.experiment is not None:
            self.experiment.add_tag("upload")
            if self.artifact is not None:
                art = self.experiment.log_artifact(self.artifact)
                self.info["version"] = (
                    f"{art.version.major}.{art.version.minor}.{art.version.patch}"
                )
                self.info["source_experiment_key"] = art.source_experiment_key
        self.store_artifact_version_to_file()
        return self.info

    def download_remote_dataset(self, remote_dataset_version_to_download: str) -> None:
        """Download a remote dataset from CometML."""
        if remote_dataset_version_to_download == "latest":
            remote_dataset_version_to_download = self.latest_artifact_version()

        stored_version = read_dataset_version_info(self.dataset_dir)
        if stored_version == remote_dataset_version_to_download:
            logger.info(
                f"Dataset version {remote_dataset_version_to_download} is already downloaded. Skipping download."
            )
        else:
            logger.info(
                f"Downloading remote dataset v{remote_dataset_version_to_download}..."
            )
            art = self.experiment.get_artifact(
                artifact_name=self.artifact_name,
                version_or_alias=remote_dataset_version_to_download,
            )
            art.download(path=self.dataset_dir, overwrite_strategy=True)
            write_dataset_version_info(
                self.dataset_dir, remote_dataset_version_to_download
            )
            self.info["version"] = remote_dataset_version_to_download
            self.info["source_experiment_key"] = art.source_experiment_key
            logger.info("Remote dataset downloaded successfully.")

    def store_artifact_version_to_file(self) -> None:
        """Store the dataset version to a file in the dataset directory."""
        version_file = os.path.join(self.dataset_dir, "artifact_version.txt")
        with open(version_file, "w") as f:
            f.write(self.info.get("version") or "unknown")

    def end_experiment(self) -> None:
        """
        End the CometML experiment and log the artifact.
        """
        if self.experiment is not None:
            self.experiment.end()
        return None


def load_dataset_from_csvs(
    x_file_path: str,
    y_file_path: str,
    read_id_file_path: str,
    batch_size: int,
    target_length: int,
    dtype: DtypeLiteral,
    num_classes: int,
    examples_per_class: Optional[int] = None,
    frac: float = 1.0,
) -> tf.data.Dataset:
    """
    Parse and load and preprocess a dataset from raw CSV files of caps data.

    Args:
    x_file_path (str): Path to features CSV file.
    y_file_path (str): Path to labels CSV file.
    read_id_file_path (str): Path to read IDs CSV file.
    batch_size (int): Batch size for the dataset.
    target_length (int): Target length of the feature sequence.
    dtype (DtypeLiteral): Data type for the features.
    num_classes (int): Number of classes in the dataset.
    examples_per_class (int, optional): Maximum number of examples per class. If None, use all examples.
    frac (float): Fraction to multiply with examples_per_class (default: 1.0).

    Returns:
    tf.data.Dataset: The processed dataset.
    """
    tf_dtype = get_dtype(dtype)

    # Load features
    x_dataset: tf.data.Dataset = tf.data.experimental.make_csv_dataset(
        x_file_path,
        batch_size=batch_size,
        column_names=[f"feature_{i}" for i in range(target_length)],
        column_defaults=[tf.float32] * target_length,
        header=True,
        num_epochs=1,
    )

    # Load labels
    y_dataset: tf.data.Dataset = tf.data.experimental.make_csv_dataset(
        y_file_path,
        batch_size=batch_size,
        column_names=["cap_class"],
        column_defaults=[tf.int32],
        header=True,
        num_epochs=1,
    )

    # Load read_ids
    read_id_dataset: tf.data.Dataset = tf.data.experimental.make_csv_dataset(
        read_id_file_path,
        batch_size=batch_size,
        column_names=["read_id"],
        column_defaults=[tf.string],
        header=True,
        num_epochs=1,
    )

    # Combine features and labels
    dataset: tf.data.Dataset = tf.data.Dataset.zip(
        (x_dataset, y_dataset, read_id_dataset)
    )

    # Reshape features and extract labels
    def reshape_and_cast(
        x: tf.Tensor, y: tf.Tensor, read_id: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        x_reshaped = tf.stack(list(x.values()), axis=1)
        x_reshaped = tf.reshape(x_reshaped, (batch_size, target_length, 1))
        x_reshaped = tf.cast(x_reshaped, tf_dtype)  # Cast to desired dtype
        y_reshaped = tf.cast(list(y.values())[0], tf.int32)
        read_id_reshaped = tf.reshape(list(read_id.values())[0], (batch_size,))
        return x_reshaped, y_reshaped, read_id_reshaped

    dataset = dataset.map(reshape_and_cast)

    if examples_per_class is not None:
        # Apply fraction to examples_per_class
        adjusted_examples_per_class = int(examples_per_class * frac)

        # Unbatch the dataset
        dataset = dataset.unbatch()

        # Group by class and limit examples
        class_datasets = [
            dataset.filter(lambda x, y, z, class_id=i: tf.equal(y, class_id)).take(
                adjusted_examples_per_class
            )
            for i in range(num_classes)
        ]

        # Combine and shuffle
        dataset = tf.data.experimental.sample_from_datasets(class_datasets)
        dataset = dataset.shuffle(buffer_size=adjusted_examples_per_class * num_classes)

        # Rebatch
        dataset = dataset.batch(batch_size)

    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def load_train_test_datasets_from_csvs(
    dataset_dir: str,
    batch_size: int,
    target_length: int,
    dtype: tf.DType,
    num_classes: int,
    examples_per_class: int,
    training_fraction: float,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load ready-made train and test datasets from CSV files."""

    train_dataset = load_dataset_from_csvs(
        x_file_path=os.path.join(dataset_dir, "train_x.csv"),
        y_file_path=os.path.join(dataset_dir, "train_y.csv"),
        read_id_file_path=os.path.join(dataset_dir, "train_read_id.csv"),
        batch_size=batch_size,
        target_length=target_length,
        dtype=dtype,
        num_classes=num_classes,
        examples_per_class=examples_per_class,
        frac=training_fraction,
    )
    logger.info("Loaded train split dataset!")
    test_fraction = 1 - training_fraction
    test_dataset = load_dataset_from_csvs(
        x_file_path=os.path.join(dataset_dir, "test_x.csv"),
        y_file_path=os.path.join(dataset_dir, "test_y.csv"),
        read_id_file_path=os.path.join(dataset_dir, "test_read_id.csv"),
        batch_size=batch_size,
        target_length=target_length,
        dtype=dtype,
        num_classes=num_classes,
        examples_per_class=examples_per_class,
        frac=test_fraction,
    )
    logger.info("Loaded test split dataset!")
    return train_dataset, test_dataset


def get_class_from_file(file_path: str) -> int:
    """Read the first data row from a CSV file and return the class ID."""
    with open(file_path) as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip header
        first_row = next(csv_reader)
        return int(first_row[1])  # Assuming cap_class is the second column


def group_files_by_class(caps_data_dir: str) -> Dict[int, List[str]]:
    """Group CSV files in the directory by their class ID."""
    class_files = defaultdict(list)
    for file in os.listdir(caps_data_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(caps_data_dir, file)
            try:
                class_id = get_class_from_file(file_path)
                class_files[class_id].append(file_path)
            except Exception as e:
                logger.warning(
                    f"Couldn't determine class for file {file}. Error: {str(e)}"
                )
    return class_files


def parse_row(
    row: Tuple[str, str, str], target_length: int, dtype: tf.DType
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Parse a row of data and convert it to the appropriate tensor format.

    Args:
        row (Tuple[str, str, str]): A tuple containing read_id, cap_class, and timeseries as strings.
        target_length (int): The desired length of the timeseries tensor.
        dtype (tf.DType): The desired data type for the timeseries tensor.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: A tuple containing the parsed and formatted tensors for
        timeseries, cap_class, and read_id.
    """
    read_id, cap_class, timeseries = row
    cap_class = tf.strings.to_number(cap_class, out_type=tf.int32)

    # Split the timeseries string and convert to float
    timeseries = tf.strings.split(timeseries, sep=",")
    timeseries = tf.strings.to_number(timeseries, out_type=tf.float32)

    # Pad or truncate the timeseries to the target length
    padded = tf.cond(
        tf.shape(timeseries)[0] >= target_length,
        lambda: timeseries[:target_length],
        lambda: tf.pad(
            timeseries,
            [[0, target_length - tf.shape(timeseries)[0]]],
            constant_values=0.0,
        ),
    )

    padded = tf.reshape(padded, (target_length, 1))

    # Cast to the desired dtype
    if dtype != tf.float32:
        padded = tf.cast(padded, dtype)

    return padded, cap_class, read_id


def csv_generator(file_path: str) -> Generator[Tuple[str, str, str], None, None]:
    """
    Generates rows from a CSV file one at a time.

    Args:
        file_path (str): Path to the CSV file.

    Yields:
        Tuple[str, str, str]: A tuple containing read_id, cap_class, and timeseries as strings.
    """
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        # Skip the header row
        next(reader)
        for row in reader:
            yield (str(row[0]), str(row[1]), str(row[2]))


def create_dataset(
    file_path: str,
    target_length: int,
    dtype: DtypeLiteral,
) -> tf.data.Dataset:
    """
    Create train and test TensorFlow datasets for a single class CSV file.

    Args:
        file_path (str): Path to the CSV file.
        target_length (int): The desired length of the timeseries tensor.
        dtype (DtypeLiteral): The desired data type for the timeseries tensor as a string.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Train and test datasets for the given class.
    """
    tf_dtype = get_dtype(dtype)
    dataset = tf.data.Dataset.from_generator(
        lambda: csv_generator(file_path),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )
    dataset = dataset.map(
        lambda x, y, z: parse_row((x, y, z), target_length, tf_dtype),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset


def create_class_dataset(
    file_paths: List[str],
    target_length: int,
    dtype: DtypeLiteral,
    examples_per_class: int,
    train_fraction: float,
) -> tf.data.Dataset:
    """Create a dataset for a single class from multiple files."""
    class_dataset = None

    for file_path in file_paths:
        dataset = create_dataset(file_path, target_length, dtype)

        if class_dataset is None:
            class_dataset = dataset
        else:
            class_dataset = class_dataset.concatenate(dataset)
    dataset = dataset.shuffle(buffer_size=10000).take(examples_per_class)

    # Split into train and test
    train_size = int(train_fraction * examples_per_class)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    return train_dataset, test_dataset


def combine_datasets(datasets: List[tf.data.Dataset]) -> tf.data.Dataset:
    """
    Combine datasets from different classes.

    Args:
        datasets (List[tf.data.Dataset]): List of datasets to combine.

    Returns:
        tf.data.Dataset: A combined dataset.
    """
    combined_dataset = datasets[0]
    for dataset in datasets[1:]:
        combined_dataset = combined_dataset.concatenate(dataset)
    return combined_dataset.shuffle(buffer_size=10000)


def interleave_class_datasets(
    class_datasets: List[tf.data.Dataset],
    num_classes: int,
) -> tf.data.Dataset:
    """
    Interleave datasets from different classes to ensure class balance.

    Args:
        class_datasets (List[tf.data.Dataset]): List of datasets, one for each class.
        num_classes (int): The number of classes in the dataset.

    Returns:
        tf.data.Dataset: An interleaved dataset with balanced class representation.
    """
    # Ensure we have the correct number of datasets
    assert (
        len(class_datasets) == num_classes
    ), "Number of datasets should match number of classes"

    def interleave_map_fn(dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(lambda x, y, z: (x, y, z))

    # Use the interleave operation to balance the classes
    interleaved_dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(
        class_datasets
    ).interleave(
        interleave_map_fn,
        cycle_length=num_classes,
        block_length=1,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return interleaved_dataset


def write_dataset_to_csv(
    dataset: tf.data.Dataset, dataset_dir: str, train_test: str
) -> None:
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    x_filename = os.path.join(dataset_dir, f"{train_test}_x.csv")
    y_filename = os.path.join(dataset_dir, f"{train_test}_y.csv")
    read_id_filename = os.path.join(dataset_dir, f"{train_test}_read_id.csv")

    with (
        open(x_filename, "w", newline="") as x_file,
        open(y_filename, "w", newline="") as y_file,
        open(read_id_filename, "w", newline="") as read_id_file,
    ):
        x_writer = csv.writer(x_file)
        y_writer = csv.writer(y_file)
        read_id_writer = csv.writer(read_id_file)

        # Write headers
        x_writer.writerow(
            [f"feature_{i}" for i in range(dataset.element_spec[0].shape[1])]
        )
        y_writer.writerow(["cap_class"])
        read_id_writer.writerow(["read_id"])

        pbar = tqdm(dataset, desc="Processing batches")

        for batch_num, (x, y, read_id) in enumerate(pbar):
            # Convert tensors to numpy arrays
            x_numpy = x.numpy()
            y_numpy = y.numpy()
            read_id_numpy = read_id.numpy()

            # Write x data (features)
            x_writer.writerows(x_numpy.reshape(x_numpy.shape[0], -1))

            # Write y data (labels)
            y_writer.writerows(y_numpy.reshape(-1, 1))

            # Write read_id data
            read_id_writer.writerows(
                [
                    [rid.decode("utf-8") if isinstance(rid, bytes) else rid]
                    for rid in read_id_numpy
                ]
            )
            pbar.set_description(f"Processed {batch_num + 1} batches")


def get_local_dataset_version(dataset_dir: str) -> Optional[str]:
    stored_version = None
    version_file = os.path.join(dataset_dir, "artifact_version.txt")
    train_x_file = os.path.join(dataset_dir, "train_x.csv")
    train_y_file = os.path.join(dataset_dir, "train_y.csv")
    test_x_file = os.path.join(dataset_dir, "test_x.csv")
    test_y_file = os.path.join(dataset_dir, "test_y.csv")
    train_exists = os.path.exists(train_x_file) and os.path.exists(train_y_file)
    test_exists = os.path.exists(test_x_file) and os.path.exists(test_y_file)
    version_file_exists = os.path.exists(version_file)

    if train_exists and test_exists and version_file_exists:
        with open(version_file) as f:
            stored_version = f.read().strip()
    return stored_version


def train_etl(
    caps_data_dir: str,
    dataset_dir: str,
    target_length: int,
    dtype: DtypeLiteral,
    examples_per_class: int,
    train_fraction: float,
    num_classes: int,
    batch_size: int,
    comet_project_name: str,
    use_remote_dataset_version: str = "",
) -> Tuple[tf.data.Dataset, tf.data.Dataset, str]:
    """
    Process the data from multiple class files, create balanced datasets,
    perform train-test split, and upload to Comet ML.

    Args:
        caps_data_dir (str): Directory containing the class CSV files.
        target_length (int): The desired length of each time series.
        dtype (DtypeLiteral): The desired data type for the timeseries tensor as a string.
        examples_per_class (int): Number of samples to use per class.
        train_fraction (float): Fraction of data to use for training.
        batch_size (int): The number of samples per batch.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, str]: The train and test datasets and the dataset version
    """

    comet_obj = CometArtifactManager(
        project_name=comet_project_name, dataset_dir=dataset_dir
    )
    current_local_version = get_local_dataset_version(dataset_dir)
    # latest_remote_version = comet_obj.latest_artifact_version()
    reprocess_dataset = False

    if use_remote_dataset_version != "":
        logger.info(f"Using remote dataset version: {use_remote_dataset_version}")
        comet_obj.download_remote_dataset(use_remote_dataset_version)
        train_dataset, test_dataset = load_train_test_datasets_from_csvs(
            dataset_dir,
            batch_size,
            target_length,
            dtype,
            num_classes,
            examples_per_class,
            train_fraction,
        )
        return train_dataset, test_dataset, use_remote_dataset_version

    if current_local_version and use_remote_dataset_version == "":
        logger.info(
            f"A dataset (v{current_local_version}) was found locally in:\n{dataset_dir}"
        )
        reprocess = input(
            "Do you want to rewrite over this local dataset, reprocess the data, and create a new dataset version? (y/n): "
        )
        if reprocess.lower() == "y" or reprocess.lower() == "yes":
            logger.info("You chose to reprocess the data.")
            logger.info("Reprocessing data...")
            reprocess_dataset = True
        else:
            logger.info("You chose not to reprocess the data.")
            logger.info("Loading local dataset...")
            train_dataset, test_dataset = load_train_test_datasets_from_csvs(
                dataset_dir,
                batch_size,
                target_length,
                dtype,
                num_classes,
                examples_per_class,
                train_fraction,
            )
            logger.info(f"Local dataset v{current_local_version} loaded successfully.")
            return train_dataset, test_dataset, current_local_version

    if reprocess_dataset or (
        not current_local_version and use_remote_dataset_version == ""
    ):
        class_files = group_files_by_class(caps_data_dir)
        train_datasets = []
        test_datasets = []
        for class_id, file_paths in class_files.items():
            train_ds, test_ds = create_class_dataset(
                file_paths,
                target_length,
                dtype,
                examples_per_class,
                train_fraction,
            )
            train_datasets.append(train_ds)
            test_datasets.append(test_ds)
            logger.info(f"Processed class {map_cap_int_to_name(class_id)}!")

        logger.info("Combining class datasets...")
        train_dataset = combine_datasets(train_datasets)
        test_dataset = combine_datasets(test_datasets)

        logger.info("Interleaving classes for ensuring class balance in each batch...")
        train_dataset = interleave_class_datasets(
            train_datasets, num_classes=num_classes
        )
        test_dataset = interleave_class_datasets(test_datasets, num_classes=num_classes)

        # Calculate total dataset size
        logger.info("Calculating dataset size")
        total_samples = examples_per_class * num_classes
        train_samples = int(train_fraction * total_samples)
        test_samples = total_samples - train_samples

        # Batch the datasets
        logger.info("Batching dataset...")
        train_dataset = train_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)

        # Prefetch for performance
        logger.info("Prefetching dataset...")
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        logger.info("Saving train/test splits to CSV files...")
        write_dataset_to_csv(train_dataset, dataset_dir, "train")
        write_dataset_to_csv(test_dataset, dataset_dir, "test")
        logger.info(
            f"Train/test splits to CSV files in the following directory:\n{dataset_dir}"
        )

        # Log dataset information to Comet ML
        comet_obj.experiment.log_parameter("target_length", target_length)
        comet_obj.experiment.log_parameter("dtype", dtype)
        comet_obj.experiment.log_parameter("examples_per_class", examples_per_class)
        comet_obj.experiment.log_parameter("train_fraction", train_fraction)
        comet_obj.experiment.log_parameter("batch_size", batch_size)
        comet_obj.experiment.log_parameter("num_classes", len(class_files))
        comet_obj.experiment.log_parameter("total_samples", total_samples)
        comet_obj.experiment.log_parameter("train_samples", train_samples)
        comet_obj.experiment.log_parameter("test_samples", test_samples)

        logger.info("Making Comet ML dataset artifacts for uploading...")
        comet_obj.make_comet_artifacts("train")
        comet_obj.make_comet_artifacts("test")

        logger.info("Uploading artifacts to Comet ML...")
        artifact_info = comet_obj.log_artifacts_to_comet()

        comet_obj.end_experiment()
        logger.info(
            f"Data processed and resulting dataset {artifact_info['version']} uploaded to Comet ML successfully."
        )
        return train_dataset, test_dataset, artifact_info["version"]

    raise RuntimeError("No valid dataset could be processed. Please check your inputs.")


if __name__ == "__main__":
    caps_data_dir = "/export/valenfs/data/processed_data/MinION/9_madcap/3_all_train_csv_202405/all_csvs"
    dataset_dir = (
        "/export/valenfs/data/processed_data/MinION/9_madcap/6_dataset_csv_202405/"
    )
    target_length = 500
    dtype: DtypeLiteral = "float16"
    examples_per_class = 10  # Set to None if you want to use all available examples
    train_fraction = 0.8
    batch_size = 12
    num_classes = 4
    comet_project_name = "dataset2"
    use_remote_dataset_version = ""
    train_etl(
        caps_data_dir,
        dataset_dir,
        target_length,
        dtype,
        examples_per_class,
        train_fraction,
        num_classes,
        batch_size,
        comet_project_name,
        use_remote_dataset_version,
    )
