import csv
import os
import subprocess
from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

from capfinder.inference_data_loader import DtypeLiteral, get_dtype
from capfinder.ml_libs import tf

# Assuming these are imported from your existing module
from capfinder.upload_download import CometArtifactManager, upload_dataset_to_comet
from capfinder.utils import map_cap_int_to_name

csv.field_size_limit(4096 * 4096)  # Set a higher field size limit (e.g., 1MB)


def read_dataset_version_info(dataset_dir: str) -> Optional[str]:
    """
    Read the dataset version information from a file.

    Args:
        dataset_dir (str): Directory containing the dataset version file.

    Returns:
        Optional[str]: The dataset version if found, None otherwise.
    """
    version_file = os.path.join(dataset_dir, "artifact_version.txt")
    if os.path.exists(version_file):
        with open(version_file) as f:
            return f.read().strip()
    return None


def write_dataset_version_info(dataset_dir: str, version: str) -> None:
    """
    Write the dataset version information to a file.

    Args:
        dataset_dir (str): Directory to write the version file.
        version (str): Version information to write.
    """
    version_file = os.path.join(dataset_dir, "artifact_version.txt")
    with open(version_file, "w") as f:
        f.write(version)


def calculate_sizes(
    total_examples: int, train_fraction: float, batch_size: int
) -> Tuple[int, int]:
    """
    Compute the train and validation sizes based on the total number of examples.

    Args:
        total_examples (int): Total number of examples in the dataset.
        train_fraction (float): Fraction of data to use for training.
        batch_size (int): Size of each batch.

    Returns:
        Tuple[int, int]: Train size and validation size, both divisible by batch_size.
    """
    train_size = int(total_examples * train_fraction)
    val_size = total_examples - train_size

    train_size = (train_size // batch_size) * batch_size
    val_size = (val_size // batch_size) * batch_size

    while train_size + val_size > total_examples:
        if train_size > val_size:
            train_size -= batch_size
        else:
            val_size -= batch_size

    return train_size, val_size


def count_examples_fast(file_path: str) -> int:
    """
    Count lines in a file using fast bash utilities, falling back to Python if necessary.

    Args:
        file_path (str): Path to the file to count lines in.

    Returns:
        int: Number of lines in the file (excluding header).
    """
    try:
        # Try using wc -l command (fast)
        result = subprocess.run(["wc", "-l", file_path], capture_output=True, text=True)
        count = int(result.stdout.split()[0]) - 1  # Subtract 1 for header
        return count
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        try:
            # Fallback to using sed and wc (slightly slower, but still fast)
            result = subprocess.run(
                f"sed '1d' {file_path} | wc -l",
                shell=True,
                capture_output=True,
                text=True,
            )
            return int(result.stdout.strip())
        except (subprocess.SubprocessError, ValueError):
            # If bash methods fail, fall back to Python method
            return count_examples_python(file_path)


def count_examples_python(file_path: str) -> int:
    """
    Count lines in a file using Python (slower but portable).

    Args:
        file_path (str): Path to the file to count lines in.

    Returns:
        int: Number of lines in the file (excluding header).
    """
    with open(file_path) as f:
        return sum(1 for _ in f) - 1  # Subtract 1 for header


def find_class_with_least_rows(class_files: Dict[int, List[str]]) -> Tuple[int, int]:
    min_class = -1
    min_rows = float("inf")

    for class_id, files in class_files.items():
        class_rows = sum(count_examples_fast(file) for file in files)
        logger.info(f"Class {class_id} has {class_rows} examples.")
        if class_rows < min_rows:
            min_rows = class_rows
            min_class = class_id

    if min_class == -1:
        raise ValueError("No valid classes found in the input dictionary.")

    return min_class, int(min_rows)


def load_train_dataset_from_csvs(
    x_file_path: str,
    y_file_path: str,
    batch_size: int,
    target_length: int,
    dtype: tf.DType,
    train_val_fraction: float = 0.8,
    use_augmentation: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """
    Load training dataset from CSV files and split into train and validation sets.

    Args:
        x_file_path (str): Path to the features CSV file.
        y_file_path (str): Path to the labels CSV file.
        batch_size (int): Size of each batch.
        target_length (int): Target length of each time series.
        dtype (tf.DType): Data type for the features.
        train_val_fraction (float, optional): Fraction of data to use for training. Defaults to 0.8.
        use_augmentation (bool): Whether to augment original training examples with warped versions

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, int, int]: Train dataset, validation dataset,
        steps per epoch, and validation steps.
    """

    def parse_fn(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = tf.io.decode_csv(x, record_defaults=[[0.0]] * target_length)
        y = tf.io.decode_csv(y, record_defaults=[[0]])
        return tf.reshape(tf.stack(x), (target_length, 1)), y[0]

    dataset = tf.data.Dataset.zip(
        (
            tf.data.TextLineDataset(x_file_path).skip(1),
            tf.data.TextLineDataset(y_file_path).skip(1),
        )
    )

    # Count total examples
    total_examples = count_examples_fast(x_file_path)
    # Calculate train and validation sizes
    train_size, val_size = calculate_sizes(
        total_examples, train_val_fraction, batch_size
    )

    # Split dataset into train and validation
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)

    # Process and augment the training dataset
    train_dataset = train_dataset.map(
        parse_fn, num_parallel_calls=tf.data.AUTOTUNE
    ).map(lambda x, y: (tf.cast(x, dtype), y))

    if use_augmentation:
        train_dataset = train_dataset.map(
            lambda x, y: augment_example(x, y, dtype)
        ).flat_map(
            lambda x: x
        )  # Flatten the dataset of datasets

    train_dataset = train_dataset.batch(batch_size, drop_remainder=True).prefetch(
        tf.data.AUTOTUNE
    )

    # Process the validation dataset (no augmentation)
    val_dataset = (
        val_dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .map(lambda x, y: (tf.cast(x, dtype), y))
        .prefetch(tf.data.AUTOTUNE)
    )

    # Recalculate steps per epoch
    steps_per_epoch = (train_size * (3 if use_augmentation else 1)) // batch_size
    validation_steps = val_size // batch_size

    return (
        train_dataset,
        val_dataset,
        steps_per_epoch,
        validation_steps,
    )


def load_test_dataset_from_csvs(
    x_file_path: str,
    y_file_path: str,
    batch_size: int,
    target_length: int,
    dtype: DtypeLiteral,
) -> tf.data.Dataset:
    """
    Load test dataset from CSV files.

    Args:
        x_file_path (str): Path to the features CSV file.
        y_file_path (str): Path to the labels CSV file.
        batch_size (int): Size of each batch.
        target_length (int): Target length of each time series.
        dtype (DtypeLiteral): Data type for the features as a string.

    Returns:
        tf.data.Dataset: Test dataset.
    """
    tf_dtype = get_dtype(dtype)

    def parse_fn(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = tf.io.decode_csv(x, record_defaults=[[0.0]] * target_length)
        y = tf.io.decode_csv(y, record_defaults=[[0]])
        return tf.reshape(tf.stack(x), (target_length, 1)), y[0]

    dataset = tf.data.Dataset.zip(
        (
            tf.data.TextLineDataset(x_file_path).skip(1),
            tf.data.TextLineDataset(y_file_path).skip(1),
        )
    )

    return (
        dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .map(lambda x, y: (tf.cast(x, tf_dtype), y))
        .prefetch(tf.data.AUTOTUNE)
    )


def create_train_val_test_datasets_from_train_test_csvs(
    dataset_dir: str,
    batch_size: int,
    target_length: int,
    dtype: tf.DType,
    train_val_fraction: float,
    use_augmentation: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int, int]:
    """
    Load ready-made train, validation, and test datasets from CSV files.

    Args:
        dataset_dir (str): Directory containing the CSV files.
        batch_size (int): Size of each batch.
        target_length (int): Target length of each time series.
        dtype (tf.DType): Data type for the features.
        train_val_fraction (float): Fraction of training data to use for validation.
        use_augmentation (bool): Whether to augment original training examples with warped versions

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int, int]:
        Train dataset, validation dataset, test dataset, steps per epoch, and validation steps.
    """
    logger.info("Loading train, val splits...")

    train_dataset, val_dataset, steps_per_epoch, validation_steps = (
        load_train_dataset_from_csvs(
            x_file_path=os.path.join(dataset_dir, "train_x.csv"),
            y_file_path=os.path.join(dataset_dir, "train_y.csv"),
            batch_size=batch_size,
            target_length=target_length,
            dtype=dtype,
            train_val_fraction=train_val_fraction,
            use_augmentation=use_augmentation,
        )
    )
    logger.info("Loading test split ...")

    test_dataset = load_test_dataset_from_csvs(
        x_file_path=os.path.join(dataset_dir, "test_x.csv"),
        y_file_path=os.path.join(dataset_dir, "test_y.csv"),
        batch_size=batch_size,
        target_length=target_length,
        dtype=dtype,
    )
    return train_dataset, val_dataset, test_dataset, steps_per_epoch, validation_steps


def get_class_from_file(file_path: str) -> int:
    """
    Read the first data row from a CSV file and return the class ID.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        int: Class ID from the first data row.
    """
    with open(file_path) as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip header
        first_row = next(csv_reader)
        return int(first_row[1])  # Assuming cap_class is the second column


def group_files_by_class(caps_data_dir: str) -> Dict[int, List[str]]:
    """
    Group CSV files in the directory by their class ID.

    Args:
        caps_data_dir (str): Directory containing the CSV files.

    Returns:
        Dict[int, List[str]]: Dictionary mapping class IDs to lists of file paths.
    """
    class_files: Dict[int, List[str]] = defaultdict(list)
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
    Padding and truncation are performed equally on both sides of the time series.

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

    # Get the current length of the timeseries
    current_length = tf.shape(timeseries)[0]

    # Function to pad the timeseries
    def pad_timeseries() -> tf.Tensor:
        pad_amount = target_length - current_length
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left
        return tf.pad(
            timeseries,
            [[pad_left, pad_right]],
            constant_values=0.0,
        )

    # Function to truncate the timeseries
    def truncate_timeseries() -> tf.Tensor:
        truncate_amount = current_length - target_length
        truncate_left = truncate_amount // 2
        truncate_right = current_length - (truncate_amount - truncate_left)
        return timeseries[truncate_left:truncate_right]

    # Pad or truncate the timeseries to the target length
    padded = tf.cond(
        current_length >= target_length, truncate_timeseries, pad_timeseries
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
    Create a TensorFlow dataset for a single class CSV file.

    Args:
        file_path (str): Path to the CSV file.
        target_length (int): The desired length of the timeseries tensor.
        dtype (DtypeLiteral): The desired data type for the timeseries tensor as a string.

    Returns:
        tf.data.Dataset: A dataset for the given class.
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
    train_test_fraction: float,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create a dataset for a single class from multiple files.

    Args:
        file_paths (List[str]): List of file paths for a single class.
        target_length (int): The desired length of the timeseries tensor.
        dtype (DtypeLiteral): The desired data type for the timeseries tensor as a string.
        examples_per_class (int): Number of examples to take per class.
        train_test_fraction (float): Fraction of data to use for training.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Train and test datasets for the given class.
    """
    class_dataset: Optional[tf.data.Dataset] = None

    for file_path in file_paths:
        dataset = create_dataset(file_path, target_length, dtype)

        if class_dataset is None:
            class_dataset = dataset
        else:
            class_dataset = class_dataset.concatenate(dataset)

    if class_dataset is None:
        raise ValueError("No valid datasets were created.")

    # Shuffle and take examples after concatenating all files
    class_dataset = class_dataset.shuffle(buffer_size=10000).take(examples_per_class)

    # Split into train and test
    train_size = int(train_test_fraction * examples_per_class)
    train_dataset = class_dataset.take(train_size)
    test_dataset = class_dataset.skip(train_size)
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
    """
    Write a dataset to CSV files.

    Args:
        dataset (tf.data.Dataset): The dataset to write.
        dataset_dir (str): The directory to write the CSV files to.
        train_test (str): Either 'train' or 'test' to indicate the dataset type.

    Returns:
        None
    """
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
    """
    Get the version of the local dataset.

    Args:
        dataset_dir (str): The directory containing the dataset.

    Returns:
        Optional[str]: The version of the local dataset, or None if not found.
    """
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


def create_warped_examples(
    signal: tf.Tensor, max_warp_factor: float = 0.3, dtype: tf.DType = tf.float32
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Create warped versions (squished and expanded) of the input signal.

    Args:
        signal (tf.Tensor): The input signal to be warped.
        max_warp_factor (float): The maximum factor by which the signal can be warped. Defaults to 0.3.
        dtype (tf.DType): The desired data type for the output tensors. Defaults to tf.float32.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing the squished and expanded versions of the input signal.
    """
    original_dtype = signal.dtype
    signal = tf.cast(signal, tf.float32)  # Convert to float32 for internal calculations

    time_steps = tf.shape(signal)[0]

    # Create squished version
    squish_factor = 1 - tf.random.uniform((), 0, max_warp_factor, seed=43)
    squished_length = tf.cast(tf.cast(time_steps, tf.float32) * squish_factor, tf.int32)
    squished = tf.image.resize(tf.expand_dims(signal, -1), (squished_length, 1))[
        :, :, 0
    ]
    pad_total = time_steps - squished_length
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    padding = [[pad_left, pad_right], [0, 0]]
    squished = tf.pad(squished, padding)

    # Create expanded version
    expand_factor = 1 + tf.random.uniform((), 0, max_warp_factor, seed=43)
    expanded_length = tf.cast(tf.cast(time_steps, tf.float32) * expand_factor, tf.int32)
    expanded = tf.image.resize(tf.expand_dims(signal, -1), (expanded_length, 1))[
        :, :, 0
    ]
    trim_total = expanded_length - time_steps
    trim_left = trim_total // 2
    trim_right = expanded_length - (trim_total - trim_left)
    expanded = expanded[trim_left:trim_right]

    # Cast back to original dtype
    squished = tf.cast(squished, original_dtype)
    expanded = tf.cast(expanded, original_dtype)

    return squished, expanded


def augment_example(x: tf.Tensor, y: tf.Tensor, dtype: tf.DType) -> tf.data.Dataset:
    """
    Augment a single example by creating warped versions and combining them with the original.

    Args:
        x (tf.Tensor): The input tensor to be augmented.
        y (tf.Tensor): The corresponding label tensor.
        dtype (tf.DType): The desired data type for the augmented tensors.

    Returns:
        tf.data.Dataset: A dataset containing the original and augmented examples with their labels.
    """
    # Apply augmentation to each example in the batch
    squished, expanded = create_warped_examples(x, 0.2, dtype=dtype)

    # Ensure all tensors have the same data type
    x = tf.cast(x, dtype)
    squished = tf.cast(squished, dtype)
    expanded = tf.cast(expanded, dtype)

    # Create a list of augmented examples
    augmented_x = [x, squished, expanded]
    augmented_y = [y, y, y]

    return tf.data.Dataset.from_tensor_slices((augmented_x, augmented_y))


def train_etl(
    caps_data_dir: str,
    dataset_dir: str,
    target_length: int,
    dtype: DtypeLiteral,
    examples_per_class: int,
    train_test_fraction: float,
    train_val_fraction: float,
    num_classes: int,
    batch_size: int,
    comet_project_name: str,
    use_remote_dataset_version: str = "",
    use_augmentation: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int, int, str]:
    """
    Process the data from multiple class files, create balanced datasets,
    perform train-test split, and upload to Comet ML.

    Args:
        caps_data_dir (str): Directory containing the class CSV files.
        dataset_dir (str): Directory to save the processed dataset.
        target_length (int): The desired length of each time series.
        dtype (DtypeLiteral): The desired data type for the timeseries tensor as a string.
        examples_per_class (int): Number of samples to use per class.
        train_test_fraction (float): Fraction of data to use for training.
        train_val_fraction (float): Fraction of training data to use for validation.
        num_classes (int): Number of classes in the dataset.
        batch_size (int): The number of samples per batch.
        comet_project_name (str): Name of the Comet ML project.
        use_remote_dataset_version (str): Version of the remote dataset to use, if any.
        use_augmentation (bool): Whether to augment original training examples with warped versions
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int, int, str]:
        The train, validation, and test datasets, steps per epoch, validation steps, and the dataset version.
    """
    comet_obj = CometArtifactManager(
        project_name=comet_project_name, dataset_dir=dataset_dir
    )
    current_local_version = get_local_dataset_version(dataset_dir)
    reprocess_dataset = False

    # Check if the remote dataset version is different from the local version
    # If yes download the remote dataset version and load it
    # If no, then load the local dataset
    if use_remote_dataset_version != "":
        if use_remote_dataset_version != current_local_version:
            logger.info(
                f"Downloading remote dataset version: v{use_remote_dataset_version}.. "
            )
            comet_obj.download_remote_dataset(use_remote_dataset_version)
        else:
            logger.info(
                "Remote version is the same as the local version. Loading local dataset..."
            )
        train_dataset, val_dataset, test_dataset, steps_per_epoch, validation_steps = (
            create_train_val_test_datasets_from_train_test_csvs(
                dataset_dir,
                batch_size,
                target_length,
                dtype,
                train_val_fraction,
                use_augmentation,
            )
        )
        write_dataset_version_info(dataset_dir, version=use_remote_dataset_version)
        return (
            train_dataset,
            val_dataset,
            test_dataset,
            steps_per_epoch,
            validation_steps,
            use_remote_dataset_version,
        )

    if current_local_version and use_remote_dataset_version == "":
        logger.info(
            f"A dataset v{current_local_version} was found locally in:\n{dataset_dir}"
        )
        reprocess = input(
            "Do you want to overwrite this local dataset by reprocess the data, and creating a new dataset version? (y/n): "
        )
        if reprocess.lower() == "y" or reprocess.lower() == "yes":
            logger.info("You chose to reprocess the data. Reprocessing data...")
            reprocess_dataset = True
        else:
            logger.info("You chose not to reprocess the data.Loading local dataset..")
            (
                train_dataset,
                val_dataset,
                test_dataset,
                steps_per_epoch,
                validation_steps,
            ) = create_train_val_test_datasets_from_train_test_csvs(
                dataset_dir,
                batch_size,
                target_length,
                dtype,
                train_val_fraction,
                use_augmentation,
            )
            logger.info(f"Local dataset v{current_local_version} loaded successfully.")
            return (
                train_dataset,
                val_dataset,
                test_dataset,
                steps_per_epoch,
                validation_steps,
                current_local_version,
            )

    if reprocess_dataset or (
        not current_local_version and use_remote_dataset_version == ""
    ):
        class_files = group_files_by_class(caps_data_dir)
        min_class, min_rows = find_class_with_least_rows(class_files)
        if examples_per_class is None:
            examples_per_class = min_rows
        else:
            examples_per_class = min(examples_per_class, min_rows)
        logger.info(
            f"Each class in the dataset will have {examples_per_class} examples"
        )

        train_datasets = []
        test_datasets = []
        for class_id, file_paths in class_files.items():
            train_ds, test_ds = create_class_dataset(
                file_paths,
                target_length,
                dtype,
                examples_per_class,
                train_test_fraction,
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
        logger.info("Calculating dataset size...")
        total_samples = examples_per_class * num_classes
        train_samples = int(train_test_fraction * total_samples)
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
        comet_obj.experiment.log_parameter("train_test_fraction", train_test_fraction)
        comet_obj.experiment.log_parameter("train_val_fraction", train_val_fraction)
        comet_obj.experiment.log_parameter("batch_size", batch_size)
        comet_obj.experiment.log_parameter("num_classes", len(class_files))
        comet_obj.experiment.log_parameter("total_samples", total_samples)
        comet_obj.experiment.log_parameter("train_samples", train_samples)
        comet_obj.experiment.log_parameter("test_samples", test_samples)

        logger.info("Making Comet ML dataset artifacts for uploading...")
        version = upload_dataset_to_comet(dataset_dir, comet_project_name)

        comet_obj.end_comet_experiment()
        logger.info(
            f"Data processed and resulting dataset {version} uploaded to Comet ML successfully."
        )

        logger.info(
            "Creating train, validation, and test datasets from dataset CSV files..."
        )
        train_dataset, val_dataset, test_dataset, steps_per_epoch, validation_steps = (
            create_train_val_test_datasets_from_train_test_csvs(
                dataset_dir,
                batch_size,
                target_length,
                dtype,
                train_val_fraction=train_val_fraction,
                use_augmentation=use_augmentation,
            )
        )

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            steps_per_epoch,
            validation_steps,
            version,
        )

    raise RuntimeError("No valid dataset could be processed. Please check your inputs.")


if __name__ == "__main__":
    caps_data_dir = "/home/valen/10-data-for-upload-to-mega/uncompressed/all_csvs"
    dataset_dir = "/home/valen/10-data-for-upload-to-mega/uncompressed/dataset"
    target_length = 500
    dtype: DtypeLiteral = "float16"
    examples_per_class = 100  # Set to None if you want to use all available examples
    train_test_fraction = 0.95
    train_val_fraction = 0.8
    batch_size = 10
    num_classes = 4
    comet_project_name = "dataset"
    use_remote_dataset_version = ""
    use_augmentation = True

    (
        train_dataset,
        val_dataset,
        test_dataset,
        steps_per_epoch,
        validation_steps,
        version,
    ) = train_etl(
        caps_data_dir,
        dataset_dir,
        target_length,
        dtype,
        examples_per_class,
        train_test_fraction,
        train_val_fraction,
        num_classes,
        batch_size,
        comet_project_name,
        use_remote_dataset_version,
        use_augmentation,
    )

    print(f"Train dataset: {train_dataset}")
    print(f"Validation dataset: {val_dataset}")
    print(f"Test dataset: {test_dataset}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print(f"Dataset version: {version}")
