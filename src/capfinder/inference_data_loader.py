from typing import Generator, Tuple

import polars as pl
from loguru import logger
from typing_extensions import Literal

from capfinder.ml_libs import float16, float32, float64, tf

DtypeLiteral = Literal["float16", "float32", "float64"]


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


def get_dtype(dtype: str) -> tf.DType:
    """
    Convert a string dtype to its corresponding TensorFlow data type.

    Args:
        dtype (str): A string representing the desired data type.

    Returns:
        tf.DType: The corresponding TensorFlow data type.

    Raises:
        ValueError: If an invalid dtype string is provided.
    """
    valid_dtypes = {
        "float16": float16,
        "float32": float32,
        "float64": float64,
    }

    if dtype in valid_dtypes:
        return valid_dtypes[dtype]
    else:
        logger.warning('You provided an invalid dtype. Using "float32" as default.')
        return float32


def csv_generator(
    file_path: str, chunk_size: int = 10000
) -> Generator[Tuple[str, str, str], None, None]:
    """
    Generate rows from a CSV file in chunks.

    Args:
        file_path (str): Path to the CSV file.
        chunk_size (int, optional): Number of rows to process in each chunk. Defaults to 10000.

    Yields:
        Tuple[str, str, str]: A tuple containing read_id, cap_class, and timeseries as strings.
    """
    df = pl.scan_csv(file_path)
    total_rows = df.select(pl.count()).collect().item()

    for start in range(0, total_rows, chunk_size):
        min(start + chunk_size, total_rows)
        chunk = df.slice(start, chunk_size).collect()
        for row in chunk.iter_rows():
            yield (str(row[0]), str(row[1]), str(row[2]))


def create_dataset(
    file_path: str, target_length: int, batch_size: int, dtype: DtypeLiteral
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        target_length (int): The desired length of the timeseries tensor.
        batch_size (int): The number of samples per batch.
        dtype (DtypeLiteral): The desired data type for the timeseries tensor as a string.

    Returns:
        tf.data.Dataset: A TensorFlow dataset that yields batches of parsed and formatted data.
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

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    logger.info("Dataset created successfully.")
    return dataset
