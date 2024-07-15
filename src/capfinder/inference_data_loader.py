from typing import Callable

from capfinder.ml_libs import tf


def parse_features(line: tf.Tensor, num_timesteps: int) -> tf.Tensor:
    """Parse features from a CSV line and reshape them.

    Parameters:
    -----------
    line : tf.Tensor
        A tensor representing a single line from the CSV file.
    num_timesteps : int
        The number of time steps in each time series.

    Returns:
    --------
    tf.Tensor
        A tensor of shape (num_timesteps, 1) containing the parsed features.
    """
    column_defaults = [[0.0]] * num_timesteps
    fields = tf.io.decode_csv(line, record_defaults=column_defaults)
    features = tf.reshape(fields, (num_timesteps, 1))  # Reshape to (timesteps, 1)
    return features


def parse_labels(line: tf.Tensor) -> tf.Tensor:
    """Parse labels from a CSV line.

    Parameters:
    -----------
    line : tf.Tensor
        A tensor representing a single line from the CSV file.

    Returns:
    --------
    tf.Tensor
        A tensor containing the parsed label.
    """
    label = tf.io.decode_csv(line, record_defaults=[[0]])
    return label[0]


def parse_read_id(line: tf.Tensor) -> tf.Tensor:
    """Parse read_id from a CSV line.

    Parameters:
    -----------
    line : tf.Tensor
        A tensor representing a single line from the CSV file.

    Returns:
    --------
    tf.Tensor
        A tensor containing the parsed read_id.
    """
    read_id = tf.io.decode_csv(line, record_defaults=[[""]])
    return read_id[0]


def load_dataset(
    file_path: str, parse_fn: Callable[[tf.Tensor], tf.Tensor], skip_header: bool = True
) -> tf.data.Dataset:
    """Load dataset from a CSV file.

    Parameters:
    -----------
    file_path : str
        The path to the CSV file.
    parse_fn : Callable[[tf.Tensor], tf.Tensor]
        The function to use for parsing each line of the CSV.
        It should take a tf.Tensor (representing a line from the CSV)
        and return a tf.Tensor (the parsed data).
    skip_header : bool, optional
        Whether to skip the header row (default is True).

    Returns:
    --------
    tf.data.Dataset
        A TensorFlow dataset containing the parsed data.
    """
    dataset = tf.data.TextLineDataset(file_path)
    if skip_header:
        dataset = dataset.skip(1)  # Skip header row
    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def load_inference_dataset(
    x_path: str,
    y_path: str,
    read_id_path: str,
    batch_size: int,
    num_timesteps: int,
) -> tf.data.Dataset:
    """Load and combine datasets for inference.

    Parameters:
    -----------
    x_path : str
        Path to the CSV file containing features.
    y_path : str
        Path to the CSV file containing labels.
    read_id_path : str
        Path to the CSV file containing read_ids.
    batch_size : int
        The size of each batch.
    num_timesteps : int
        The number of time steps in each time series.

    Returns:
    --------
    tf.data.Dataset
        A combined dataset with features, labels, and read_ids, padded and batched.
    """
    features_dataset = load_dataset(x_path, lambda x: parse_features(x, num_timesteps))
    labels_dataset = load_dataset(y_path, parse_labels)
    read_id_dataset = load_dataset(read_id_path, parse_read_id)

    # Combine datasets
    dataset = tf.data.Dataset.zip((features_dataset, labels_dataset, read_id_dataset))

    # Batch and prefetch
    dataset = dataset.padded_batch(
        batch_size, padded_shapes=([num_timesteps, 1], [], []), drop_remainder=False
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
