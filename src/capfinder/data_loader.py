from typing import Tuple

import tensorflow as tf


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


def load_feature_dataset(file_path: str, num_timesteps: int) -> tf.data.Dataset:
    """Load feature dataset from a CSV file.

    Parameters:
    -----------
    file_path : str
        The path to the CSV file containing features.
    num_timesteps : int
        The number of time steps in each time series.

    Returns:
    --------
    tf.data.Dataset
        A TensorFlow dataset containing the parsed features.
    """
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.skip(1)  # Skip header row
    dataset = dataset.map(
        lambda x: parse_features(x, num_timesteps), num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset


def load_label_dataset(file_path: str) -> tf.data.Dataset:
    """Load label dataset from a CSV file.

    Parameters:
    -----------
    file_path : str
        The path to the CSV file containing labels.

    Returns:
    --------
    tf.data.Dataset
        A TensorFlow dataset containing the parsed labels.
    """
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.skip(1)  # Skip header row
    dataset = dataset.map(parse_labels, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def combine_datasets(
    features_dataset: tf.data.Dataset,
    labels_dataset: tf.data.Dataset,
    batch_size: int,
    num_timesteps: int,
) -> tf.data.Dataset:
    """Combine feature and label datasets with padded batching.

    Parameters:
    -----------
    features_dataset : tf.data.Dataset
        The dataset containing features.
    labels_dataset : tf.data.Dataset
        The dataset containing labels.
    batch_size : int
        The size of each batch.
    num_timesteps : int
        The number of time steps in each time series.

    Returns:
    --------
    tf.data.Dataset
        A combined dataset with features and labels, padded and batched.
    """
    dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
    dataset = dataset.padded_batch(
        batch_size, padded_shapes=([num_timesteps, 1], []), drop_remainder=True
    )
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def load_datasets(
    train_x_path: str,
    train_y_path: str,
    val_x_path: str,
    val_y_path: str,
    batch_size: int,
    num_timesteps: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load and combine train and validation datasets.

    Parameters:
    -----------
    train_x_path : str
        Path to the CSV file containing training features.
    train_y_path : str
        Path to the CSV file containing training labels.
    val_x_path : str
        Path to the CSV file containing validation features.
    val_y_path : str
        Path to the CSV file containing validation labels.
    batch_size : int
        The size of each batch.
    num_timesteps : int
        The number of time steps in each time series.

    Returns:
    --------
    Tuple[tf.data.Dataset, tf.data.Dataset]
        A tuple containing the combined training dataset and validation dataset.
    """
    train_features_dataset = load_feature_dataset(train_x_path, num_timesteps)
    train_labels_dataset = load_label_dataset(train_y_path)
    val_features_dataset = load_feature_dataset(val_x_path, num_timesteps)
    val_labels_dataset = load_label_dataset(val_y_path)

    train_dataset = combine_datasets(
        train_features_dataset, train_labels_dataset, batch_size, num_timesteps
    )
    val_dataset = combine_datasets(
        val_features_dataset, val_labels_dataset, batch_size, num_timesteps
    )

    return train_dataset, val_dataset
