import csv
import hashlib
import os
from importlib import resources
from importlib.metadata import version
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import polars as pl
from loguru import logger
from prefect import flow, task
from prefect.engine import TaskRunContext
from prefect.tasks import task_input_hash
from tqdm import tqdm
from typing_extensions import Literal

from capfinder import model as model_module
from capfinder.collate import collate_bam_pod5
from capfinder.inference_data_loader import create_dataset
from capfinder.logger_config import configure_logger, configure_prefect_logging
from capfinder.ml_libs import keras, tf
from capfinder.report import generate_report
from capfinder.utils import log_header, log_output, log_step, map_cap_int_to_name

# Define custom types
TrainData = Tuple[
    np.ndarray,  # x_train
    np.ndarray,  # y_train
    pl.Series,  # read_id_train
]
DtypeLiteral = Literal["float16", "float32", "float64"]
DtypeNumpy = Union[np.float16, np.float32, np.float64]


def custom_cache_key_fn(context: TaskRunContext, parameters: dict) -> str:
    """
    Generate a custom cache key based on input parameters.

    Args:
        context (TaskRunContext): Prefect context (unused in this function).
        parameters (dict): Dictionary of parameters used for cache key generation.

    Returns:
        str: The generated cache key.
    """
    dataset_hash = hashlib.md5(str(parameters["dataset"]).encode()).hexdigest()
    model_hash = hashlib.md5(str(parameters["model"]).encode()).hexdigest()
    output_dir_hash = hashlib.md5(parameters["output_dir"].encode()).hexdigest()
    combined_hash = hashlib.md5(
        f"{dataset_hash}_{model_hash}_{output_dir_hash}".encode()
    ).hexdigest()
    return combined_hash


def count_csv_rows(file_path: str) -> int:
    """
    Quickly count the number of rows in a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        int: Number of rows in the CSV file (excluding the header).
    """
    with Path(file_path).open() as f:
        return sum(1 for _ in f) - 1  # Subtract 1 to exclude the header


@task(cache_key_fn=custom_cache_key_fn)
def reconfigure_logging_task(output_dir: str, debug_code: bool) -> None:
    """
    Reconfigure logging settings for both application and Prefect.

    Args:
        output_dir (str): Directory where logs will be saved.
        debug_code (bool): Flag to determine if code locations should be shown in logs.
    """
    configure_logger(output_dir, show_location=debug_code)
    configure_prefect_logging(show_location=debug_code)


def get_model(
    model_path: Optional[str] = None, load_optimizer: bool = False
) -> keras.Model:
    """
    Load and return a model from the given model path or use the default model.

    Args:
        model_path (Optional[str]): Path to the custom model file. If None, use the default model.
        load_optimizer (bool): Whether to load the optimizer with the model.

    Returns:
        keras.Model: The loaded Keras model.
    """
    if model_path is None:
        model_file = resources.files(model_module).joinpath("cnn_lstm-classifier.keras")
        with resources.as_file(model_file) as default_model_path:
            model = keras.models.load_model(default_model_path, compile=False)
    else:
        model = keras.models.load_model(model_path, compile=False)

    logger.info(
        f"Model loaded successfully from {'default path' if model_path is None else model_path}"
    )
    return model


@task(cache_key_fn=task_input_hash)
def batched_inference(
    dataset: tf.data.Dataset,
    model: keras.Model,
    output_dir: str,
    csv_file_path: str,
) -> str:
    """
    Perform batched inference on a dataset using a given model and save predictions to a CSV file.

    Args:
        dataset (tf.data.Dataset): The input dataset to perform inference on.
        model (keras.Model): The Keras model to use for making predictions.
        output_dir (str): The directory where the output CSV file will be saved.
        csv_file_path (str): Path to the original CSV file used to create the dataset.

    Returns:
        str: The path to the output CSV file containing the predictions.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, "predictions.csv")

    total_reads = count_csv_rows(csv_file_path)
    logger.info(f"Total reads to perform cap predictions on: {total_reads}")

    with open(output_csv_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["read_id", "predicted_cap"])

        pbar = tqdm(total=total_reads, unit="reads")

        processed_reads = 0
        try:
            for batch in dataset:
                x, _, read_id = batch
                preds = model.predict(x, verbose=0)
                batch_pred_classes = tf.argmax(preds, axis=1).numpy()

                rows_to_write = [
                    [rid.decode("utf-8"), map_cap_int_to_name(pred_class)]
                    for rid, pred_class in zip(read_id.numpy(), batch_pred_classes)
                ]

                csvwriter.writerows(rows_to_write)

                batch_size = len(read_id)
                processed_reads += batch_size
                pbar.update(batch_size)

                if processed_reads >= total_reads:
                    break
        except tf.errors.OutOfRangeError:
            logger.warning(
                "Dataset iterator exhausted before processing all expected reads."
            )
        finally:
            pbar.close()

    logger.info(f"Batched inference completed! Processed {processed_reads} reads.")
    if processed_reads != total_reads:
        logger.warning(
            f"Number of processed samples ({processed_reads}) "
            f"differs from expected total ({total_reads})."
        )

    return output_csv_path


@task(cache_key_fn=task_input_hash)
def collate_bam_pod5_wrapper(
    bam_filepath: str,
    pod5_dir: str,
    num_cpus: int,
    reference: str,
    cap_class: int,
    cap0_pos: int,
    train_or_test: str,
    plot_signal: bool,
    output_dir: str,
) -> tuple[str, str]:
    """
    Wrapper for collating BAM and POD5 files.

    Args:
        bam_filepath (str): Path to the BAM file.
        pod5_dir (str): Directory containing POD5 files.
        num_cpus (int): Number of CPUs to use for processing.
        reference (str): Reference sequence.
        cap_class (int): CAP class identifier.
        cap0_pos (int): Position of CAP0.
        train_or_test (str): Indicates whether data is for training or testing.
        plot_signal (bool): Flag to plot the signal.
        output_dir (str): Directory where output files will be saved.

    Returns:
        tuple[str, str]: Paths to the data and metadata files.
    """
    data_path, metadata_path = collate_bam_pod5(
        bam_filepath=bam_filepath,
        pod5_dir=pod5_dir,
        num_processes=num_cpus,
        reference=reference,
        cap_class=cap_class,
        cap0_pos=cap0_pos,
        train_or_test=train_or_test,
        plot_signal=plot_signal,
        output_dir=output_dir,
    )
    return data_path, metadata_path


@task(cache_key_fn=task_input_hash)
def generate_report_wrapper(
    metadata_file: str, predictions_file: str, output_csv: str, output_html: str
) -> None:
    """
    Wrapper for generating the report.

    Args:
        metadata_file (str): Path to the metadata file.
        predictions_file (str): Path to the predictions file.
        output_csv (str): Path to save the output CSV.
        output_html (str): Path to save the output HTML.
    """
    generate_report(
        metadata_file,
        predictions_file,
        output_csv,
        output_html,
    )
    os.remove(predictions_file)


@flow(name="prepare-inference-data")
def prepare_inference_data(
    bam_filepath: str,
    pod5_dir: str,
    num_cpus: int,
    output_dir: str,
    dtype: DtypeLiteral,
    reference: str = "GCTTTCGTTCGTCTCCGGACTTATCGCACCACCTATCCATCATCAGTACTGT",
    cap0_pos: int = 52,
    train_or_test: str = "test",
    plot_signal: bool = False,
    cap_class: int = -99,
    target_length: int = 500,
    batch_size: int = 32,
    custom_model_path: Optional[str] = None,
    debug_code: bool = False,
    refresh_cache: bool = False,
) -> tuple[str, str]:
    """
    Prepare inference data by processing BAM and POD5 files, and generate features for the model.

    Args:
        bam_filepath (str): Path to the BAM file.
        pod5_dir (str): Directory containing POD5 files.
        num_cpus (int): Number of CPUs to use for processing.
        output_dir (str): Directory where output files will be saved.
        dtype (DtypeLiteral): Data type for the features.
        reference (str): Reference sequence.
        cap0_pos (int): Position of CAP0.
        train_or_test (str): Indicates whether data is for training or testing.
        plot_signal (bool): Flag to plot the signal.
        cap_class (int): CAP class identifier.
        target_length (int): Length of the target sequence.
        batch_size (int): Size of the data batches.
        custom_model_path (Optional[str]): Path to a custom model file. If None, use the default model.
        debug_code (bool): Flag to enable debugging information in logs.
        refresh_cache (bool): Flag to refresh cached data.

    Returns:
        tuple[str, str]: Paths to the output CSV and HTML files.
    """
    configure_prefect_logging(show_location=debug_code)
    os.makedirs(output_dir, exist_ok=True)

    log_step(1, 5, "Extracting Cap Signal by collating BAM and POD5 files")
    data_path, metadata_path = collate_bam_pod5_wrapper.with_options(
        refresh_cache=refresh_cache
    )(
        bam_filepath=bam_filepath,
        pod5_dir=pod5_dir,
        num_cpus=num_cpus,
        reference=reference,
        cap_class=cap_class,
        cap0_pos=cap0_pos,
        train_or_test=train_or_test,
        plot_signal=plot_signal,
        output_dir=os.path.join(output_dir, "0_raw_cap_signal_data"),
    )

    log_step(2, 5, "Creating TensorFlow dataset")
    dataset = create_dataset(data_path, target_length, batch_size, dtype)

    log_step(
        3, 5, f"Loading the {'custom' if custom_model_path else 'pre-trained'} model"
    )
    model = get_model(custom_model_path)

    log_step(4, 5, "Performing batch inference for cap type prediction")
    predictions_csv_path = batched_inference.with_options(refresh_cache=refresh_cache)(
        dataset,
        model,
        output_dir=os.path.join(output_dir, "1_cap_predictions"),
        csv_file_path=data_path,
    )

    log_step(5, 5, "Generating report")
    output_csv_path = os.path.join(
        output_dir, "1_cap_predictions", "cap_predictions.csv"
    )
    output_html_path = os.path.join(
        output_dir, "1_cap_predictions", "cap_analysis_report.html"
    )
    generate_report_wrapper.with_options(refresh_cache=refresh_cache)(
        metadata_file=metadata_path,
        predictions_file=predictions_csv_path,
        output_csv=output_csv_path,
        output_html=output_html_path,
    )
    return output_csv_path, output_html_path


def predict_cap_types(
    bam_filepath: str,
    pod5_dir: str,
    num_cpus: int,
    output_dir: str,
    dtype: DtypeLiteral,
    reference: str = "GCTTTCGTTCGTCTCCGGACTTATCGCACCACCTATCCATCATCAGTACTGT",
    cap0_pos: int = 52,
    train_or_test: str = "test",
    plot_signal: bool = False,
    cap_class: int = -99,
    target_length: int = 500,
    batch_size: int = 32,
    custom_model_path: Optional[str] = None,
    debug_code: bool = False,
    refresh_cache: bool = False,
    formatted_command: Optional[str] = None,
) -> None:
    """
    Predict CAP types by preparing the inference data and running the prediction workflow.

    Args:
        bam_filepath (str): Path to the BAM file.
        pod5_dir (str): Directory containing POD5 files.
        num_cpus (int): Number of CPUs to use for processing.
        output_dir (str): Directory where output files will be saved.
        dtype (DtypeLiteral): Data type for the features.
        reference (str): Reference sequence.
        cap0_pos (int): Position of CAP0.
        train_or_test (str): Indicates whether data is for training or testing.
        plot_signal (bool): Flag to plot the signal.
        cap_class (int): CAP class identifier.
        target_length (int): Length of the target sequence.
        batch_size (int): Size of the data batches.
        custom_model_path (Optional[str]): Path to a custom model file. If None, use the default model.
        debug_code (bool): Flag to enable debugging information in logs.
        refresh_cache (bool): Flag to refresh cached data.
        formatted_command (Optional[str]): The formatted command string to be logged.
    """
    log_filepath = configure_logger(
        os.path.join(output_dir, "logs"), show_location=debug_code
    )
    configure_prefect_logging(show_location=debug_code)
    version_info = version("capfinder")
    log_header(f"Using Capfinder v{version_info}")
    logger.info(formatted_command)
    output_csv_path, output_html_path = prepare_inference_data(
        bam_filepath,
        pod5_dir,
        num_cpus,
        output_dir,
        dtype,
        reference,
        cap0_pos,
        train_or_test,
        plot_signal,
        cap_class,
        target_length,
        batch_size,
        custom_model_path,
        debug_code,
        refresh_cache,
    )
    grey = "\033[90m"
    reset = "\033[0m"
    log_output(
        f"Cap predictions have been saved to the following path:\n {grey}{output_csv_path}{reset}\nThe log file has been saved to:\n {grey}{log_filepath}{reset}\nThe analysis report has been saved to:\n {grey}{output_html_path}{reset}"
    )
    log_header("Processing finished!")


if __name__ == "__main__":
    bam_filepath = "/export/valenfs/data/processed_data/MinION/9_madcap/1_data/8_20231114_randomCAP1v3_rna004/1_basecall_subset/sorted.calls.bam"
    pod5_dir = "/export/valenfs/data/raw_data/minion/2024_cap_ligation_data_v3_oligo/20240521_cap1/20231114_randomCAP1v3_rna004/"
    num_cpus = 3
    output_dir = "/export/valenfs/data/processed_data/MinION/9_madcap/1_data/8_20231114_randomCAP1v3_rna004/test_OTE_vizs_july55"
    dtype: DtypeLiteral = "float16"
    reference = "GCTTTCGTTCGTCTCCGGACTTATCGCACCACCTATCCATCATCAGTACTGT"
    cap0_pos = 52
    train_or_test = "test"
    plot_signal = True
    cap_class = -99
    target_length = 500
    batch_size = 10
    debug_code = False
    refresh_cache = True
    formatted_command = ""
    custom_model_path = None  # Set this to a path if you want to use a custom model

    predict_cap_types(
        bam_filepath,
        pod5_dir,
        num_cpus,
        output_dir,
        dtype,
        reference,
        cap0_pos,
        train_or_test,
        plot_signal,
        cap_class,
        target_length,
        batch_size,
        custom_model_path,
        debug_code,
        refresh_cache,
        formatted_command,
    )
