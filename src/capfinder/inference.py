import csv
import hashlib
import os
from importlib import resources
from importlib.metadata import version
from typing import List, Optional, Tuple, Union

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
from capfinder.inference_data_loader import make_batched_dataset
from capfinder.logger_config import configure_logger, configure_prefect_logging
from capfinder.ml_libs import keras, tf
from capfinder.report import generate_report
from capfinder.train_etl import concatenate_dataframes, load_csv, make_x_y_read_id_sets
from capfinder.utils import (
    get_dtype,
    log_header,
    log_output,
    log_step,
    map_cap_int_to_name,
)

# Define custom types
TrainData = Tuple[
    np.ndarray,  # x_train
    np.ndarray,  # y_train
    pl.Series,  # read_id_train
]
DtypeLiteral = Literal["float16", "float32", "float64"]
DtypeNumpy = Union[np.float16, np.float32, np.float64]


def reconfigure_logging_task(output_dir: str, debug_code: bool) -> None:
    """
    Reconfigure logging settings for both application and Prefect.

    Args:
        output_dir (str): Directory where logs will be saved.
        debug_code (bool): Flag to determine if code locations should be shown in logs.
    """
    configure_logger(output_dir, show_location=debug_code)
    configure_prefect_logging(show_location=debug_code)


def get_model(model_name: str, load_optimizer: bool = False) -> keras.Model:
    """
    Load and return a model from the given model name.

    Args:
        model_name (str): Name of the model file.
        load_optimizer (bool): Whether to load the optimizer with the model.

    Returns:
        keras.Model: The loaded Keras model.
    """
    model_file = resources.files(model_module).joinpath(model_name)
    with resources.as_file(model_file) as model_path:
        model = keras.models.load_model(model_path, compile=False)

    logger.info("Model loaded successfully in memory.")
    return model


@task(cache_key_fn=task_input_hash)
def list_csv_files(directory: str) -> List[str]:
    """
    List CSV files in the specified directory that match the pattern.

    Args:
        directory (str): Directory to search for CSV files.

    Returns:
        List[str]: List of paths to CSV files.
    """
    try:
        all_files = os.listdir(directory)
        csv_files = [
            os.path.join(directory, file)
            for file in all_files
            if file.startswith("data__cap") and file.endswith(".csv")
        ]
        return csv_files
    except Exception as e:
        logger.error(f"An error occurred while listing files: {e}")
        return []


@task(cache_key_fn=task_input_hash)
def save_to_file(
    x: np.ndarray, y: np.ndarray, read_id: pl.Series, output_dir: str
) -> None:
    """
    Save arrays and series to CSV files in the specified directory.

    Args:
        x (np.ndarray): Array of features.
        y (np.ndarray): Array of labels.
        read_id (pl.Series): Series of read IDs.
        output_dir (str): Directory where files will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    dfp_x = pl.from_numpy(data=x.reshape(x.shape[:2]), orient="row")
    dfp_y = pl.from_numpy(data=y.reshape(y.shape[:2]), orient="row")
    dfp_id = pl.DataFrame(read_id)

    x_path = os.path.join(output_dir, "x.csv")
    dfp_x.write_csv(file=x_path)
    y_path = os.path.join(output_dir, "y.csv")
    dfp_y.write_csv(file=y_path)
    id_path = os.path.join(output_dir, "read_id.csv")
    dfp_id.write_csv(file=id_path)


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


@task(cache_key_fn=custom_cache_key_fn)
def batched_inference(
    dataset: tf.data.Dataset, model: keras.Model, output_dir: str
) -> str:
    """
    Perform inference on the dataset in batches and save predictions to a CSV file.

    Args:
        dataset (tf.data.Dataset): Tensorflow batched dataset.
        model (keras.Model): Pre-trained model for inference.
        output_dir (str): Directory where predictions will be saved.

    Returns:
        str: Path to the CSV file containing the predictions.
    """
    total_batches = dataset.reduce(0, lambda x, _: x + 1).numpy()
    y_pred: List[int] = []
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, "predictions.csv")

    with open(output_csv_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["read_id", "predicted_cap"])
        pbar = tqdm(dataset, total=total_batches, desc="Processing batches...")
        for batch_num, (x, _, read_id) in enumerate(pbar, start=1):
            preds = model.predict(x, verbose=0)
            batch_pred_classes = np.argmax(preds, axis=1)
            y_pred.extend(batch_pred_classes)
            for rid, pred_class in zip(read_id.numpy(), batch_pred_classes):
                csvwriter.writerow(
                    [rid.decode("utf-8"), map_cap_int_to_name(pred_class)]
                )
            pbar.set_description(f"Processing batch {batch_num}/{total_batches}")
            pbar.set_postfix({"Last batch shape": x.shape})
    logger.info("Batched inference completed!")
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


def step_info(cur_step: int, tot_steps: int, info: str) -> int:
    """
    Log the current step information and increment the step count.

    Args:
        cur_step (int): Current step number.
        tot_steps (int): Total number of steps.
        info (str): Information about the current step.

    Returns:
        int: The incremented step number.
    """
    next_step = cur_step + 1
    logger.info(f"Step {next_step}/{tot_steps}: {info}")
    return next_step


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
        debug_code (bool): Flag to enable debugging information in logs.
        refresh_cache (bool): Flag to refresh cached data.

    Returns:
        tuple[str, str]: Paths to the output CSV and HTML files.
    """
    configure_prefect_logging(show_location=debug_code)
    os.makedirs(output_dir, exist_ok=True)
    tot_steps = 9
    log_step(1, tot_steps, "Extracting Cap Signal by collating BAM and POD5 files")
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

    log_step(2, tot_steps, "Discovering CSV files from step 1")
    csv_files = list_csv_files.with_options(refresh_cache=refresh_cache)(
        os.path.join(output_dir, "0_raw_cap_signal_data")
    )

    log_step(3, tot_steps, "Reading CSV files")
    loaded_dataframes = [load_csv(file_path) for file_path in csv_files]

    log_step(4, tot_steps, "Concatenating CSV files")
    concatenated_df = concatenate_dataframes(loaded_dataframes)
    x, y, read_id = make_x_y_read_id_sets(
        concatenated_df, target_length, dtype_n=get_dtype(dtype)
    )

    log_step(5, tot_steps, "Generating features for the model")
    save_to_file.with_options(refresh_cache=refresh_cache)(
        x,
        y,
        read_id,
        output_dir=os.path.join(output_dir, "1_processed_cap_signal_data"),
    )

    log_step(6, tot_steps, "Batching the features")
    dataset = make_batched_dataset(
        x_path=os.path.join(output_dir, "1_processed_cap_signal_data", "x.csv"),
        y_path=os.path.join(output_dir, "1_processed_cap_signal_data", "y.csv"),
        read_id_path=os.path.join(
            output_dir, "1_processed_cap_signal_data", "read_id.csv"
        ),
        batch_size=batch_size,
        num_timesteps=target_length,
    )

    log_step(7, tot_steps, "Loading the pre-trained model")
    model = get_model("cnn_lstm-classifier.keras")

    log_step(8, tot_steps, "Performing batch inference for cap type prediction")
    predictions_csv_path = batched_inference.with_options(refresh_cache=refresh_cache)(
        dataset, model, output_dir=os.path.join(output_dir, "3_cap_predictions")
    )

    log_step(9, tot_steps, "Generating report")
    output_csv_path = os.path.join(
        output_dir, "3_cap_predictions", "cap_predictions.csv"
    )
    output_html_path = os.path.join(
        output_dir, "3_cap_predictions", "cap_analysis_report.html"
    )
    generate_report(
        metadata_file=metadata_path,
        predictions_file=predictions_csv_path,
        output_csv=output_csv_path,
        output_html=output_html_path,
    )
    os.remove(predictions_csv_path)
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
        debug_code (bool): Flag to enable debugging information in logs.
        refresh_cache (bool): Flag to refresh cached data.
        formatted_command (Optional[str]): The formatted command string to be logged.
    """
    log_filepath = configure_logger(output_dir, show_location=debug_code)
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
    output_dir = "/export/valenfs/data/processed_data/MinION/9_madcap/1_data/8_20231114_randomCAP1v3_rna004/test_OTE_vizs_july12"
    dtype: DtypeLiteral = "float16"
    reference = "GCTTTCGTTCGTCTCCGGACTTATCGCACCACCTATCCATCATCAGTACTGT"
    cap0_pos = 52
    train_or_test = "test"
    plot_signal = True
    cap_class = -99
    target_length = 500
    batch_size = 4
    debug_code = False
    refresh_cache = False
    formatted_command = ""

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
        debug_code,
        refresh_cache,
        formatted_command,
    )
