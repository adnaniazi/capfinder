"""
The main workhorse which collates information from the BAM file and the POD5 files,
aligns OTE to extracts the signal for the region of interest (ROI) for
training or testing purposes. It also plots the ROI signal if requested.

Author: Adnan M. Niazi
Date: 2024-02-28
"""

import contextlib
import csv
import datetime
import multiprocessing
import os
import signal
from dataclasses import dataclass
from typing import Any, Dict, Union

from loguru import logger
from mpire import WorkerPool  # type: ignore

from capfinder.align import align
from capfinder.bam import get_total_records, process_bam_records
from capfinder.find_ote_test import process_read as extract_roi_coords_test
from capfinder.find_ote_train import process_read as extract_roi_coords_train
from capfinder.index import fetch_filepath_using_filename, index
from capfinder.logger_config import configure_logger
from capfinder.plot import plot_roi_signal
from capfinder.process_pod5 import (
    extract_roi_signal,
    find_base_locs_in_signal,
    pull_read_from_pod5,
)
from capfinder.utils import open_database

# Create a lock for synchronization
lock = multiprocessing.Lock()

# Shared dictionary
shared_dict = multiprocessing.Manager().dict()
shared_dict["good_reads_count"] = 0
shared_dict["bad_reads_count"] = 0
shared_dict["good_reads_dir"] = 0
shared_dict["bad_reads_dir"] = 0


@dataclass
class FASTQRecord:
    """
    Simulates a FASTQ record object.

    Params:
        id: str
            Read ID.
        seq: str
            Read sequence.

    Returns:
        FASTQRecord: FASTQRecord
            A FASTQRecord object.
    """

    id: str
    seq: str


class DatabaseHandler:
    def __init__(
        self, database_path: str, plots_csv_filepath: Union[str, None]
    ) -> None:
        """Initializes the database handler"""
        self.database_path = database_path
        self.plots_csv_filepath = plots_csv_filepath
        if self.plots_csv_filepath:
            self.csvfile = open(self.plots_csv_filepath, "a", newline="")

    def init_func(self, worker_state: Dict[str, Any]) -> None:
        """Initializes the database connection for each worker."""
        worker_state["db_connection"], worker_state["db_cursor"] = open_database(
            self.database_path
        )
        # Write the header row to the CSV file
        if self.plots_csv_filepath:
            csv_writer = csv.writer(self.csvfile)
            csv_writer.writerow(["read_id", "plot_filepath"])
            worker_state["csv_writer"] = csv_writer
            worker_state["csvfile"] = self.csvfile  # Store csvfile in worker_state

    def exit_func(self, worker_state: Dict[str, Any]) -> None:
        """Closes the database connection and the CSV file."""
        conn = worker_state.get("db_connection")
        if conn:
            conn.close()

        csvfile = worker_state.get("csvfile")
        if self.plots_csv_filepath and csvfile:
            csvfile.close()


def collate_bam_pod5_worker(
    worker_state: Dict[str, Any],
    bam_data: Dict[str, Any],
    reference: str,
    cap0_pos: int,
    train_or_test: str,
    plot_signal: bool,
    output_dir: str,
) -> None:
    """Worker function that collates information from POD5 and BAM file, finds the
    FASTA coordinates of  region of interest (ROI) and and extracts its signal.

    Params:
        worker_state: dict
            Dictionary containing the database connection and cursor.
        bam_data: dict
            Dictionary containing the BAM record information.
        reference: str
            Reference sequence.
        cap0_pos: int
            Position of the cap0 base in the reference sequence.
        train_or_test: str
            Whether to extract ROI for training or testing.
        plot_signal: bool
            Whether to plot the ROI signal.
        output_dir: str
            Path to the output directory.

    Returns:
        roi_data: dict
            Dictionary containing the ROI signal and fasta sequence.

    """
    # 1. Get read info from bam record
    read_id = bam_data["read_id"]
    pod5_filename = bam_data["pod5_filename"]
    parent_read_id = bam_data["parent_read_id"]
    # 2. Find the pod5 filepath corresponding to the pod5_filename in the database
    pod5_filepath = fetch_filepath_using_filename(
        worker_state["db_connection"], worker_state["db_cursor"], pod5_filename
    )

    # 3. Pull the read data from the multi-pod5 file
    # If the read is a split read, pull the parent read data
    if parent_read_id == "":
        pod5_data = pull_read_from_pod5(read_id, pod5_filepath)
    else:
        pod5_data = pull_read_from_pod5(parent_read_id, pod5_filepath)

    # 4. Extract the locations of each new base in signal coordinates
    base_locs_in_signal = find_base_locs_in_signal(bam_data)

    # 5. Get alignment of OTE with the read
    # Simulate a FASTQ record object
    read_fasta = bam_data["read_fasta"]
    fastq_record = FASTQRecord(read_id, read_fasta)
    if train_or_test.lower() == "train":
        aln_res = extract_roi_coords_train(
            record=fastq_record, reference=reference, cap0_pos=cap0_pos
        )
    elif train_or_test.lower() == "test":
        aln_res = extract_roi_coords_test(
            record=fastq_record, reference=reference, cap0_pos=cap0_pos
        )
    else:
        logger.warning(
            "Invalid train_or_test argument. Must be either 'train' or 'test'."
        )
        return None

    # 6. Extract signal data for the ROI
    start_base_idx_in_fasta = aln_res["left_flanking_region_start_fastq_pos"]
    end_base_idx_in_fasta = aln_res["right_flanking_region_start_fastq_pos"]

    roi_data = extract_roi_signal(
        signal=pod5_data["signal_pa"],
        base_locs_in_signal=base_locs_in_signal,
        fasta=read_fasta,
        experiment_type=pod5_data["experiment_type"],
        start_base_idx_in_fasta=start_base_idx_in_fasta,
        end_base_idx_in_fasta=end_base_idx_in_fasta,
        num_left_clipped_bases=bam_data["num_left_clipped_bases"],
    )

    # 7. Add additional information to the ROI data
    roi_data["start_base_idx_in_fasta"] = start_base_idx_in_fasta
    roi_data["end_base_idx_in_fasta"] = end_base_idx_in_fasta
    roi_data["read_id"] = read_id

    # 8. Plot the ROI signal if requested
    # Save plot in directories of 100 plots each separated into
    # good and bad categories. Good reads mean those that have
    # the OTE in them and bad reads mean those that do not.
    if plot_signal:
        read_type = (
            "bad_reads"
            if start_base_idx_in_fasta is None and end_base_idx_in_fasta is None
            else "good_reads"
        )
        count_key = f"{read_type}_count"
        dir_key = f"{read_type}_dir"
        with lock:
            shared_dict[count_key] = shared_dict.get(count_key, 0) + 1
            if shared_dict[count_key] > 100:
                worker_state[
                    "csvfile"
                ].flush()  # write the rows in the buffer to the csv file
                shared_dict[dir_key] = shared_dict.get(dir_key, 0) + 1
                shared_dict[count_key] = 1
                os.makedirs(
                    os.path.join(
                        output_dir, "plots", read_type, str(shared_dict[dir_key])
                    ),
                    exist_ok=True,
                )
            # Get the current timestamp
            # We append the timestamp to the name of the plot file
            # so that we can handle multiple plots for the same read
            # due to multiple alginments (secondary/supp.) in SAM files
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            plot_filepath = os.path.join(
                output_dir,
                "plots",
                read_type,
                str(shared_dict[dir_key]),
                f"{read_id}__{timestamp}.html",
            )
            # Write the data for this plot
            worker_state["csv_writer"].writerow([read_id, plot_filepath])
        # Suppress the output of align() function
        with contextlib.redirect_stdout(None):
            _, _, chunked_aln_str, alignment_score = align(
                query_seq=read_fasta, target_seq=reference, pretty_print_alns=True
            )
        plot_roi_signal(
            pod5_data,
            bam_data,
            roi_data,
            start_base_idx_in_fasta,
            end_base_idx_in_fasta,
            plot_filepath,
            chunked_aln_str,
            alignment_score,
        )
    return None


def collate_bam_pod5(
    bam_filepath: str,
    pod5_dir: str,
    num_processes: int,
    reference: str,
    cap0_pos: int,
    train_or_test: str,
    plot_signal: bool,
    output_dir: str,
) -> None:
    # 1. Initial configuration
    configure_logger(output_dir)
    logger.info("Computing bam total records")
    num_bam_records = get_total_records(bam_filepath)
    logger.info(f"Found {num_bam_records}.")

    # 2. Make index database if it does not exist
    database_path = os.path.join(output_dir, "database.db")
    if not os.path.exists(database_path):
        logger.info("Index database not found. Creating a new one.")
        index(pod5_dir, output_dir)

    # 3. If plots are requested, create the CSV file and the directories
    plots_csv_filepath = None  # Initialize the variable here

    if plot_signal:
        good_reads_plots_dir = os.path.join(output_dir, "plots", "good_reads", "0")
        bad_reads_plots_dir = os.path.join(output_dir, "plots", "bad_reads", "0")
        # create the directories if they do not exist
        os.makedirs(good_reads_plots_dir, exist_ok=True)
        os.makedirs(bad_reads_plots_dir, exist_ok=True)
        # Define the path to the CSV file within the "plots" directory
        plots_csv_filepath = os.path.join(output_dir, "plots", "plotpaths.csv")

    # 4. Initialize the database handler
    db_handler = DatabaseHandler(database_path, plots_csv_filepath)

    # 5. Set the signal handler for SIGINT
    # This is useful when the use presses Ctrl+C to stop the program.
    # The program saves the CSV file and exits.
    def signal_handler(signum: signal.Signals, frame: Any) -> None:
        print("KeyboardInterrupt detected. Closing the CSV file.")
        if db_handler.plots_csv_filepath:
            csvfile = db_handler.csvfile
            if csvfile:
                csvfile.close()
        exit(1)  # Exit the program

    signal.signal(signal.SIGINT, signal_handler)  # type: ignore

    # 6. Process the BAM file row-by-row using multiple processes
    try:
        with WorkerPool(n_jobs=num_processes, use_worker_state=True) as pool:
            pool.map(
                collate_bam_pod5_worker,
                [
                    (
                        bam_data,
                        reference,
                        cap0_pos,
                        train_or_test,
                        plot_signal,
                        output_dir,
                    )
                    for bam_data in process_bam_records(bam_filepath)
                ],
                iterable_len=num_bam_records,
                worker_init=db_handler.init_func,  # Passing method of db_handler object
                worker_exit=db_handler.exit_func,  # Passing method of db_handler object
                progress_bar=True,
            )

    except Exception as e:
        # Handle the exception (e.g., log the error)
        logger.error(f"An error occurred: {e}")
    finally:
        # Close the CSV file when an error occurs to save the progress so far
        if plots_csv_filepath:
            csvfile = db_handler.csvfile
            if csvfile:
                csvfile.close()
    return None


if __name__ == "__main__":
    bam_filepath = "/export/valenfs/data/processed_data/MinION/9_madcap/1_data/8_20231114_randomCAP1v3_rna004/1_basecall/sorted.calls.bam"
    pod5_dir = "/export/valenfs/data/raw_data/minion/20231114_randomCAP1v3_rna004"
    num_processes = 120
    reference = "GCTTTCGTTCGTCTCCGGACTTATCGCACCACCTATCCATCATCAGTACTGT"
    cap0_pos = 52
    train_or_test = "test"
    output_dir = "/export/valenfs/data/processed_data/MinION/9_madcap/1_data/8_20231114_randomCAP1v3_rna004/test_OTE_vizs6"
    plot_signal = True
    collate_bam_pod5(
        bam_filepath,
        pod5_dir,
        num_processes,
        reference,
        cap0_pos,
        train_or_test,
        plot_signal,
        output_dir,
    )
