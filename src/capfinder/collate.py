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

import numpy as np
from loguru import logger
from mpire import WorkerPool

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
from capfinder.utils import map_cap_int_to_name, open_database

csv.field_size_limit(4096 * 4096)  # Set a higher field size limit (e.g., 1MB)

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
        self,
        cap_class: int,
        num_processes: int,
        database_path: str,
        plots_csv_filepath: Union[str, None],
        output_dir: str,
    ) -> None:
        """Initializes the index database handler"""
        self.cap_class = cap_class
        self.database_path = database_path
        self.plots_csv_filepath = plots_csv_filepath
        self.num_processes = num_processes
        self.output_dir = output_dir

        # Open the plots CSV file in append mode
        if self.plots_csv_filepath:
            self.csvfile = open(self.plots_csv_filepath, "a", newline="")

    def init_func(self, worker_id: int, worker_state: Dict[str, Any]) -> None:
        """Opens the database connection and CSV files"""

        # 1. Open the database connection and cursor
        worker_state["db_connection"], worker_state["db_cursor"] = open_database(
            self.database_path
        )

        # 2. Write the header row to the plots CSV file
        if self.plots_csv_filepath:
            csv_writer = csv.writer(self.csvfile)
            csv_writer.writerow(["read_id", "plot_filepath"])
            worker_state["csv_writer"] = csv_writer
            worker_state["csvfile"] = self.csvfile  # Store csvfile in worker_state

        # Define paths to data and metadata CSV files
        data_file_path = os.path.join(self.output_dir, f"data_tmp_{worker_id}.csv")
        metadata_file_path = os.path.join(
            self.output_dir, f"metadata_tmp_{worker_id}.csv"
        )

        # 3. Open data_file_path in append mode and write the header row if the file is empty
        data_file = open(data_file_path, "a", newline="")
        data_writer = csv.writer(data_file)
        if data_file.tell() == 0:  # Check if the file is empty
            data_writer.writerow(
                ["read_id", "cap_class", "timeseries"]
            )  # Replace with your actual header

        # Save the data file path and writer to worker_state
        worker_state["data_file"] = data_file
        worker_state["data_writer"] = data_writer

        # 4. Open metadata_file_path in append mode and write the header row if the file is empty
        metadata_file = open(metadata_file_path, "a", newline="")
        metadata_writer = csv.writer(metadata_file)
        if metadata_file.tell() == 0:  # Check if the file is empty
            metadata_writer.writerow(
                [
                    "read_id",
                    "parent_read_id",
                    "pod5_file",
                    "read_type",
                    "roi_fasta",
                    "roi_start",
                    "roi_end",
                    "fasta_length",
                    "fasta",
                ]
            )  # Replace with your actual header

        # Save the metadata file path and writer to worker_state
        worker_state["metadata_file"] = metadata_file
        worker_state["metadata_writer"] = metadata_writer

    def exit_func(self, worker_id: int, worker_state: Dict[str, Any]) -> None:
        """Closes the database connection and the CSV file."""
        conn = worker_state.get("db_connection")
        if conn:
            conn.close()

        # Close the plots csv file
        csvfile = worker_state.get("csvfile")
        if self.plots_csv_filepath and csvfile:
            csvfile.close()

        # Close the data file
        worker_state["data_file"].close()

        # Close the metadata file
        worker_state["metadata_file"].close()

    def merge_data(self) -> None:
        """Merges the data and metadata CSV files."""
        self._merge_csv_files(data_or_metadata="data")
        self._merge_csv_files(data_or_metadata="metadata")

    def _merge_csv_files(self, data_or_metadata: str) -> None:
        """Merges the data and metadata CSV files.

        Args:
            data_or_metadata (str): Whether to merge data or metadata CSV files.

        Returns:
            None
        """
        cap_name = map_cap_int_to_name(self.cap_class)
        data_path = os.path.join(self.output_dir, f"{data_or_metadata}__{cap_name}.csv")
        # delete if the file already exists
        if os.path.exists(data_path):
            logger.info(f"Overwriting existing {data_or_metadata} CSV file.")
            os.remove(data_path)
        with open(data_path, "w", newline="") as output_csv:
            writer = csv.writer(output_csv)
            for i in range(self.num_processes):
                ind_csv_file = os.path.join(
                    self.output_dir, f"{data_or_metadata}_tmp_{i}.csv"
                )
                # Open each CSV file and read its contents
                with open(ind_csv_file) as input_csv:
                    reader = csv.reader(input_csv)

                    # If it's the first file, write the header to the output file
                    if i == 0:
                        header = next(reader)
                        writer.writerow(header)
                    else:
                        next(reader)

                    # Write the remaining rows to the output file
                    for row in reader:
                        writer.writerow(row)
                os.remove(ind_csv_file)
        logger.info(f"Successfully merged {data_or_metadata} CSV file.")


def collate_bam_pod5_worker(
    worker_id: int,
    worker_state: Dict[str, Any],
    bam_data: Dict[str, Any],
    reference: str,
    cap_class: int,
    cap0_pos: int,
    train_or_test: str,
    plot_signal: bool,
    output_dir: str,
) -> None:
    """Worker function that collates information from POD5 and BAM file, finds the
    FASTA coordinates of  region of interest (ROI) and and extracts its signal.

    Params:
        worker_id: int
            Worker ID.
        worker_state: dict
            Dictionary containing the database connection and cursor.
        bam_data: dict
            Dictionary containing the BAM record information.
        reference: str
            Reference sequence.
        cap_class: int
            Class label for the RNA cap
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

    # Check that the read is not empty
    if read_fasta is None:
        logger.warning(f"Read {read_id} has empty FASTA. Skipping the read.")
        return None

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

    # 8. Find if a read is good or bad
    read_type = (
        "bad_reads"
        if start_base_idx_in_fasta is None and end_base_idx_in_fasta is None
        else "good_reads"
    )

    # 9. Save the train/test and metadata information
    # We need to store train/test data only for the good reads
    precision = 8

    # Define a vectorized function for formatting (if applicable)
    def format_value(x: float) -> str:
        return f"{x:.{precision}f}"

    vectorized_formatter = np.vectorize(format_value)

    if read_type == "good_reads":
        roi_signal: np.ndarray = roi_data["roi_signal"]
        if roi_signal.size == 0:
            read_type = "bad_reads"
        else:
            timeseries_str = ",".join(vectorized_formatter(roi_data["roi_signal"]))
            worker_state["data_writer"].writerow([read_id, cap_class, timeseries_str])

    # We need to store metadata for all reads (good and bad)
    if read_fasta is not None:
        read_length = len(read_fasta)
    else:
        read_length = 0

    worker_state["metadata_writer"].writerow(
        [
            read_id,
            parent_read_id,
            pod5_filepath,
            read_type.rstrip("s"),
            roi_data["roi_fasta"],
            roi_data["start_base_idx_in_fasta"],
            roi_data["end_base_idx_in_fasta"],
            read_length,
            read_fasta,
        ]
    )

    # 10. Plot the ROI signal if requested
    # Save plot in directories of 100 plots each separated into
    # good and bad categories. Good reads mean those that have
    # the OTE in them and bad reads mean those that do not.
    if plot_signal:
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
    cap_class: int,
    cap0_pos: int,
    train_or_test: str,
    plot_signal: bool,
    output_dir: str,
) -> None:
    """
    Collates information from the BAM file and the POD5 files,
    aligns OTE to extracts the signal for the
    region of interest (ROI) for training or testing purposes.
    It also plots the ROI signal if requested.

    Params:
    -------
    bam_filepath : str
        Path to the BAM file.
    pod5_dir : str
        Path to the directory containing the POD5 files.
    num_processes : int
        Number of processes to use for parallel processing.
    reference : str
        Reference sequence.
    cap_class : int
        Class label for the RNA cap.

        Valid options for `cap_class` and their corresponding meanings are:

        * 0: "cap_0" - Represents the absence of a specific modification at the 5' end.
        * 1: "cap_1" (default) - The most common cap structure, typically containing a 7-methylguanosine (m7G) modification.
        * 2: "cap_2" - Less common cap structure, often containing a 2'-O-methyl modification.
        * 3: "cap_2-1" - Combination of cap_2 and cap_1 modifications.
        * 4: "cap_TMG" - Cap structure with trimethylguanosine (TMG) modification.
        * 5: "cap_NAD" - Cap structure with nicotinamide adenine dinucleotide (NAD) modification.
        * 6: "cap_FAD" - Cap structure with flavin adenine dinucleotide (FAD) modification.
        * -99: "cap_unknown" - Indicates an unknown or undetermined cap structure.

    cap0_pos : int
        Position of the cap N1 base in the reference sequence (0-based).

    train_or_test : str
        Whether to extract ROI for training or testing.
        Valid options are'train' or 'test'.

    plot_signal : bool
        Whether to plot the ROI signal.

    output_dir : str
        Path to the output directory.

        Will Contain:
        * A CSV file (data__cap_x.csv) containing the extracted ROI signal data.
        * A CSV file (metadata__cap_x.csv) containing the complete metadata information.
        * A log file (capfinder_vXYZ_datatime.log) containing the logs of the program.
        * A directory (plots) containing the plots of the ROI signal if plot_signal is set to True.
            * good_reads: Directory that contains the plots for the good reads.
            * bad_reads: Directory that contains the plots for the bad reads.
            * plotpaths.csv: CSV file containing the paths to the plots based on the read ID.

    Returns:
    --------
    None
    """
    # 1. Initial configuration
    configure_logger(output_dir)
    logger.info("Computing BAM total records...")
    num_bam_records = get_total_records(bam_filepath)
    logger.info(f"Found {num_bam_records} BAM records!")

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
    db_handler = DatabaseHandler(
        cap_class, num_processes, database_path, plots_csv_filepath, output_dir
    )

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
        logger.info("Processing BAM file using multiple processes...")
        with WorkerPool(
            n_jobs=num_processes, use_worker_state=True, pass_worker_id=True
        ) as pool:
            pool.map(
                collate_bam_pod5_worker,
                [
                    (
                        bam_data,
                        reference,
                        cap_class,
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

        # Merge the data and metadata CSV files
        db_handler.merge_data()
        logger.success("All steps fininshed successfully!")
    return None


if __name__ == "__main__":
    bam_filepath = "/export/valenfs/data/processed_data/MinION/9_madcap/1_data/8_20231114_randomCAP1v3_rna004/1_basecall_subset/sorted.calls.bam"
    pod5_dir = "/export/valenfs/data/raw_data/minion/2024_cap_ligation_data_v3_oligo/20240521_cap1/20231114_randomCAP1v3_rna004/"
    num_processes = 3
    reference = "GCTTTCGTTCGTCTCCGGACTTATCGCACCACCTATCCATCATCAGTACTGT"
    cap0_pos = 52
    train_or_test = "test"
    output_dir = "/export/valenfs/data/processed_data/MinION/9_madcap/1_data/8_20231114_randomCAP1v3_rna004/test_OTE_vizs_jun5"
    plot_signal = True
    cap_class = 1
    collate_bam_pod5(
        bam_filepath,
        pod5_dir,
        num_processes,
        reference,
        cap_class,
        cap0_pos,
        train_or_test,
        plot_signal,
        output_dir,
    )

    # bam_filepath = "/export/valenfs/data/processed_data/MinION/9_madcap/1_data_old/7_20231025_capjump_rna004/2_alignment/1_basecalled/sorted.calls.bam"
    # # bam_filepath = "/export/valenfs/data/processed_data/MinION/9_madcap/1_data/7_20231025_capjump_rna004/1_basecall_subset/sorted.calls.bam"
    # pod5_dir = "/export/valenfs/data/raw_data/minion/7_20231025_capjump_rna004/20231025_CapJmpCcGFP_RNA004/20231025_1536_MN29576_FAX71885_5b8c42a6"
    # num_processes = 120
    # reference = "TTCGTCTCCGGACTTATCGCACCACCTAT"
    # cap0_pos = 43  # 59
    # train_or_test = "test"
    # output_dir = "/export/valenfs/data/processed_data/MinION/9_madcap/1_data/7_20231025_capjump_rna004/output_full12"
    # plot_signal = True
    # cap_class = 1
    # collate_bam_pod5(
    #     bam_filepath,
    #     pod5_dir,
    #     num_processes,
    #     reference,
    #     cap_class,
    #     cap0_pos,
    #     train_or_test,
    #     plot_signal,
    #     output_dir,
    # )
