"""
This module helps us to visualize the alignments of reads to a reference sequence.
The module reads a FASTQ file or folder of FASTQ files,
processes each read in parallel, and writes the output to a file.
The output file contains the read ID, average quality, sequence,
alignment score, and alignment string.

This module is useful in understandig the output of Parasail alignment.

Author: Adnan M. Niazi
Date: 2024-02-28
"""

import contextlib
import os
from typing import Any, List, Sequence, Union

from Bio import SeqIO
from mpire import WorkerPool

from capfinder.align import align
from capfinder.utils import file_opener


def calculate_average_quality(quality_scores: Sequence[Union[int, float]]) -> float:
    """
    Calculate the average quality score for a read.
    Args:
        quality_scores (Sequence[Union[int, float]]): A list of quality scores for a read.
    Returns:
        average_quality (float): The average quality score for a read.
    """
    average_quality = sum(quality_scores) / len(quality_scores)
    return average_quality


def process_read(record: Any, reference: str) -> str:
    """
    Process a single read from a FASTQ file. The function calculates average read quality,
    alignment score, and alignment string. The output is a string that can be written to a file.

    Args:
        record (Any): A single read from a FASTQ file.
        reference (str): The reference sequence to align the read to.
    Returns:
        output_string (str): A string containing the read ID, average quality, sequence,
                            alignment score, and alignment string.
    """
    read_id = record.id
    quality_scores = record.letter_annotations["phred_quality"]
    average_quality = round(calculate_average_quality(quality_scores))
    sequence = str(record.seq)
    with contextlib.redirect_stdout(None):
        _, _, chunked_aln_str, alignment_score = align(
            query_seq=sequence, target_seq=reference, pretty_print_alns=True
        )

    output_string = f">{read_id} {average_quality:.0f}\n{sequence}\n\n"
    output_string += f"Alignment Score: {alignment_score}\n"
    output_string += f"{chunked_aln_str}\n"

    return output_string


def write_ouput(output_list: List[str], output_filepath: str) -> None:
    """
    Write a list of strings to a file.

    Args:
        output_list (list): A list of strings to write to a file.
        output_filepath (str): The path to the output file.

    Returns:
        None
    """
    if os.path.exists(output_filepath):
        os.remove(output_filepath)
    with open(output_filepath, "a") as f:
        f.writelines("\n".join(output_list))


def process_fastq_file(
    fastq_filepath: str, reference: str, num_processes: int, output_folder: str
) -> None:
    """
    Process a single FASTQ file. The function reads the FASTQ file, processes each read in parallel.
    The output is a file containing the read ID, average quality, sequence, alignment score, and alignment string.

    Args:
        fastq_filepath (str): The path to the FASTQ file.
        reference (str): The reference sequence to align the read to.
        num_processes (int): The number of processes to use for parallel processing.
        output_folder (str): The folder where the output file will be stored.

    Returns:
        None
    """

    # Make output file name
    # Make output_folder if it does not exist already
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    directory, filename = os.path.split(fastq_filepath)
    filename_no_extension, extension = os.path.splitext(filename)
    output_filepath = os.path.join(
        output_folder, f"{filename_no_extension}_alignments.txt"
    )

    with file_opener(fastq_filepath) as fastq_file:
        records = list(SeqIO.parse(fastq_file, "fastq"))
        total_records = len(records)

        with WorkerPool(n_jobs=num_processes) as pool:
            results = pool.map(
                process_read,
                [(item, reference) for item in records],
                iterable_len=total_records,
                progress_bar=True,
            )
            write_ouput(results, output_filepath)


def process_fastq_folder(
    folder_path: str, reference: str, num_processes: int, output_folder: str
) -> None:
    """
    Process all FASTQ files in a folder. The function reads all FASTQ files in a folder, processes each read in parallel.

    args:
        folder_path (str): The path to the folder containing FASTQ files.
        reference (str): The reference sequence to align the read to.
        num_processes (int): The number of processes to use for parallel processing.
        output_folder (str): The folder where the output file will be stored.

    returns:
        None
    """
    # List all files in the folder
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith((".fastq", ".fastq.gz")):
                file_path = os.path.join(root, file_name)
                process_fastq_file(file_path, reference, num_processes, output_folder)


def process_fastq_path(
    path: str, reference: str, num_processes: int, output_folder: str
) -> None:
    """
    Process a FASTQ file or folder of FASTQ files based on the provided path.

    args:
        path (str): The path to the FASTQ file or folder.
        reference (str): The reference sequence to align the read to.
        num_processes (int): The number of processes to use for parallel processing.
        output_folder (str): The folder where the output file will be stored.

    returns:
        None
    """
    if os.path.isfile(path):
        # Process a single FASTQ file
        process_fastq_file(path, reference, num_processes, output_folder)
    elif os.path.isdir(path):
        # Process all FASTQ files in a folder
        process_fastq_folder(path, reference, num_processes, output_folder)
    else:
        print("Invalid path. Please provide a valid FASTQ file or folder path.")


def visualize_alns(
    path: str, reference: str, num_processes: int, output_folder: str
) -> None:
    """
    Main function to visualize alignments.

    Args:
        path (str): The path to the FASTQ file or folder.
        reference (str): The reference sequence to align the read to.
        num_processes (int): The number of processes to use for parallel processing.
        output_folder (str): The folder where the output file will be stored.

    Returns:
        None
    """
    process_fastq_path(path, reference, num_processes, output_folder)


if __name__ == "__main__":
    # Example usage:
    # Specify the path to the FASTQ file or folder
    path = "/export/valenfs/data/processed_data/MinION/9_madcap/1_data/2_20230829_randomcap02/minimap_aligments/attempt1"  # Replace with your file or folder path

    # Specify the number of processes for parallel processing
    num_processes = 1  # Adjust as needed

    # Specify the folder where worker output files will be stored
    output_folder = "/export/valenfs/data/processed_data/MinION/9_madcap/1_data/2_20230829_randomcap02/minimap_aligments/attempt1"  # Replace with your desired folder path

    # Define the alignment reference
    reference = "CCGGACTTATCGCACCACCTATCCATCATCAGTACTGTNNNNNNCCTGGTAACTGGGAC"

    # Call the function to process the FASTQ file or folder
    visualize_alns(path, reference, num_processes, output_folder)
