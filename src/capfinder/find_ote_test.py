"""
The module contains the code to find OTE sequence in test data --
where we only know the context to the left of the NNNNNN region --
and its location with high-confidence.
The modules can process one read at a time or all reads in a FASTQ
file or folder of FASTQ files.

Author: Adnan M. Niazi
Date: 2024-02-28
"""

import contextlib
import csv
import os
from typing import Any, Dict, List, Tuple

from Bio import SeqIO
from mpire import WorkerPool

from capfinder.align import align
from capfinder.utils import file_opener

# Length of the NNNNNN region in the training reference
N_REGION_LEN = 6

# Length of the flanking region before the cap
BEFORE_N_REGION_WINDOW_LEN = 10

# Number of flanking bases on each of the left and right
# right sides of N1N2 cap bases
NUM_CAP_FLANKING_BASES = 5


def make_coordinates(aln_str: str, ref_str: str) -> List[int]:
    """
    Walk along the alignment string and make an incrementing index
    where there is a match, mismatch, and deletions. For gaps in
    the alignment string, it output a -1 in the index list.

    Args:
        aln_str (str): The alignment string.
        ref_str (str): The reference string.

    Returns:
        coord_list (list): A list of indices corresponding to the alignment string.
    """
    # Make index coordinates along the alignment string
    coord_list = []
    cnt = 0
    for idx, aln_chr in enumerate(aln_str):
        if aln_chr != " ":
            coord_list.append(cnt)
            cnt += 1
        else:
            if ref_str[idx] != "-":  # handle deletions
                coord_list.append(cnt)
                cnt += 1
            else:
                coord_list.append(-1)

    # Go in reverse in the coord_list, and put -1 of all the places
    # where there is a gap in the alignment string. Break out when the
    # first non-gap character is encountered.
    for idx in range(len(coord_list) - 1, -1, -1):
        if aln_str[idx] == " ":
            coord_list[idx] = -1
        else:
            break
    return coord_list


def cnt_match_mismatch_gaps(aln_str: str) -> Tuple[int, int, int]:
    """
    Takes an alignment string and counts the number of matches, mismatches, and gaps.

    Args:
        aln_str (str): The alignment string.

    Returns:
        match_cnt (int): The number of matches in the alignment string.
        mismatch_cnt (int): The number of mismatches in the alignment string.
        gap_cnt (int): The number of gaps in the alignment string.
    """
    match_cnt = 0
    mismatch_cnt = 0
    gap_cnt = 0
    for aln_chr in aln_str:
        if aln_chr == "|":
            match_cnt += 1
        elif aln_chr == "/":
            mismatch_cnt += 1
        elif aln_chr == " ":
            gap_cnt += 1
    return match_cnt, mismatch_cnt, gap_cnt


def has_good_aln_in_n_region(match_cnt: int, mismatch_cnt: int, gap_cnt: int) -> bool:
    """
    Checks if the alignment in the NNNNNN region is good.

    Args:
        match_cnt (int): The number of matches in the NNNNNN region.
        mismatch_cnt (int): The number of mismatches in the NNNNNN region.
        gap_cnt (int): The number of gaps in the NNNNNN region.

    Returns:
        bool: True if the alignment in the NNNNNN region is good, False otherwise.
    """
    # For a good alignment in NNNNNN region, the number of mismatches should be
    # greater than the number of gaps
    if mismatch_cnt >= gap_cnt:
        return True
    else:
        return False


def has_good_aln_in_5prime_flanking_region(
    match_cnt: int, mismatch_cnt: int, gap_cnt: int
) -> bool:
    """
    Checks if the alignment in the flanking region before the cap is good.

    Args:
        match_cnt (int): The number of matches in the flanking region.
        mismatch_cnt (int): The number of mismatches in the flanking region.
        gap_cnt (int): The number of gaps in the flanking region.

    Returns:
        bool: True if the alignment in the flanking region is good, False otherwise.
    """
    if (mismatch_cnt > match_cnt) or (gap_cnt > match_cnt):
        return False
    else:
        return True


def process_read(record: Any, reference: str, cap0_pos: int) -> Dict[str, Any]:
    """
    Process a single read from a FASTQ file. The function alnigns the read to the reference,
    and checks if the alignment in the NNNNNN region and the flanking regions is good. If the
    alignment is good, then the function returns the read ID, alignment score, and the
    positions of the left flanking region, cap0 base, and the right flanking region in the
    read's FASTQ sequence. If the alignment is bad, then the function returns the read ID,
    alignment score, and the reason why the alignment is bad.

    Args:
        record (SeqRecord): A single read from a FASTQ file.
        reference (str): The reference sequence to align the read to.
        cap0_pos (int): The position of the first cap base in the reference sequence (0-indexed).

    Returns:
        out_ds (dict): A dictionary containing the following keys:
            read_id (str): The identifier of the sequence read.
            read_type (str): The type of the read, which can be 'good' or 'bad'
            reason (str or None): The reason for the failed alignment, if available.
            alignment_score (float): The alignment score for the read.
            left_flanking_region_start_fastq_pos (int or None): The starting position of the left flanking region
            in the FASTQ file, if available.
            cap_n1_minus_1_read_fastq_pos (int or None): The position of the caps N1 base in the FASTQ file (0-indexed), if available.
            right_flanking_region_start_fastq_pos (int or None): The starting position of the right flanking region
            in the FASTQ file, if available.
    """
    # Get alignment
    sequence = str(record.seq)
    fasta_length = len(sequence)

    with contextlib.redirect_stdout(None):
        qry_str, aln_str, ref_str, aln_score = align(
            query_seq=sequence, target_seq=reference, pretty_print_alns=False
        )

    # define a data structure to return when the read OTE is not found
    out_ds_failed = {
        "read_id": record.id,
        "read_type": "bad",
        "reason": None,
        "alignment_score": aln_score,
        "left_flanking_region_start_fastq_pos": None,
        "cap_n1_minus_1_read_fastq_pos": None,
        "right_flanking_region_start_fastq_pos": None,
        "roi_fasta": None,
        "fasta_length": fasta_length,
    }

    # For low quality alignments, return None
    if aln_score < 20:
        out_ds_failed["reason"] = "low_aln_score"
        return out_ds_failed

    # Make index coordinates along the reference
    coord_list = make_coordinates(aln_str, ref_str)

    # Check if the the first base 5 prime of the cap N1 base is in
    # the coordinates list. If not then the alignment did not
    # even reach the cap, so it is a bad read then.
    try:
        cap_n1_minus_1_idx = coord_list.index(cap0_pos - 1)
    except Exception:
        out_ds_failed["reason"] = "aln_does_not_reach_the_cap_base"
        return out_ds_failed

    # 2. Define regions in which to check for good alignment
    before_nnn_region = (
        cap_n1_minus_1_idx - BEFORE_N_REGION_WINDOW_LEN,
        cap_n1_minus_1_idx + 1,
    )

    # 3. Extract alignment strings for each region
    aln_str_before_nnn_region = aln_str[before_nnn_region[0] : before_nnn_region[1]]

    # 4. Count matches, mismatches, and gaps in each region
    bn_match_cnt, bn_mismatch_cnt, bn_gap_cnt = cnt_match_mismatch_gaps(
        aln_str_before_nnn_region
    )

    # 5. Is there a good alignment region flanking 5' of the cap?
    has_good_aln_before_n_region = has_good_aln_in_5prime_flanking_region(
        bn_match_cnt, bn_mismatch_cnt, bn_gap_cnt
    )
    if not (has_good_aln_before_n_region):
        out_ds_failed["reason"] = "bad_alignment_before_the_cap"
        return out_ds_failed

    # Find the position of cap N1-1 base in read's sequence (0-based indexing)
    cap_n1_minus_1_read_fastq_pos = (
        qry_str[:cap_n1_minus_1_idx].replace("-", "").count("")
    )

    # To reach the 5' end of the left flanking region, we need to find
    # to over the alignment string on count the matches and mismatches
    # but not the gaps
    idx = cap_n1_minus_1_idx
    cnt = 0
    while True:
        if cnt == NUM_CAP_FLANKING_BASES:
            break
        else:
            if aln_str[idx] != " ":
                cnt += 1
                left_flanking_region_start_idx = idx
        idx -= 1

    left_flanking_region_start_fastq_pos = (
        qry_str[:left_flanking_region_start_idx].replace("-", "").count("") - 1
    )
    right_flanking_region_start_fastq_pos = (
        cap_n1_minus_1_read_fastq_pos + NUM_CAP_FLANKING_BASES + 1
    )

    roi_fasta = sequence[
        left_flanking_region_start_fastq_pos:right_flanking_region_start_fastq_pos
    ]

    out_ds_passed = {
        "read_id": record.id,
        "read_type": "good",
        "reason": "good_alignment_in_cap-flanking_regions",
        "alignment_score": aln_score,
        "left_flanking_region_start_fastq_pos": left_flanking_region_start_fastq_pos,
        "cap_n1_minus_1_read_fastq_pos": cap_n1_minus_1_read_fastq_pos,
        "right_flanking_region_start_fastq_pos": right_flanking_region_start_fastq_pos,
        "roi_fasta": roi_fasta,
        "fasta_length": fasta_length,
    }

    # A fix to avoid outputting blank ROI when the read is shorter than
    # computed ROI coordinates
    sl = len(sequence)
    if (left_flanking_region_start_fastq_pos) > sl or (
        right_flanking_region_start_fastq_pos
    ) > sl:
        output = out_ds_failed
    else:
        output = out_ds_passed

    return output


def process_fastq_file(
    fastq_filepath: str,
    reference: str,
    cap0_pos: int,
    num_processes: int,
    output_folder: str,
) -> None:
    """
    Process a single FASTQ file. The function reads the FASTQ file, and processes each read in parallel.

    Args:
        fastq_filepath (str): The path to the FASTQ file.
        reference (str): The reference sequence to align the read to.
        num_processes (int): The number of processes to use for parallel processing.
        output_folder (str): The folder where worker output files will be stored.

    Returns:
        None
    """

    # Make output file name
    directory, filename = os.path.split(fastq_filepath)
    filename_no_extension, extension = os.path.splitext(filename)
    os.path.join(output_folder, f"{filename_no_extension}.txt")

    with file_opener(fastq_filepath) as fastq_file:
        records = list(SeqIO.parse(fastq_file, "fastq"))
        total_records = len(records)

        with WorkerPool(n_jobs=num_processes) as pool:
            results = pool.map(
                process_read,
                [(record, reference, cap0_pos) for record in records],
                iterable_len=total_records,
                progress_bar=True,
            )
            write_csv(
                results,
                output_filepath=os.path.join(
                    output_folder,
                    filename_no_extension + "_test_ote_search_results.csv",
                ),
            )


def write_csv(resutls_list: List[dict], output_filepath: str) -> None:
    """
    Take a list of dictionaries and write them to a CSV file.

    Args:
        resutls_list (list): A list of dictionaries.
        output_filepath (str): The path to the output CSV file.

    Returns:
        None
    """
    # Specify the CSV column headers based on the dictionary keys
    fieldnames = resutls_list[0].keys()

    # Create and write to the CSV file
    with open(output_filepath, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write the data rows
        writer.writerows(resutls_list)


def process_fastq_folder(
    folder_path: str,
    reference: str,
    cap0_pos: int,
    num_processes: int,
    output_folder: str,
) -> None:
    """
    Process all FASTQ files in a folder. The function reads all FASTQ files in a folder,
    and feeds one FASTQ at a time which to a prcessing function that processes reads in this
    FASTQ file in parallel.

    Args:
        folder_path (str): The path to the folder containing FASTQ files.
        reference (str): The reference sequence to align the read to.
        cap0_pos (int): The position of the first cap base (N1) in the reference sequence (0-indexed).
        num_processes (int): The number of processes to use for parallel processing.
        output_folder (str): The folder where worker output files will be stored.

    Returns:
        None
    """
    # List all files in the folder
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith((".fastq", ".fastq.gz")):
                file_path = os.path.join(root, file_name)
                process_fastq_file(
                    file_path, reference, cap0_pos, num_processes, output_folder
                )


def dispatcher(
    input_path: str,
    reference: str,
    cap0_pos: int,
    num_processes: int,
    output_folder: str,
) -> None:
    """
    Check if the input path is a file or folder, and call the appropriate function to process the input.

    Args:
        input_path (str): The path to the FASTQ file or folder.
        reference (str): The reference sequence to align the read to.
        num_processes (int): The number of processes to use for parallel processing.
        output_folder (str): The folder where worker output files will be stored.

    Returns:
        None
    """
    if os.path.isfile(input_path):
        process_fastq_file(
            input_path, reference, cap0_pos, num_processes, output_folder
        )
    elif os.path.isdir(input_path):
        process_fastq_folder(
            input_path, reference, cap0_pos, num_processes, output_folder
        )
    else:
        raise ValueError("Error! Invalid path type. Path must be a file or folder.")


def find_ote_test(
    input_path: str,
    reference: str,
    cap0_pos: int,
    num_processes: int,
    output_folder: str,
) -> None:
    """
    Main function to process a FASTQ file or folder of FASTQ files to find OTEs
    in the reads. The function is suitable only for testing data where only the OTE
    sequence is known and the N1N2 cap bases and any bases 3' of them are unknown.

    Args:
        input_path (str): The path to the FASTQ file or folder.
        reference (str): The reference sequence to align the read to.
        num_processes (int): The number of processes to use for parallel processing.
        output_folder (str): The folder where worker output files will be stored.
    Returns:
        None
    """
    dispatcher(input_path, reference, cap0_pos, num_processes, output_folder)


if __name__ == "__main__":
    # Example usage:
    # Specify the path to the FASTQ file or folder
    path = "/export/valenfs/data/raw_data/minion/7_20231025_capjump_rna004/20231025_CapJmpCcGFP_RNA004/20231025_1536_MN29576_FAX71885_5b8c42a6"  # Replace with your file or folder path

    # Specify the number of processes for parallel processing
    num_processes = 100  # Adjust as needed

    # Specify the folder where worker output files will be stored
    output_folder = "/export/valenfs/data/processed_data/MinION/9_madcap/1_data/7_20231025_capjump_rna004/3_processed_output"  # Replace with your desired folder path

    # Define the alignment reference
    reference = "GAGAUGAGCUUUCGUUCGUCUCCGGACUUAUCGCACCACCUAUCC"
    cap0_pos = 45

    # Call the function to process the FASTQ file or folder
    find_ote_test(path, reference, cap0_pos, num_processes, output_folder)
