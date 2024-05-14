"""
We can only read BAM records one at a time from a BAM file.
PySAM does not allow random access of BAM records. The module
prepares and yields the BAM record information for each read.

Author: Adnan M. Niazi
Date: 2024-02-28
"""

import os
import re
from typing import Any, Dict, Generator

import pysam


def find_hard_clipped_bases(cigar_string: str) -> tuple[int, int]:
    # Use regular expressions to find left and right hard clipping
    left_hard_clipping = re.search(r"^(\d+)H", cigar_string)
    right_hard_clipping = re.search(r"(\d+)H$", cigar_string)

    # Extract the counts from the matches
    left_count = int(left_hard_clipping.group(1)) if left_hard_clipping else 0
    right_count = int(right_hard_clipping.group(1)) if right_hard_clipping else 0

    return left_count, right_count


def generate_bam_records(
    bam_filepath: str,
) -> Generator[pysam.AlignedSegment, None, None]:
    """Yield each record from a BAM file. Also creates an index (.bai)
    file if one does not exist already.

    Params:
        bam_filepath: str
            Path to the BAM file.

    Yields:
        record: pysam.AlignedSegment
            A BAM record.
    """
    index_filepath = f"{bam_filepath}.bai"

    if not os.path.exists(index_filepath):
        pysam.index(bam_filepath)  # type: ignore

    with pysam.AlignmentFile(bam_filepath, "rb") as bam_file:
        yield from bam_file


def get_total_records(bam_filepath: str) -> int:
    """Returns the total number of records in a BAM file.

    Params:
        bam_filepath: str
            Path to the BAM file.

    Returns:
        total_records: int
            Total number of records in the BAM file.
    """
    bam_file = pysam.AlignmentFile(bam_filepath)
    total_records = sum(1 for _ in bam_file)
    bam_file.close()
    return total_records


def get_signal_info(record: pysam.AlignedSegment) -> Dict[str, Any]:
    """Returns the signal info from a BAM record.

    Params:
        record: pysam.AlignedSegment
            A BAM record.

    Returns:
        signal_info: Dict[str, Any]
            Dictionary containing signal info for a read.
    """
    signal_info = {}
    tags_dict = dict(record.tags)  # type: ignore
    moves_table = tags_dict["mv"]
    moves_step = moves_table.pop(0)
    signal_info["moves_table"] = moves_table
    signal_info["moves_step"] = moves_step
    signal_info["read_id"] = record.query_name
    signal_info["start_sample"] = tags_dict["ts"]
    signal_info["num_samples"] = tags_dict["ns"]
    signal_info["quality_score"] = tags_dict["qs"]
    signal_info["channel"] = tags_dict["ch"]
    signal_info["signal_mean"] = tags_dict["sm"]
    signal_info["signal_sd"] = tags_dict["sd"]
    signal_info["is_qcfail"] = record.is_qcfail
    signal_info["is_reverse"] = record.is_reverse
    signal_info["is_forward"] = record.is_forward
    signal_info["is_mapped"] = record.is_mapped
    signal_info["is_supplementary"] = record.is_supplementary
    signal_info["is_secondary"] = record.is_secondary
    signal_info["read_quality"] = record.qual  # type: ignore
    signal_info["read_fasta"] = record.query_sequence
    signal_info["mapping_quality"] = record.mapping_quality
    signal_info["parent_read_id"] = tags_dict.get("pi", "")
    signal_info["split_point"] = tags_dict.get("sp", 0)
    signal_info["time_stamp"] = tags_dict.get("st")
    signal_info["pod5_filename"] = tags_dict.get("fn")
    (
        signal_info["num_left_clipped_bases"],
        signal_info["num_right_clipped_bases"],
    ) = find_hard_clipped_bases(str(record.cigarstring))
    return signal_info


def process_bam_records(bam_filepath: str) -> Generator[Dict[str, Any], None, None]:
    """Top level function to process a BAM file.
    Yields signal info for each read in the BAM file.

    Params:
        bam_filepath: str
            Path to the BAM file to process.

    Yields:
        signal_info: Generator[Dict[str, Any], None, None]
            Dictionary containing signal info for a read.
    """
    for record in generate_bam_records(bam_filepath):
        yield get_signal_info(record)


if __name__ == "__main__":
    bam_filepath = "/export/valenfs/data/processed_data/MinION/9_madcap/1_data/3_20230829_randomcap01/1_basecall/calls.bam"
    for read_info in process_bam_records(bam_filepath):
        print(read_info)
