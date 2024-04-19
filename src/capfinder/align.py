"""
The module aligns a reference sequence to a read sequence using Parasail.
The module also provides functions to generate alignment strings and chunks for
pretty printing.


Author: Adnan M. Niazi
Date: 2024-02-28
"""

import re
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Tuple

import parasail

from capfinder.constants import CIGAR_CODES, CODE_TO_OP, OP_TO_CODE

CigarTuplesPySam = List[Tuple[int, int]]
CigarTuplesSam = List[Tuple[str, int]]

CIGAR_STRING_PATTERN = re.compile(r"(\d+)" + f"([{''.join(CIGAR_CODES)}])")


@dataclass
class PairwiseAlignment:
    """
    Pairwise alignment with semi-global alignment allowing for gaps at the
    start and end of the query sequence.
    """

    ref_start: int
    ref_end: int
    query_start: int
    query_end: int
    cigar_pysam: CigarTuplesPySam
    cigar_sam: CigarTuplesSam

    def __init__(
        self,
        ref_start: int,
        ref_end: int,
        query_start: int,
        query_end: int,
        cigar_pysam: CigarTuplesPySam,
        cigar_sam: CigarTuplesSam,
    ):
        """
        Initializes a PairwiseAlignment object.

        Args:
            ref_start (int): The starting position of the alignment in the reference sequence.
            ref_end (int): The ending position of the alignment in the reference sequence.
            query_start (int): The starting position of the alignment in the query sequence.
            query_end (int): The ending position of the alignment in the query sequence.
            cigar_pysam (CigarTuplesPySam): A list of tuples representing the CIGAR string in the Pysam format.
            cigar_sam (CigarTuplesSam): A list of tuples representing the CIGAR string in the SAM format.
        """
        self.ref_start = ref_start
        self.ref_end = ref_end
        self.query_start = query_start
        self.query_end = query_end
        self.cigar_pysam = cigar_pysam
        self.cigar_sam = cigar_sam


def cigartuples_from_string(cigarstring: str) -> CigarTuplesPySam:
    """
    Returns pysam-style list of (op, count) tuples from a cigarstring.
    """
    return [
        (CODE_TO_OP[m.group(2)], int(m.group(1)))
        for m in re.finditer(CIGAR_STRING_PATTERN, cigarstring)
    ]


def parasail_align(*, query: str, ref: str) -> Any:
    """
    Semi-global alignment allowing for gaps at the start and end of the query
    sequence.

    :param query: str
    :param ref: str
    :return: PairwiseAlignment
    """
    alignment_result = parasail.sg_trace_scan_32(query, ref, 10, 2, parasail.dnafull)
    return alignment_result


def trim_parasail_alignment(alignment_result: Any) -> PairwiseAlignment:
    """
    Trim the alignment result to remove leading and trailing gaps.
    """

    try:
        ref_start = 0
        ref_end = alignment_result.len_ref
        query_start = 0
        query_end = alignment_result.len_query
        fixed_start = False
        fixed_end = False

        cigar_string = alignment_result.cigar.decode.decode()
        cigar_tuples = deque(cigartuples_from_string(cigar_string))

        while not (fixed_start and fixed_end):
            first_op, first_length = cigar_tuples[0]
            if first_op in (1, 4):  # insert, soft-clip, increment query start
                query_start += first_length
                cigar_tuples.popleft()
            elif first_op == 2:  # delete, increment reference start
                ref_start += first_length
                cigar_tuples.popleft()
            else:
                fixed_start = True

            last_op, last_length = cigar_tuples[-1]
            if last_op in (1, 4):  # decrement the query end
                query_end -= last_length
                cigar_tuples.pop()
            elif last_op == 2:  # decrement the ref_end
                ref_end -= last_length
                cigar_tuples.pop()
            else:
                fixed_end = True

        cigar_pysam = list(cigar_tuples)
        cigar_sam = [(OP_TO_CODE[str(k)], v) for k, v in cigar_pysam]

        return PairwiseAlignment(
            ref_start=ref_start,
            ref_end=ref_end,
            query_start=query_start,
            query_end=query_end,
            cigar_pysam=cigar_pysam,
            cigar_sam=cigar_sam,
        )
    except IndexError as e:
        raise RuntimeError(
            "failed to find match operations in pairwise alignment"
        ) from e


def make_alignment_strings(
    query: str, target: str, alignment: PairwiseAlignment
) -> Tuple[str, str, str]:
    """
    Generate alignment strings for the given query and target sequences based on a PairwiseAlignment object.

    Args:
        query (str): The query sequence.
        target (str): The target/reference sequence.
        alignment (PairwiseAlignment): An object representing the alignment between query and target sequences.

    Returns:
        Tuple[str, str, str]: A tuple containing three strings:
            1. The aligned target sequence with gaps.
            2. The aligned query sequence with gaps.
            3. The visual representation of the alignment with '|' for matches, '/' for mismatches,
               and ' ' for gaps or insertions.
    """
    ref_start = alignment.ref_start
    ref_end = alignment.ref_end
    query_start = alignment.query_start
    cigar_sam = alignment.cigar_sam

    # Initialize the strings
    aln_query = ""
    aln_target = ""
    aln = ""
    target_count = 0
    query_count = 0

    # Handle the start
    if query_start != 0:
        aln_target += "-" * query_start
        aln_query += query[:query_start]
        aln += " " * query_start
        query_count += query_start

    if ref_start != 0:
        aln_target += target[:ref_start]
        target_count = ref_start
        aln_query += "-" * ref_start
        aln += " " * ref_start

    # Handle the middle
    for operation, length in cigar_sam:
        # Match: advance both target and query counts
        if operation in ("=", "M"):
            aln_target += target[target_count : target_count + length]
            aln_query += query[query_count : query_count + length]
            aln += "|" * length
            target_count += length
            query_count += length

        # Insertion: advance query count only
        elif operation == "I":
            aln_target += "-" * length
            aln_query += query[query_count : query_count + length]
            aln += " " * length
            query_count += length

        # Deletion or gaps: advance target count only
        # see: https://jef.works/blog/2017/03/28/CIGAR-strings-for-dummies/
        elif operation in ("D", "N"):
            aln_target += target[target_count : target_count + length]
            aln_query += "-" * length
            aln += " " * length
            target_count += length

        # Mismatch: advance both target and query counts
        elif operation == "X":
            aln_target += target[target_count : target_count + length]
            aln_query += query[query_count : query_count + length]
            aln += "/" * length
            target_count += length
            query_count += length

    # Handle the end
    ql = len(query)
    tl = len(target)
    target_remainder = tl - ref_end
    if target_remainder:
        aln_target += target[target_count:]
        aln_query += target_remainder * "-"
        aln += target_remainder * " "

    end_dash_len = ql - query_count
    if end_dash_len:
        aln_target += "-" * end_dash_len
        aln_query += query[query_count:]
        aln += " " * end_dash_len
        query_count += query_start

    return aln_query, aln, aln_target


def make_alignment_chunks(
    target: str, query: str, alignment: str, chunk_size: int
) -> str:
    """
    Divide three strings (target, query, and alignment) into chunks of the specified length
    and print them as triplets with the specified prefixes and a one-line gap between each triplet.

    Args:
        target (str): The target/reference string.
        query (str): The query string.
        alignment (str): The alignment string.
        chunk_size (int): The desired chunk size.

    Returns:
        aln_string (str): The aligned strings in chunks with the specified prefix.
    """
    # Check if chunk size is valid
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero")

    # Divide the strings into chunks
    target_chunks = [
        target[i : i + chunk_size] for i in range(0, len(target), chunk_size)
    ]
    query_chunks = [query[i : i + chunk_size] for i in range(0, len(query), chunk_size)]
    alignment_chunks = [
        alignment[i : i + chunk_size] for i in range(0, len(alignment), chunk_size)
    ]

    # Iterate over the triplets and print them
    aln_string = ""
    for t_chunk, q_chunk, a_chunk in zip(target_chunks, query_chunks, alignment_chunks):
        aln_string += f"QRY: {q_chunk}\n"
        aln_string += f"ALN: {a_chunk}\n"
        aln_string += f"REF: {t_chunk}\n\n"

    return aln_string


# Main function call
def align(
    query_seq: str, target_seq: str, pretty_print_alns: bool
) -> Tuple[str, str, str, int]:
    """
    Main function call to align two sequences and print the alignment.

    Args:
        query_seq (str): The query sequence.
        target_seq (str): The target/reference sequence.
        pretty_print_alns (bool): Whether to print the alignment in a pretty format.

    Returns:
        Tuple[str, str, str]: A tuple containing three strings:
            1. The aligned query sequence with gaps.
            2. The visual representation of the alignment with '|' for matches, '/' for mismatches,
                and ' ' for gaps or insertions.
            3. The aligned target sequence with gaps.
            4. The alignment score.

    """
    # Perform the alignment
    alignment = parasail_align(query=query_seq, ref=target_seq)
    alignment_score = alignment.score
    alignment = trim_parasail_alignment(alignment)
    # Generate the aligned strings
    aln_query, aln, aln_target = make_alignment_strings(
        query_seq, target_seq, alignment
    )

    # Print the alignment in a pretty format if required
    if pretty_print_alns:
        print("Alignment score:", alignment_score)
        chunked_aln_str = make_alignment_chunks(
            aln_target, aln_query, aln, chunk_size=40
        )
        print(chunked_aln_str)
        return (
            "",
            "",
            chunked_aln_str,
            alignment_score,
        )
    else:
        return aln_query, aln, aln_target, alignment_score


if __name__ == "__main__":
    query_seq = "GAAAGAGATAGAGTTACCATTCTATCATAATTAATATTATCGTCTACAACACCATCCATCCTTCCATCAACTAATTCTACCTTTACTG"
    target_seq = "CCGGACTTATCGCACCACCTATCCATCATCAGTACTGTNNNNNNCCTGGTAACTGGGAC"

    pretty_print_alns = True
    align(query_seq=query_seq, target_seq=target_seq, pretty_print_alns=True)
