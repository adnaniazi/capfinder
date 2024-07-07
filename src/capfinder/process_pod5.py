"""
Given read_id and pod5 filepath, this file preprocesses the signal data,
and extracts the signal data for a region of interest (ROI).

Author: Adnan M. Niazi
Date: 2024-02-28
"""

from typing import Any, Dict, Optional, TypedDict

import numpy as np
import numpy.typing as npt
import pod5 as p5


class ROIData(TypedDict):
    roi_fasta: Optional[str]
    roi_signal: np.ndarray
    signal_start: Optional[int]
    signal_end: Optional[int]
    plot_signal: np.ndarray
    roi_signal_for_plot: Optional[Any]
    base_locs_in_signal: npt.NDArray[np.int32]
    start_base_idx_in_fasta: Optional[int]
    end_base_idx_in_fasta: Optional[int]
    read_id: Optional[str]


def pull_read_from_pod5(read_id: str, pod5_filepath: str) -> Dict[str, Any]:
    """Returns a single read from a pod5 file.

    Params:
        read_id: str
            The read_id of the read to be extracted.
        pod5_filepath: str
            Path to the pod5 file.

    Returns:
        dict: Dictionary containing information about the extracted read.
            - 'sample_rate': Sample rate of the read.
            - 'sequencing_kit': Sequencing kit used.
            - 'experiment_type': Experiment type.
            - 'local_basecalling': Local basecalling information.
            - 'signal': Signal data.
            - 'signal_pa': Signal data for the positive strand.
            - 'end_reason': Reason for the end of the read.
            - 'sample_count': Number of samples in the read.
            - 'channel': Pore channel information.
            - 'well': Pore well information.
            - 'pore_type': Pore type.
            - 'writing_software': Software used for writing.
            - 'scale': Scaling factor for the signal.
            - 'shift': Shift factor for the signal.

    """
    signal_dict = {}
    with p5.Reader(pod5_filepath) as reader:
        read = next(reader.reads(selection=[read_id]))
        # Get the signal data and sample rate
        signal_dict["sample_rate"] = read.run_info.sample_rate
        signal_dict["sequencing_kit"] = read.run_info.sequencing_kit
        signal_dict["experiment_type"] = read.run_info.context_tags["experiment_type"]
        signal_dict["local_basecalling"] = read.run_info.context_tags[
            "local_basecalling"
        ]
        signal_dict["signal"] = read.signal
        signal_dict["signal_pa"] = read.signal_pa
        signal_dict["end_reason"] = read.end_reason.reason.name
        signal_dict["sample_count"] = read.sample_count
        signal_dict["channel"] = read.pore.channel
        signal_dict["well"] = read.pore.well
        signal_dict["pore_type"] = read.pore.pore_type
        signal_dict["writing_software"] = reader.writing_software
        signal_dict["scale"] = read.tracked_scaling.scale
        signal_dict["shift"] = read.tracked_scaling.shift
    return signal_dict


def find_base_locs_in_signal(bam_data: dict) -> npt.NDArray[np.int32]:
    """
    Finds the locations of each new base in the signal.

    Params:
        bam_data (dict): Dictionary containing information from the BAM file.

    Returns:
        npt.NDArray[np.int32]: Array of locations of each new base in the signal.
    """
    start_sample = bam_data["start_sample"]
    split_point = bam_data["split_point"]

    # we map the moves from 3' to 5' to the signal
    # and start from the start sample or its sum with the split point
    # if the read is split
    start_sample = start_sample + split_point

    moves_step = bam_data["moves_step"]
    moves_table = np.array(bam_data["moves_table"])

    # Where do moves occur in the signal coordinates?
    moves_indices = np.arange(
        start_sample, start_sample + moves_step * len(moves_table), moves_step
    )

    # We only need locs where a move of 1 occurs
    base_locs_in_signal: npt.NDArray[np.int32] = moves_indices[moves_table != 0]

    return base_locs_in_signal


def z_normalize(data: np.ndarray) -> npt.NDArray[np.float64]:
    """Normalize the input data using Z-score normalization.

    Params:
        data (np.ndarray): Input data to be Z-score normalized.

    Returns:
        npt.NDArray[np.float64]: Z-score normalized data.

    Note:
        Z-score normalization (or Z normalization) transforms the data
        to have a mean of 0 and a standard deviation of 1.
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    z_normalized_data: npt.NDArray[np.float64] = (data - mean) / std_dev
    return z_normalized_data


def clip_extreme_values(
    z_normalized_data: npt.NDArray[np.float64], num_std_dev: float = 4.0
) -> npt.NDArray[np.float64]:
    """Clip extreme values in the Z-score normalized data.

    Clips values outside the specified number of standard deviations from
    the mean. This function takes Z-score normalized data as input, along
    with an optional parameter to set the number of standard deviations.

    Params:
        z_normalized_data (npt.NDArray[np.float64]): Z-score normalized data.
        num_std_dev (float, optional): Number of standard deviations to use
            as the limit. Defaults to 4.0.

    Returns:
        npt.NDArray[np.float64]: Clipped data within the specified range.

    Example:
        >>> z_normalized_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> clipped_data = clip_extreme_values(z_normalized_data, num_std_dev=3.0)
    """

    lower_limit = -num_std_dev
    upper_limit = num_std_dev
    clipped_data: npt.NDArray[np.float64] = np.clip(
        z_normalized_data, lower_limit, upper_limit
    )
    return clipped_data


def preprocess_signal_data(signal: np.ndarray) -> npt.NDArray[np.float64]:
    """
    Preprocesses the signal data.

    Params:
        signal (np.ndarray): Signal data.

    Returns:
        signal (npt.NDArray[np.float64]): Preprocessed signal data.
    """
    signal = z_normalize(signal)
    signal = clip_extreme_values(signal)
    return signal


def extract_roi_signal(
    signal: np.ndarray,
    base_locs_in_signal: npt.NDArray[np.int32],
    fasta: str,
    experiment_type: str,
    start_base_idx_in_fasta: int,
    end_base_idx_in_fasta: int,
    num_left_clipped_bases: int,
) -> ROIData:
    """
    Extracts the signal data for a region of interest (ROI).

    Params:
        signal (np.ndarray): Signal data.
        base_locs_in_signal (npt.NDArray[np.int32]): Array of locations of each new base in the signal.
        fasta (str): Fasta sequence of the read.
        experiment_type (str): Type of experiment (rna or dna).
        start_base_idx_in_fasta (int): Index of the first base in the ROI.
        end_base_idx_in_fasta (int): Index of the last base in the ROI.
        num_left_clipped_bases (int): Number of bases clipped from the left.

    Returns:
        ROIData: Dictionary containing the ROI signal and fasta sequence.
    """
    signal = preprocess_signal_data(signal)
    roi_data: ROIData = {
        "roi_fasta": None,
        "roi_signal": np.array([], dtype=np.float64),
        "signal_start": None,
        "signal_end": None,
        "plot_signal": signal,  # Assuming signal is defined somewhere
        "roi_signal_for_plot": None,
        "base_locs_in_signal": base_locs_in_signal,  # Assuming base_locs_in_signal is defined somewhere
        "start_base_idx_in_fasta": None,
        "end_base_idx_in_fasta": None,
        "read_id": None,
    }

    # Check for valid inputs
    if end_base_idx_in_fasta is None and start_base_idx_in_fasta is None:
        return roi_data

    if end_base_idx_in_fasta > len(fasta) or start_base_idx_in_fasta < 0:
        return roi_data

    if experiment_type not in ("rna", "dna"):
        return roi_data

    start_base_idx_in_fasta += num_left_clipped_bases
    end_base_idx_in_fasta += num_left_clipped_bases
    if experiment_type == "rna":
        rev_base_locs_in_signal = base_locs_in_signal[::-1]
        signal_end = rev_base_locs_in_signal[start_base_idx_in_fasta - 1]
        signal_start = rev_base_locs_in_signal[end_base_idx_in_fasta - 1]
        roi_data["roi_fasta"] = fasta[
            start_base_idx_in_fasta
            - num_left_clipped_bases : end_base_idx_in_fasta
            - num_left_clipped_bases
        ]
    else:
        # TODO: THE LOGIC IS NOT TESTED
        signal_start = base_locs_in_signal[
            start_base_idx_in_fasta - 1
        ]  # TODO: Confirm -1
        signal_end = base_locs_in_signal[end_base_idx_in_fasta - 1]  # TODO: Confirm -1
        roi_data["roi_fasta"] = fasta[start_base_idx_in_fasta:end_base_idx_in_fasta]

    # Signal data is 3'-> 5' for RNA 5' -> 3' for DNA
    # The ROI FASTA is always 5' -> 3' irrespective of the experiment type
    roi_data["roi_signal"] = signal[signal_start:signal_end]

    # Make roi signal for plot and pad it with NaNs outside the ROI
    plot_signal = np.copy(signal)
    plot_signal[:signal_start] = np.nan
    plot_signal[signal_end:] = np.nan
    roi_data["signal_start"] = signal_start
    roi_data["signal_end"] = signal_end
    roi_data["roi_signal_for_plot"] = plot_signal
    roi_data["base_locs_in_signal"] = base_locs_in_signal

    return roi_data


if __name__ == "__main__":
    bam_filepath = "/export/valenfs/data/processed_data/MinION/9_madcap/1_data/3_20230829_randomcap01/1_basecall/calls.bam"
    database_path = "/export/valenfs/data/processed_data/MinION/9_madcap/1_data/3_20230829_randomcap01/database.db"
