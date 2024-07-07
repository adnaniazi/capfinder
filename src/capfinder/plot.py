"""
The modules helps in plotting the entire read signal, signal for ROI,
and the base annotations. It also prints alignments. All this information
is useful in understanding if the OTE-finding algorthim is homing-in on the
correct region of interest (ROI).

The plot is saved as an HTML file.

Author: Adnan M. Niazi
Date: 2024-02-28
"""

import numpy as np
from bokeh.layouts import column
from bokeh.models import Div, WheelZoomTool  # type: ignore[attr-defined]
from bokeh.plotting import figure, output_file, save
from loguru import logger

from capfinder.process_pod5 import ROIData


def append_dummy_sequence(
    fasta_sequence: str, num_left_clipped_bases: int, num_right_clipped_bases: int
) -> str:
    """Append/prepend 'H' to the left/right of the FASTA sequence based on soft-clipping counts

    Args:
        fasta_sequence (str): FASTA sequence
        num_left_clipped_bases (int): Number of bases soft-clipped from the left
        num_right_clipped_bases (int): Number of bases soft-clipped from the right

    Returns:
        modified_sequence (str): FASTA sequence with 'H' appended/prepended to the left/right

    """
    modified_sequence = (
        "H" * num_left_clipped_bases + fasta_sequence + "H" * num_right_clipped_bases
    )
    return modified_sequence


def plot_roi_signal(
    pod5_data: dict,
    bam_data: dict,
    roi_data: ROIData,
    start_base_idx_in_fasta: int,
    end_base_idx_in_fasta: int,
    plot_filepath: str,
    chunked_aln_str: str,
    alignment_score: int,
) -> None:
    read_id = bam_data["read_id"]
    parent_read_id = bam_data["parent_read_id"]
    # Prepare plotting data
    y1 = roi_data.get("plot_signal")
    plot_roi_cond = (
        start_base_idx_in_fasta is not None
        and end_base_idx_in_fasta is not None
        and roi_data["roi_signal_for_plot"] is not None
    )
    if plot_roi_cond:
        y2 = roi_data["roi_signal_for_plot"]

    x = np.arange(len(y1))  # type: ignore[arg-type]
    moves = roi_data["base_locs_in_signal"]

    # Map bases to colors
    base_colors = {
        "H": "white",
        "G": "#7F58AF",
        "T": "#64C5EB",
        "C": "#E84D8A",
        "A": "#FEB326",
    }

    # Create a new plot
    plot_height = 300
    xwheel_zoom = WheelZoomTool(dimensions="width")  # zoom only in x-dimension
    p = figure(
        title=read_id,
        height=plot_height,
        tools=[xwheel_zoom, "xpan, save, reset, box_zoom"],
        sizing_mode="stretch_width",
        active_scroll=xwheel_zoom,
        x_axis_label="Sample #",
        y_axis_label="Current (pA)",
    )

    # Hide vertical and horizontal grid lines
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # Orient FASTA sequence based on experiment type
    experiment_type = pod5_data["experiment_type"]
    fasta_sequence = bam_data["read_fasta"]

    # Fix FASTA for hard-clipped reads that lack the original bases due to hard clipping
    fasta_sequence = append_dummy_sequence(
        fasta_sequence,
        bam_data["num_left_clipped_bases"],
        bam_data["num_right_clipped_bases"],
    )

    if experiment_type != "dna":
        fasta_sequence = fasta_sequence[::-1]

    # Plot the whole signal irrespective of whether it is a fused read or not
    start_array = moves
    end_array = np.append(
        moves[1:],
        bam_data["start_sample"] + bam_data["split_point"] + bam_data["num_samples"],
    )

    # Get the base colors and legend labels in a vectorized manner
    bases = np.array(list(fasta_sequence))
    color_array = np.vectorize(base_colors.get)(bases)

    # Draw rectangles for base-annotation all at once
    p.quad(
        top=[max(y1)] * len(start_array),  # type: ignore[arg-type]
        bottom=[min(y1)] * len(start_array),  # type: ignore[arg-type]
        left=start_array.tolist(),
        right=end_array.tolist(),
        color=color_array.tolist(),
        alpha=0.7,
    )

    # Debugging
    if len(start_array.tolist()) != len(color_array.tolist()):
        logger.debug("Array misze mismatch")
        logger.debug(f"read_id: {read_id}")
        logger.debug(f"moves {len(moves)}")
        logger.debug(f"FASTA sequence len: {len(fasta_sequence)}")
        logger.debug(f"FASTA sequence: {fasta_sequence}")
        logger.debug(f"plot_filepath {plot_filepath}")
        num_left_clipped_bases = bam_data["num_left_clipped_bases"]
        num_right_clipped_bases = bam_data["num_right_clipped_bases"]
        logger.debug(f"left clip {num_left_clipped_bases}")
        logger.debug(f"right clip {num_right_clipped_bases}")

    # Plot full signal and ROI signal if it exists
    p.line(x, y1, legend_label="Signal PA", line_width=2, color="black")
    if plot_roi_cond:
        p.line(x, y2, legend_label="ROI signal", line_width=2, color="#89BD21")

    legend_html = """
    <div style="display: flex; justify-content: center; width: 100%;">
        <div style="padding: 10px; border: none; display: inline-block;">
            <div style="display: flex; flex-direction: row; align-items: center;">
                <div style="width: 10px; height: 10px; background-color: {G_color}; margin-right: 5px;"></div>
                <div style="margin-right: 10px;">G</div>
                <div style="width: 10px; height: 10px; background-color: {T_color}; margin-right: 5px;"></div>
                <div style="margin-right: 10px;">T</div>
                <div style="width: 10px; height: 10px; background-color: {C_color}; margin-right: 5px;"></div>
                <div style="margin-right: 10px;">C</div>
                <div style="width: 10px; height: 10px; background-color: {A_color}; margin-right: 5px;"></div>
                <div>A</div>
            </div>
        </div>
    </div>
    """.format(
        G_color=base_colors["G"],
        T_color=base_colors["T"],
        C_color=base_colors["C"],
        A_color=base_colors["A"],
    )

    legend = Div(text=legend_html)

    # Insert invisible word breaks '&#8203;' in FASTA sequence every 10 bases for easy wraping of text
    formatted_read_fasta = "".join(
        [
            char + "&#8203;" if (i + 1) % 10 == 0 else char
            for i, char in enumerate(bam_data["read_fasta"])
        ]
    )
    formatted_rev_read_fasta = "".join(
        [
            char + "&#8203;" if (i + 1) % 10 == 0 else char
            for i, char in enumerate(bam_data["read_fasta"][::-1])
        ]
    )
    chunked_aln_str_br = chunked_aln_str.replace("\n", "<br>")

    # Handle None values
    if roi_data["signal_end"] is None or roi_data["signal_start"] is None:
        roi_length_in_samples = None
        translocation_rate = None
        rev_roi_fasta = None
        roi_fasta = None
    else:
        roi_fasta = roi_data["roi_fasta"]
        if roi_fasta is None:
            len_roi_fasta = 0
            translocation_rate = None
            rev_roi_fasta = None
        else:
            len_roi_fasta = len(roi_fasta)
            if len_roi_fasta > 0:
                translocation_rate = int(
                    (roi_data["signal_end"] - roi_data["signal_start"]) / len_roi_fasta
                )
                rev_roi_fasta = roi_fasta[::-1]
            else:
                translocation_rate = None
                rev_roi_fasta = None
        roi_length_in_samples = roi_data["signal_end"] - roi_data["signal_start"]

    # Create a Div widget with text containing the values of important variables
    text = f"""
    <div>
        <style>
            table tr:nth-child(even) {{background-color: #f2f2f2;}}
            table td, table th {{padding: 2px; text-align: left;}} /* Left-align table cell content */
            .wrap-text pre {{white-space: pre-wrap; word-wrap: break-word; word-break: break-all;}} /* Wrap text in pre tag */
            table td:first-child {{white-space: nowrap;}} /* Set white-space property to nowrap for the first column */
        </style>
        <table border="1" style="border-collapse: collapse; width: 98vw;"> <!-- Set the table width to 100% -->
            <tr>
                <th>Information</th>
                <th class="wrap-text">Value</th>
            </tr>
            <tr>
                <td>Read ID - Parent Read ID</td>
                <td class="wrap-text"><pre>{read_id} {parent_read_id}</pre></td>
            </tr>
            <tr>
                <td>Reversed FASTA (3' -> 5')</td>
                <td class="wrap-text"><pre>{formatted_rev_read_fasta}</pre></td>
            </tr>
            <tr>
                <td>FASTA (5' -> 3')</td>
                <td class="wrap-text"><pre>{formatted_read_fasta}</pre></td>
            </tr>
            <tr>
                <td>ROI start base in FASTA</td>
                <td class="wrap-text">{start_base_idx_in_fasta}</td>
            </tr>
            <tr>
                <td>ROI end base in FASTA</td>
                <td class="wrap-text">{end_base_idx_in_fasta}</td>
            </tr>
            <tr>
                <td>ROI start sample in signal</td>
                <td class="wrap-text">{roi_data['signal_start']}</td>
            </tr>
            <tr>
                <td>ROI end sample in signal</td>
                <td class="wrap-text">{roi_data['signal_end']}</td>
            </tr>
            <tr>
                <td>ROI length in samples</td>
                <td class="wrap-text">{roi_length_in_samples}</td>
            </tr>
            <tr>
                <td>Translocation rate</td>
                <td class="wrap-text">{translocation_rate}</td>
            </tr>
            <tr>
                <td>Basecalling start sample</td>
                <td class="wrap-text">{bam_data['start_sample']}</td>
            </tr>
            <tr>
                <td>Reversed ROI FASTA (3' -> 5')</td>
                <td class="wrap-text"><pre>{rev_roi_fasta}</pre></td>
            </tr>
            <tr>
                <td>ROI FASTA (5' -> 3')</td>
                <td class="wrap-text"><pre>{roi_fasta}</pre></td>
            </tr>
            <tr>
                <td>Alignment score</td>
                <td class="wrap-text">{alignment_score}</td>
            </tr>
            <tr>
                <td>Alignment</td>
                <td class="wrap-text"> <pre>{chunked_aln_str_br}</pre> </td>
            </tr>
        </table>
    </div>
    """

    div_info = Div(
        text=text,
        styles={
            "padding-left": "20px",
            "word-wrap": "break-word",
            "word-break": "break-all",
        },
    )

    # Combine the plot and the Div widget in a layout
    layout = column(p, legend, div_info, sizing_mode="stretch_width")
    output_file(plot_filepath)
    save(layout)
