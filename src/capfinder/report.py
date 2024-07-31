import csv
import math
import re
from collections import defaultdict
from typing import DefaultDict, Dict, Iterator, Optional, Tuple

import bokeh
import numpy as np
import pandas as pd
from bokeh.embed import components
from bokeh.layouts import column, row
from bokeh.models import (  # type: ignore
    ColumnDataSource,
    CustomJS,
    HoverTool,
    RadioButtonGroup,
)
from bokeh.plotting import figure
from loguru import logger

# Custom color palette derived from the provided images
custom_colors = [
    "#FFD9E0",  # Blossom
    "#C1E1C1",  # Pistachio
    "#AEC6CF",  # Serene
    "#FFF6B8",  # Canary
    "#E6D2C3",  # Latte
    "#8DA290",  # Sage Green
    "#BEC7B4",  # Ash Gray
    "#DEE2D9",  # Pale Gray
    "#FCF1D8",  # Cream
    "#F2CBBB",  # Peach
]


def custom_sort_key(cap: str) -> Tuple[float, str, str]:
    if cap == "OTE_not_found":
        return (float("inf"), "", "")  # This ensures OTE_not_found is always last

    # Extract the numeric part and the alphabetic part
    match = re.match(r"cap_(\d+)(-\d+)?(.*)$", cap)
    if match:
        num = int(match.group(1))
        suffix = match.group(2) or ""
        alpha = match.group(3) or ""
        return (num, suffix, alpha)
    else:
        # For caps that don't match the expected pattern, sort them after numeric ones
        return (float("inf") - 1, "", cap)


def create_bar_chart(source: ColumnDataSource) -> figure:
    p = figure(
        x_range=source.data["predictions"],
        height=400,
        width=600,
        title="Distribution of Cap Predictions",
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    p.vbar(
        x="predictions",
        top="counts",
        width=0.9,
        source=source,
        line_color="white",
        fill_color="color",
    )
    p.xgrid.grid_line_color = None
    p.y_range.start = 0  # type: ignore
    p.xaxis.axis_label = "Predicted Cap"
    p.yaxis.axis_label = "Count"
    p.xaxis.major_label_orientation = 0.7
    hover = HoverTool(
        tooltips=[
            ("Cap", "@predictions"),
            ("Count", "@counts"),
            ("Percentage", "@percentages{0.2f}%"),
        ]
    )
    p.add_tools(hover)
    return p


def create_pie_chart(source: ColumnDataSource) -> figure:
    data = source.data
    total = sum(data["counts"])
    data["angle"] = [count / total * 2 * math.pi for count in data["counts"]]
    data["cumulative_angle"] = np.cumsum(data["angle"]).tolist()
    data["start_angle"] = [0] + data["cumulative_angle"][:-1]  # type: ignore

    p = figure(
        height=400,
        width=600,
        title="Distribution of Cap Predictions",
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        x_range=(-1.1, 1.1),
        y_range=(-1.1, 1.1),
    )
    p.wedge(
        x=0,
        y=0,
        radius=0.9,
        start_angle="start_angle",
        end_angle="cumulative_angle",
        line_color="white",
        fill_color="color",
        legend_field="predictions",
        source=source,
    )
    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None
    hover = HoverTool(
        tooltips=[
            ("Cap", "@predictions"),
            ("Count", "@counts"),
            ("Percentage", "@percentages{0.2f}%"),
        ]
    )
    p.add_tools(hover)
    p.legend.location = "center_right"
    p.legend.click_policy = "hide"
    return p


def read_csv_in_chunks(
    file_path: str, chunk_size: int = 100000
) -> Iterator[pd.DataFrame]:
    """Generator function to read CSV file in chunks."""
    yield from pd.read_csv(file_path, chunksize=chunk_size)


def create_file_index(
    file_path: str, id_column: str, chunk_size: int = 100000
) -> Dict[str, int]:
    """Create an index of id_column values and their file positions."""
    index = {}
    with open(file_path) as file:
        reader = csv.reader(file)
        headers = next(reader)  # Read and skip the header row
        id_index = headers.index(id_column)

        for pos, rw in enumerate(
            reader, start=1
        ):  # Start enumeration from 1 to account for header
            index[rw[id_index]] = pos

    return index


def get_row_by_id(
    file_path: str, read_id: str, index: Dict[str, int]
) -> Optional[Dict[str, str]]:
    """Retrieve a row from the file based on the read_id."""
    if read_id not in index:
        return None

    with open(file_path) as file:
        reader = csv.reader(file)
        headers = next(reader)  # Read and skip the header row
        file.seek(0)
        next(reader)  # Skip the header row again
        for _ in range(index[read_id]):
            rw = next(reader)

        return dict(zip(headers, rw))


def generate_report(
    metadata_file: str,
    predictions_file: str,
    output_csv: str,
    output_html: str,
    chunk_size: int = 100000,
) -> None:
    # Step 1: Create an index of read_ids and their file positions
    logger.info("Creating file indexes...")
    metadata_index = create_file_index(metadata_file, "read_id", chunk_size)
    predictions_index = create_file_index(predictions_file, "read_id", chunk_size)

    # Step 2: Get all unique read_ids
    all_read_ids = set(metadata_index.keys()) | set(predictions_index.keys())

    # Initialize cap type counts and total reads count
    cap_type_counts: DefaultDict[str, int] = defaultdict(int)
    total_reads = len(all_read_ids)

    logger.info(f"Processing {total_reads} reads...")

    # Step 3: Process the data
    with open(output_csv, "w", newline="") as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(["read_id", "pod5_file", "predicted_cap"])  # Write header

        for read_id in all_read_ids:
            metadata_row = get_row_by_id(metadata_file, read_id, metadata_index)
            predictions_row = get_row_by_id(
                predictions_file, read_id, predictions_index
            )

            pod5_file = metadata_row["pod5_file"] if metadata_row else None
            predicted_cap = (
                predictions_row["predicted_cap"] if predictions_row else "OTE_not_found"
            )

            if (
                predicted_cap != "predicted_cap"
            ):  # Ensure we're not including the header
                csv_writer.writerow([read_id, pod5_file, predicted_cap])
                cap_type_counts[predicted_cap] += 1

    logger.info("Data processing complete. Generating analysis...")

    # Data Analysis
    ote_not_found_count = cap_type_counts["OTE_not_found"]
    with_cap_count = total_reads - ote_not_found_count

    ote_not_found_percentage = round((ote_not_found_count / total_reads * 100), 2)
    with_cap_percentage = round((with_cap_count / total_reads * 100), 2)

    # Convert cap_type_counts to a pandas Series for easier manipulation
    cap_prediction_counts = pd.Series(cap_type_counts)
    cap_prediction_percentages = round((cap_prediction_counts / total_reads * 100), 2)

    # Combine counts and percentages
    cap_predictions_summary = pd.DataFrame(
        {"count": cap_prediction_counts, "percentage": cap_prediction_percentages}
    )

    # Sort the index (Cap predictions) using the custom sort key
    sorted_index = sorted(cap_predictions_summary.index, key=custom_sort_key)
    cap_predictions_summary = cap_predictions_summary.reindex(sorted_index)

    # Create Bokeh plots
    predictions = cap_predictions_summary.index.tolist()
    counts = cap_predictions_summary["count"].tolist()
    percentages = cap_predictions_summary["percentage"].tolist()

    # Assign colors from the custom palette
    colors = custom_colors[: len(predictions)]
    if len(predictions) > len(custom_colors):
        colors = custom_colors * (len(predictions) // len(custom_colors) + 1)
        colors = colors[: len(predictions)]

    source = ColumnDataSource(
        {
            "predictions": predictions,
            "counts": counts,
            "percentages": percentages,
            "color": colors,
        }
    )

    bar_chart = create_bar_chart(source)
    pie_chart = create_pie_chart(source)
    pie_chart.visible = False  # type: ignore

    # Create radio buttons for chart selection
    radio_button_group = RadioButtonGroup(labels=["Bar Chart", "Pie Chart"], active=0)

    # Create a CustomJS callback to switch between charts
    callback = CustomJS(
        args={
            "bar_chart": bar_chart,
            "pie_chart": pie_chart,
            "radio_button_group": radio_button_group,
        },
        code="""
        if (radio_button_group.active === 0) {
            bar_chart.visible = true;
            pie_chart.visible = false;
        } else {
            bar_chart.visible = false;
            pie_chart.visible = true;
        }
    """,
    )

    radio_button_group.js_on_change("active", callback)

    # Combine plots and radio buttons
    layout = column(radio_button_group, row(bar_chart, pie_chart))

    # Get the HTML components
    script, div = components(layout)

    # Get Bokeh version
    bokeh_version = bokeh.__version__

    # Generate HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cap Data Analysis Report</title>
        <link href="https://cdn.bokeh.org/bokeh/release/bokeh-{bokeh_version}.min.css" rel="stylesheet" type="text/css">
        <link href="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-{bokeh_version}.min.css" rel="stylesheet" type="text/css">
        <script src="https://cdn.bokeh.org/bokeh/release/bokeh-{bokeh_version}.min.js"></script>
        <script src="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-{bokeh_version}.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #FFFFFF;
            }}
            h1, h2 {{
                color: #2c3e50;
                border-bottom: 2px solid #2c3e50;
                padding-bottom: 10px;
            }}
            h1 {{
                text-align: center;
                font-size: 2.5em;
                margin-bottom: 30px;
            }}
            h2 {{
                font-size: 1.8em;
                margin-top: 30px;
            }}
            .container {{
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 20px;
            }}
            .table-container, .plot-container {{
                background-color: #FFFFFF;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-bottom: 20px;
            }}
            .table-container {{
                flex: 1;
                min-width: 300px;
            }}
            .plot-container {{
                flex: 2;
                min-width: 400px;
            }}
            .cap-predictions-table {{
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-top: 20px;
            }}
            .cap-predictions-table th,
            .cap-predictions-table td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #e0e0e0;
            }}
            .cap-predictions-table thead {{
                background-color: #C1E1C1;
                color: #333;
                font-weight: bold;
            }}
            .cap-predictions-table th {{
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-size: 14px;
            }}
            .cap-predictions-table tbody tr:nth-child(even) {{
                background-color: #f8f8f8;
            }}
            .cap-predictions-table tbody tr:nth-child(odd) {{
                background-color: #ffffff;
            }}
            .cap-predictions-table tbody tr:hover {{
                background-color: #e8f4e8;
                transition: background-color 0.3s ease;
            }}
            .cap-predictions-table td:first-child {{
                font-weight: bold;
                color: #2c3e50;
            }}
            .cap-predictions-table td:nth-child(2),
            .cap-predictions-table td:nth-child(3) {{
                text-align: right;
            }}
            ul {{
                list-style-type: none;
                padding: 0;
            }}
            li {{
                background-color: #f6f5f5;
                margin-bottom: 10px;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            .conclusion {{
                background-color: #f4fcff;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-top: 30px;
            }}
        </style>
    </head>
    <body>
        <h1>Capfinder Cap Data Analysis Report</h1>

        <h2>Summary Statistics</h2>
        <ul>
            <li>Total number of reads: <strong>{total_reads}</strong></li>
            <li>Number of reads with Cap predictions: <strong>{with_cap_count}</strong> ({with_cap_percentage}%)</li>
            <li>Number of reads without Cap predictions: <strong>{ote_not_found_count}</strong> ({ote_not_found_percentage}%)</li>
        </ul>

        <div class="container">
            <div class="table-container">
                <h2>Cap Predictions</h2>
                <table class="cap-predictions-table">
                    <thead>
                        <tr>
                            <th>Predicted Cap</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(f"<tr><td>{cap}</td><td>{int(row['count'])}</td><td>{row['percentage']:.2f}%</td></tr>" for cap, row in cap_predictions_summary.iterrows())}
                    </tbody>
                </table>
            </div>
            <div class="plot-container">
                <h2>Distribution of Cap Predictions</h2>
                {div}
            </div>
        </div>

        <div class="note">
            <h2>Note</h2>
            <p>
                Please note that OTE_not_found indicates reads for which we could not predict cap types for the following reasons:
            </p>
            <ol>
                <li>Perhaps the motor protein dropped off before it even reached OTE. This is very likely in long reads.</li>
                <li>The OTE was sequenced but contained so much basecalling errors that we are not able to home into the cap region with high enough confidence.</li>
            </ol>
        </div>
        {script}
    </body>
    </html>
    """

    # Save the HTML report
    with open(output_html, "w") as f:
        f.write(html_report)

    logger.info("Analysis complete!")


if __name__ == "__main__":
    generate_report(
        metadata_file="/export/valenfs/data/processed_data/MinION/9_madcap/1_data/8_20231114_randomCAP1v3_rna004/test_OTE_vizs_july20/raw_cap_signal_data/metadata__cap_unknown.csv",
        predictions_file="/export/valenfs/data/processed_data/MinION/9_madcap/1_data/8_20231114_randomCAP1v3_rna004/test_OTE_vizs_july20/cap_predictions/predictions.csv",
        output_csv="/export/valenfs/data/processed_data/MinION/9_madcap/1_data/8_20231114_randomCAP1v3_rna004/test_OTE_vizs_july20/cap_predictions/merged_cap_data.csv",
        output_html="/export/valenfs/data/processed_data/MinION/9_madcap/1_data/8_20231114_randomCAP1v3_rna004/test_OTE_vizs_july20/cap_predictions/cap_analysis_report.html",
    )
