import csv
import math
import re
import sqlite3
from collections import defaultdict
from typing import DefaultDict, List, Tuple

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
from tqdm.auto import tqdm

# Custom color palette derived from the provided images
custom_colors = [
    "#FFD9E0",
    "#C1E1C1",
    "#AEC6CF",
    "#FFF6B8",
    "#E6D2C3",
    "#8DA290",
    "#BEC7B4",
    "#DEE2D9",
    "#FCF1D8",
    "#F2CBBB",
]


def custom_sort_key(cap: str) -> Tuple[float, str, str]:
    if cap == "OTE_not_found":
        return (float("inf"), "", "")
    match = re.match(r"cap_(\d+)(-\d+)?(.*)$", cap)
    if match:
        num = int(match.group(1))
        suffix = match.group(2) or ""
        alpha = match.group(3) or ""
        return (num, suffix, alpha)
    else:
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


def create_database(db_path: str) -> sqlite3.Connection:
    """Create a new SQLite database and return the connection."""
    conn = sqlite3.connect(db_path)
    return conn


def create_table(conn: sqlite3.Connection, table_name: str, columns: List[str]) -> None:
    """Create a table in the SQLite database."""
    cursor = conn.cursor()
    columns_str = ", ".join(columns)
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str})")
    conn.commit()


def count_csv_rows(csv_file: str) -> int:
    """Count the number of rows in a CSV file."""
    with open(csv_file) as f:
        return sum(1 for _ in f) - 1  # Subtract 1 to account for header


def csv_to_sqlite(
    csv_file: str,
    db_conn: sqlite3.Connection,
    table_name: str,
    chunk_size: int = 100000,
) -> None:
    """Import CSV data into SQLite database in chunks with progress bar."""
    create_table(
        db_conn,
        table_name,
        ["read_id TEXT PRIMARY KEY", "pod5_file TEXT", "predicted_cap TEXT"],
    )

    cursor = db_conn.cursor()
    total_rows = count_csv_rows(csv_file)

    with open(csv_file) as f:
        csv_reader = csv.DictReader(f)
        chunk = []
        with tqdm(total=total_rows, unit="reads") as pbar:
            for rw in csv_reader:
                chunk.append(
                    (
                        rw["read_id"],
                        rw.get("pod5_file", ""),
                        rw.get("predicted_cap", ""),
                    )
                )
                if len(chunk) >= chunk_size:
                    cursor.executemany(
                        f"INSERT OR REPLACE INTO {table_name} (read_id, pod5_file, predicted_cap) VALUES (?, ?, ?)",
                        chunk,
                    )
                    db_conn.commit()
                    pbar.update(len(chunk))
                    chunk = []

            if chunk:  # Insert any remaining rows
                cursor.executemany(
                    f"INSERT OR REPLACE INTO {table_name} (read_id, pod5_file, predicted_cap) VALUES (?, ?, ?)",
                    chunk,
                )
                db_conn.commit()
                pbar.update(len(chunk))


def join_tables(
    conn: sqlite3.Connection, output_csv: str, chunk_size: int = 100000
) -> None:
    """Join metadata and predictions tables and save to CSV in chunks with progress bar."""
    query = """
    SELECT m.read_id, m.pod5_file, COALESCE(p.predicted_cap, 'OTE_not_found') as predicted_cap
    FROM metadata m
    LEFT JOIN predictions p ON m.read_id = p.read_id
    """

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM metadata")
    total_rows = cursor.fetchone()[0]

    cursor.execute(query)

    with open(output_csv, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["read_id", "pod5_file", "predicted_cap"])  # Write header

        with tqdm(total=total_rows, unit="reads") as pbar:
            while True:
                results = cursor.fetchmany(chunk_size)
                if not results:
                    break
                csv_writer.writerows(results)
                pbar.update(len(results))


def get_cap_type_counts(conn: sqlite3.Connection) -> DefaultDict[str, int]:
    """Get cap type counts from the joined data."""
    query = """
    SELECT COALESCE(predicted_cap, 'OTE_not_found') as cap, COUNT(*) as count
    FROM (
        SELECT m.read_id, COALESCE(p.predicted_cap, 'OTE_not_found') as predicted_cap
        FROM metadata m
        LEFT JOIN predictions p ON m.read_id = p.read_id
    )
    GROUP BY cap
    """
    cursor = conn.cursor()
    cursor.execute(query)

    results = defaultdict(int)
    for cap, count in cursor:
        results[cap] = count

    return results


def generate_report(
    metadata_file: str,
    predictions_file: str,
    output_csv: str,
    output_html: str,
    chunk_size: int = 100000,
) -> None:

    logger.info("Creating SQLite database...")
    db_conn = create_database(":memory:")

    logger.info("Importing metadata to SQLite...")
    csv_to_sqlite(metadata_file, db_conn, "metadata", chunk_size)

    logger.info("Importing predictions to SQLite...")
    csv_to_sqlite(predictions_file, db_conn, "predictions", chunk_size)

    logger.info("Joining tables and saving to CSV...")
    join_tables(db_conn, output_csv, chunk_size)

    cap_type_counts = get_cap_type_counts(db_conn)

    total_reads = sum(cap_type_counts.values())
    ote_not_found_count = cap_type_counts["OTE_not_found"]
    with_cap_count = total_reads - ote_not_found_count

    ote_not_found_percentage = round((ote_not_found_count / total_reads * 100), 2)
    with_cap_percentage = round((with_cap_count / total_reads * 100), 2)

    cap_prediction_counts = pd.Series(cap_type_counts)
    cap_prediction_percentages = round((cap_prediction_counts / total_reads * 100), 2)

    cap_predictions_summary = pd.DataFrame(
        {"count": cap_prediction_counts, "percentage": cap_prediction_percentages}
    )

    sorted_index = sorted(cap_predictions_summary.index, key=custom_sort_key)
    cap_predictions_summary = cap_predictions_summary.reindex(sorted_index)

    predictions = cap_predictions_summary.index.tolist()
    counts = cap_predictions_summary["count"].tolist()
    percentages = cap_predictions_summary["percentage"].tolist()

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

    radio_button_group = RadioButtonGroup(labels=["Bar Chart", "Pie Chart"], active=0)

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

    layout = column(radio_button_group, row(bar_chart, pie_chart))

    script, div = components(layout)

    bokeh_version = bokeh.__version__
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

    # Close the database connection
    db_conn.close()


if __name__ == "__main__":
    generate_report(
        metadata_file="/export/valenfs/data/processed_data/MinION/9_madcap/1_data/8_20231114_randomCAP1v3_rna004/test_OTE_vizs_july20/raw_cap_signal_data/metadata__cap_unknown.csv",
        predictions_file="/export/valenfs/data/processed_data/MinION/9_madcap/1_data/8_20231114_randomCAP1v3_rna004/test_OTE_vizs_july20/cap_predictions/predictions.csv",
        output_csv="/export/valenfs/data/processed_data/MinION/9_madcap/1_data/8_20231114_randomCAP1v3_rna004/test_OTE_vizs_july20/cap_predictions/merged_cap_data.csv",
        output_html="/export/valenfs/data/processed_data/MinION/9_madcap/1_data/8_20231114_randomCAP1v3_rna004/test_OTE_vizs_july20/cap_predictions/cap_analysis_report.html",
    )
