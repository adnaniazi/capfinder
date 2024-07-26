import json
import os
import shlex
import sys
import textwrap
from importlib.metadata import version
from typing import List, Optional

import typer
from loguru import logger
from typing_extensions import Annotated

version_info = version("capfinder")

app = typer.Typer(
    help=f"""Capfinder v{version_info}: A Python package for decoding RNA cap types using an encoder-based deep learning model.\n
    """,
    add_completion=True,
    rich_markup_mode="rich",
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"Capfinder v{version_info}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", callback=version_callback, is_eager=True
    ),
) -> None:
    pass


@app.command()
def extract_cap_signal(
    bam_filepath: Annotated[
        str, typer.Option("--bam_filepath", "-b", help="Path to the BAM file")
    ] = "",
    pod5_dir: Annotated[
        str,
        typer.Option(
            "--pod5_dir", "-p", help="Path to directory containing POD5 files"
        ),
    ] = "",
    reference: Annotated[
        str,
        typer.Option("--reference", "-r", help="Reference Sequence (5' -> 3')"),
    ] = "GCTTTCGTTCGTCTCCGGACTTATCGCACCACCTATCCATCATCAGTACTGT",
    cap_class: Annotated[
        int,
        typer.Option(
            "--cap_class",
            "-c",
            help="""\n
    Integer-based class label for the RNA cap type. \n
    - 0 represents Cap_0 \n
    - 1 represents Cap 1 \n
    - 2 represents Cap 2 \n
    - 3 represents Cap2-1 \n
    - 4 represents TMG Cap \n
    - 5 represents NAD Cap \n
    - 6 represents FAD Cap \n
    - -99 represents an unknown cap(s). \n
    """,
        ),
    ] = -99,
    cap_n1_pos0: Annotated[
        int,
        typer.Option(
            "--cap_n1_pos0",
            "-p",
            help="0-based index of 1st nucleotide (N1) of cap in the reference",
        ),
    ] = 52,
    train_or_test: Annotated[
        str,
        typer.Option(
            "--train_or_test",
            "-t",
            help="set to train or test depending on whether it is training or testing data",
        ),
    ] = "test",
    output_dir: Annotated[
        str,
        typer.Option(
            "--output_dir",
            "-o",
            help=textwrap.dedent(
                """
        Path to the output directory which will contain: \n
            ├── A CSV file (data__cap_x.csv) containing the extracted ROI signal data.\n
            ├── A CSV file (metadata__cap_x.csv) containing the complete metadata information.\n
            ├── A log file (capfinder_vXYZ_datatime.log) containing the logs of the program.\n
            └── (Optional) plots directory containing cap signal plots, if plot_signal is set to True.\n
            \u200B    ├── good_reads: Directory that contains the plots for the good reads.\n
            \u200B    ├── bad_reads: Directory that contains the plots for the bad reads.\n
            \u200B    └── plotpaths.csv: CSV file containing the paths to the plots based on the read ID.\n"""
            ),
        ),
    ] = "",
    n_workers: Annotated[
        int,
        typer.Option(
            "--n_workers", "-n", help="Number of CPUs to use for parallel processing"
        ),
    ] = 1,
    plot_signal: Annotated[
        Optional[bool],
        typer.Option(
            "--plot_signal/--no_plot_signal",
            help="Whether to plot extracted cap signal or not",
        ),
    ] = None,
) -> None:
    """
    Extracts signal corresponding to the RNA cap type using BAM and POD5 files. Also, generates plots if required.

    Example command:
    capfinder extract-cap-signal \\
        --bam_filepath /path/to/sorted.bam \\
        --pod5_dir /path/to/pod5_dir \\
        --reference GCTTTCGTTCGTCTCCGGACTTATCGCACCACCTATCCATCATCAGTACTGT \\
        --cap_class -99 \\
        --cap_n1_pos0 52 \\
        --train_or_test test \\
        --output_dir /path/to/output_dir \\
        --n_workers 10 \\
        --no_plot_signal

    capfinder extract-cap-signal \\
        --bam_filepath /path/to/sorted.bam \\
        --pod5_dir /path/to/pod5_dir \\
        --reference GCTTTCGTTCGTCTCCGGACTTATCGCACCACCTATCCATCATCAGTACTGTNNNNNNCGATGTAACTGGGACATGGTGAGCAATCAGGGAAAAAAAAAAAAAAA \\
        --cap_class 0 \\
        --cap_n1_pos0 52 \\
        --train_or_test train \\
        --output_dir /path/to/output_dir \\
        --n_workers 10 \\
        --no_plot_signal
    """
    from capfinder.collate import collate_bam_pod5

    ps = False
    if plot_signal is None:
        ps = False
    elif plot_signal:
        ps = True
    else:
        ps = False

    collate_bam_pod5(
        bam_filepath=bam_filepath,
        pod5_dir=pod5_dir,
        num_processes=n_workers,
        reference=reference,
        cap_class=cap_class,
        cap0_pos=cap_n1_pos0,
        train_or_test=train_or_test,
        plot_signal=ps,
        output_dir=output_dir,
    )


@app.command()
def make_train_dataset(
    csv_dir: Annotated[
        str,
        typer.Option(
            "--csv_dir",
            "-c",
            help="Directory containing all the cap signal data files (data__cap_x.csv)",
        ),
    ] = "",
    save_dir: Annotated[
        str,
        typer.Option(
            "--save_dir",
            "-s",
            help="Directory where the processed data will be saved as csv files.",
        ),
    ] = "",
    target_length: Annotated[
        int,
        typer.Option(
            "--target_length",
            "-t",
            help="Number of signal points in cap signal to consider. If the signal is shorter, it will be padded with zeros. If the signal is longer, it will be truncated.",
        ),
    ] = 500,
    dtype: Annotated[
        str,
        typer.Option(
            "--dtype",
            "-d",
            help="Data type to transform the dataset to Valid values are 'float16', 'float32', or 'float64'.",
        ),
    ] = "float32",
    n_workers: Annotated[
        int,
        typer.Option(
            "--n_workers", "-n", help="Number of CPUs to use for parallel processing"
        ),
    ] = 1,
) -> None:
    """
    Prepares dataset for training the ML model.

    Example command:
    capfinder make-train-dataset \\
        --csv_dir /path/to/csv_dir \\
        --save_dir /path/to/save_dir \\
        --target_length 500 \\
        --dtype float16 \\
        --n_workers 10
    """
    from typing import cast

    from capfinder.train_etl import DtypeLiteral, train_etl

    dt: DtypeLiteral = "float16"
    if dtype in {"float16", "float32", "float64"}:
        dt = cast(
            DtypeLiteral, dtype
        )  # This is safe because input_str must be one of the Literal values
    else:
        logger.warning(
            f"Invalid dtype literal: {dtype}. Allowed values are 'float16', 'float32', 'float64'. Using 'float16' as default."
        )

    train_etl(
        data_dir=csv_dir,
        save_dir=save_dir,
        target_length=target_length,
        dtype=dt,
        n_workers=n_workers,
    )


@app.command()
def create_train_config(
    file_path: Annotated[
        str,
        typer.Option(
            "--file_path", "-f", help="File path to save the JSON configuration file"
        ),
    ] = "",
) -> None:
    """Creats a dummy JSON configuration file at the specified path. Edit it to suit your needs."""
    config = {
        "etl_params": {
            "data_dir": "/export/valenfs/data/processed_data/MinION/9_madcap/dummy_data/real_data2/",
            "save_dir": "/export/valenfs/data/processed_data/MinION/9_madcap/dummy_data/saved_data/",
            "target_length": 500,
            "dtype": "float16",
            "n_workers": 10,
            "n_classes": 4,
            "use_local_dataset": False,
            "remote_dataset_version": "8.0.0",
        },
        "tune_params": {
            "comet_project_name": "capfinder_tfr_tune",
            "patience": 0,
            "max_epochs_hpt": 3,
            "max_trials": 5,
            "factor": 2,
            "batch_size": 64,
            "seed": 42,
            "tuning_strategy": "hyperband",
            "overwrite": True,
        },
        "train_params": {
            "comet_project_name": "capfinder_tfr_train",
            "patience": 2,
            "max_epochs_final_model": 10,
            "batch_size": 64,
        },
        "model_save_dir": "/export/valenfs/data/processed_data/MinION/9_madcap/models/",
        "model_type": "cnn_lstm",
    }

    with open(file_path, "w") as file:
        json.dump(config, file, indent=4)


@app.command()
def train_model(
    config_file: Annotated[
        str,
        typer.Option(
            "--file_path",
            "-f",
            help="""Path to the JSON configuration file containing the parameters for the training pipeline.""",
        ),
    ] = "",
) -> None:
    """Trains the model using the parameters in the JSON configuration file."""
    from capfinder.training import run_training_pipeline

    # Load the configuration file
    with open(config_file) as file:
        config = json.load(file)

    etl_params = config["etl_params"]
    tune_params = config["tune_params"]
    train_params = config["train_params"]
    model_save_dir = config["model_save_dir"]
    model_type = config["model_type"]

    # Run the training pipeline with the loaded parameters
    run_training_pipeline(
        etl_params=etl_params,
        tune_params=tune_params,
        train_params=train_params,
        model_save_dir=model_save_dir,
        model_type=model_type,
    )


@app.command()
def predict_cap_types(
    bam_filepath: Annotated[
        str, typer.Option("--bam_filepath", "-b", help="Path to the BAM file")
    ] = "",
    pod5_dir: Annotated[
        str,
        typer.Option(
            "--pod5_dir", "-p", help="Path to directory containing POD5 files"
        ),
    ] = "",
    output_dir: Annotated[
        str,
        typer.Option(
            "--output_dir",
            "-o",
            help="Path to the output directory for prediction results and logs",
        ),
    ] = "",
    reference: Annotated[
        str,
        typer.Option("--reference", "-r", help="Reference Sequence (5' -> 3')"),
    ] = "GCTTTCGTTCGTCTCCGGACTTATCGCACCACCTATCCATCATCAGTACTGT",
    cap_n1_pos0: Annotated[
        int,
        typer.Option(
            "--cap_n1_pos0",
            "-p",
            help="0-based index of 1st nucleotide (N1) of cap in the reference",
        ),
    ] = 52,
    n_cpus: Annotated[
        int,
        typer.Option(
            "--n_cpus", "-n", help="Number of CPUs to use for parallel processing"
        ),
    ] = 1,
    dtype: Annotated[
        str,
        typer.Option(
            "--dtype",
            "-d",
            help="Data type for model input. Valid values are 'float16', 'float32', or 'float64'.",
        ),
    ] = "float16",
    target_length: Annotated[
        int,
        typer.Option(
            "--target_length",
            "-t",
            help="Number of signal points in cap signal to consider",
        ),
    ] = 500,
    batch_size: Annotated[
        int,
        typer.Option("--batch_size", "-bs", help="Batch size for model inference"),
    ] = 128,
    plot_signal: Annotated[
        bool,
        typer.Option(
            "--plot_signal/--no_plot_signal",
            help="Whether to plot extracted cap signal or not",
        ),
    ] = False,
    debug_code: Annotated[
        bool,
        typer.Option(
            "--debug/--no-debug",
            help="Enable debug mode for more detailed logging",
        ),
    ] = False,
    refresh_cache: Annotated[
        bool,
        typer.Option(
            "--refresh-cache/--no-refresh-cache",
            help="Refresh the cache for intermediate results",
        ),
    ] = False,
) -> None:
    """
    Predicts RNA cap types using BAM and POD5 files.

    Example command:
        capfinder predict-cap-types \\
        --bam_filepath /path/to/sorted.bam \\
        --pod5_dir /path/to/pod5_dir \\
        --output_dir /path/to/output_dir \\
        --batch_size 256 \\
        --no_plot_signal \\
        --no-debug \\
        --no-refresh-cache

    Complete command (only if you know what you are doing):
        capfinder predict-cap-types \\
        --bam_filepath /path/to/sorted.bam \\
        --pod5_dir /path/to/pod5_dir \\
        --output_dir /path/to/output_dir \\
        --reference GCTTTCGTTCGTCTCCGGACTTATCGCACCACCTATCCATCATCAGTACTGT \\
        --cap_n1_pos0 52 \\
        --n_cpus 10 \\
        --dtype float16 \\
        --target_length 500 \\
        --batch_size 256 \\
        --no_plot_signal \\
        --no-debug \\
        --no-refresh-cache
    """
    from typing import cast

    from capfinder.inference import predict_cap_types
    from capfinder.train_etl import DtypeLiteral

    dt: DtypeLiteral = "float16"
    if dtype in {"float16", "float32", "float64"}:
        dt = cast(
            DtypeLiteral, dtype
        )  # This is safe because input_str must be one of the Literal values
    else:
        logger.warning(
            f"Invalid dtype literal: {dtype}. Allowed values are 'float16', 'float32', 'float64'. Using 'float16' as default."
        )

    predict_cap_types(
        bam_filepath=bam_filepath,
        pod5_dir=pod5_dir,
        num_cpus=n_cpus,
        output_dir=output_dir,
        dtype=dt,
        reference=reference,
        cap0_pos=cap_n1_pos0,
        train_or_test="test",
        plot_signal=plot_signal,
        cap_class=-99,
        target_length=target_length,
        batch_size=batch_size,
        debug_code=debug_code,
        refresh_cache=refresh_cache,
    )


@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None and not any(
        arg in sys.argv for arg in ["--help", "-h"]
    ):
        # ANSI color codes
        grey = "\033[90m"
        reset = "\033[0m"

        # Get the full path of the Python executable
        python_path = sys.executable

        def color_path(arg: str) -> str:
            return (
                f"{grey}{arg}{reset}" if os.path.exists(os.path.dirname(arg)) else arg
            )

        # Reconstruct the full command with the Python path
        command_parts: List[str] = [f"{color_path(shlex.quote(python_path))} -m"]

        # Group 'capfinder' with the subcommand
        if len(sys.argv) > 2:
            command_parts.append(f"capfinder {sys.argv[1]}")
            current_arg = None
            for arg in sys.argv[2:]:
                if arg.startswith("--"):
                    if current_arg:
                        command_parts.append(current_arg)
                    current_arg = arg
                else:
                    if current_arg:
                        current_arg += f" {color_path(shlex.quote(arg))}"
                        command_parts.append(current_arg)
                        current_arg = None
                    else:
                        command_parts.append(color_path(shlex.quote(arg)))
            if current_arg:
                command_parts.append(current_arg)
        else:
            command_parts.append("capfinder")
            command_parts.extend(color_path(shlex.quote(arg)) for arg in sys.argv[1:])

        # Format the command with grouped arguments per line, adding backslashes
        formatted_command = " \\\n".join(command_parts)

        logger.info(f"You have issued the following command:\n{formatted_command}")


if __name__ == "__main__":
    app()
