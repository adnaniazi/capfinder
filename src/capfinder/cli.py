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

from capfinder.utils import CAP_MAPPING  # noqa
from capfinder.utils import (
    CUSTOM_MAPPING_PATH,
    DEFAULT_CAP_MAPPING,
    get_next_available_cap_number,
    is_cap_name_unique,
    load_custom_mapping,
    save_custom_mapping,
    update_cap_mapping,
)

version_info = version("capfinder")
formatted_command_global = None


app = typer.Typer(
    help=f"""Capfinder v{version_info}: A Python package for decoding RNA cap types using an encoder-based deep learning model.\n
    """,
    add_completion=True,
    rich_markup_mode="rich",
)


# Initialize cap mapping when the CLI starts
caps_app = typer.Typer()


app.add_typer(
    caps_app, name="capmap", help="Manages mapping of caps to interger labels"
)


@caps_app.command("add")
def add_cap(cap_int: int, cap_name: str) -> None:
    """Add a new cap mapping or update an existing one."""
    global CAP_MAPPING

    next_available = get_next_available_cap_number()

    # Check if the cap name is unique
    existing_cap_int = is_cap_name_unique(cap_name)
    if existing_cap_int is not None and existing_cap_int != cap_int:
        typer.echo(
            f"Error: The cap name '{cap_name}' is already used for cap number {existing_cap_int}."
        )
        typer.echo("Please use a unique name for each cap.")
        return

    if cap_int in CAP_MAPPING:
        update_cap_mapping({cap_int: cap_name})
        typer.echo(f"Updated existing mapping: {cap_int} -> {cap_name}")
    elif cap_int == next_available:
        update_cap_mapping({cap_int: cap_name})
        typer.echo(f"Added new mapping: {cap_int} -> {cap_name}")
    else:
        typer.echo(f"Error: The next available cap number is {next_available}.")
        typer.echo(
            f"Please use {next_available} as the cap number to maintain continuity."
        )
        return
    typer.echo(f"Custom mappings saved to: {CUSTOM_MAPPING_PATH}")


@caps_app.command("remove")
def remove_cap(cap_int: int) -> None:
    """Remove a cap mapping."""
    global CAP_MAPPING

    if cap_int in CAP_MAPPING:
        del CAP_MAPPING[cap_int]
        save_custom_mapping(CAP_MAPPING)
        typer.echo(f"Removed mapping for cap integer: {cap_int}")
        typer.echo(f"Custom mappings saved to: {CUSTOM_MAPPING_PATH}")
    else:
        typer.echo(f"No mapping found for cap integer: {cap_int}")


@caps_app.command("list")
def list_caps() -> None:
    """List all current cap mappings."""
    load_custom_mapping()  # Reload the mappings from the file
    global CAP_MAPPING

    if not CAP_MAPPING:
        typer.echo("No cap mappings found. Using default mappings:")
        for cap_int, cap_name in sorted(DEFAULT_CAP_MAPPING.items()):
            typer.echo(f"{cap_int}: {cap_name}")
    else:
        typer.echo("Current cap mappings:")
        for cap_int, cap_name in sorted(CAP_MAPPING.items()):
            typer.echo(f"{cap_int}: {cap_name}")

    next_available = get_next_available_cap_number()
    typer.echo(f"\nNext available cap number: {next_available}")
    typer.echo(f"\nCustom mappings file location: {CUSTOM_MAPPING_PATH}")


@caps_app.command("reset")
def reset_caps() -> None:
    """Reset cap mappings to default."""
    global CAP_MAPPING
    CAP_MAPPING = DEFAULT_CAP_MAPPING.copy()
    save_custom_mapping(CAP_MAPPING)
    typer.echo("Cap mappings reset to default.")
    typer.echo(f"Default mappings saved to: {CUSTOM_MAPPING_PATH}")


@caps_app.command("config")
def show_config() -> None:
    """Show the location of the configuration file."""
    typer.echo(f"Custom mappings file location: {CUSTOM_MAPPING_PATH}")
    if CUSTOM_MAPPING_PATH.exists():
        typer.echo("The file exists and contains custom mappings.")
    else:
        typer.echo(
            "The file does not exist yet. It will be created when you add a custom mapping."
        )
        logger.warning(f"Config file does not exist at {CUSTOM_MAPPING_PATH}")


@caps_app.command("help")
def cap_help() -> None:
    """Display help information about cap mapping management."""
    typer.echo("Cap Mapping Management Help")
    typer.echo("----------------------------")
    typer.echo(
        "Capfinder allows you to customize cap mappings. These mappings persist across runs."
    )
    typer.echo(f"\nYour custom mappings are stored in: {CUSTOM_MAPPING_PATH}")
    typer.echo("\nAvailable commands:")
    typer.echo("  capfinder capmap add <int> <name>  : Add or update a cap mapping")
    typer.echo("  capfinder capmap remove <int>      : Remove a cap mapping")
    typer.echo("  capfinder capmap list              : List all current cap mappings")
    typer.echo("  capfinder capmap reset             : Reset cap mappings to default")
    typer.echo(
        "  capfinder capmap config            : Show the location of the configuration file"
    )
    typer.echo("\nExamples:")
    typer.echo("  capfinder capmap add 7 new_cap_type")
    typer.echo("  capfinder capmap remove 7")
    typer.echo("  capfinder capmap list")
    typer.echo(
        "\nNote: Changes to cap mappings are immediately saved and will persist across runs."
    )
    typer.echo(
        "When adding a new cap, you must use the next available number in the sequence."
    )


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"Capfinder v{version_info}")
        raise typer.Exit()


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
            └── (Optional) plots directory containing cap signal plots, if --plot-signal is used.\n
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
            "--plot-signal/--no-plot-signal",
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
            "max_examples": None,
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
    # reference: Annotated[
    #     str,
    #     typer.Option("--reference", "-r", help="Reference Sequence (5' -> 3')"),
    # ] = "GCTTTCGTTCGTCTCCGGACTTATCGCACCACCTATCCATCATCAGTACTGT",
    # cap_n1_pos0: Annotated[
    #     int,
    #     typer.Option(
    #         "--cap_n1_pos0",
    #         "-p",
    #         help="0-based index of 1st nucleotide (N1) of cap in the reference",
    #     ),
    # ] = 52,
    n_cpus: Annotated[
        int,
        typer.Option(
            "--n_cpus",
            "-n",
            help=textwrap.dedent(
                """\
            Number of CPUs to use for parallel processing.
            We use multiple CPUs during processing for POD5 file and BAM data (Step 1/5).
            For faster processing of this data (POD5 & BAM), increase the number of CPUs.
            For inference (Step 4/5), only a single CPU is used no matter how many CPUs you have specified.
            For faster inference, have a GPU available (it will be detected automatically) and set dtype to 'float16'."""
            ),
        ),
    ] = 1,
    dtype: Annotated[
        str,
        typer.Option(
            "--dtype",
            "-d",
            help=textwrap.dedent(
                """\
            Data type for model input. Valid values are 'float16', 'float32', or 'float64'.
            If you do not have a GPU, use 'float32' or 'float64' for better performance.
            If you have a GPU, use 'float16' for faster inference."""
            ),
        ),
    ] = "float16",
    # target_length: Annotated[
    #     int,
    #     typer.Option(
    #         "--target_length",
    #         "-t",
    #         help="Number of signal points in cap signal to consider",
    #     ),
    # ] = 500,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch_size",
            "-bs",
            help=textwrap.dedent(
                """\
            Batch size for model inference.
            Larger batch sizes can speed up inference but require more memory."""
            ),
        ),
    ] = 128,
    plot_signal: Annotated[
        bool,
        typer.Option(
            "--plot-signal/--no-plot-signal",
            help=textwrap.dedent(
                """\
                "Whether to plot extracted cap signal or not.
                Saving plots can help you plot the read's signal, and plot the signal for cap and flanking bases(&#177;5)."""
            ),
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
        --n_cpus 10 \\
        --dtype float16 \\
        --batch_size 256 \\
        --no-plot-signal \\
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

    global formatted_command_global

    predict_cap_types(
        bam_filepath=bam_filepath,
        pod5_dir=pod5_dir,
        num_cpus=n_cpus,
        output_dir=output_dir,
        dtype=dt,
        reference="GCTTTCGTTCGTCTCCGGACTTATCGCACCACCTATCCATCATCAGTACTGT",
        cap0_pos=52,
        train_or_test="test",
        plot_signal=plot_signal,
        cap_class=-99,
        target_length=500,
        batch_size=batch_size,
        debug_code=debug_code,
        refresh_cache=refresh_cache,
        formatted_command=formatted_command_global,  # Pass the formatted command here
    )
    logger.success("Finished predicting cap types!")


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        None, "--version", callback=version_callback, is_eager=True
    ),
) -> None:
    global formatted_command_global

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
        formatted_command_global = " \\\n".join(command_parts)
        return None


if __name__ == "__main__":
    app()
