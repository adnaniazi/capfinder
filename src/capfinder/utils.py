"""
The module contains some common utility functions used in the capfinder package.

Author: Adnan M. Niazi
Date: 2024-02-28
"""

import gzip
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import IO, Dict, Optional, Tuple, Type, Union, cast

import numpy as np
from comet_ml import Experiment  # Import CometML before keras
from loguru import logger


def initialize_comet_ml_experiment(project_name: str) -> Experiment:
    """
    Initialize a CometML experiment for logging.

    This function creates a CometML Experiment instance using the provided
    project name and the COMET_API_KEY environment variable.

    Parameters:
    -----------
    project_name: str
        The name of the CometML project.

    Returns:
    --------
    Experiment:
        An instance of the CometML Experiment class.

    Raises:
    -------
    ValueError:
        If the project_name is empty or None, or if the COMET_API_KEY is not set.
    RuntimeError:
        If there's an error initializing the experiment.
    """
    if not project_name:
        raise ValueError("Project name cannot be empty or None")

    comet_api_key = os.getenv("COMET_API_KEY")

    if not comet_api_key:
        logger.error(
            "CometML API key is not set. Please set the COMET_API_KEY environment variable."
        )
        logger.info("Example: export COMET_API_KEY='YOUR_API_KEY'")
        raise ValueError("COMET_API_KEY environment variable is not set")

    try:
        experiment = Experiment(
            api_key=comet_api_key,
            project_name=project_name,
            auto_output_logging="native",
            auto_histogram_weight_logging=True,
            auto_histogram_gradient_logging=False,
            auto_histogram_activation_logging=False,
            display_summary_level=0,
        )
        logger.info(
            f"Successfully initialized CometML experiment for project: {project_name}"
        )
        return experiment
    except Exception as e:
        logger.error(f"Failed to initialize CometML experiment: {str(e)}")
        raise RuntimeError(f"Failed to initialize CometML experiment: {str(e)}") from e


def file_opener(filename: str) -> Union[IO[str], IO[bytes]]:
    """
    Open a file for reading. If the file is compressed, use gzip to open it.

    Args:
        filename (str): The path to the file to open.

    Returns:
        file object: A file object that can be used for reading.
    """
    if filename.endswith(".gz"):
        # Compressed FASTQ file (gzip)
        return gzip.open(filename, "rt")
    else:
        # Uncompressed FASTQ file
        return open(filename)


def open_database(
    database_path: str,
) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    """
    Open the database connection based on the database path.

    Params:
        database_path (str): Path to the database.

    Returns:
        conn (sqlite3.Connection): Connection object for the database.
        cursor (sqlite3.Cursor): Cursor object for the database.
    """
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    return conn, cursor


def get_dtype(dtype: str) -> Type[np.floating]:
    """
    Returns the numpy floating type corresponding to the provided dtype string.

    If the provided dtype string is not valid, a warning is logged and np.float32 is returned as default.

    Parameters:
    dtype (str): The dtype string to convert to a numpy floating type.

    Returns:
    Type[np.floating]: The corresponding numpy floating type.
    """
    valid_dtypes = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
    }

    if dtype in valid_dtypes:
        dt = valid_dtypes[dtype]
    else:
        logger.warning('You provided an invalid dtype. Using "float32" as default.')
        dt = np.float32

    return cast(Type[np.floating], dt)  # Cast dt to the expected type


def get_terminal_width() -> int:
    """
    Get the width of the terminal.

    Returns:
        int: The width of the terminal in columns. Defaults to 80 if not available.
    """
    return shutil.get_terminal_size((80, 20)).columns


def log_header(text: str) -> None:
    """
    Log a centered header surrounded by '=' characters.

    Args:
        text (str): The text to be displayed in the header.

    Returns:
        None
    """
    width = get_terminal_width()
    header = f"\n{'=' * width}\n{text.center(width)}\n{'=' * width}"
    logger.info(header)


def log_subheader(text: str) -> None:
    """
    Log a centered subheader surrounded by '-' characters.

    Args:
        text (str): The text to be displayed in the header.

    Returns:
        None
    """
    width = get_terminal_width()
    header = f"\n{'-' * width}\n{text.center(width)}\n{'-' * width}"
    logger.info(header)


def log_step(step_num: int, total_steps: int, description: str) -> None:
    """
    Log a step in a multi-step process.

    Args:
        step_num (int): The current step number.
        total_steps (int): The total number of steps.
        description (str): A description of the current step.

    Returns:
        None
    """
    width = get_terminal_width()
    step = (
        f"\n{'-' * width}\nStep {step_num}/{total_steps}: {description}\n{'-' * width}"
    )
    logger.info(step)


def log_substep(text: str) -> None:
    """
    Log a substep or bullet point.

    Args:
        text (str): The text of the substep to be logged.

    Returns:
        None
    """
    logger.info(f"  • {text}")


def log_output(description: str) -> None:
    """
    Log a step in a multi-step process.

    Args:
        description (str): A description of the current step.

    Returns:
        None
    """
    width = get_terminal_width()
    text = f"\n{'-' * width}\n{description}"
    logger.info(text)


DEFAULT_CAP_MAPPING: Dict[int, str] = {
    -99: "cap_unknown",
    0: "cap_0",
    1: "cap_1",
    2: "cap_2",
    3: "cap_2-1",
}
# Global cap mapping that will be used in the application
global CAP_MAPPING
CAP_MAPPING: Dict[int, str] = {}

# Use pathlib for cross-platform compatibility
CONFIG_DIR = Path.home() / ".capfinder"
CUSTOM_MAPPING_PATH = CONFIG_DIR / "custom_mapping.json"


def ensure_config_dir() -> None:
    """Ensure the configuration directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def initialize_cap_mapping() -> None:
    """Initialize the cap mapping file if it doesn't exist."""
    global CAP_MAPPING
    ensure_config_dir()
    if not CUSTOM_MAPPING_PATH.exists() or CUSTOM_MAPPING_PATH.stat().st_size == 0:
        save_custom_mapping(DEFAULT_CAP_MAPPING)
    load_custom_mapping()


def map_cap_int_to_name(cap_class: int) -> str:
    """Map the integer representation of the CAP class to the CAP name."""
    global CAP_MAPPING

    return CAP_MAPPING.get(cap_class, f"Unknown cap: {cap_class}")


def update_cap_mapping(new_mapping: Dict[int, str]) -> None:
    """Update the CAP_MAPPING with new entries."""
    global CAP_MAPPING
    CAP_MAPPING.update(new_mapping)
    save_custom_mapping(CAP_MAPPING)


def load_custom_mapping() -> None:
    """Load custom mapping from JSON file if it exists."""
    global CAP_MAPPING
    try:
        if CUSTOM_MAPPING_PATH.exists():
            with CUSTOM_MAPPING_PATH.open("r") as f:
                loaded_mapping = json.load(f)
            # Convert string keys back to integers
            CAP_MAPPING = {int(k): v for k, v in loaded_mapping.items()}
        else:
            CAP_MAPPING = DEFAULT_CAP_MAPPING.copy()
    except json.JSONDecodeError:
        logger.error(
            "Failed to decode JSON from custom mapping file. Using default mapping."
        )
        CAP_MAPPING = DEFAULT_CAP_MAPPING.copy()
    except Exception as e:
        logger.error(
            f"Unexpected error loading custom mapping: {e}. Using default mapping."
        )
        CAP_MAPPING = DEFAULT_CAP_MAPPING.copy()

    if not CAP_MAPPING:
        logger.warning("Loaded mapping is empty. Using default mapping.")
        CAP_MAPPING = DEFAULT_CAP_MAPPING.copy()


def save_custom_mapping(mapping: Dict[int, str]) -> None:
    """Save the given mapping to JSON file."""
    ensure_config_dir()
    try:
        with CUSTOM_MAPPING_PATH.open("w") as f:
            json.dump(mapping, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save custom mapping: {e}")
        raise


def get_next_available_cap_number() -> int:
    """
    Find the next available cap number in the sequence.

    Returns:
    int: The next available cap number.
    """
    global CAP_MAPPING

    existing_caps = set(CAP_MAPPING.keys())
    existing_caps.discard(-99)  # Remove the special 'unknown' cap
    if not existing_caps:
        return 0
    max_cap = max(existing_caps)
    next_cap = max_cap + 1
    return next_cap


def is_cap_name_unique(new_cap_name: str) -> Optional[int]:
    """
    Check if the given cap name is unique among existing cap mappings.

    Args:
    new_cap_name (str): The new cap name to check for uniqueness.

    Returns:
    Optional[int]: The integer label of the existing cap with the same name, if any. None otherwise.
    """
    global CAP_MAPPING
    for cap_int, cap_name in CAP_MAPPING.items():
        if cap_name.lower() == new_cap_name.lower():
            return cap_int
    return None


initialize_cap_mapping()
