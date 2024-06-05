"""
The module contains some common utility functions used in the capfinder package.

Author: Adnan M. Niazi
Date: 2024-02-28
"""

import gzip
import sqlite3
from typing import IO, Tuple, Type, Union, cast

import numpy as np
from loguru import logger


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


def map_cap_int_to_name(cap_class: int) -> str:
    """Map the integer representation of the CAP class to the CAP name.

    Args:
        cap_class (int): Integer representation of the CAP class.

    Returns:
        cap_name (str): The name of the CAP class.
    """
    cap_mapping = {
        0: "cap_0",
        1: "cap_1",
        2: "cap_2",
        3: "cap_2-1",
        4: "cap_TMG",
        5: "cap_NAD",
        6: "cap_FAD",
        -99: "cap_unknown",
    }
    return cap_mapping[cap_class]


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
