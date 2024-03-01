"""
The module contains some common utility functions used in the capfinder package.

Author: Adnan M. Niazi
Date: 2024-02-28
"""
import gzip
import sqlite3
from typing import IO, Tuple, Union


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
