"""
We cannot random access a record in a BAM file, we can only iterate through it. That is our starting point.
For each record in BAM file, we need to find the corresponding record in POD5 file. For that we need a
mapping between POD5 file and read_ids. This is why we need to build an index of POD5 files. This module
helps us to build an index of POD5 files and stores it in a SQLite database.

Author: Adnan M. Niazi
Date: 2024-02-28
"""

import os
import sqlite3
from typing import Any, Generator, Tuple

from loguru import logger
from tqdm import tqdm


def initialize_database(
    database_path: str,
) -> Tuple[sqlite3.Cursor, sqlite3.Connection]:
    """
    Intializes the database connection based on the database path.

    Params:
        database_path (str): Path to the database.

    Returns:
        cursor (sqlite3.Cursor): Cursor object for the database.
        conn (sqlite3.Connection): Connection object for the database.
    """
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS pod5_index (pod5_filename TEXT PRIMARY KEY, pod5_filepath TEXT)"""
    )
    return cursor, conn


def write_database(
    data: Tuple[str, str], cursor: sqlite3.Cursor, conn: sqlite3.Connection
) -> None:
    """
    Write the index to a database.

    Params:
        data Tuple[str, str]): Tuples of fileroot and file
        cursor (sqlite3.Cursor): Cursor object for the database.
        conn (sqlite3.Connection): Connection object for the database.

    Returns:
        None
    """
    cursor.execute("INSERT or REPLACE INTO pod5_index VALUES (?, ?)", data)
    conn.commit()


def generate_pod5_path_and_name(
    pod5_path: str,
) -> Generator[Tuple[str, str], None, None]:
    """Traverse the directory and yield all the names+extension and
    fullpaths of the pod5 files.

    Params:
        pod5_path (str): Path to a POD5 file/directory of POD5 files.

    Yields:
        Tuple[str, str]: Tuple containing the name+extension and full path of a POD5 file.
    """

    if os.path.isdir(pod5_path):
        for root, _dirs, files in os.walk(pod5_path):
            for file in files:
                if file.endswith(".pod5"):
                    yield (file, os.path.join(root, file))
    elif os.path.isfile(pod5_path) and pod5_path.endswith(".pod5"):
        root = os.path.basename(pod5_path)
        file = os.path.dirname(pod5_path)
        yield (file, os.path.join(root, file))


def find_database_size(database_path: str) -> Any:
    """
    Find the number of records in the database.

    Params:
        database_path (str): Path to the database.

    Returns:
        size (Any): Number of records in the database.
    """
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM pod5_index")
    result = cursor.fetchone()
    size = result[0] if result is not None else 0
    return size


def fetch_filepath_using_filename(
    conn: sqlite3.Connection, cursor: sqlite3.Cursor, pod5_filename: str
) -> Any:
    """
    Retrieve the pod5_filepath based on pod5_filename from the database.

    Params:
        conn (sqlite3.Connection): Connection object for the database.
        cursor (sqlite3.Cursor): Cursor object for the database.
        pod5_filename (str): The pod5_filename to be searched for.

    Returns:
        pod5_filepath (Any): The corresponding pod5_filepath if found, else None.
    """
    try:
        # Execute the SQL query to retrieve the pod5_filepath based on pod5_filename
        cursor.execute(
            "SELECT pod5_filepath FROM pod5_index WHERE pod5_filename = ?",
            (pod5_filename,),
        )
        result = cursor.fetchone()

        # Return the result (pod5_filepath) or None if not found
        return result[0] if result else None

    except sqlite3.Error as e:
        logger.error(f"Error: {e}")
        return None


def index(pod5_path: str, output_dir: str) -> None:
    """
    Builds an index mapping read_ids to POD5 file paths.

    Params:
        pod5_path (str): Path to a POD5 file or directory of POD5 files.
        output_dir (str): Path where database.db file is written to.

    Returns:
        None
    """

    database_path = os.path.join(output_dir, "database.db")
    cursor, conn = initialize_database(database_path)
    total_files = sum(1 for _ in generate_pod5_path_and_name(pod5_path))
    logger.info(f"Indexing {total_files} POD5 files")
    for data in tqdm(
        generate_pod5_path_and_name(pod5_path),
        total=total_files,
        desc="",
        unit="files",
    ):
        write_database(data, cursor, conn)
    logger.success("Indexing complete")
    conn.close()


if __name__ == "__main__":
    pod5_path = "/export/valenfs/data/raw_data/minion/20230829_randomcap02/"
    output_dir = "/export/valenfs/data/processed_data/MinION/9_madcap/1_data/2_20230829_randomcap02/visualizations2"

    index(pod5_path, output_dir)
