import gzip
from typing import IO, Union


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
