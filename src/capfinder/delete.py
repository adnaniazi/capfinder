import csv
from typing import Generator, Tuple

# Assuming these are imported from your existing module


def csv_generator(file_path: str) -> Generator[Tuple[str, str, str], None, None]:
    """
    Generates rows from a CSV file one at a time.

    Args:
        file_path (str): Path to the CSV file.

    Yields:
        Tuple[str, str, str]: A tuple containing read_id, cap_class, and timeseries as strings.
    """
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            yield (str(row[0]), str(row[1]), str(row[2]))


for row in csv_generator(
    "/export/valenfs/data/processed_data/MinION/9_madcap/3_all_train_csv_202405/all_csvs/data__cap_2-1.csv"
):
    print(row)
