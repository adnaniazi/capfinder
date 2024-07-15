# from capfinder.train_etl import (
#     list_csv_files,
#     load_csv,
#     concatenate_dataframes,
#     make_x_y_read_id_sets,
# )
# from capfinder.inference_data_loader import load_inference_dataset
# from typing import Any, List, Optional, Tuple, Type, Union
# import os
# import numpy as np
# import polars as pl
# from loguru import logger
# from polars import DataFrame
# from prefect import flow, task
# from prefect.engine import TaskRunContext
# from prefect.tasks import task_input_hash
# from typing_extensions import Literal

# from capfinder.utils import get_dtype

# # Define custom types
# TrainData = Tuple[
#     np.ndarray,  # x_train
#     np.ndarray,  # y_train
#     pl.Series,  # read_id_train
# ]
# DtypeLiteral = Literal["float16", "float32", "float64"]
# DtypeNumpy = Union[np.float16, np.float32, np.float64]


# def save_to_file(x, y, read_id, output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir, exist_ok=True)

#     dfp_x = pl.from_numpy(data=x.reshape(x.shape[:2]), orient="row")
#     dfp_y = pl.from_numpy(data=y.reshape(y.shape[:2]), orient="row")
#     dfp_id = pl.DataFrame(output_dir)

#     x_path = os.path.join(output_dir, "x.csv")
#     dfp_x.write_csv(file=x_path)
#     y_path = os.path.join(output_dir, "y.csv")
#     dfp_y.write_csv(file=y_path)
#     id_path = os.path.join(output_dir, "read_id.csv")
#     dfp_id.write_csv(file=id_path)


# # Top level task because Prefect does not allow flow level
# # results to be cached to memory. We create a flow as a task
# # to save the result to memory.
# # https://github.com/PrefectHQ/prefect/issues/7288
# @flow(name="prepare-inference-data")
# def prepare_inference_data(
#     data_dir: str, output_dir: str, target_length: int, dtype_n: Type[np.floating]
# ) -> TrainData:
#     """Pipeline task to load and concatenate CSV files.

#     Parameters
#     ----------
#     data_dir: str
#         Path to the directory containing raw CSV files.

#     output_dir: str
#         Path to the directory where the processed data will be saved.

#     target_length: int
#         The desired length of each time series.

#     dtype_n: Type[np.floating]
#         The data type to use for the features.
#     """

#     csv_files = list_csv_files(data_dir)
#     loaded_dataframes = [load_csv(file_path) for file_path in csv_files]
#     concatenated_df = concatenate_dataframes(loaded_dataframes)
#     x, y, read_id = make_x_y_read_id_sets(concatenated_df, target_length, dtype_n)
#     save_to_file(x, y, read_id, output_dir)


# def main(
#     cap_signal_dir: str,
#     output_dir: str,
#     dtype: DtypeLiteral,
#     target_length: int = 500,
#     batch_size: int = 1024,
#     model_version: str = "latest",
# ):

#     prepare_inference_data(
#         data_dir=cap_signal_dir,
#         output_dir=output_dir,
#         target_length=target_length,
#         dtype=dtype,
#     )
#     load_inference_dataset(
#         x_path=os.path.join(output_dir, "x.csv"),
#         y_path=os.path.join(output_dir, "y.csv"),
#         read_id_path=os.path.join(output_dir, "read_id.csv"),
#         batch_size=batch_size,
#         num_timesteps=target_length,
#     )
