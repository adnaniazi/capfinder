If you need to integrate Capfinder directly into your Python scripts instead of using it via the terminal, you can utilize the Python API as follows:

### Extract Cap Signal

```python
from capfinder.cli import app

app(
    ["extract-cap-signal",
     "--bam_filepath", "/path/to/bam",
     "--pod5_dir", "/path/to/pod5",
     "--reference", "GCTTTCGTTCGTCTCCGGACTTATCGCACCACCTATCCATCATCAGTACTGT",
     "--cap_class", "1",
     "--cap_n1_pos0", "52",
     "--train_or_test", "test",
     "--output_dir", "/path/to/output"]
)
```

This example demonstrates how to use the Capfinder Python API to extract the signal corresponding to the RNA cap type from BAM and POD5 files. The `extract-cap-signal` command is invoked using the `app.run()` method, passing in the necessary parameters such as the paths to the BAM and POD5 files, the reference sequence, the cap class, the position of the cap N1 base, whether the data is for training or testing, and the output directory.

### Prepare the Training Dataset

```python
app(
    ["make-train-dataset",
     "--caps_data_dir", "/path/to/csv",
     "--output_dir", "/path/to/save",
     "--target_length", "500",
     "--dtype", "float16"]
)
```

This example shows how to use the Capfinder Python API to prepare the dataset for training the machine learning model. The `make-train-dataset` command is called, where you specify the directory containing the cap signal CSV files, the output directory to save the processed dataset, the target length for the input sequences, and the data type to use for the dataset. This command can be run independently or is automatically invoked by the `train-model` command.


### Create a Training Configuration File

```python
app(
    ["create-train-config",
     "--file_path", "/path/to/config.json"]
)
```

This example demonstrates how to use the Capfinder Python API to create a JSON configuration file for the training pipeline. The `create-train-config` command is called, and you provide the file path where the configuration file should be saved.


### Train the Model

```python
app(
    ["train-model",
     "--config_file", "/path/to/config.json"]
)
```

This example shows how to use the Capfinder Python API to train the model. The `train-model` command is called, and you provide the path to the JSON configuration file that contains the training parameters.



### Predict Cap Types

```python
app(
    ["predict-cap-types",
     "--bam_filepath", "/path/to/bam",
     "--pod5_dir", "/path/to/pod5",
     "--output_dir", "/path/to/output",
     "--n_cpus", "10",
     "--dtype", "float16",
     "--batch_size", "256",
     "--plot-signal",
     "--debug",
     "--refresh-cache"]
)
```

This example demonstrates how to use the Capfinder Python API to predict the RNA cap types. The `predict-cap-types` command is called, and you provide the paths to the BAM and POD5 files, the output directory, the number of CPUs to use, the data type to use for the model input, the batch size for inference, and options to control signal plotting, debugging, and cache refreshing.


### Manage Cap Mappings

```python
app(
    ["capmap", "add", "--cap_int", "7", "--cap_name", "new_cap_type"]
)

app(
    ["capmap", "remove", "--cap_int", "7"]
)

app(
    ["capmap", "list"]
)

app(
    ["capmap", "reset"]
)

app(
    ["capmap", "config"]
)
```

These examples show how to use the Capfinder Python API to manage the cap mappings. The `capmap` commands are used to add a new cap mapping, remove an existing cap mapping, list all current cap mappings, reset the cap mappings to the default, show the location of the cap mapping configuration file, and display the help information for cap mapping management.
