import os
import shutil
import signal
import subprocess
from datetime import datetime
from typing import Dict, Literal, Optional, Tuple, Type, Union

import comet_ml
import jax
from comet_ml import Experiment  # Import CometML before keras

os.environ["KERAS_BACKEND"] = (
    "jax"  # the placement is important, it cannot be after keras import
)

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_tuner import BayesianOptimization, Hyperband, Objective, RandomSearch
from loguru import logger
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from capfinder.cnn_lstm_model import CapfinderHyperModel as CNNLSTMModel
from capfinder.data_loader import load_datasets
from capfinder.encoder_model import CapfinderHyperModel as EncoderModel
from capfinder.train_etl import train_etl
from capfinder.utils import map_cap_int_to_name

# Declare and initialize global stop_training flag
global stop_training
stop_training = False

ModelType = Literal["cnn_lstm", "encoder"]


def get_model(model_type: ModelType) -> Type[CNNLSTMModel] | Type[EncoderModel]:
    if model_type == "cnn_lstm":
        return CNNLSTMModel
    elif model_type == "encoder":
        return EncoderModel


def handle_interrupt(
    signum: Optional[int] = None, frame: Optional[object] = None
) -> None:
    """
    Handles interrupt signals (e.g., Ctrl+C) by setting a global flag to stop training.

    Args:
        signum: The signal number (optional).
        frame: The current stack frame (optional).

    Returns:
        None
    """
    global stop_training
    stop_training = True


class InterruptCallback(keras.callbacks.Callback):
    """
    Callback to interrupt training based on a global flag.
    """

    def on_train_batch_end(
        self, batch: int, logs: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Checks the global `stop_training` flag at the end of each batch.
        If True, interrupts training and logs a message.

        Args:
            batch: The current batch index (integer).
            logs: Optional dictionary of training metrics at the end of the batch (default: None).

        Returns:
            None
        """
        global stop_training
        if stop_training:
            logger.info("Training interrupted by user during batch.")
            self.model.stop_training = True

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        """
        Checks the global `stop_training` flag at the end of each epoch.
        If True, interrupts training and logs a message.

        Args:
            epoch: The current epoch index (integer).
            logs: Optional dictionary of training metrics at the end of the epoch (default: None).

        Returns:
            None
        """
        global stop_training
        if stop_training:
            logger.info(
                "Training interrupted by user at the end of epoch %d.", epoch + 1
            )
            self.model.stop_training = True


# Define a function to generate a unique filename with a datetime suffix
def generate_unique_name(base_name: str, extension: str) -> str:
    """Generate a unique filename with a datetime suffix.

    Parameters:
    -----------
    base_name: str
        The base name of the file.
    extension: str
        The file extension.

    Returns:
    --------
    str
        The unique filename with the datetime suffix.
    """
    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Append the date and time to the base name
    unique_filename = f"{base_name}_{current_datetime}{extension}"
    return unique_filename


def save_model(
    model: keras.Model, base_name: str, extension: str, save_dir: str
) -> str:
    """
    Save the given model to a specified directory.

    Parameters:
    -----------
    model: keras.Model
        The model to be saved.
    base_name: str
        The base name for the saved model file.
    extension: str
        The file extension for the saved model file.
    save_dir: str
        The directory where the model should be saved.

    Returns:
    --------
    str
        The full path where the model was saved.
    """
    # Generate a unique filename for the model
    model_filename = generate_unique_name(base_name, extension)

    # Construct the full path where the model should be saved
    model_save_path = os.path.join(save_dir, model_filename)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Save the model to the specified path
    model.save(model_save_path)

    # Return the save path
    return model_save_path


def set_jax_as_backend() -> None:
    """Set JAX as the backend for Keras distributed training.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    # Set the Keras backend to JAX
    logger.info(f"Backend for training: {keras.backend.backend()}")

    # Retrieve available CPU devices
    devices = jax.devices()

    # Log available devices
    for device in devices:
        logger.info(f"Device available for training: {device}, Type: {device.platform}")

    # Define a 1D device mesh for data parallelism
    mesh_1d = keras.distribution.DeviceMesh(
        shape=(len(devices),), axis_names=["data"], devices=devices
    )

    # Create a DataParallel distribution
    data_parallel = keras.distribution.DataParallel(device_mesh=mesh_1d)

    # Set the global distribution
    keras.distribution.set_distribution(data_parallel)


def initialize_comet_ml_experiment(project_name: str) -> Experiment:
    """Initialize a CometML experiment for logging.

    Parameters:
    -----------
    project_name: str
        The name of the CometML project.

    Returns:
    --------
    Experiment:
        An instance of the CometML Experiment class.
    """
    comet_api_key = os.getenv("COMET_API_KEY")
    comet_ml.init(project_name=project_name, api_key=comet_api_key)
    if comet_api_key:
        logger.info("Found CometML API key!")
        experiment = Experiment(
            auto_output_logging="native",
            auto_histogram_weight_logging=True,
            auto_histogram_gradient_logging=False,
            auto_histogram_activation_logging=False,
        )
    else:
        experiment = None
        logger.error(
            """CometML API key is not set.
            Please set it as an environment variable using
            export COMET_API_KEY="YOUR_API_KEY"."""
        )

    return experiment


def initialize_tuner(
    hyper_model: "CNNLSTMModel | EncoderModel",
    tune_params: dict,
    model_save_dir: str,
    model_type: ModelType,
) -> Union[Hyperband, BayesianOptimization, RandomSearch]:
    """Initialize a Keras Tuner object based on the specified tuning strategy.

    Parameters:
    -----------
    hyper_model: CapfinderHyperModel
        An instance of the CapfinderHyperModel class.
    tune_params: dict
        A dictionary containing the hyperparameters for tuning.
    model_save_dir: str
        The directory where the model should be saved.
    comet_project_name: str
    model_type: ModelType
        Type of the model to be trained.

    Returns:
    --------
    Union[Hyperband, BayesianOptimization, RandomSearch]:
        An instance of the Keras Tuner class based on the specified tuning strategy.
    """

    tuning_strategy = tune_params["tuning_strategy"].lower()
    if tuning_strategy not in ["random_search", "bayesian_optimization", "hyperband"]:
        tuning_strategy = "hyperband"
        logger.warning(
            "Invalid tuning strategy. Using Hyperband. Valid options are: 'random_search', 'bayesian_optimization', and 'hyperband'"
        )

    if tuning_strategy == "hyperband":
        tuner = Hyperband(
            hypermodel=hyper_model.build,
            objective=Objective("val_sparse_categorical_accuracy", direction="max"),
            max_epochs=tune_params["max_epochs_hpt"],
            factor=tune_params["factor"],
            overwrite=tune_params["overwrite"],
            directory=model_save_dir,
            seed=tune_params["seed"],
            project_name=tune_params["comet_project_name"] + "_" + model_type,
        )
    elif tuning_strategy == "bayesian_optimization":
        tuner = BayesianOptimization(
            hypermodel=hyper_model.build,
            objective=Objective("val_sparse_categorical_accuracy", direction="max"),
            max_trials=tune_params["max_trials"],
            overwrite=tune_params["overwrite"],
            directory=model_save_dir,
            seed=tune_params["seed"],
            project_name=tune_params["comet_project_name"] + "_" + model_type,
        )
    elif tuning_strategy == "random_search":
        tuner = RandomSearch(
            hypermodel=hyper_model.build,
            objective=Objective("val_sparse_categorical_accuracy", direction="max"),
            max_trials=tune_params["max_trials"],
            overwrite=tune_params["overwrite"],
            directory=model_save_dir,
            seed=tune_params["seed"],
            project_name=tune_params["comet_project_name"] + "_" + model_type,
        )
    return tuner


def kill_gpu_processes() -> None:
    """
    Terminates processes running on the NVIDIA GPU and sets the Keras dtype policy to float16.

    This function checks if the `nvidia-smi` command exists and, if found, attempts
    to terminate all Python processes utilizing the GPU. If no NVIDIA GPU is found,
    the function skips the termination step. It also sets the Keras global policy to
    mixed_float16 for faster training.

    Returns:
        None
    """
    if shutil.which("nvidia-smi") is None:
        print("No NVIDIA GPU found. Skipping GPU process termination.")
        return

    # set the keras dtype policy to float16 for faster training
    keras.mixed_precision.set_global_policy("mixed_float16")

    try:
        # Get the list of GPU processes
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        lines = result.stdout.split("\n")

        # Parse the lines to find PIDs of processes using the GPU
        for line in lines:
            if "python" in line:  # Adjust this if other processes need to be terminated
                parts = line.split()
                pid = parts[4]
                print(f"Terminating process with PID: {pid}")
                subprocess.run(["kill", "-9", pid])
    except Exception as e:
        print(f"Error occurred while terminating GPU processes: {str(e)}")


def get_shape_with_batch_size(
    dataset: tf.data.Dataset, batch_size: int
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    features_spec, labels_spec = dataset.element_spec
    feature_shape = list(features_spec.shape)
    label_shape = list(labels_spec.shape)

    # Replace None with the actual batch size
    feature_shape[0] = batch_size
    label_shape[0] = batch_size

    return tuple(feature_shape), tuple(label_shape)


def check_and_download_artifact(etl_params: dict) -> dict:
    dataset_info = {}
    experiment = initialize_comet_ml_experiment(project_name="capfinder-datasets")

    art = experiment.get_artifact(
        artifact_name="cap_data",
        version_or_alias=etl_params["remote_dataset_version"],
    )

    current_art_version = f"{art.version.major}.{art.version.minor}.{art.version.patch}"

    version_file = os.path.join(etl_params["save_dir"], "artifact_version.txt")

    # Check if version file exists and read the stored version
    if os.path.exists(version_file):
        with open(version_file) as f:
            stored_version = f.read().strip()

        # If versions match, skip download
        if stored_version == current_art_version:
            print(
                f"Artifact version {art.version} already downloaded. Skipping download."
            )
            dataset_info["version"] = current_art_version
            dataset_info["etl_experiment_url"] = (
                f"Used data version {current_art_version} from COMETs data registry"
            )
            experiment.end()
            return dataset_info

    # If versions don't match or file doesn't exist, download artifact
    art.download(path=etl_params["save_dir"], overwrite_strategy=True)

    # Save the new version to the file
    with open(version_file, "w") as f:
        f.write(current_art_version)

    dataset_info["version"] = current_art_version
    dataset_info["etl_experiment_url"] = (
        f"Used data version {current_art_version} from COMETs data registry"
    )

    experiment.end()
    return dataset_info


def run_training_pipeline(
    etl_params: dict,
    tune_params: dict,
    train_params: dict,
    model_save_dir: str,
    model_type: ModelType,
) -> None:
    kill_gpu_processes()
    # set_jax_as_backend()

    """
    #################################################
    #############          ETL         ##############
    #################################################
    """

    if etl_params["use_local_dataset"]:
        x_train, y_train, _, x_test, y_test, _, dataset_info = train_etl(
            etl_params["data_dir"],
            etl_params["save_dir"],
            etl_params["target_length"],
            etl_params["dtype"],
            etl_params["n_workers"],
        )
    else:  # use remote dataset
        dataset_info = check_and_download_artifact(etl_params)

    train_x_file_path = os.path.join(etl_params["save_dir"], "train_x.csv")
    train_y_file_path = os.path.join(etl_params["save_dir"], "train_y.csv")
    test_x_file_path = os.path.join(etl_params["save_dir"], "test_x.csv")
    test_y_file_path = os.path.join(etl_params["save_dir"], "test_y.csv")

    # Load datasets
    batch_size = tune_params["batch_size"]
    train_dataset, test_dataset = load_datasets(
        train_x_file_path,
        train_y_file_path,
        test_x_file_path,
        test_y_file_path,
        batch_size,
        num_timesteps=etl_params["target_length"],
    )

    logger.info("Dataset loaded successfully!")
    train_feature_shape, train_label_shape = get_shape_with_batch_size(
        train_dataset, tune_params["batch_size"]
    )
    logger.info(f"x_train shape: {train_feature_shape}")
    logger.info(f"y_train shape: {train_label_shape}")
    test_feature_shape, test_label_shape = get_shape_with_batch_size(
        test_dataset, tune_params["batch_size"]
    )
    logger.info(f"x_test shape: {test_feature_shape}")
    logger.info(f"y_test shape: {test_label_shape}")
    logger.info(f"Dataset version: {dataset_info['version']}")

    """
    #################################################
    #############        TUNE          ##############
    #################################################
    """

    tune_experiment = initialize_comet_ml_experiment(
        project_name=tune_params["comet_project_name"] + "_" + model_type
    )
    tune_experiment_url = tune_experiment.url
    if model_type not in ["cnn_lstm", "encoder"]:
        raise ValueError("Invalid model type. Expected 'cnn_lstm' or 'encoder'.")
    model = get_model(model_type)

    hyper_model = model(
        input_shape=(etl_params["target_length"], 1), n_classes=etl_params["n_classes"]
    )

    tuner = initialize_tuner(hyper_model, tune_params, model_save_dir, model_type)

    # tensorboard_save_path = os.path.join(model_save_dir, "tensorboard_logs_encoder", model_type)
    # logger.info(
    #     f"Run tensorboard as following:\ntensorboard --logdir {tensorboard_save_path}"
    # )

    # Split train into train-val sets
    dataset_size = train_dataset.reduce(0, lambda x, _: x + 1).numpy()

    train_size = int(0.8 * dataset_size)  # 80% for training
    dataset_copy = train_dataset
    train_dataset = dataset_copy.take(train_size)
    val_dataset = dataset_copy.skip(train_size)
    val_size = dataset_size - train_size

    # Apply repeat to the datasets
    train_dataset = train_dataset.repeat()
    val_dataset = val_dataset.repeat()

    try:
        tuner.search(
            train_dataset,
            steps_per_epoch=train_size,
            validation_data=val_dataset,
            validation_steps=val_size,
            epochs=tune_params["max_epochs_hpt"],
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=tune_params["patience"], restore_best_weights=True
                ),
                # keras.callbacks.TensorBoard(log_dir=tensorboard_save_path),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    verbose=1,
                    mode="min",
                    min_lr=1e-6,
                ),
            ],
        )
    except KeyboardInterrupt:
        logger.info("Hyperparameter search was canceled.")
    finally:
        # Retrieve the best hyperparameters found during the search
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    log_params = {
        "ETL params": etl_params,
        "Tune params": tune_params,
        "Train params": train_params,
        "Data Shapes": {
            "x_train": train_feature_shape,
            "y_train": train_label_shape,
            "x_test": test_feature_shape,
            "y_test": test_label_shape,
        },
        "Dataset version": dataset_info["version"],
        "Best Hyperparameters": {
            f"{key}": value for key, value in best_hp.values.items()
        },
    }

    if tune_experiment:
        tune_experiment.log_parameters(log_params)
        tune_experiment.end()

    """
    #################################################
    #############        TRAIN         ##############
    #################################################
    """

    train_experiment = initialize_comet_ml_experiment(
        project_name=train_params["comet_project_name"] + "_" + model_type
    )
    comet_callback = train_experiment.get_callback(framework="keras")

    if train_experiment:
        train_experiment.log_parameters(log_params)

    # Final model training
    best_model = hyper_model.build(best_hp)
    best_encoder_model = hyper_model.encoder_model

    logger.info("Training on best hyperparameters...")

    # Instantiate the InterruptCallback
    interrupt_callback = InterruptCallback()
    signal.signal(
        signal.SIGINT, handle_interrupt
    )  # location is important for this line

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=train_params["patience"],
        verbose=1,
        mode="auto",
        restore_best_weights=True,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, verbose=1, mode="min", min_lr=1e-6
    )

    # Callbacks
    best_model_path = os.path.join(model_save_dir, "best_model_weights.weights.h5")
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        best_model_path,
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )

    best_model.fit(
        train_dataset,
        steps_per_epoch=train_size,
        validation_data=val_dataset,
        validation_steps=val_size,
        epochs=train_params["max_epochs_final_model"],
        batch_size=train_params["batch_size"],
        callbacks=[
            early_stopping,
            reduce_lr,
            comet_callback,
            model_checkpoint,
            interrupt_callback,
        ],
    )
    logger.success("Training on best hyperparameters done!")
    best_model.load_weights(best_model_path)

    # Evaluate the best model on the test data
    y_true_test = []
    y_pred_test = []

    logger.info("Making predictions on test set for confusion matrix...")
    # Iterate through the test_dataset to collect true labels and predictions
    total_test_batches = test_dataset.reduce(0, lambda x, _: x + 1).numpy()
    pbar = tqdm(
        test_dataset, total=total_test_batches, desc="Processing test dataset batches"
    )
    for batch_num, (features, labels) in enumerate(pbar, start=1):
        predictions = best_model.predict(
            features, verbose=0
        )  # Assuming `best_model` is your trained model
        y_true_test.extend(labels.numpy())
        y_pred_test.extend(np.argmax(predictions, axis=1))
        # Update the progress bar description
        pbar.set_description(f"Processing batch {batch_num}/{total_test_batches}")
        pbar.set_postfix({"Last batch shape": features.shape})

    # Convert lists to numpy arrays
    y_true_test = np.array(y_true_test)  # type: ignore
    y_pred_test = np.array(y_pred_test)  # type: ignore
    logger.success("Predictions on test set done!")

    # Assuming `map_cap_int_to_name` maps integers to class names
    class_labels = [map_cap_int_to_name(i) for i in range(etl_params["n_classes"])]

    # Evaluate the best model on the test data
    logger.info("Evaluate final model performance on test dataset")
    test_loss, test_acc = best_model.evaluate(test_dataset)
    logger.success("Model evaluation on test set done!")
    logger.info(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

    # Log the test set confusion matrix to the Comet ML dashboard pane
    train_experiment.log_confusion_matrix(
        title="Test Confusion Matrix",
        y_true=y_true_test,
        y_predicted=y_pred_test,
        labels=class_labels,
    )

    # Save the best model
    classifier_name = "classifier"
    encoder_name = "encoder"
    model_save_path = save_model(
        best_model,
        classifier_name,
        ".keras",
        os.path.join(model_save_dir, "classifier"),
    )
    if best_encoder_model is not None:
        encoder_save_path = save_model(
            best_encoder_model,
            encoder_name,
            ".keras",
            os.path.join(model_save_dir, "encoder"),
        )

    if train_experiment:
        train_experiment.log_model("Classifier", model_save_path)
        logger.info(f"Best classifier model saved at: \n{model_save_path}")
        if best_encoder_model is not None:
            train_experiment.log_model("Encoder", encoder_save_path)
            logger.info(f"Best encoder model saved at: \n{encoder_save_path}")

    # Log the confusion matrix
    # Initialize empty lists for true and predicted labels
    y_true = []
    y_pred = []
    total_batches = train_size  # Assuming this is the correct number of batches

    # Predict using the model on training data
    logger.info("Making predictions on training set for confusion matrix...")
    train_dataset = train_dataset.take(train_size)  # because it is repeated

    pbar = tqdm(train_dataset, total=total_batches, desc="Processing batches")
    for batch_num, (features, labels) in enumerate(pbar, start=1):
        predictions = best_model.predict(
            features, verbose=0  # Set to 0 to avoid nested progress bars
        )
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

        # Update the progress bar description
        pbar.set_description(f"Processing batch {batch_num}/{total_batches}")
        pbar.set_postfix({"Last batch shape": features.shape})

    # Convert lists to numpy arrays
    y_true = np.array(y_true)  # type: ignore
    y_pred = np.array(y_pred)  # type: ignore

    # Assuming `map_cap_int_to_name` maps integers to class names
    class_labels = [map_cap_int_to_name(i) for i in range(etl_params["n_classes"])]

    # Compute confusion matrix
    conf_matrix_train = confusion_matrix(
        y_true, y_pred, labels=range(etl_params["n_classes"])
    )
    logger.success("Done making predictions on training set for confusion matrix!")

    # Create DataFrame for better visualization
    conf_matrix_train_df = pd.DataFrame(
        conf_matrix_train, index=class_labels, columns=class_labels
    )
    conf_matrix_str_train = conf_matrix_train_df.to_string()

    if train_experiment:
        train_experiment.log_text(
            text=dataset_info["etl_experiment_url"],
            metadata={"Description": "ETL Experiment URL"},
        )
        train_experiment.log_text(
            text=tune_experiment_url,
            metadata={"Description": "Tune Experiment URL"},
        )
        train_experiment.log_text(
            text=conf_matrix_str_train,
            metadata={"Description": "Train Confusion Matrix"},
        )
        # train_experiment.log_text(
        #     text=f"tensorboard --logdir {tensorboard_save_path}",
        #     metadata={"Description": "Tensorboard command"},
        # )

        # Log the test set confusion matrix to the Comet ML dashboard pane
        train_experiment.log_confusion_matrix(
            title="Test Confusion Matrix",
            y_true=y_true_test,
            y_predicted=y_pred_test,
            labels=class_labels,
        )

        train_experiment.log_text(
            text=f"{test_acc}",
            metadata={"Description": "Test dataset accuracy"},
        )

        train_experiment.end()


if __name__ == "__main__":
    # Configure settings here
    etl_params = {
        "data_dir": "/export/valenfs/data/processed_data/MinION/9_madcap/4_make_train_dataset_202405/dataset",
        "save_dir": "/export/valenfs/data/processed_data/MinION/9_madcap/dummy_data/saved_data3/",
        "target_length": 500,  # length of time series
        "dtype": "float16",  # data type of the time series
        "n_workers": 10,  # number of workers for parallel processing (by prefect)
        "n_classes": 4,  # number of classes in the dataset
        "use_local_dataset": False,  # set to False to use the online dataset, otherwise the local dataset will be used and will be uplaoded to comet
        "remote_dataset_version": "8.0.0",  # version of the online dataset to use
    }

    tune_params = {
        "comet_project_name": "capfinder_tfr_tune-delete",
        "patience": 0,
        "max_epochs_hpt": 3,
        "max_trials": 5,  # for random_search, and bayesian_optimization. For hyperband this has no effect
        "factor": 2,
        "batch_size": 4,
        "seed": 42,
        "tuning_strategy": "hyperband",  # "hyperband" or "random_search" or "bayesian_optimization"
        "overwrite": True,
    }

    train_params = {
        "comet_project_name": "capfinder_tfr_train-delete",
        "patience": 20,
        "max_epochs_final_model": 1,
        "batch_size": 4,
    }

    model_save_dir = (
        "/export/valenfs/data/processed_data/MinION/9_madcap/5_trained_models_202405/"
    )
    model_type: ModelType = "encoder"  # cnn_lstm

    # Run the training pipeline
    run_training_pipeline(
        etl_params=etl_params,
        tune_params=tune_params,
        train_params=train_params,
        model_save_dir=model_save_dir,
        model_type=model_type,
    )
