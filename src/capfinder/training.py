import os
import signal
from datetime import datetime
from typing import Dict, Optional, Union

import comet_ml
import jax
import keras
import numpy as np
import pandas as pd
from comet_ml import Experiment  # Import CometML before keras
from keras_tuner import BayesianOptimization, Hyperband, Objective, RandomSearch
from loguru import logger
from sklearn.metrics import confusion_matrix

from capfinder.model import CapfinderHyperModel
from capfinder.train_etl import train_etl
from capfinder.utils import get_dtype, map_cap_int_to_name

# Uncomment the following lines to use JAX as the backend for Keras
# Also uncomment the set_jax_as_backend() code line in the run_training_pipeline function
# Tensorboard does not work properly with JAX backend

# os.environ["KERAS_BACKEND"] = (
#     "jax"  # the placement is important, it cannot be after keras import
# )


# Declare and initialize global stop_training flag
global stop_training
stop_training = False


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
            auto_histogram_gradient_logging=True,
            auto_histogram_activation_logging=True,
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
    hyper_model: CapfinderHyperModel,
    tune_params: dict,
    model_save_dir: str,
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
            project_name=tune_params["comet_project_name"],
        )
    elif tuning_strategy == "bayesian_optimization":
        tuner = BayesianOptimization(
            hypermodel=hyper_model.build,
            objective=Objective("val_sparse_categorical_accuracy", direction="max"),
            max_trials=tune_params["max_trials"],
            overwrite=tune_params["overwrite"],
            directory=model_save_dir,
            seed=tune_params["seed"],
            project_name=tune_params["comet_project_name"],
        )
    elif tuning_strategy == "random_search":
        tuner = RandomSearch(
            hypermodel=hyper_model.build,
            objective=Objective("val_sparse_categorical_accuracy", direction="max"),
            max_trials=tune_params["max_trials"],
            overwrite=tune_params["overwrite"],
            directory=model_save_dir,
            seed=tune_params["seed"],
            project_name=tune_params["comet_project_name"],
        )
    return tuner


def run_training_pipeline(
    etl_params: dict,
    tune_params: dict,
    train_params: dict,
    model_save_dir: str,
) -> None:
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
        dataset_info = {}
        experiment = initialize_comet_ml_experiment(project_name="capfinder-datasets")
        art = experiment.get_artifact(
            artifact_name="cap_data",
            version_or_alias=etl_params["remote_dataset_version"],
        )
        art.download(path=etl_params["save_dir"], overwrite_strategy=True)

        x_train = np.genfromtxt(
            os.path.join(etl_params["save_dir"], "train_x.csv"),
            delimiter=",",
            dtype=get_dtype(etl_params["dtype"]),
            skip_header=1,
        )
        y_train = np.genfromtxt(
            os.path.join(etl_params["save_dir"], "train_y.csv"),
            delimiter=",",
            dtype=np.int8,
            skip_header=1,
        )
        x_test = np.genfromtxt(
            os.path.join(etl_params["save_dir"], "test_x.csv"),
            delimiter=",",
            dtype=get_dtype(etl_params["dtype"]),
            skip_header=1,
        )
        y_test = np.genfromtxt(
            os.path.join(etl_params["save_dir"], "test_y.csv"),
            delimiter=",",
            dtype=np.int8,
            skip_header=1,
        )
        x_train = x_train.reshape(-1, x_train.shape[1], 1)
        x_test = x_test.reshape(-1, x_test.shape[1], 1)
        dataset_info["version"] = art.version
        dataset_info["etl_experiment_url"] = (
            f"Used data version {art.version} from COMETs data registry"
        )
        experiment.end()

    # Convert the integer labels to one-hot encoded labels
    # y_train = keras.utils.to_categorical(y_train, etl_params["n_classes"])
    # y_test = keras.utils.to_categorical(y_test, etl_params["n_classes"])

    logger.info("Dataset loaded successfully!")
    logger.info(f"x_train shape: {x_train.shape}")
    logger.info(f"x_test shape: {x_test.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"y_test shape: {y_test.shape}")
    logger.info(f"Dataset version: {dataset_info["version"]}")

    """
    #################################################
    #############        TUNE          ##############
    #################################################
    """

    tune_experiment = initialize_comet_ml_experiment(
        project_name=tune_params["comet_project_name"]
    )
    tune_experiment_url = tune_experiment.url
    # Hyperparameter tuning
    hyper_model = CapfinderHyperModel(
        input_shape=(etl_params["target_length"], 1), n_classes=etl_params["n_classes"]
    )

    tuner = initialize_tuner(hyper_model, tune_params, model_save_dir)

    tensorboard_save_path = os.path.join(
        model_save_dir,
        "tensorboard_logs",
        generate_unique_name(base_name="log", extension=""),
    )
    logger.info(
        f"Run tensorboard as following:\ntensorboard --logdir {tensorboard_save_path}"
    )

    try:
        tuner.search(
            x_train,
            y_train,
            validation_split=0.2,
            epochs=tune_params["max_epochs_hpt"],
            batch_size=tune_params["batch_size"],
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=tune_params["patience"], restore_best_weights=True
                ),
                keras.callbacks.TensorBoard(log_dir=tensorboard_save_path),
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
            "x_train": x_train.shape,
            "y_train": y_train.shape,
            "x_test": x_test.shape,
            "y_test": y_test.shape,
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
        project_name=train_params["comet_project_name"]
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

    best_model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=train_params["max_epochs_final_model"],
        batch_size=train_params["batch_size"],
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=train_params["patience"],
                verbose=1,
                mode="auto",
                restore_best_weights=True,
            ),
            comet_callback,
            interrupt_callback,
        ],
    )

    # Predict using the model on training data
    y_pred_probs_train = best_model.predict(x_train)
    y_pred_train = np.argmax(y_pred_probs_train, axis=1)

    # Evaluate the best model on the test data
    test_loss, test_acc = best_model.evaluate(x_test, y_test)
    y_pred_probs_test = best_model.predict(x_test)
    y_pred_test = np.argmax(y_pred_probs_test, axis=1)

    logger.info(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

    # Save the best model
    classifier_name = "classifier"
    encoder_name = "encoder"
    model_save_path = save_model(
        best_model,
        classifier_name,
        ".keras",
        os.path.join(model_save_dir, "classifier"),
    )
    encoder_save_path = save_model(
        best_encoder_model,
        encoder_name,
        ".keras",
        os.path.join(model_save_dir, "encoder"),
    )

    if train_experiment:
        train_experiment.log_model("Classifier", model_save_path)
        train_experiment.log_model("Encoder", encoder_save_path)
        logger.info(f"Best classifier model saved at: \n{model_save_path}")
        logger.info(f"Best encoder model saved at: \n{encoder_save_path}")

    # Log the confusion matrix
    class_labels = [map_cap_int_to_name(i) for i in range(etl_params["n_classes"])]
    conf_matrix_train = confusion_matrix(
        y_true=y_train, y_pred=y_pred_train, labels=range(etl_params["n_classes"])
    )
    conf_matrix_df_train = pd.DataFrame(
        conf_matrix_train, index=class_labels, columns=class_labels
    )
    conf_matrix_str_train = conf_matrix_df_train.to_string()

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
        train_experiment.log_text(
            text=f"tensorboard --logdir {tensorboard_save_path}",
            metadata={"Description": "Tensorboard command"},
        )

        # Log the test set confusion matrix to the Comet ML dashboard pane
        train_experiment.log_confusion_matrix(
            title="Test Confusion Matrix",
            y_true=y_test,
            y_predicted=y_pred_test,
            labels=class_labels,
        )

        train_experiment.end()


if __name__ == "__main__":
    # Configure settings here
    etl_params = {
        "data_dir": "/export/valenfs/data/processed_data/MinION/9_madcap/dummy_data/real_data2/",
        "save_dir": "/export/valenfs/data/processed_data/MinION/9_madcap/dummy_data/saved_data/",
        "target_length": 500,  # length of time series
        "dtype": "float16",  # data type of the time series
        "n_workers": 10,  # number of workers for parallel processing (by prefect)
        "n_classes": 4,  # number of classes in the dataset
        "use_local_dataset": True,  # set to False to use the online dataset, otherwise the local dataset will be used and will be uplaoded to comet
        "remote_dataset_version": "latest",  # version of the online dataset to use
    }

    tune_params = {
        "comet_project_name": "capfinder_tfr_tune",
        "patience": 0,
        "max_epochs_hpt": 3,
        "max_trials": 5,  # for random_search, and bayesian_optimization. For hyperband this has no effect
        "factor": 2,
        "batch_size": 64,
        "seed": 42,
        "tuning_strategy": "hyperband",  # "hyperband" or "random_search" or "bayesian_optimization"
        "overwrite": True,
    }

    train_params = {
        "comet_project_name": "capfinder_tfr_train",
        "patience": 2,
        "max_epochs_final_model": 10,
        "batch_size": 64,
    }

    model_save_dir = "/export/valenfs/data/processed_data/MinION/9_madcap/models/"

    # Run the training pipeline
    run_training_pipeline(
        etl_params=etl_params,
        tune_params=tune_params,
        train_params=train_params,
        model_save_dir=model_save_dir,
    )
