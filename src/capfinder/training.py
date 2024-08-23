import os
import shutil
import signal
import subprocess
from datetime import datetime
from importlib.metadata import version
from typing import Dict, Literal, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from comet_ml import Experiment  # Import CometML before keras

os.environ["COMET_LOGGING_FILE"] = "/dev/null"

from loguru import logger
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from capfinder.attention_cnnlstm_model import (
    CapfinderHyperModel as AttentionCNNLSTMModel,
)
from capfinder.cnn_lstm_model import CapfinderHyperModel as CNNLSTMModel
from capfinder.cyclic_learing_rate import (
    CometLRLogger,
    CustomProgressCallback,
    CyclicLR,
    SGDRScheduler,
)
from capfinder.encoder_model import CapfinderHyperModel as EncoderModel
from capfinder.logger_config import configure_logger, configure_prefect_logging
from capfinder.ml_libs import jax  # noqa
from capfinder.ml_libs import (
    BayesianOptimization,
    Hyperband,
    Objective,
    RandomSearch,
    keras,
    tf,
)
from capfinder.resnet_model import ResNetTimeSeriesHyper as ResnetModel
from capfinder.train_etl import train_etl
from capfinder.utils import (
    initialize_comet_ml_experiment,
    log_header,
    log_output,
    log_subheader,
    map_cap_int_to_name,
)

# Declare and initialize global stop_training flag
global stop_training
stop_training = False

ModelType = Literal["attention_cnn_lstm", "cnn_lstm", "encoder", "resnet"]


def get_model(
    model_type: ModelType,
) -> (
    Type[AttentionCNNLSTMModel]
    | Type[CNNLSTMModel]
    | Type[EncoderModel]
    | Type[ResnetModel]
):
    if model_type == "cnn_lstm":
        return CNNLSTMModel
    elif model_type == "encoder":
        return EncoderModel
    elif model_type == "resnet":
        return ResnetModel
    elif model_type == "attention_cnn_lstm":
        return AttentionCNNLSTMModel


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
            te = epoch + 1
            logger.info(f"Training interrupted by user at the end of epoch {te}")
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
    logger.info(f"Best model saved to:{model_save_path}")

    # Return the save path
    return model_save_path


def set_data_distributed_training() -> None:
    """
    Set JAX as the backend for Keras training, with distributed training if multiple CUDA devices are available.

    This function checks for available CUDA devices and sets up distributed training only if more than one is found.

    Returns:
    --------
    None
    """
    # Set the Keras backend to JAX
    logger.info(f"Backend for training: {keras.backend.backend()}")

    # Retrieve available devices
    all_devices = jax.devices()
    cuda_devices = [d for d in all_devices if d.platform == "gpu"]

    # Log available devices
    for device in all_devices:
        logger.info(f"Device available: {device}, Type: {device.platform}")

    if len(cuda_devices) > 1:
        keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info(
            f"({len(cuda_devices)}) CUDA devices detected. Setting up data distributed training."
        )

        # Define a 1D device mesh for data parallelism using only CUDA devices
        mesh_1d = keras.distribution.DeviceMesh(
            shape=(len(cuda_devices),), axis_names=["data"], devices=cuda_devices
        )

        # Create a DataParallel distribution
        data_parallel = keras.distribution.DataParallel(device_mesh=mesh_1d)

        # Set the global distribution
        keras.distribution.set_distribution(data_parallel)

        logger.info("Distributed training setup complete.")
    elif len(cuda_devices) == 1:
        keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info(
            "Single CUDA device detected. Using standard (non-distributed) training."
        )
    else:
        logger.info("No CUDA devices detected. Training will proceed on CPU.")
        keras.mixed_precision.set_global_policy("float32")


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
        logger.info("Using Hyperband tuning strategy...")
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
        logger.info("Using Bayesian Optimization tuning strategy...")
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
        logger.info("Using Random Search tuning strategy...")
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
        logger.info("No NVIDIA GPU found. Skipping GPU process termination.")
        return

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
        logger.warning(f"Error occurred while terminating GPU processes: {str(e)}")


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


def count_examples(dataset: tf.data.Dataset, dataset_name: str) -> int:
    """
    Count the number of individual examples in a dataset.

    Args:
    dataset (tf.data.Dataset): The dataset to count examples from.
    dataset_name (str): The name of the dataset.

    Returns:
    int: The number of examples in the dataset.
    """
    count = sum(
        1 for _ in tqdm(dataset, desc=f"Examples in {dataset_name}", unit="examples")
    )
    return count


def count_batches(dataset: tf.data.Dataset, dataset_name: str) -> int:
    """
    Count the number of individual examples in a dataset.

    Args:
    dataset (tf.data.Dataset): The dataset to count examples from.
    dataset_name (str): The name of the dataset.

    Returns:
    int: The number of examples in the dataset.
    """
    count = sum(
        1 for _ in tqdm(dataset, desc=f"Batches in {dataset_name}", unit="batches")
    )
    return count


def select_lr_scheduler(
    lr_scheduler_params: dict, train_size: int
) -> Union[keras.callbacks.ReduceLROnPlateau, CyclicLR, SGDRScheduler]:
    """
    Selects and configures the learning rate scheduler based on the provided parameters.

    Args:
        lr_scheduler_params (dict): Configuration parameters for the learning rate scheduler.
        train_size (int): Number of training examples, used for step size calculations.

    Returns:
        Union[keras.callbacks.ReduceLROnPlateau, CyclicLR, SGDRScheduler]: The selected learning rate scheduler.
    """
    scheduler_type = lr_scheduler_params["type"]

    if scheduler_type == "reduce_lr_on_plateau":
        rlr_params = lr_scheduler_params["reduce_lr_on_plateau"]
        return keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=rlr_params["factor"],
            patience=rlr_params["patience"],
            verbose=1,
            mode="min",
            min_lr=rlr_params["min_lr"],
        )

    elif scheduler_type == "cyclic_lr":
        clr_params = lr_scheduler_params["cyclic_lr"]
        return CyclicLR(
            base_lr=clr_params["base_lr"],
            max_lr=clr_params["max_lr"],
            step_size=train_size * clr_params["step_size_factor"],
            mode=clr_params["mode"],
        )

    elif scheduler_type == "sgdr":
        sgdr_params = lr_scheduler_params["sgdr"]
        return SGDRScheduler(
            min_lr=sgdr_params["min_lr"],
            max_lr=sgdr_params["max_lr"],
            steps_per_epoch=train_size,
            lr_decay=sgdr_params["lr_decay"],
            cycle_length=sgdr_params["cycle_length"],
            mult_factor=sgdr_params["mult_factor"],
        )

    else:
        logger.warning(
            f"Unknown scheduler type: {scheduler_type}. Using ReduceLROnPlateau as default."
        )
        return keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
            mode="min",
            min_lr=1e-6,
        )


def count_batches_fast(dataset: tf.data.Dataset, dataset_name: str) -> int:
    count = dataset.reduce(0, lambda x, _: x + 1)
    example_count = int(count.numpy())  # Explicitly cast to int
    logger.info(f"Batches in {dataset_name}: {example_count}")
    return example_count


def run_training_pipeline(
    etl_params: dict,
    tune_params: dict,
    train_params: dict,
    shared_params: dict,
    lr_scheduler_params: dict,
    debug_code: bool,
    formatted_command: Optional[str],
) -> None:
    log_filepath = configure_logger(
        os.path.join(shared_params["output_dir"], "logs"), show_location=debug_code
    )
    configure_prefect_logging(show_location=debug_code)
    version_info = version("capfinder")
    log_header(f"Using Capfinder v{version_info}")
    logger.info(formatted_command)

    log_subheader("Step 0: Compute setup")

    logger.info("Starting training pipeline...")
    kill_gpu_processes()
    logger.info("Setting up data distributed training (if multiple GPUs available)...")
    set_data_distributed_training()

    etl_params["dataset_dir"] = os.path.join(shared_params["output_dir"], "dataset")
    if not os.path.exists(etl_params["dataset_dir"]):
        os.makedirs(etl_params["dataset_dir"], exist_ok=True)

    """
    #################################################
    #############          ETL         ##############
    #################################################
    """

    log_subheader("Step 1: Extract, Transform, Load (ETL)")

    # Prepare training datasets using ETL pipeline
    (
        train_dataset,
        val_dataset,
        test_dataset,
        steps_per_epoch,
        validation_steps,
        dataset_version,
    ) = train_etl(
        etl_params["caps_data_dir"],
        etl_params["dataset_dir"],
        shared_params["target_length"],
        shared_params["dtype"],
        etl_params["examples_per_class"],
        shared_params["train_test_fraction"],
        shared_params["train_val_fraction"],
        shared_params["num_classes"],
        shared_params["batch_size"],
        etl_params["comet_project_name"],
        etl_params["use_remote_dataset_version"],
        shared_params["use_augmentation"],
    )

    logger.info("Dataset loaded successfully!")
    train_feature_shape, train_label_shape = get_shape_with_batch_size(
        train_dataset, shared_params["batch_size"]
    )
    logger.info(f"x_train shape: {train_feature_shape}")
    logger.info(f"y_train shape: {train_label_shape}")
    test_feature_shape, test_label_shape = get_shape_with_batch_size(
        test_dataset, shared_params["batch_size"]
    )
    logger.info(f"x_test shape: {test_feature_shape}")
    logger.info(f"y_test shape: {test_label_shape}")
    logger.info(f"Dataset version: {dataset_version}")

    """
    #################################################
    #############        TUNE          ##############
    #################################################
    """
    log_subheader("Step 2: Hyperparameter Tuning")

    logger.info("Press CTRL + C once to stop the hyperparameter search at any point.")

    model_type = shared_params["model_type"]
    if model_type not in ["attention_cnn_lstm", "cnn_lstm", "encoder", "resnet"]:
        raise ValueError(
            "Invalid model type. Expected 'attention_cnn_lstm', 'cnn_lstm' or 'encoder' or 'renset'."
        )
    model = get_model(model_type)
    tune_params["comet_project_name"] = (
        tune_params["comet_project_name"] + "_" + model_type
    )
    train_params["comet_project_name"] = (
        train_params["comet_project_name"] + "_" + model_type
    )

    tune_experiment = initialize_comet_ml_experiment(
        project_name=tune_params["comet_project_name"]
    )
    tune_experiment_url = tune_experiment.url

    hyper_model = model(
        input_shape=(shared_params["target_length"], 1),
        n_classes=shared_params["num_classes"],
    )

    model_save_dir = os.path.join(shared_params["output_dir"], "tuner_models")
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir, exist_ok=True)
    tuner = initialize_tuner(hyper_model, tune_params, model_save_dir, model_type)

    logger.info("Counting test dataset batches...")
    total_batches_test = count_batches_fast(test_dataset, "test dataset")

    logger.info(
        f"\nTrain set size: {steps_per_epoch} batches\nValidation set size: {validation_steps} batches\nTest set size: {total_batches_test} batches"
    )

    train_dataset = train_dataset.repeat()
    val_dataset = val_dataset.repeat()

    # tensorboard_save_path = os.path.join(model_save_dir, "tensorboard_logs_encoder", model_type)
    # logger.info(
    #     f"Run tensorboard as following:\ntensorboard --logdir {tensorboard_save_path}"
    # )
    logger.info(
        f"Starting {tune_params['max_trials']} trials of hyperparameter search..."
    )
    try:
        tuner.search(
            train_dataset.map(lambda x, y: (x, y)),
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset.map(lambda x, y: (x, y)),
            validation_steps=validation_steps,
            epochs=tune_params["max_epochs_hpt"],
            callbacks=[
                # keras.callbacks.TensorBoard(log_dir=tensorboard_save_path),
                keras.callbacks.EarlyStopping(
                    patience=tune_params["patience"], restore_best_weights=True
                ),
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
        logger.info("Hyperparameter search was canceled by user.")
    finally:
        # Retrieve the best hyperparameters found during the search
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    log_params = {
        "ETL params": etl_params,
        "Tune params": tune_params,
        "Train params": train_params,
        "Shared params": shared_params,
        "Data Shapes": {
            "x_train": train_feature_shape,
            "y_train": train_label_shape,
            "x_test": test_feature_shape,
            "y_test": test_label_shape,
        },
        "Dataset version": dataset_version,
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
    log_subheader("Step 3: Final model training")

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

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=train_params["patience"],
        verbose=1,
        mode="auto",
        restore_best_weights=True,
    )

    # Select the learning rate scheduler
    lr_scheduler = select_lr_scheduler(lr_scheduler_params, steps_per_epoch)
    custom_progress = CustomProgressCallback()
    comet_lr_logger = CometLRLogger(train_experiment)
    best_model_dir = os.path.join(shared_params["output_dir"], "best_model")
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir, exist_ok=True)
    best_model_weights_path = os.path.join(
        best_model_dir, "best_model_weights.weights.h5"
    )

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        best_model_weights_path,
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )
    # Prepare callbacks
    callbacks = [
        early_stopping,
        lr_scheduler,  # Add the selected learning rate scheduler
        custom_progress,
        comet_lr_logger,
        comet_callback,
        model_checkpoint,
        interrupt_callback,
    ]

    best_model.fit(
        train_dataset.map(lambda x, y: (x, y)),
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset.map(lambda x, y: (x, y)),
        validation_steps=validation_steps,
        epochs=train_params["max_epochs_final_model"],
        batch_size=shared_params["batch_size"],
        callbacks=callbacks,
    )
    logger.success("Training on best hyperparameters done!")
    best_model.load_weights(best_model_weights_path)

    log_subheader("Step 4: Performance Evaluation")

    # Evaluate the best model on the test data
    batch_size = shared_params["batch_size"]
    y_true_test = np.zeros(total_batches_test * batch_size)
    y_pred_test = np.zeros(total_batches_test * batch_size)

    logger.info("Making predictions on test set for confusion matrix...")
    # Iterate through the test_dataset to collect true labels and predictions

    pbar = tqdm(
        test_dataset, total=total_batches_test, desc="Processing test dataset batches"
    )
    for batch_num, (features, labels) in enumerate(pbar):
        predictions = best_model.predict(features, verbose=0)
        start_idx = batch_num * batch_size
        end_idx = (batch_num + 1) * batch_size
        y_true_test[start_idx:end_idx] = labels.numpy()
        y_pred_test[start_idx:end_idx] = np.argmax(predictions, axis=1)
        pbar.set_description(f"Processing batch {batch_num + 1}/{total_batches_test}")

    # Convert lists to numpy arrays
    y_true_test = np.array(y_true_test, dtype=np.int32)
    y_pred_test = np.array(y_pred_test, dtype=np.int32)
    logger.success("Predictions on test set done!")

    # Assuming `map_cap_int_to_name` maps integers to class names
    class_labels = [map_cap_int_to_name(i) for i in range(shared_params["num_classes"])]

    # Evaluate the best model on the test data
    logger.info("Evaluate final model performance on test dataset")
    test_loss, test_acc = best_model.evaluate(test_dataset.map(lambda x, y: (x, y)))
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
        os.path.join(best_model_dir, "classifier"),
    )
    if best_encoder_model is not None:
        encoder_save_path = save_model(
            best_encoder_model,
            encoder_name,
            ".keras",
            os.path.join(best_model_dir, "encoder"),
        )

    if train_experiment:
        train_experiment.log_model("Classifier", model_save_path)
        logger.info(f"Best classifier model saved at: \n{model_save_path}")
        train_experiment.log_model("Classifier weights", best_model_weights_path)
        logger.info(f"Best classifier weights saved at: \n{best_model_weights_path}")

        if best_encoder_model is not None:
            train_experiment.log_model("Encoder", encoder_save_path)
            logger.info(f"Best encoder model saved at: \n{encoder_save_path}")

    # Log the confusion matrix
    # Predict using the model on training data
    logger.info("Making predictions on training set for confusion matrix...")
    train_dataset = train_dataset.take(steps_per_epoch)  # because it is repeated
    y_true = np.zeros(steps_per_epoch * batch_size, dtype=np.int32)
    y_pred = np.zeros(steps_per_epoch * batch_size, dtype=np.int32)

    pbar = tqdm(train_dataset, total=steps_per_epoch, desc="Processing batches")
    for batch_num, (features, labels) in enumerate(pbar):
        predictions = best_model.predict(features, verbose=0)
        start_idx = batch_num * batch_size
        end_idx = (batch_num + 1) * batch_size
        y_true[start_idx:end_idx] = labels.numpy()
        y_pred[start_idx:end_idx] = np.argmax(predictions, axis=1)
        pbar.set_description(f"Processing batch {batch_num + 1}/{steps_per_epoch}")

    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Assuming `map_cap_int_to_name` maps integers to class names
    class_labels = [map_cap_int_to_name(i) for i in range(shared_params["num_classes"])]

    # Compute confusion matrix
    conf_matrix_train = confusion_matrix(
        y_true, y_pred, labels=range(shared_params["num_classes"])
    )
    logger.success("Done making predictions on training set for confusion matrix!")

    # Create DataFrame for better visualization
    conf_matrix_train_df = pd.DataFrame(
        conf_matrix_train, index=class_labels, columns=class_labels
    )
    conf_matrix_str_train = conf_matrix_train_df.to_string()

    if train_experiment:
        # train_experiment.log_text(
        #     text=dataset_info["etl_experiment_url"],
        #     metadata={"Description": "ETL Experiment URL"},
        # )
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

        train_experiment.log_asset(
            file_data=log_filepath,
            file_name="Logfile",
            step=0,
        )

        train_experiment.end()

    grey = "\033[90m"
    reset = "\033[0m"
    logger.success("Training pipeline ran successfully!")
    log_output(
        f"Dataset was saved to the following path:\n {grey}{etl_params['dataset_dir']}{reset}\nBest model's .keras file have been saved to the following path:\n {grey}{model_save_path}{reset}\nModel weights file was saved to the following path:\n {grey}{best_model_weights_path}{reset}\nThe log file has been saved to:\n {grey}{log_filepath}{reset}"
    )
    log_header("Processing finished!")


if __name__ == "__main__":
    # Configure settings here
    etl_params = {
        "use_remote_dataset_version": "",  # version of the online dataset to use
        "caps_data_dir": "/home/valen/10-data-for-upload-to-mega/uncompressed/all_csvs",
        "examples_per_class": 5000,  # maximum number of examples to use from the dataset
        "comet_project_name": "dataset2",
    }

    tune_params = {
        "comet_project_name": "capfinder_tune_delete",
        "patience": 0,
        "max_epochs_hpt": 3,
        "max_trials": 1,  # for random_search, and bayesian_optimization. For hyperband this has no effect
        "factor": 2,
        "seed": 42,
        "tuning_strategy": "bayesian_optimization",  # "hyperband" or "random_search" or "bayesian_optimization"
        "overwrite": True,
    }

    train_params = {
        "comet_project_name": "capfinder_train_delete",
        "patience": 120,
        "max_epochs_final_model": 3,
    }

    shared_params = {
        "num_classes": 4,
        "model_type": "cnn_lstm",  # attention_cnn_lstm, cnn_lstm, resnet, encoder
        "batch_size": 500,
        "target_length": 500,
        "dtype": "float16",
        "train_test_fraction": 0.95,
        "train_val_fraction": 0.8,
        "use_augmentation": False,
        "output_dir": "/home/valen/10-data-for-upload-to-mega/uncompressed/output",
    }

    # Learning Rate Scheduler Configuration
    lr_scheduler_params = {
        "type": "reduce_lr_on_plateau",  # Choose one: "reduce_lr_on_plateau", "cyclic_lr", or "sgdr"
        # ReduceLROnPlateau parameters
        "reduce_lr_on_plateau": {"factor": 0.5, "patience": 5, "min_lr": 1e-6},
        # Cyclic LR parameters
        "cyclic_lr": {
            "base_lr": 1e-3,
            "max_lr": 5e-2,
            "step_size_factor": 8,  # step_size will be train_size * step_size_factor
            "mode": "triangular2",
        },
        # SGDR parameters
        "sgdr": {
            "min_lr": 1e-3,
            "max_lr": 2e-3,
            "lr_decay": 0.9,
            "cycle_length": 5,
            "mult_factor": 1.5,
        },
    }
    debug_code = True

    run_training_pipeline(
        etl_params=etl_params,
        tune_params=tune_params,
        train_params=train_params,
        shared_params=shared_params,
        lr_scheduler_params=lr_scheduler_params,
        debug_code=debug_code,
        formatted_command="",
    )
