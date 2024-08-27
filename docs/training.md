# Training the Capfinder Classifier

Now that we have cap signal data for all cap types, we're ready to train the classifier using our training pipeline. This process consists of three main stages: ETL (Extract, Transform, Load), Hyperparameter Tuning, and Final Training. Each stage plays a crucial role in developing an accurate and efficient classifier.

## Training Pipeline Overview

1. **ETL Stage**: Prepares the cap signal data for training.
2. **Tuning Stage**: Determines optimal hyperparameters for the chosen model.
3. **Training Stage**: Trains the final model using the best hyperparameters.

Let's delve into each stage in detail:

### 1. ETL Stage

The ETL stage is crucial for preparing our cap signal data, which is in the form of time series, for training. Here's what happens:

- **Signal Processing**: Each signal is either truncated or zero-padded to a uniform length, which we call `target_length`. We typically set this to 500 data points.
- **Balanced Dataset Creation**: `examples_per_class` controls sample size per class for a balanced dataset. When specified, it selects that many examples from each class. If `null`, it automatically uses the size of the smallest class for all classes, ensuring equal representation and preventing bias in model training.
- **Batched Loading**: Data is loaded into memory in class-balanced batches, controlled by the `batch_size` parameter. This approach allows us to handle cap signal files larger than available memory.
- **Batch Size Considerations**: We recommend a batch size of 1024 or higher for efficiency. However, if you encounter GPU memory issues (indicated by a `Killed` message during hyperparameter tuning), try lowering the batch size. Some models, like the `encoder` type, are large and may require powerful GPUs and smaller batch sizes (as low as 16).
- **Data Augmentation**: Users can set `use_augmentation` to add time warped versions of traning data during data.  When `True`, it adds two versions (squished and expanded) of each original training example, applying random warping between 0-20%. This triples the training set size and increases tuning and training time by approximately 3x. The augmentation enhances classifier robustness to RNA translocation speed variations, potentially improving classification accuracy. Users should weigh the increased training time against potential performance benefits when deciding to use this option.
- **Data Versioning**: On the first run, the ETL stage creates train and test datasets, automatically versions them, and uploads them to Comet ML. This requires a valid Comet ML API key (free for academic users). Subsequent runs can load this preprocessed dataset, which is faster than reprocessing the original CSV files.

### 2. Tuning Stage

The tuning stage is where we optimize the hyperparameters of our chosen deep learning model:

- **Model Selection**: You can choose from several model types: `attention_cnn_lstm`, `cnn_lstm`, `encoder`, `resnet`. The `encoder` model is the most computationally demanding, while `cnn_lstm` is the least demanding and is our default pretrained model due to its simplicity and speed.
- **Hyperparameter Optimization**: This stage uses techniques to find the best hyperparameter values. You can choose the optimization strategy from Random Search (`random_search`), Bayesian optimization (`bayesian_optimization`), or Hyperband (`hyperband`) using the `tuning_strategy` parameter in the configuration file.
- **Comet ML Integration**: A Comet ML experiment (specified by `comet_project_name` in the tune parameters) tracks performance metrics during tuning and logs the best hyperparameters.
- **Flexibility**: You can cancel the tuning stage at any time using CTRL+C. The best hyperparameters found up to that point will be used for the final training in the thrid stage.
- **Epoch Strategy**: It's often best to keep the number of epochs low during tuning to explore more parameter combinations. The most promising hyperparameters usually show good performance early on.

### 3. Training Stage

After finding the best hyperparameters, we proceed to the final training stage:

- **Extended Training**: We use the best hyperparameters to train the model for a longer duration, allowing it to reach its performance plateau.
- **Adaptive Stopping**: Capfinder includes mechanisms to automatically stop training when performance ceases to improve. It then loads the weights from the most promising epoch.
- **Model Evaluation**: The final trained model is tested on a held-out test set, providing an unbiased assessment of its performance.
- **Result Logging**: Performance metrics and a confusion matrix are logged in Comet ML and the log file.
- **Model Saving**: The final model weights are saved in .keras format for future use.

## Setting up your COMET ML API Key

Before initiating any training process, it's essential to set up your COMET ML account and API key. This allows Capfinder to log experiments, track metrics, and store models.

### Steps to Set Up COMET ML:

1. **Create an Account**:
    - Visit [COMET ML](https://www.comet.com/site/pricing/)
    - Sign up for an account (Free tier is available for individual use)

2. **Generate API Key**:
    - Once logged in, navigate to your account settings
    - Look for the "API Keys" section
    - Generate a new API key

3. **Export API Key**:
    - After generating the API key, you need to make it available to Capfinder
    - Export it as an environment variable in your terminal

### Example Command:

To export your COMET ML API key, use the following command in your terminal:

```bash
export COMET_API_KEY="your-api-key-here"
```

## Creating a Training Configuration File

Before running the training pipeline, you need to create a JSON configuration file. Use the following command to generate a template:

!!! example

    ```bash
    capfinder create-train-config --file_path /path/to/your/config.json
    ```

This will create a JSON file with default values. Edit this file to suit your specific needs.

## Running the Training Pipeline

Once you've customized your configuration file, start the training process with:

!!! example

    ```bash
    capfinder train-model --config_file /path/to/your/config.json
    ```

!!! tip "Prematurely quiting hyperparameter tuning"

    Hyperparameter tuning takes a lot of time and will run until `max_trials` number of trials have been executed. If you are feeling impatient, or think that you have acheived good enough accuracy already, you can interrupt tuning by Pressing `CTRL+C` once at any time during tuning. The best hyperparameter upto that time point will used for the subsequent final classifier training.

!!! tip "Prematurely quiting final classifier training"

    The final classifier will be trained for multiple epochs as specified in the `max_epochs_final_model` setting. You can interrupt the final model training at any point in time by Pressing `CTRL+C` once. The models for the best epoch will be restored and this final model will be saved as as `.keras` file. It's weights will also be saved in an `.h5` file.

## Configuration File Parameters

Let's break down the configuration file parameters:

### ETL Parameters

```json
"etl_params": {
    "use_remote_dataset_version": "latest",
    "caps_data_dir": "/dir/",
    "examples_per_class": 100000,
    "comet_project_name": "dataset"
}
```

- `use_remote_dataset_version`: Version of the remote dataset to use. Set to `""` to use/create a local dataset.
- `caps_data_dir`: Directory containing cap signal data files for all classes. This is the directory where all the [cap signal csv file for all the classes are stored](all_caps_data.md)
- `examples_per_class`: Maximum number of examples to use per class. If `null`, it automatically uses the size of the smallest class for all classes, ensuring equal representation and preventing bias in model training.
- `comet_project_name`: Name of the Comet ML project for dataset logging.

### Tuning Parameters

```json
"tune_params": {
    "comet_project_name": "capfinder_tune",
    "patience": 0,
    "max_epochs_hpt": 3,
    "max_trials": 5,
    "factor": 2,
    "seed": 42,
    "tuning_strategy": "hyperband",
    "overwrite": false
}
```

- `comet_project_name`: Name of the Comet ML project for hyperparameter tuning.
- `patience`: Number of epochs with no improvement before stopping.
- `max_epochs_hpt`: Maximum epochs for each trial during tuning.
- `max_trials`: Maximum number of trials for hyperparameter search.
- `factor`: Reduction factor for Hyperband algorithm.
- `seed`: Random seed for reproducibility.
- `tuning_strategy`: Choose from "hyperband", "random_search", or "bayesian_optimization".
- `overwrite`: Whether to overwrite previous tuning results.

### Training Parameters

```json
"train_params": {
    "comet_project_name": "capfinder_train",
    "patience": 120,
    "max_epochs_final_model": 300
}
```

- `comet_project_name`: Name of the Comet ML project for model training.
- `patience`: Number of epochs with no improvement before stopping.
- `max_epochs_final_model`: Maximum epochs for training the final model.

### Shared Parameters

```json
"shared_params": {
    "num_classes": 4,
    "model_type": "cnn_lstm",
    "batch_size": 32,
    "target_length": 500,
    "dtype": "float16",
    "train_test_fraction": 0.95,
    "train_val_fraction": 0.8,
    "use_augmentation": false,
    "output_dir": "/dir/"
}
```

- `num_classes`: Number of classes in the dataset.
- `model_type`: Choose from "attention_cnn_lstm", "cnn_lstm", "encoder", "resnet".
- `batch_size`: Batch size for training. Adjust based on GPU memory.
- `target_length`: Target length for input sequences.
- `dtype`: Data type for model parameters. Options: "float16", "float32", "float64".
- `train_test_fraction`: Fraction of data to use for training vs. testing. Testing set is the holdout set.
- `train_val_fraction`: Fraction of training data to use for training vs. validation.
- `use_augmentation`: Whether to augment real training data with time-warped (squished and expanded) data
- `output_dir`: Directory to save output files.

### Learning Rate Scheduler Parameters

The `lr_scheduler_params` section allows you to choose and configure one of three learning rate scheduling strategies. It's important to note that these schedulers are only used during the final training stage and not during hyperparameter tuning.

1. **Reduce LR on Plateau (`reduce_lr_on_plateau`)**:
    This scheduler reduces the learning rate when a metric has stopped improving. It's useful for fine-tuning the model in later stages of training.
    - `factor`: The factor by which the learning rate will be reduced (e.g., 0.5 means halving the learning rate).
    - `patience`: Number of epochs with no improvement after which learning rate will be reduced.
    - `min_lr`: Lower bound on the learning rate.

2. **Cyclic Learning Rate (`cyclic_lr`)**:
    This scheduler cyclically varies the learning rate between two boundaries. It can help the model to escape local minima and find better optima.
    - `base_lr`: Lower boundary of learning rate (initial learning rate).
    - `max_lr`: Upper boundary of learning rate in the cycle.
    - `step_size_factor`: Determines the number of iterations in a cycle.
    - `mode`: The cycling mode (e.g., "triangular2" for a triangular cycle that decreases the cycle amplitude by half after each cycle).

3. **Stochastic Gradient Descent with Restarts (SGDR) (`sgdr`)**:
    This scheduler implements a cosine annealing learning rate schedule with periodic restarts. It can help the model to escape local minima and converge to a better optimum.
    - `min_lr`: Minimum learning rate.
    - `max_lr`: Maximum learning rate.
    - `lr_decay`: Factor to decay the maximum learning rate after each cycle.
    - `cycle_length`: Number of epochs in the initial cycle.
    - `mult_factor`: Factor to increase the cycle length after each full cycle.

Choose the scheduler type by setting the `type` parameter to one of "reduce_lr_on_plateau", "cyclic_lr", or "sgdr", and then configure the specific parameters for your chosen scheduler. Remember, these schedulers are applied only during the final training phase, not during hyperparameter tuning.

### Debug Mode

```json
"debug_code": false
```
- Set to `true` to enable debug mode for more detailed logging.

## Important Considerations

1. **Batch Size**: Start with a batch size of 1024 or higher. If you encounter memory issues, especially with larger models like "encoder", reduce the batch size.

2. **GPU Monitoring**: Use commands like `nvidia-smi` to monitor GPU resources and adjust batch size accordingly.

3. **Comet ML**: Ensure you have a valid Comet ML API key (free for academic users) for data versioning and experiment tracking.

4. **Model Types**:
    - "cnn_lstm": Simplest and fastest for training and inference. Achieves good accuracy.
    - "encoder": Most computationally demanding. We had very little success with this model
    - "attention_cnn_lstm" and "resnet": Intermediate in terms of computational requirements.

5. **Hyperparameter Tuning**: Can be interrupted at any time using CTRL+C. The best hyperparameters up to that point will be used for final training.

6. **Final Training**: Uses the best hyperparameters to train the model for a longer duration, allowing it to reach performance plateau.

7. **Automatic Stopping**: The training process will automatically stop when performance ceases to improve, loading the weights from the most promising epoch.

8. **Model Evaluation**: The final model is tested on a held-out test set, providing an unbiased performance assessment.

9. **Output**: Performance metrics, confusion matrix, and the final model (in .keras format) are saved and logged in Comet ML.

By following this guide and adjusting the configuration parameters as needed, you can effectively train and evaluate your Capfinder classifier. The process is designed to be flexible, allowing you to balance between computational resources, time constraints, and model performance based on your specific requirements.
