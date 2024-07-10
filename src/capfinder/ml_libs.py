import os

os.environ["KERAS_BACKEND"] = "jax"

# from keras import Model

import logging
import warnings

# Disable TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
logging.getLogger("tensorflow").setLevel(logging.FATAL)

# Filter out specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*cuFFT.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*cuDNN.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*cuBLAS.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*TensorRT.*")

# Now import TensorFlow
