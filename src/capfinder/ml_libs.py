import logging  # noqa
import os  # noqa
import warnings  # noqa

import jax  # noqa

os.environ["KERAS_BACKEND"] = "jax"

import keras  # noqa
import keras.ops as K  # noqa
from keras import layers  # noqa
from keras.callbacks import Callback  # noqa
from keras.layers import LSTM  # noqa
from keras.layers import BatchNormalization  # noqa
from keras.layers import Conv1D  # noqa
from keras.layers import Dense  # noqa
from keras.layers import Dropout  # noqa
from keras.layers import Input  # noqa
from keras.layers import MaxPooling1D  # noqa
from keras.models import Model  # noqa
from keras.optimizers import Adam  # noqa
from keras_tuner import (  # noqa
    BayesianOptimization,
    Hyperband,
    HyperModel,
    HyperParameters,
    Objective,
    RandomSearch,
)

# Disable TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
logging.getLogger("tensorflow").setLevel(logging.FATAL)

# Filter out specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*cuFFT.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*cuDNN.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*cuBLAS.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*TensorRT.*")

# Now import TensorFlow
import tensorflow as tf  # noqa
from tensorflow import float16, float32, float64  # noqa
