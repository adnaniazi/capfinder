import math

from keras.layers import (
    LSTM,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Input,
    MaxPooling1D,
)
from keras.models import Model
from keras.optimizers import Adam
from keras_tuner import HyperModel, HyperParameters


class CapfinderHyperModel(HyperModel):
    def __init__(self, input_shape: tuple[int, int], n_classes: int):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.encoder_model = None  # Initialize an attribute to store the encoder model

    def build(self, hp: HyperParameters) -> Model:
        inputs = Input(shape=self.input_shape)
        x = inputs

        # Calculate the maximum number of conv layers based on input size
        max_conv_layers = int(math.log2(self.input_shape[0])) - 1
        conv_layers = hp.Int("conv_layers", 1, max_conv_layers)

        # Convolutional layers
        for i in range(conv_layers):
            # Dynamically adjust the range for filters based on the layer depth
            max_filters = min(128, 32 * (2 ** (i + 1)))
            filters = hp.Int(f"filters_{i}", 32, max_filters, step=32)

            # Dynamically adjust the kernel size based on the current feature map size
            current_size = x.shape[1]
            max_kernel_size = min(7, current_size)
            kernel_size = hp.Choice(
                f"kernel_size_{i}", list(range(3, max_kernel_size + 1, 2))
            )

            x = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(hp.Float(f"dropout_{i}", 0.1, 0.5, step=0.1))(x)
            x = BatchNormalization()(x)

        # Calculate the maximum number of LSTM layers based on remaining sequence length
        max_lstm_layers = int(math.log2(x.shape[1])) + 1
        lstm_layers = hp.Int("lstm_layers", 1, max_lstm_layers)

        # LSTM layers
        for i in range(lstm_layers):
            return_sequences = i < lstm_layers - 1
            lstm_units = hp.Int(
                f"lstm_units_{i}", 32, min(128, 32 * (2 ** (i + 1))), step=32
            )
            x = LSTM(units=lstm_units, return_sequences=return_sequences)(x)
            x = Dropout(hp.Float(f"lstm_dropout_{i}", 0.1, 0.5, step=0.1))(x)
            x = BatchNormalization()(x)

        # Fully connected layer
        dense_units = hp.Int("dense_units", 16, min(128, x.shape[-1] * 2), step=16)
        x = Dense(units=dense_units, activation="relu")(x)
        x = Dropout(hp.Float("dense_dropout", 0.1, 0.5, step=0.1))(x)
        outputs = Dense(self.n_classes, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=Adam(
                learning_rate=hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )

        return model
