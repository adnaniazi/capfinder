import math
from typing import Any, Tuple

from capfinder.ml_libs import (
    LSTM,
    Adam,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    HyperModel,
    Input,
    K,
    MaxPooling1D,
    Model,
    layers,
)


class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def build(self, input_shape: Tuple[int, ...]) -> None:
        self.W = self.add_weight(
            name="att_weight", shape=(input_shape[-1], 1), initializer="normal"
        )
        self.b = self.add_weight(
            name="att_bias", shape=(input_shape[1], 1), initializer="zeros"
        )
        super().build(input_shape)

    def call(self, x: Any) -> Any:
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, int]:
        return (input_shape[0], input_shape[-1])


class CapfinderHyperModel(HyperModel):
    """
    Hypermodel for the Capfinder CNN-LSTM with Attention architecture.

    This model is designed for time series classification tasks, specifically for
    identifying RNA cap types. It combines Convolutional Neural Networks (CNNs) for
    local feature extraction, Long Short-Term Memory (LSTM) networks for sequence
    processing, and an attention mechanism to focus on the most relevant parts of
    the input sequence.

    The architecture is flexible and allows for hyperparameter tuning of the number
    of layers, units, and other key parameters.

    Attributes:
        input_shape (Tuple[int, ...]): The shape of the input data.
        n_classes (int): The number of classes for classification.
        encoder_model (Optional[Model]): Placeholder for a potential encoder model.

    Methods:
        build(hp): Constructs and returns a Keras model based on the provided
                   hyperparameters.
    """

    def __init__(self, input_shape: Tuple[int, int], n_classes: int) -> None:
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.encoder_model = None

    def build(self, hp: Any) -> Model:
        inputs = Input(shape=self.input_shape)
        x = inputs

        # Calculate the maximum number of conv layers based on input size
        max_conv_layers = min(
            int(math.log2(self.input_shape[0])) - 1, 5
        )  # Limit to 5 layers max
        conv_layers = hp.Int("conv_layers", 1, max_conv_layers)

        # Convolutional layers
        for i in range(conv_layers):
            # Dynamically adjust the range for filters based on the layer depth
            max_filters = min(256, 32 * (2 ** (i + 1)))
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

            # Only apply MaxPooling if the current size is greater than 2
            if current_size > 2:
                x = MaxPooling1D(pool_size=2)(x)

            x = Dropout(hp.Float(f"dropout_{i}", 0.1, 0.5, step=0.1))(x)
            x = BatchNormalization()(x)

        # Calculate the maximum number of LSTM layers based on remaining sequence length
        current_seq_length = x.shape[1]
        max_lstm_layers = min(
            int(math.log2(current_seq_length)) + 1, 3
        )  # Limit to 3 LSTM layers max
        lstm_layers = hp.Int("lstm_layers", 1, max_lstm_layers)

        # LSTM layers
        for i in range(lstm_layers):
            return_sequences = i < lstm_layers - 1
            max_lstm_units = min(256, 32 * (2 ** (i + 1)))
            lstm_units = hp.Int(f"lstm_units_{i}", 32, max_lstm_units, step=32)
            x = LSTM(
                units=lstm_units,
                return_sequences=return_sequences or i == lstm_layers - 1,
            )(x)
            x = Dropout(hp.Float(f"lstm_dropout_{i}", 0.1, 0.5, step=0.1))(x)
            x = BatchNormalization()(x)

        # Attention layer
        x = AttentionLayer()(x)

        # Fully connected layer
        max_dense_units = min(256, x.shape[-1] * 2)
        dense_units = hp.Int("dense_units", 16, max_dense_units, step=16)
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
