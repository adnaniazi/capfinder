from typing import Tuple

from capfinder.ml_libs import HyperModel, HyperParameters, Model, keras, layers, tf


class ResNetBlockHyper(layers.Layer):
    def __init__(self, filters: int, kernel_size: int, strides: int = 1):
        super().__init__()
        self.conv1 = layers.Conv1D(
            filters, kernel_size, strides=strides, padding="same"
        )
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(filters, kernel_size, padding="same")
        self.bn2 = layers.BatchNormalization()

        self.shortcut = layers.Conv1D(filters, 1, strides=strides, padding="same")
        self.bn_shortcut = layers.BatchNormalization()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = keras.activations.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        shortcut = self.shortcut(inputs)
        shortcut = self.bn_shortcut(shortcut)

        x = layers.add([x, shortcut])
        return keras.activations.relu(x)

    def compute_output_shape(
        self, input_shape: Tuple[int, int]
    ) -> Tuple[int, int, int]:
        return input_shape[0], input_shape[1], self.conv1.filters


class ResNetTimeSeriesHyper(HyperModel):
    """
    A HyperModel class for building a ResNet-style neural network for time series classification.

    This class defines a tunable ResNet architecture that can be optimized using Keras Tuner.
    It creates a model with an initial convolutional layer, followed by a variable number of
    ResNet blocks, and ends with global average pooling and dense layers.

    Attributes:
        input_shape (Tuple[int, int]): The shape of the input data (timesteps, features).
        n_classes (int): The number of classes for classification.

    Methods:
        build(hp): Builds and returns a compiled Keras model based on the provided hyperparameters.
    """

    def __init__(self, input_shape: Tuple[int, int], n_classes: int):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.encoder_model = None

    def build(self, hp: HyperParameters) -> Model:
        """
        Build and compile a ResNet model based on the provided hyperparameters.

        This method constructs a ResNet architecture with tunable hyperparameters including
        the number of filters, kernel sizes, number of ResNet blocks, dense layer units,
        dropout rate, and learning rate.

        Args:
            hp (hp.HyperParameters): A HyperParameters object used to define the search space.

        Returns:
            Model: A compiled Keras model ready for training.
        """
        inputs = keras.Input(shape=self.input_shape)

        # Initial convolution
        initial_filters = hp.Int(
            "initial_filters", min_value=32, max_value=128, step=32
        )
        x = layers.Conv1D(
            initial_filters,
            kernel_size=hp.Choice("initial_kernel", values=[3, 5, 7]),
            padding="same",
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = keras.activations.relu(x)
        x = layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(x)

        # ResNet blocks
        num_blocks_per_stage = hp.Int("num_blocks_per_stage", min_value=2, max_value=4)
        num_stages = hp.Int("num_stages", min_value=2, max_value=4)

        for stage in range(num_stages):
            filters = hp.Int(
                f"filters_stage_{stage}", min_value=64, max_value=256, step=64
            )
            for block in range(num_blocks_per_stage):
                kernel_size = hp.Choice(
                    f"kernel_stage_{stage}_block_{block}", values=[3, 5, 7]
                )
                strides = 2 if block == 0 and stage > 0 else 1
                x = ResNetBlockHyper(filters, kernel_size, strides)(x)

        # Global pooling and output
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(
            hp.Int("dense_units", min_value=32, max_value=256, step=32),
            activation="relu",
        )(x)
        x = layers.Dropout(hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1))(
            x
        )
        outputs = layers.Dense(self.n_classes, activation="softmax")(x)

        model = Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
                )
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )

        return model
