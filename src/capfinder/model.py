from typing import List, Optional, Tuple

import keras
from keras import Model, layers
from keras_tuner import HyperModel, HyperParameters


def transformer_encoder(
    inputs: keras.layers.Layer,
    head_size: int,
    num_heads: int,
    ff_dim: int,
    dropout: Optional[float] = 0.0,
) -> keras.layers.Layer:
    """
    Create a transformer encoder block.

    The transformer encoder block consists of a multi-head attention layer
    followed by layer normalization and a feed-forward network.

    Parameters:
    ----------
    inputs : keras.layers.Layer
        The input layer or tensor for the encoder block.

    head_size : int
        The size of the attention heads.

    num_heads : int
        The number of attention heads.

    ff_dim : int
        The dimensionality of the feed-forward network.

    dropout : float, optional
        The dropout rate applied after the attention layer and within the feed-forward network. Default is 0.0.

    Returns:
    -------
    keras.layers.Layer
        The output layer of the encoder block, which can be used as input for the next layer in a neural network.
    """
    # Multi-head attention layer with dropout and layer normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed-forward network with convolutional layers, dropout, and layer normalization
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def build_model(
    input_shape: Tuple[int, int],
    head_size: int,
    num_heads: int,
    ff_dim: int,
    num_transformer_blocks: int,
    mlp_units: List[int],
    n_classes: int,
    dropout: float = 0.0,
    mlp_dropout: float = 0.0,
) -> Tuple[keras.Model, keras.Model]:
    """
    Build a transformer-based neural network model and return the encoder output.

    Parameters:
    input_shape : Tuple[int, int]
        The shape of the input data.

    head_size : int
        The size of the attention heads in the transformer encoder.

    num_heads : int
        The number of attention heads in the transformer encoder.

    ff_dim : int
        The dimensionality of the feed-forward network in the transformer encoder.

    num_transformer_blocks : int
        The number of transformer encoder blocks in the model.

    mlp_units : List[int]
        A list containing the number of units for each layer in the MLP.

    n_classes : int
        The number of output classes (for classification tasks).

    dropout : float, optional
        The dropout rate applied in the transformer encoder.

    mlp_dropout : float, optional
        The dropout rate applied in the MLP.

    Returns:
    Tuple[keras.Model, keras.Model]:
        A tuple containing the full model and the encoder model.
    """
    # Input layer
    inputs = keras.Input(shape=input_shape)

    # Apply transformer encoder blocks and save the output of the encoder
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Save the encoder output
    encoder_output = x

    # Apply global average pooling
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)

    # Add multi-layer perceptron (MLP) layers
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    # Add softmax output layer
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    # Construct the full model
    model = keras.Model(inputs, outputs)

    # Create a model that produces only the encoder output
    encoder_model = keras.Model(inputs, encoder_output)

    # Return the full model and the encoder model
    return model, encoder_model


# Custom HyperModel class to wrap the model building function
class CapfinderHyperModel(HyperModel):
    def __init__(self, input_shape: Tuple[int, int], n_classes: int):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.encoder_model = None  # Initialize an attribute to store the encoder model

    def build(self, hp: HyperParameters) -> Model:
        # Call the model builder function and obtain the full model and encoder model
        model, encoder_model = build_model(
            input_shape=self.input_shape,
            head_size=hp.Int("head_size", min_value=32, max_value=512, step=32),
            num_heads=hp.Int("num_heads", min_value=1, max_value=8, step=1),
            ff_dim=hp.Int("ff_dim", min_value=4, max_value=16, step=4),
            num_transformer_blocks=hp.Int(
                "num_transformer_blocks", min_value=2, max_value=8, step=1
            ),
            mlp_units=[hp.Int("mlp_units_1", min_value=64, max_value=256, step=32)],
            n_classes=self.n_classes,
            mlp_dropout=hp.Float("mlp_dropout", min_value=0.1, max_value=0.5, step=0.1),
            dropout=hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1),
        )

        # Store the encoder model as an instance attribute for later access
        self.encoder_model = encoder_model

        # Compile the model
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["sparse_categorical_accuracy"],
        )

        # Return only the full model to Keras Tuner
        return model
