from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


def build_model(
    image_size: int = 224,
    dense_units: int = 512,
    dropout: float = 0.5,
) -> tf.keras.Model:
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(image_size, image_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs)


def unfreeze_last_layers(model: tf.keras.Model, n_layers: int = 40) -> int:
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        return 0

    start = max(0, len(base_model.layers) - n_layers)
    unfrozen = 0
    for i, layer in enumerate(base_model.layers):
        should_train = i >= start and not isinstance(layer, layers.BatchNormalization)
        layer.trainable = should_train
        if should_train:
            unfrozen += 1
    return unfrozen


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model
