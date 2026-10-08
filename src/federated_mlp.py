"""Federated MLP architecture recovered from the TensorFlow Federated experiment."""

import tensorflow as tf


def create_keras_mlp_model(feature_dim, num_classes):
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(feature_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])


def keras_loss():
    return tf.keras.losses.SparseCategoricalCrossentropy()


def keras_metrics():
    return [tf.keras.metrics.SparseCategoricalAccuracy()]

# The recovered notebook split the SMOTE-balanced training set into client shards
# and trained this model with TensorFlow Federated weighted FedAvg. The full
# orchestration depends on the installed TensorFlow Federated version, so the
# portable model definition is kept here while the repository README documents
# the original federated experiment.
