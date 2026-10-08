"""LSTM classifier architecture recovered from the NSL-KDD experiments."""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def build_lstm(input_features: int, n_classes: int):
    model = Sequential([
        LSTM(64, input_shape=(1, input_features), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(n_classes, activation='softmax'),
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )
    return model


def reshape_for_lstm(X):
    """Recovered experiment treats each feature vector as one timestep."""
    return X.reshape((X.shape[0], 1, X.shape[1]))


def training_callback():
    return EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=5,
        restore_best_weights=True,
    )
