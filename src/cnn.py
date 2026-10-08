"""1D CNN architecture recovered from the NSL-KDD experiments."""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping


def build_cnn(input_features: int, n_classes: int):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(input_features, 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(n_classes, activation='softmax'),
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )
    return model


def reshape_for_cnn(X):
    return X.reshape((X.shape[0], X.shape[1], 1))


def training_callback():
    return EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=5,
        restore_best_weights=True,
    )
