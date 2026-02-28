import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

def build_lstm_model(
    sequence_length: int = 60,
    n_features: int = 6,
    units: list = [128, 64],
    dropout_rate: float = 0.2,
) -> Sequential:
    model = Sequential()
    for i, u in enumerate(units):
        model.add(LSTM(
            u,
            return_sequences=(i < len(units) - 1),
            input_shape=(sequence_length, n_features) if i == 0 else None,
            kernel_regularizer=l2(1e-4),
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='huber',
        metrics=['mae'],
    )
    return model
