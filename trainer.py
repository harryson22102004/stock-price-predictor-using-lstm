import os
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from src.model import build_lstm_model

class Trainer:
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        model = build_lstm_model(
            sequence_length=X_train.shape[1],
            n_features=X_train.shape[2],
        )
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
            ModelCheckpoint(
                os.path.join(self.model_dir, "best_model.keras"),
                monitor='val_loss', save_best_only=True,
            ),
        ]
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
        )
        return model, history
