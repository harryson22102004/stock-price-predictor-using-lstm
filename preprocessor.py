import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

class StockPreprocessor:
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._fitted = False

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        # Bollinger Bands
        sma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        df['BB_Upper'] = sma20 + (std20 * 2)
        df['BB_Lower'] = sma20 - (std20 * 2)
        # Volume change
        df['Volume_Change'] = df['Volume'].pct_change()
        return df.dropna()

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        features = ['Close', 'Volume', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']
        data = df[features].values
        scaled = self.scaler.fit_transform(data)
        self._fitted = True
        return self._create_sequences(scaled)

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(data[i, 0])  # Predict Close price
        return np.array(X), np.array(y)

    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        dummy = np.zeros((len(predictions), self.scaler.n_features_in_))
        dummy[:, 0] = predictions.flatten()
        return self.scaler.inverse_transform(dummy)[:, 0]
