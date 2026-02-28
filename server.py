from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from src.preprocessor import StockPreprocessor
from src.data_fetcher import fetch_stock_data
import pickle

app = Flask(__name__)
model = load_model("./models/best_model.keras")
preprocessor = pickle.load(open("./models/preprocessor.pkl", "rb"))

@app.route("/predict", methods=["GET"])
def predict():
    ticker = request.args.get("ticker", "AAPL")
    days = int(request.args.get("days", 7))
    df = fetch_stock_data(ticker, period="6mo")
    df = preprocessor.add_technical_indicators(df)
    features = ['Close', 'Volume', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']
    recent = preprocessor.scaler.transform(df[features].values[-60:])
    predictions = []
    seq = recent.copy()
    for _ in range(days):
        pred = model.predict(seq.reshape(1, 60, -1), verbose=0)
        predictions.append(float(pred[0, 0]))
        seq = np.roll(seq, -1, axis=0)
        seq[-1, 0] = pred[0, 0]
    prices = preprocessor.inverse_transform_predictions(
        np.array(predictions)
    )
    return jsonify({"ticker": ticker, "predictions": prices.tolist()})

if __name__ == "__main__":
    app.run(port=5000, debug=False)
