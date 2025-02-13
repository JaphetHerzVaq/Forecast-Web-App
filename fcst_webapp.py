from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# Obtener datos históricos de una acción
def scrape_stock_data(comp_name):
    yf_data = yf.download(comp_name, start="2024-01-01", end=datetime.now().strftime("%Y-%m-%d"))
    return yf_data

# Preparar datos para el modelo
def prepare_data(prices, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled_prices) - time_step - 1):
        X.append(scaled_prices[i:(i + time_step), 0])
        y.append(scaled_prices[i + time_step, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)  
    return X, y, scaler

# Entrenar el modelo
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq)
            labels = labels.unsqueeze(1)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
    return model

# Predecir precios futuros
def predict_next_30_days(model, data, scaler, time_step=60):
    model.eval()
    predictions = []
    last_sequence = data[-time_step:]

    with torch.no_grad():
        for _ in range(30):
            last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
            last_sequence_scaled = torch.tensor(last_sequence_scaled, dtype=torch.float32).reshape(1, time_step, 1)
            predicted_price_scaled = model(last_sequence_scaled).numpy()
            predicted_price = scaler.inverse_transform(predicted_price_scaled)
            predictions.append(predicted_price[0][0])
            last_sequence = np.append(last_sequence[1:], predicted_price)
    return predictions

# Guardar y mostrar la gráfica de predicciones
def plot_results(historical_data, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data, label='Historical Prices', color='blue')
    
    future_dates = [datetime.now().date() + timedelta(days=i) for i in range(1, 31)]
    plt.plot(future_dates, predictions, '--', label='Predicted Prices', color='red')

    plt.title('Stock Price Prediction for Next 30 Days')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Guardar la imagen en la carpeta static
    plt.savefig("static/stock_prediction.png")
    plt.close()

# Ruta principal con formulario HTML
@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    ticker = None
    image_url = None

    if request.method == "POST":
        ticker = request.form["ticker"]
        stock_data = scrape_stock_data(ticker)
        
        if stock_data.empty:
            return render_template("index.html", error="No se encontraron datos para esta acción.")

        prices = stock_data['Close'].values.reshape(-1, 1)

        # Preparar los datos
        time_step = 60
        X, y, scaler = prepare_data(prices, time_step)

        # Convertir datos a tensores
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Crear DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Entrenar modelo
        model = LSTMModel()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model = train_model(model, train_loader, criterion, optimizer, num_epochs=10)

        # Hacer predicciones
        predicted_prices = predict_next_30_days(model, prices, scaler, time_step)
        future_dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 31)]

        predictions = list(zip(future_dates, predicted_prices))

        # Generar la gráfica
        plot_results(stock_data['Close'], predicted_prices)
        image_url = "static/stock_prediction.png"

    return render_template("index.html", predictions=predictions, ticker=ticker, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

