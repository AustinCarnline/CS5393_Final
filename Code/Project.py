import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

plt.style.use('ggplot')

# 1. Data Collection and Preprocessing
def fetch_and_preprocess_data(ticker="AAPL", start_date="2010-01-01", end_date="2023-11-13"):
    """Fetch stock data and preprocess it with technical indicators"""
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Calculate technical indicators
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = compute_rsi(data['Close'])
    data['MACD'], data['Signal_Line'] = compute_macd(data['Close'])
    data['Volume_Change'] = data['Volume'].pct_change()
    
    # Drop NA values created by indicators
    data = data.dropna()
    
    return data

def compute_rsi(series, window=14):
    """Compute Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, slow=26, fast=12, signal=9):
    """Compute MACD and Signal line"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# 2. Feature Engineering and Sequence Creation
def create_sequences(data, features, target, seq_length):
    """Create input sequences and target values"""
    X, y = [], []
    data_values = data[features].values
    target_values = data[target].values
    
    for i in range(len(data_values) - seq_length):
        X.append(data_values[i:(i + seq_length)])
        y.append(target_values[i + seq_length])
    
    return np.array(X), np.array(y)

def prepare_data(data, features, target='Close', seq_length=60, test_size=0.2):
    """Prepare data for modeling"""
    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    
    # Create sequences
    X, y = create_sequences(pd.DataFrame(scaled_data, columns=features), 
                          features, target, seq_length)
    
    # Split into train/test sets
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler

# 3. Model Building
def build_lstm_model(input_shape, units=50, dropout_rate=0.2):
    """Build LSTM model"""
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                loss='mean_squared_error',
                metrics=['mae'])
    return model

def build_gru_model(input_shape, units=50, dropout_rate=0.2):
    """Build GRU model"""
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape,
           activation='tanh', recurrent_activation='sigmoid'),
        Dropout(dropout_rate),
        GRU(units, return_sequences=False,
           activation='tanh', recurrent_activation='sigmoid'),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                loss='mean_squared_error',
                metrics=['mae'])
    return model

# 4. Model Training and Evaluation
def train_model(model, X_train, y_train, validation_split=0.1, epochs=100, batch_size=32):
    """Train model with early stopping"""
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    return history

def evaluate_model(model, X_test, y_test, scaler, original_data):
    """Evaluate model performance and visualize results"""
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions
    # Create dummy array for inverse transform
    dummy_array = np.zeros((len(predictions), len(features)))
    dummy_array[:, 0] = predictions.flatten()  # Assuming Close is first feature
    predictions = scaler.inverse_transform(dummy_array)[:, 0]
    
    # Inverse transform actual values
    dummy_array[:, 0] = y_test.flatten()
    y_test_actual = scaler.inverse_transform(dummy_array)[:, 0]
    
    # Calculate metrics
    mse = mean_squared_error(y_test_actual, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, predictions)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # Plot results
    test_dates = original_data.index[-len(y_test_actual):]
    
    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, y_test_actual, label='Actual Price', color='blue')
    plt.plot(test_dates, predictions, label='Predicted Price', color='red', linestyle='--')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.show()
    
    return rmse, mae

# Main Execution
if __name__ == "__main__":
    # Configuration
    TICKER = "AAPL"
    FEATURES = ['Close', 'Volume', 'SMA_50', 'RSI', 'MACD', 'Volume_Change']
    TARGET = 'Close'
    SEQ_LENGTH = 60
    TEST_SIZE = 0.2
    
    # 1. Get and preprocess data
    stock_data = fetch_and_preprocess_data(TICKER)
    
    # 2. Prepare data for modeling
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        stock_data, FEATURES, TARGET, SEQ_LENGTH, TEST_SIZE
    )
    
    # 3. Build and train LSTM model
    print("Training LSTM Model...")
    lstm_model = build_lstm_model((SEQ_LENGTH, len(FEATURES)))
    lstm_history = train_model(lstm_model, X_train, y_train)
    
    # Evaluate LSTM
    print("\nLSTM Model Evaluation:")
    lstm_rmse, lstm_mae = evaluate_model(
        lstm_model, X_test, y_test, scaler, stock_data
    )
    
    # 4. Build and train GRU model
    print("\nTraining GRU Model...")
    gru_model = build_gru_model((SEQ_LENGTH, len(FEATURES)))
    gru_history = train_model(gru_model, X_train, y_train)
    
    # Evaluate GRU
    print("\nGRU Model Evaluation:")
    gru_rmse, gru_mae = evaluate_model(
        gru_model, X_test, y_test, scaler, stock_data
    )
    
    # 5. Compare models
    print("\nModel Comparison:")
    print(f"LSTM - RMSE: {lstm_rmse:.2f}, MAE: {lstm_mae:.2f}")
    print(f"GRU - RMSE: {gru_rmse:.2f}, MAE: {gru_mae:.2f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(lstm_history.history['loss'], label='LSTM Train')
    plt.plot(lstm_history.history['val_loss'], label='LSTM Validation')
    plt.plot(gru_history.history['loss'], label='GRU Train')
    plt.plot(gru_history.history['val_loss'], label='GRU Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(lstm_history.history['mae'], label='LSTM Train')
    plt.plot(lstm_history.history['val_mae'], label='LSTM Validation')
    plt.plot(gru_history.history['mae'], label='GRU Train')
    plt.plot(gru_history.history['val_mae'], label='GRU Validation')
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()