import os
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
from datetime import datetime

# Create reports directory, if it does not exist already
os.makedirs('reports', exist_ok=True)

plt.style.use('ggplot')

# 1. Data Collection and Preprocessing
def fetch_and_preprocess_data(ticker="NVDA", start_date="2010-01-01", end_date="2023-11-13"):
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

def evaluate_model(model, X_test, y_test, scaler, original_data, features, model_name, ticker):
    """Evaluate model performance and generate report components"""
    start_time = datetime.now()
    predictions = model.predict(X_test)
    test_time = (datetime.now() - start_time).total_seconds()
    
    # Inverse transform
    dummy_array = np.zeros((len(predictions), len(features)))
    dummy_array[:, 0] = predictions.flatten()
    predictions = scaler.inverse_transform(dummy_array)[:, 0]
    
    dummy_array[:, 0] = y_test.flatten()
    y_test_actual = scaler.inverse_transform(dummy_array)[:, 0]
    
    # Calculate metrics
    mse = mean_squared_error(y_test_actual, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, predictions)
    
    # Generate plots
    test_dates = original_data.index[-len(y_test_actual):]
    plot_filename = save_prediction_plot(test_dates, y_test_actual, predictions, model_name, ticker)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'test_time': test_time,
        'predictions': predictions,
        'actual': y_test_actual,
        'plot_file': plot_filename
    }

def generate_report(ticker, model_name, rmse, mae, model_summary, 
                   train_time, test_time, params, additional_notes=""):
    """Generate a performance report and save to file"""
    report_filename = f"reports/{ticker}_{model_name}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"STOCK PREDICTION MODEL PERFORMANCE REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Ticker: {ticker}\n")
        f.write(f"Model Type: {model_name}\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("MODEL METRICS\n")
        f.write("-"*60 + "\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("MODEL ARCHITECTURE\n")
        f.write("-"*60 + "\n")
        f.write(model_summary + "\n\n")
        
        f.write("-"*60 + "\n")
        f.write("TRAINING DETAILS\n")
        f.write("-"*60 + "\n")
        f.write(f"Training Time: {train_time:.2f} seconds\n")
        f.write(f"Testing Time: {test_time:.2f} seconds\n")
        f.write(f"Model Parameters:\n")
        for key, value in params.items():
            f.write(f"  {key}: {value}\n")
        
        if additional_notes:
            f.write("\n" + "-"*60 + "\n")
            f.write("ADDITIONAL NOTES\n")
            f.write("-"*60 + "\n")
            f.write(additional_notes + "\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*60 + "\n")
    
    print(f"Report generated: {report_filename}")
    return report_filename

def save_training_plot(history, model_name, ticker):
    """Save training history plot as image"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    
    plot_filename = f"reports/{ticker}_{model_name}_training_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    return plot_filename

def save_prediction_plot(test_dates, y_test_actual, predictions, model_name, ticker):
    """Save prediction plot as image"""
    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, y_test_actual, label='Actual Price', color='blue')
    plt.plot(test_dates, predictions, label='Predicted Price', color='red', linestyle='--')
    plt.title(f'{ticker} Stock Price Prediction ({model_name})')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    
    plot_filename = f"reports/{ticker}_{model_name}_prediction_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_filename)
    plt.close()
    return plot_filename


# Main Execution
if __name__ == "__main__":
    # Configuration
    TICKER = "NVDA"
    FEATURES = ['Close', 'Volume', 'SMA_50', 'RSI', 'MACD', 'Volume_Change']
    TARGET = 'Close'
    SEQ_LENGTH = 60
    TEST_SIZE = 0.2
    
    # Data collection and preprocessing
    stock_data = fetch_and_preprocess_data(TICKER)
    X_train, X_test, y_train, y_test, scaler = prepare_data(stock_data, FEATURES, TARGET, SEQ_LENGTH, TEST_SIZE)
    
    # LSTM Model
    lstm_start = datetime.now()
    lstm_model = build_lstm_model((SEQ_LENGTH, len(FEATURES)))
    lstm_history = train_model(lstm_model, X_train, y_train)
    lstm_train_time = (datetime.now() - lstm_start).total_seconds()
    
    # Evaluate LSTM
    lstm_results = evaluate_model(lstm_model, X_test, y_test, scaler, stock_data, FEATURES, "LSTM", TICKER)
    save_training_plot(lstm_history, "LSTM", TICKER)
    
    # Generate LSTM report
    lstm_report = generate_report(
        ticker=TICKER,
        model_name="LSTM",
        rmse=lstm_results['rmse'],
        mae=lstm_results['mae'],
        model_summary=str(lstm_model.summary()),
        train_time=lstm_train_time,
        test_time=lstm_results['test_time'],
        params={
            'Sequence Length': SEQ_LENGTH,
            'Features': FEATURES,
            'LSTM Units': 50,
            'Dropout Rate': 0.2,
            'Epochs': 100,
            'Batch Size': 32
        },
        additional_notes="LSTM model with two LSTM layers and dropout regularization."
    )
    
    # GRU Model
    gru_start = datetime.now()
    gru_model = build_gru_model((SEQ_LENGTH, len(FEATURES)))
    gru_history = train_model(gru_model, X_train, y_train)
    gru_train_time = (datetime.now() - gru_start).total_seconds()
    
    # Evaluate GRU
    gru_results = evaluate_model(gru_model, X_test, y_test, scaler, stock_data, FEATURES, "GRU", TICKER)
    save_training_plot(gru_history, "GRU", TICKER)
    
    # Generate GRU report
    gru_report = generate_report(
        ticker=TICKER,
        model_name="GRU",
        rmse=gru_results['rmse'],
        mae=gru_results['mae'],
        model_summary=str(gru_model.summary()),
        train_time=gru_train_time,
        test_time=gru_results['test_time'],
        params={
            'Sequence Length': SEQ_LENGTH,
            'Features': FEATURES,
            'GRU Units': 50,
            'Dropout Rate': 0.2,
            'Epochs': 100,
            'Batch Size': 32
        },
        additional_notes="GRU model with two GRU layers and tanh/sigmoid activations."
    )
    
    # Comparative report
    comparative_report = f"reports/{TICKER}_COMPARATIVE_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(comparative_report, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"{TICKER} MODEL COMPARISON REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Comparison Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("PERFORMANCE METRICS COMPARISON\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Metric':<20} {'LSTM':<15} {'GRU':<15}\n")
        f.write(f"{'RMSE':<20} {lstm_results['rmse']:<15.4f} {gru_results['rmse']:<15.4f}\n")
        f.write(f"{'MAE':<20} {lstm_results['mae']:<15.4f} {gru_results['mae']:<15.4f}\n")
        f.write(f"{'Training Time (s)':<20} {lstm_train_time:<15.2f} {gru_train_time:<15.2f}\n")
        f.write(f"{'Testing Time (s)':<20} {lstm_results['test_time']:<15.4f} {gru_results['test_time']:<15.4f}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("CONCLUSION\n")
        f.write("-"*60 + "\n")
        f.write("Lower RMSE/MAE values indicate better performance. Compare training times\n")
        f.write("for computational efficiency considerations.\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*60 + "\n")
    
    print(f"\nAll reports generated:")
    print(f"- LSTM Report: {lstm_report}")
    print(f"- GRU Report: {gru_report}")
    print(f"- Comparative Report: {comparative_report}")