import os
import yfinance as yf
import pandas as pd
import numpy as np

RAW_PATH = "data/raw_aapl.csv"
PROCESSED_PATH = "data/processed_aapl.csv"

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df, window=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr


# DATA LOADING PIPELINE

def download_raw_data():
    print("Downloading raw AAPL data from Yahoo Finance...")
    df = yf.download("AAPL", start="2020-01-01", end="2024-12-31")

    df = df.reset_index()

    df.to_csv(RAW_PATH, index=False)
    print("Saved:", RAW_PATH)
    return df


def load_raw_data():
    if os.path.exists(RAW_PATH):
        print("Loading raw data from CSV...")
        return pd.read_csv(RAW_PATH)
    else:
        return download_raw_data()


def process_data(df):
    df = df.copy()
    
    # Ensure Date is present
    if 'Date' not in df.columns and df.index.name == 'Date':
        df = df.reset_index()
    
    # Technical Indicators
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["ATR"] = compute_atr(df)

    df = df.dropna()

    # Save original Close for trading calculations
    df["Close_original"] = df["Close"]
    
    # Normalize numerical features (for state representation only)
    features = ["Close", "MA_5", "MA_20", "RSI", "ATR", "Volume"]
    df[features] = (df[features] - df[features].mean()) / df[features].std()

    # Save processed CSV including Date
    df.to_csv(PROCESSED_PATH, index=False)
    print("Saved processed data:", PROCESSED_PATH)

    return df


def load_processed_data():
    if os.path.exists(PROCESSED_PATH):
        print("Loading processed data from CSV...")
        df = pd.read_csv(PROCESSED_PATH)
        
        # Check for duplicate dates BEFORE setting as index
        print(f"Duplicate dates: {df['Date'].duplicated().sum()}")
        if df['Date'].duplicated().any():
            print("Removing duplicate dates...")
            df = df.drop_duplicates(subset='Date', keep='last')
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Ensure all numeric columns are float type
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                          'MA_5', 'MA_20', 'RSI', 'ATR']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.set_index('Date')
        df = df.sort_index()
        
        print(f"Loaded {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}")
        return df
    
    print("No processed file found. Processing now...")
    raw = load_raw_data()
    processed = process_data(raw)
    
    # Apply same datetime index conversion
    processed['Date'] = pd.to_datetime(processed['Date'])
    processed = processed.set_index('Date')
    processed = processed.sort_index()
    
    return processed


def train_val_test_split(df):
    df = df.sort_index()
    
    if df.index.duplicated().any():
        print("Warning: Duplicate dates found. Keeping last occurrence.")
        df = df[~df.index.duplicated(keep='last')]
    
    if not df.index.is_monotonic_increasing:
        print("Warning: Index not monotonic after sorting. Forcing sort.")
        df = df.sort_index()
    
    print(f"Splitting data from {df.index.min().date()} to {df.index.max().date()}")
    
    # Use more explicit datetime slicing
    train = df[(df.index >= '2020-01-01') & (df.index <= '2022-12-31')]
    val = df[(df.index >= '2023-01-01') & (df.index <= '2023-06-30')]
    test = df[(df.index >= '2023-07-01') & (df.index <= '2024-12-31')]
    
    print(f"Train: {len(train)} rows ({train.index.min().date()} to {train.index.max().date()})")
    print(f"Val: {len(val)} rows ({val.index.min().date()} to {val.index.max().date()})")
    print(f"Test: {len(test)} rows ({test.index.min().date()} to {test.index.max().date()})")

    return train, val, test


if __name__ == "__main__":
    df = load_processed_data()
    train, val, test = train_val_test_split(df)

    print("\nFinal shapes:")
    print("Train:", train.shape)
    print("Val:", val.shape)
    print("Test:", test.shape)