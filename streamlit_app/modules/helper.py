import yfinance as yf
import pandas as pd
import requests
import time
from datetime import datetime

def read_crypto_tickers():
    """
    Read cryptocurrency tickers with enhanced error handling
    """
    crypto_tickers = [
        {"ticker": "BTC-USD", "name": "Bitcoin"},
        {"ticker": "ETH-USD", "name": "Ethereum"}, 
        {"ticker": "BNB-USD", "name": "Binance Coin"},
        {"ticker": "XRP-USD", "name": "Ripple"},
        {"ticker": "ADA-USD", "name": "Cardano"},
        {"ticker": "DOGE-USD", "name": "Dogecoin"},
        {"ticker": "SOL-USD", "name": "Solana"},
        {"ticker": "DOT-USD", "name": "Polkadot"},
        {"ticker": "AVAX-USD", "name": "Avalanche"},
        {"ticker": "MATIC-USD", "name": "Polygon"}
    ]
    return crypto_tickers

def download_data(ticker, max_retries=2):
    """
    Download cryptocurrency data with enhanced error handling and retry mechanism
    """
    print(f"🔍 Downloading data for {ticker}...")  # Debug print
    
    for attempt in range(max_retries):
        try:
            # Create ticker object
            ticker_obj = yf.Ticker(ticker)
            
            # Try with shorter period first to avoid delisted error
            data = ticker_obj.history(period="2y", interval="1d", auto_adjust=True)
            
            if data.empty:
                print(f"⚠️ Data empty for {ticker}, attempt {attempt + 1}")
                # Try with different period
                data = ticker_obj.history(period="1y", interval="1d", auto_adjust=True)
                
            if data.empty:
                print(f"❌ Still no data for {ticker}")
                time.sleep(1)
                continue
                
            print(f"✅ Successfully downloaded {len(data)} records for {ticker}")
            
            # Reset index to get Date as column
            data.reset_index(inplace=True)
            
            # Ensure we have the required columns
            if 'Close' not in data.columns:
                print(f"❌ Close column missing for {ticker}")
                return pd.DataFrame(), None
                
            # Set Date as index
            data.set_index('Date', inplace=True)
            
            # Return data and close column name
            return data, 'Close'
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed for {ticker}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"💥 All attempts failed for {ticker}")
                return pd.DataFrame(), None
    
    return pd.DataFrame(), None

def download_data_with_fallback(ticker):
    """
    Download data with multiple fallback strategies
    """
    # Try main method first
    data, close_col = download_data(ticker)
    
    if not data.empty:
        return data, close_col
    
    # If main method fails, try alternative tickers
    fallback_tickers = {
        'BTC-USD': 'BTC-USD',
        'ETH-USD': 'ETH-USD', 
        'BNB-USD': 'BNB-USD',
        'XRP-USD': 'XRP-USD',
        'ADA-USD': 'ADA-USD',
        'DOGE-USD': 'DOGE-USD',
        'SOL-USD': 'SOL-USD',
        'DOT-USD': 'DOT-USD',
        'AVAX-USD': 'AVAX-USD',
        'MATIC-USD': 'MATIC-USD'
    }
    
    if ticker in fallback_tickers:
        print(f"🔄 Trying fallback for {ticker}...")
        return download_data(fallback_tickers[ticker])
    
    return pd.DataFrame(), None

def format_table(df):
    """
    Format dataframe for display
    """
    if df.empty:
        return df
        
    df_formatted = df.copy()
    
    # Format date column if exists
    if 'Date' in df_formatted.columns:
        df_formatted['Date'] = pd.to_datetime(df_formatted['Date']).dt.strftime('%Y-%m-%d')
    
    # Format price columns if exist
    price_columns = ['Predicted Price', 'Close', 'Open', 'High', 'Low']
    for col in price_columns:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
    
    return df_formatted

def validate_data(data, ticker):
    """
    Validate downloaded data
    """
    if data.empty:
        return False, "Data kosong"
    
    if len(data) < 60:
        return False, f"Data historis tidak cukup. Diperlukan minimal 60 data, tersedia {len(data)}"
    
    if 'Close' not in data.columns:
        return False, "Column 'Close' tidak ditemukan dalam data"
    
    return True, "Data valid"
