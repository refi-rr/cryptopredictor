import pandas as pd
import requests
import time
from datetime import datetime, timedelta

def read_crypto_tickers():
    """
    Read cryptocurrency tickers with CoinGecko compatible IDs
    """
    crypto_tickers = [
        {"ticker": "bitcoin", "name": "Bitcoin", "symbol": "BTC"},
        {"ticker": "ethereum", "name": "Ethereum", "symbol": "ETH"},
        {"ticker": "binancecoin", "name": "Binance Coin", "symbol": "BNB"},
        {"ticker": "ripple", "name": "Ripple", "symbol": "XRP"},
        {"ticker": "cardano", "name": "Cardano", "symbol": "ADA"},
        {"ticker": "dogecoin", "name": "Dogecoin", "symbol": "DOGE"},
        {"ticker": "solana", "name": "Solana", "symbol": "SOL"},
        {"ticker": "polkadot", "name": "Polkadot", "symbol": "DOT"},
        {"ticker": "avalanche-2", "name": "Avalanche", "symbol": "AVAX"},
        {"ticker": "matic-network", "name": "Polygon", "symbol": "MATIC"},
        {"ticker": "chainlink", "name": "Chainlink", "symbol": "LINK"},
        {"ticker": "litecoin", "name": "Litecoin", "symbol": "LTC"},
        {"ticker": "hyperliquid", "name": "Hyperliquid", "symbol": "HYPE"},
        {"ticker": "aster-2", "name": "Aster", "symbol": "ASTER"}

    ]
    return crypto_tickers

def download_data_from_coingecko(coin_id, days=730):  # Default 2 years
    """
    Download cryptocurrency data from CoinGecko API
    """
    try:
        print(f"🔍 Downloading data for {coin_id} from CoinGecko...")
        
        # CoinGecko API endpoint for historical market data
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            return pd.DataFrame(), None
            
        data = response.json()
        
        # Extract prices from response
        if 'prices' not in data or not data['prices']:
            print(f"❌ No price data found for {coin_id}")
            return pd.DataFrame(), None
            
        prices = data['prices']
        
        # Convert to DataFrame
        df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
        
        # Add OHLC data if available (using close for all since CoinGecko only provides close in market_chart)
        df['Open'] = df['Close']
        df['High'] = df['Close']
        df['Low'] = df['Close']
        df['Volume'] = 0  # Placeholder
        
        print(f"✅ Successfully downloaded {len(df)} records for {coin_id}")
        return df, 'Close'
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return pd.DataFrame(), None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return pd.DataFrame(), None

def download_data_with_fallback(coin_id):
    """
    Download data with multiple fallback strategies
    """
    # Try main method first
    data, close_col = download_data_from_coingecko(coin_id)
    
    if not data.empty:
        return data, close_col
    
    # If 2 years fails, try 1 year
    print("🔄 Trying with shorter period...")
    data, close_col = download_data_from_coingecko(coin_id, days=365)
    
    if not data.empty:
        return data, close_col
        
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

def validate_data(data, coin_id):
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
