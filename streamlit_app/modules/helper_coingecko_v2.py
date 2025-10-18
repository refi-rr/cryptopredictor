import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import numpy as np

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
        {"ticker": "bitcoin-cash", "name": "Bitcoin Cash", "symbol": "BCH"},
        {"ticker": "stellar", "name": "Stellar", "symbol": "XLM"},
        {"ticker": "uniswap", "name": "Uniswap", "symbol": "UNI"}
    ]
    return crypto_tickers

def download_data_from_coingecko(coin_id, days=1095, interval='daily'):
    """
    Download cryptocurrency data from CoinGecko API with enhanced features
    """
    try:
        print(f"🔍 Downloading {days} days of data for {coin_id} from CoinGecko...")
        
        # CoinGecko API endpoint for historical market data
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': interval
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=60)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code} - {response.text}")
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
        
        # Add OHLC data (using close for OHL since CoinGecko only provides close in market_chart)
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

def download_data_with_fallback(coin_id, days=1095):
    """
    Download data with multiple fallback strategies for maximum reliability
    """
    print(f"🔄 Starting data download for {coin_id} with {days} days...")
    
    # Try main method first with requested days
    data, close_col = download_data_from_coingecko(coin_id, days=days)
    
    if not data.empty:
        return data, close_col
    
    # If main method fails, try with reduced days
    fallback_periods = [730, 365, 180, 90]
    
    for period in fallback_periods:
        print(f"🔄 Trying fallback period: {period} days")
        data, close_col = download_data_from_coingecko(coin_id, days=period)
        
        if not data.empty:
            print(f"✅ Success with {period} days period")
            return data, close_col
        time.sleep(1)
    
    print(f"💥 All download attempts failed for {coin_id}")
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
