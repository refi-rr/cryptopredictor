import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from modules.helper import read_crypto_tickers, download_data, format_table

st.set_page_config(page_title="Crypto Price Predictor", page_icon="🤑")

# Sidebar inputs
try:
    crypto_data = read_crypto_tickers()
except Exception as e:
    st.error(f"Error fetching cryptocurrency data: {e}")
    st.stop()

# Perbaikan utama: Safe crypto selection dengan error handling
selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", [f"{d['ticker']} ({d['name']})" for d in crypto_data])

# REPLACE THIS PROBLEMATIC LINE (Line 23):
# selected_ticker = [d['ticker'] for d in crypto_data if selected_crypto.startswith(d['ticker'])][0]

# DENGAN KODE YANG LEBIH AMAN:
try:
    # Cari ticker yang match
    matching_tickers = [d['ticker'] for d in crypto_data if selected_crypto.startswith(d['ticker'])]
    
    if not matching_tickers:
        st.error(f"❌ Ticker tidak ditemukan untuk: {selected_crypto}")
        st.info(f"📋 Pilihan yang tersedia: {[d['ticker'] for d in crypto_data]}")
        st.stop()
    
    selected_ticker = matching_tickers[0]
    st.sidebar.success(f"✅ Selected: {selected_ticker}")
    
except Exception as e:
    st.error(f"Error selecting ticker: {e}")
    st.stop()

# Extract name dengan error handling
try:
    selected_name = selected_crypto.split('(')[1][:-1]
except IndexError:
    selected_name = selected_ticker  # Fallback ke ticker jika parsing gagal

epochs = st.sidebar.slider("Epochs", 1, 50, 10)
days_to_predict = st.sidebar.slider("Prediction Range (Days)", 1, 90, 30)

if st.sidebar.button("Run Prediction"):
    try:
        st.text(f"Fetching data for {selected_ticker} ({selected_name})...")
        crypto_df, close_column = download_data(selected_ticker)
        
        # Validasi data yang didownload
        if crypto_df.empty:
            st.error(f"Data untuk {selected_ticker} kosong. Tidak dapat melanjutkan prediksi.")
            st.stop()
            
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        st.stop()

    st.subheader(f"Historical Prices for {selected_ticker} ({selected_name})")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=crypto_df.index, y=crypto_df[close_column], mode='lines', name="Closing Price"))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        showlegend=True
    )
    st.plotly_chart(fig)

    # Data preparation dengan validasi
    data = crypto_df.filter([close_column])
    dataset = data.values

    if len(dataset) == 0:
        st.error("Filtered data is empty. Cannot proceed with predictions.")
        st.stop()

    # Validasi cukup data untuk training
    if len(dataset) < 100:
        st.warning(f"⚠️ Data historis terbatas ({len(dataset)} records). Prediksi mungkin kurang akurat.")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    train_len = int(len(dataset) * 0.8)

    # Validasi panjang training data
    if train_len < 60:
        st.error(f"❌ Data training tidak cukup. Diperuhkan minimal 60 data, tersedia {train_len}.")
        st.stop()

    train_data = scaled_data[:train_len]
    X_train, y_train = [], []
    for i in range(60, len(train_data)):
        X_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    # Validasi training samples
    if len(X_train) == 0:
        st.error("❌ Tidak cukup data untuk training model.")
        st.stop()

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build and train model dengan progress indicator
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    st.text("Training model...")
    with st.spinner(f"Training model dengan {epochs} epochs..."):
        model.fit(X_train, y_train, batch_size=1, epochs=epochs, verbose=0)
    st.success("✅ Model training completed!")

    # Predictions dengan error handling
    try:
        test_data = scaled_data[train_len - 60:]
        X_test, y_test = [], dataset[train_len:]
        for i in range(60, len(test_data)):
            X_test.append(test_data[i-60:i, 0])

        if len(X_test) == 0:
            st.error("❌ Tidak cukup data untuk testing.")
            st.stop()

        X_test = np.array(X_test).reshape((len(X_test), 60, 1))
        predictions = scaler.inverse_transform(model.predict(X_test))

        # Future predictions
        future_input = scaled_data[-60:]
        future_prices = []
        
        with st.spinner("Generating future predictions..."):
            for _ in range(days_to_predict):
                pred_input = future_input.reshape((1, 60, 1))
                pred_price = model.predict(pred_input, verbose=0)
                future_prices.append(pred_price[0, 0])
                future_input = np.append(future_input[1:], pred_price, axis=0)

        future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))
        future_dates = pd.date_range(start=crypto_df.index[-1] + pd.Timedelta(days=1), periods=days_to_predict)
        future_predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_prices.flatten()})

        # Future Predictions Plot
        st.subheader(f"Forecasted Prices for {selected_ticker} ({selected_name})")
        fig = go.Figure()

        # Historical prices
        fig.add_trace(go.Scatter(
            x=crypto_df.index,
            y=crypto_df[close_column],
            mode='lines',
            name="Historical Price"
        ))

        # Forecast prices
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_prices.flatten(),
            mode='lines',
            name="Forecast",
            line=dict(color="red", dash='dash')
        ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            showlegend=True
        )
        st.plotly_chart(fig)
        
        # Display predicted prices
        st.subheader(f"Predicted Future Prices for {selected_ticker} ({selected_name})")
        st.dataframe(format_table(future_predictions_df))
        
    except Exception as e:
        st.error(f"❌ Error selama prediksi: {e}")
        st.stop()
