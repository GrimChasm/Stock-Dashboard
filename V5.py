import streamlit as st
from openbb import obb
import pandas as pd
import plotly.graph_objs as go
import pandas_ta as ta

# --- Helper Functions ---

def get_asset_df(symbol, is_crypto=False, start_date=None, end_date=None):
    if is_crypto:
        result = obb.crypto.price.historical(symbol, start_date=start_date, end_date=end_date)
    else:
        result = obb.equity.price.historical(symbol, start_date=start_date, end_date=end_date)
    df = result.to_dataframe()
    return df

def add_indicators(df, rsi_length, macd_fast, macd_slow, macd_signal, ema_windows):
    df['RSI'] = ta.rsi(df['close'], length=rsi_length)
    macd = ta.macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    if macd is not None:
        df['MACD'] = macd.iloc[:, 0]
        df['MACD_signal'] = macd.iloc[:, 1]
        df['MACD_hist'] = macd.iloc[:, 2]
    for window in ema_windows:
        df[f'EMA_{window}'] = ta.ema(df['close'], length=window)
    return df

# --- Streamlit App ---

st.title("🚀 Multi-Asset Interactive Dashboard (Stocks & Crypto) + Alerts")

symbols_input = st.sidebar.text_input("Enter Symbols (comma-separated, e.g., AAPL,BTC-USD)", value="AAPL,BTC-USD")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.Timestamp.today())

rsi_length = st.sidebar.number_input("RSI Length", min_value=2, value=14)
macd_fast = st.sidebar.number_input("MACD Fast", min_value=2, value=12)
macd_slow = st.sidebar.number_input("MACD Slow", min_value=2, value=26)
macd_signal = st.sidebar.number_input("MACD Signal", min_value=1, value=9)
ema_windows_input = st.sidebar.text_input("EMA Windows (comma-separated)", value="20,50")
ema_windows = [int(x.strip()) for x in ema_windows_input.split(",") if x.strip()]

indicator_toggles = {
    "Show RSI": st.sidebar.checkbox("Show RSI", value=True),
    "Show MACD": st.sidebar.checkbox("Show MACD", value=True),
    "Show EMA": st.sidebar.checkbox("Show EMA", value=True),
    "Show Volume": st.sidebar.checkbox("Show Volume", value=True)
}

theme = st.sidebar.selectbox("Chart Theme", ["plotly_dark", "plotly_white", "ggplot2", "seaborn"])

# Instead of button, just run this directly, so updates on any change
for symbol in symbols:
    st.header(f"📊 {symbol}")
    is_crypto = any(sub in symbol.upper() for sub in ['BTC', 'ETH', 'DOGE', 'SOL', 'ADA', '-USD'])
    try:
        with st.spinner(f"Loading data for {symbol}..."):
            df = get_asset_df(symbol, is_crypto, start_date, end_date)
            df = add_indicators(df, rsi_length, macd_fast, macd_slow, macd_signal, ema_windows)

        alerts = []
        if df['RSI'].iloc[-1] > 70:
            alerts.append(f"⚠️ {symbol}: RSI Overbought ({df['RSI'].iloc[-1]:.2f})")
        if df['RSI'].iloc[-1] < 30:
            alerts.append(f"⚠️ {symbol}: RSI Oversold ({df['RSI'].iloc[-1]:.2f})")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))

        if indicator_toggles["Show EMA"]:
            for window in ema_windows:
                fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{window}'], mode='lines', name=f'EMA {window}'))

        if indicator_toggles["Show Volume"]:
            fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', yaxis='y2', marker_color='lightgrey', opacity=0.3))

        if indicator_toggles["Show RSI"]:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', yaxis='y3'))

        if indicator_toggles["Show MACD"]:
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', yaxis='y4'))
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], mode='lines', name='MACD Signal', yaxis='y4'))
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD Histogram', yaxis='y4', marker_color='grey', opacity=0.5))

        fig.update_layout(
            template=theme,
            title=f"{symbol} Chart with Indicators",
            xaxis=dict(rangeslider=dict(visible=False)),
            yaxis=dict(title='Price'),
            yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
            yaxis3=dict(title='RSI', anchor='free', overlaying='y', side='right', position=1),
            yaxis4=dict(title='MACD', anchor='free', overlaying='y', side='right', position=0.85),
            legend=dict(orientation='h')
        )

        st.plotly_chart(fig, use_container_width=True)

        if alerts:
            for alert in alerts:
                st.warning(alert)

        st.download_button("Download Data CSV", df.to_csv().encode('utf-8'), file_name=f"{symbol}_data.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error loading {symbol}: {e}")
