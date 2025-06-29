import os

# Set a safe, writable directory
os.environ["OPENBB_CACHE_DIR"] = "/tmp/openbb_cache"
import streamlit as st
from openbb import obb
import pandas as pd
import plotly.graph_objs as go
import pandas_ta as ta
import requests

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

    bb = ta.bbands(df['close'], length=20)
    if bb is not None:
        df['BB_upper'] = bb.iloc[:, 0]
        df['BB_middle'] = bb.iloc[:, 1]
        df['BB_lower'] = bb.iloc[:, 2]
    return df

def detect_signals(df):
    buy_signals = []
    sell_signals = []
    for i in range(1, len(df)):
        if df['close'].iloc[i-1] < df['BB_lower'].iloc[i-1] and df['close'].iloc[i] > df['BB_lower'].iloc[i]:
            buy_signals.append((df.index[i], df['close'].iloc[i]))
        elif df['close'].iloc[i-1] > df['BB_upper'].iloc[i-1] and df['close'].iloc[i] < df['BB_upper'].iloc[i]:
            sell_signals.append((df.index[i], df['close'].iloc[i]))
    return buy_signals, sell_signals

def get_asset_summary(symbol):
    summary = obb.equity.fundamental.metrics(symbol)
    return summary.to_dict() if summary else {}

def fetch_sentiment(symbol):
    try:
        result = obb.news.sentiment(symbol)
        sentiments = result.to_dataframe()
        if not sentiments.empty:
            sentiments = sentiments.head(5)[['published_at', 'summary', 'sentiment_label']]
            return sentiments
    except:
        pass
    return pd.DataFrame()

def generate_ai_insights(df):
    insights = []

    if df['close'].iloc[-1] > df['close'].iloc[0]:
        insights.append("📈 The asset shows an upward trend over the selected period.")
    else:
        insights.append("📉 The asset shows a downward trend over the selected period.")

    returns = df['close'].pct_change().dropna()
    volatility = returns.std()
    if volatility > 0.04:
        insights.append(f"⚠️ High volatility detected (Std: {volatility:.2%}). Suitable for experienced traders.")
    elif volatility < 0.01:
        insights.append(f"🔍 Low volatility (Std: {volatility:.2%}). Indicates a stable price movement.")
    else:
        insights.append(f"📊 Moderate volatility observed (Std: {volatility:.2%}).")

    avg_volume = df['volume'].tail(30).mean()
    last_volume = df['volume'].iloc[-1]
    if last_volume > 1.5 * avg_volume:
        insights.append(f"📢 Surge in volume (Current: {last_volume:.0f}, Avg: {avg_volume:.0f}).")
    elif last_volume < 0.5 * avg_volume:
        insights.append(f"💤 Low volume activity (Current: {last_volume:.0f}, Avg: {avg_volume:.0f}).")

    return insights

# --- Streamlit App ---

st.set_page_config(page_title="Asset Dashboard", layout="wide")

custom_css = """
    <style>
        body[data-theme="Dark"] {
            background-color: #0e1117;
            color: white;
        }
        body[data-theme="Light"] {
            background-color: #ffffff;
            color: black;
        }
    </style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

st.title("🚀 Multi-Asset Interactive Dashboard (Stocks & Crypto) + Alerts")

symbols_input = st.sidebar.text_input("Enter Symbols (comma-separated, e.g., MSFT,AMZN,GOOG)", value="MSFT,AMZN,GOOG")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.Timestamp.today())

rsi_length = st.sidebar.number_input("RSI Length", min_value=2, value=14)
macd_fast = st.sidebar.number_input("MACD Fast", min_value=2, value=12)
macd_slow = st.sidebar.number_input("MACD Slow", min_value=2, value=26)
macd_signal = st.sidebar.number_input("MACD Signal", min_value=1, value=9)
ema_windows_input = st.sidebar.text_input("EMA Windows (comma-separated)", value="20,50")
ema_windows = [int(x.strip()) for x in ema_windows_input.split(",") if x.strip()]

theme_choice = st.sidebar.radio("Select Theme", ["Dark", "Light"])
theme = "plotly_dark" if theme_choice == "Dark" else "plotly_white"
st.markdown(f"<script>document.body.setAttribute('data-theme', '{theme_choice}');</script>", unsafe_allow_html=True)

indicator_toggles = {
    "Show RSI": st.sidebar.checkbox("Show RSI", value=True),
    "Show MACD": st.sidebar.checkbox("Show MACD", value=True),
    "Show EMA": st.sidebar.checkbox("Show EMA", value=True),
    "Show BBands": st.sidebar.checkbox("Show Bollinger Bands", value=True),
    "Show Volume": st.sidebar.checkbox("Show Volume", value=True),
    "Show Buy/Sell Signals": st.sidebar.checkbox("Show Buy/Sell Signals", value=True)
}

for symbol in symbols:
    st.header(f"📊 {symbol}")
    is_crypto = any(sub in symbol.upper() for sub in ['BTC', 'ETH', 'DOGE', 'SOL', 'ADA', '-USD'])

    try:
        df = get_asset_df(symbol, is_crypto, start_date, end_date)
        df = add_indicators(df, rsi_length, macd_fast, macd_slow, macd_signal, ema_windows)
        buy_signals, sell_signals = detect_signals(df) if indicator_toggles["Show Buy/Sell Signals"] else ([], [])

        current_price = df['close'].iloc[-1]
        st.subheader(f"💲 Current Price: {current_price:.2f}")

        summary_data = get_asset_summary(symbol)
        if summary_data:
            st.markdown("**Technical Summary**")
            cols = st.columns(4)
            labels = [
                ("Previous Close", "previousClose"),
                ("Open", "open"),
                ("Day's Range", "dayRange"),
                ("52 Week Range", "fiftyTwoWeekRange"),
                ("Volume", "volume"),
                ("Average Volume", "averageVolume"),
                ("Market Cap", "marketCap"),
                ("PE Ratio", "peRatio"),
                ("EPS", "eps"),
                ("Earnings Date", "earningsDate")
            ]
            for i, (label, key) in enumerate(labels):
                cols[i % 4].markdown(f"**{label}:** {summary_data.get(key, 'N/A')}")

        alerts = []
        if df['RSI'].iloc[-1] > 70:
            alerts.append(f"⚠️ {symbol}: RSI Overbought ({df['RSI'].iloc[-1]:.2f})")
        if df['RSI'].iloc[-1] < 30:
            alerts.append(f"⚠️ {symbol}: RSI Oversold ({df['RSI'].iloc[-1]:.2f})")

        insights = generate_ai_insights(df)
        with st.expander("🧠 AI-Generated Insights"):
            for insight in insights:
                st.markdown(insight)

        sentiment_df = fetch_sentiment(symbol)
        if not sentiment_df.empty:
            with st.expander("📰 News Sentiment"):
                for _, row in sentiment_df.iterrows():
                    st.markdown(f"**{row['published_at']}** - `{row['sentiment_label']}`\n> {row['summary']}")

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

        if indicator_toggles["Show BBands"]:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], mode='lines', name='BB Upper', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_middle'], mode='lines', name='BB Mid', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], mode='lines', name='BB Lower', line=dict(dash='dot')))

        if indicator_toggles["Show Buy/Sell Signals"]:
            if buy_signals:
                fig.add_trace(go.Scatter(x=[x[0] for x in buy_signals], y=[x[1] for x in buy_signals],
                                         mode='markers', marker_symbol='triangle-up', marker_color='green', marker_size=10, name='Buy Signal'))
            if sell_signals:
                fig.add_trace(go.Scatter(x=[x[0] for x in sell_signals], y=[x[1] for x in sell_signals],
                                         mode='markers', marker_symbol='triangle-down', marker_color='red', marker_size=10, name='Sell Signal'))

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
