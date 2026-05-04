
# TradeEdge Pro
# Install:
# pip install streamlit yfinance pandas numpy plotly requests

import math
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="TradeEdge Pro", page_icon="📈", layout="wide")


# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
<style>
.stApp { background-color:#f3f4f6; color:#111827; }
.block-container { padding-top:1rem; padding-bottom:2rem; }

.sticky-header {
    position: sticky;
    top: 0;
    z-index: 999;
    background: #ffffff;
    padding: 18px;
    border: 1px solid #d1d5db;
    border-radius: 0 0 18px 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.10);
}

.floating-reset {
    position: fixed;
    bottom: 22px;
    right: 22px;
    z-index: 9999;
}

.signal-box {
    padding: 28px;
    border-radius: 20px;
    text-align: center;
    margin-top: 18px;
    margin-bottom: 22px;
    color: white;
    box-shadow: 0 12px 30px rgba(0,0,0,0.20);
}

.signal-title {
    font-size: 42px;
    font-weight: 900;
    letter-spacing: 1px;
    color: white;
}

.signal-subtitle {
    font-size: 20px;
    margin-top: 10px;
    color: white;
}

.news-card {
    background:#ffffff;
    border:1px solid #d1d5db;
    border-radius:14px;
    padding:16px;
    margin-bottom:12px;
    color:#111827;
}

.news-card a {
    color:#2563eb;
    text-decoration:none;
    font-weight:700;
}

div[data-testid="stMetric"] {
    background:#ffffff;
    border:1px solid #d1d5db;
    border-radius:16px;
    padding:16px;
    box-shadow:0 6px 18px rgba(0,0,0,0.08);
}

button { border-radius:12px !important; }

.stTabs [data-baseweb="tab-list"] { gap:8px; }

.stTabs [data-baseweb="tab"] {
    background:#ffffff;
    border:1px solid #d1d5db;
    border-radius:12px;
    padding:10px 16px;
    color:#111827;
}

.stTabs [aria-selected="true"] {
    background:#2563eb !important;
    color:white !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# CONSTANTS
# =========================================================
JOURNAL_FILE = "trade_journal.csv"

POPULAR_TICKERS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Nvidia": "NVDA",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Google / Alphabet": "GOOGL",
    "Meta": "META",
    "Netflix": "NFLX",
    "AMD": "AMD",
    "Palantir": "PLTR",
    "Robinhood": "HOOD",
    "Ford": "F",
    "Disney": "DIS",
    "Costco": "COST",
    "Walmart": "WMT",
    "Coca-Cola": "KO",
    "JPMorgan": "JPM",
    "Bank of America": "BAC",
    "Goldman Sachs": "GS",
    "Coinbase": "COIN",
    "Boeing": "BA",
    "Target": "TGT",
    "Nike": "NKE",
    "SPY ETF": "SPY",
    "QQQ ETF": "QQQ",
    "IWM ETF": "IWM",
    "DIA ETF": "DIA",
}

DEFAULT_SCAN_TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META",
    "NFLX", "AMD", "PLTR", "HOOD", "COIN", "SPY", "QQQ", "IWM"
]


# =========================================================
# SESSION STATE
# =========================================================
if "ticker" not in st.session_state:
    st.session_state.ticker = "AAPL"
if "last_alert" not in st.session_state:
    st.session_state.last_alert = ""
if "manual_ticker" not in st.session_state:
    st.session_state.manual_ticker = ""
if "selected_company" not in st.session_state:
    st.session_state.selected_company = "Apple"


def reset_app():
    st.session_state.manual_ticker = ""
    st.session_state.last_alert = ""


# =========================================================
# JOURNAL
# =========================================================
def load_journal() -> pd.DataFrame:
    cols = [
        "Date/Time", "Ticker", "Trade Type", "Strike", "Expiration",
        "Entry Price", "Exit Price", "Contracts", "Profit/Loss", "Win/Loss"
    ]

    try:
        df = pd.read_csv(JOURNAL_FILE)
        for col in cols:
            if col not in df.columns:
                df[col] = np.nan
        return df[cols]
    except Exception:
        return pd.DataFrame(columns=cols)


def save_journal(df: pd.DataFrame) -> None:
    df.to_csv(JOURNAL_FILE, index=False)


def calculate_pl(entry_price: float, exit_price: float, contracts: int) -> float:
    try:
        return round((float(exit_price) - float(entry_price)) * 100 * int(contracts), 2)
    except Exception:
        return 0.0


# =========================================================
# HELPERS
# =========================================================
def safe_format(value, fmt="{:.2f}") -> str:
    try:
        if pd.isna(value):
            return "N/A"
        return fmt.format(value)
    except Exception:
        return "N/A"


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def clean_iv_value(iv) -> float:
    try:
        if pd.isna(iv) or float(iv) < 0.01:
            return 0.30
        return float(iv)
    except Exception:
        return 0.30


def black_scholes_greeks(stock_price, strike, days_to_expiration, iv, option_type):
    try:
        if stock_price <= 0 or strike <= 0 or days_to_expiration <= 0 or iv <= 0:
            return np.nan, np.nan, np.nan, np.nan

        t = days_to_expiration / 365
        r = 0.045

        d1 = (
            math.log(stock_price / strike)
            + (r + 0.5 * iv ** 2) * t
        ) / (iv * math.sqrt(t))

        d2 = d1 - iv * math.sqrt(t)
        pdf_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)

        if option_type == "CALL":
            delta = normal_cdf(d1)
            theta = (
                -stock_price * pdf_d1 * iv / (2 * math.sqrt(t))
                - r * strike * math.exp(-r * t) * normal_cdf(d2)
            ) / 365
        else:
            delta = normal_cdf(d1) - 1
            theta = (
                -stock_price * pdf_d1 * iv / (2 * math.sqrt(t))
                + r * strike * math.exp(-r * t) * normal_cdf(-d2)
            ) / 365

        gamma = pdf_d1 / (stock_price * iv * math.sqrt(t))
        vega = stock_price * pdf_d1 * math.sqrt(t) / 100

        return delta, gamma, theta, vega
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def estimate_probability_of_profit(stock_price, strike, premium, days_to_expiration, iv, option_type):
    try:
        if stock_price <= 0 or strike <= 0 or premium <= 0 or days_to_expiration <= 0 or iv <= 0:
            return np.nan

        t = days_to_expiration / 365

        if option_type == "CALL":
            breakeven = strike + premium
            z = math.log(breakeven / stock_price) / (iv * math.sqrt(t))
            pop = 1 - normal_cdf(z)
        else:
            breakeven = strike - premium
            if breakeven <= 0:
                return np.nan
            z = math.log(breakeven / stock_price) / (iv * math.sqrt(t))
            pop = normal_cdf(z)

        return max(0, min(1, pop))
    except Exception:
        return np.nan


def calculate_iv_rank(options_df: pd.DataFrame):
    try:
        iv_series = options_df["Clean IV"].replace(0, np.nan).dropna()
        if iv_series.empty:
            return np.nan

        iv_min = iv_series.min()
        iv_max = iv_series.max()
        current_iv = iv_series.median()

        if iv_max == iv_min:
            return 50.0

        iv_rank = ((current_iv - iv_min) / (iv_max - iv_min)) * 100
        return max(0, min(100, iv_rank))
    except Exception:
        return np.nan


def choose_best_contract(options_df: pd.DataFrame):
    try:
        if options_df.empty:
            return None

        sorted_df = options_df.sort_values(
            by=["Option Score", "volume", "openInterest"],
            ascending=[False, False, False]
        )
        return sorted_df.iloc[0]
    except Exception:
        return None


# =========================================================
# DATA LOADERS
# =========================================================
@st.cache_data(ttl=60)
def load_data(symbol, selected_period, interval):
    if interval in ["15m", "30m"] and selected_period in ["6mo", "1y", "2y"]:
        selected_period = "60d"

    if interval == "1h" and selected_period == "2y":
        selected_period = "730d"

    data = yf.download(
        symbol,
        period=selected_period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


@st.cache_data(ttl=120)
def load_info(symbol):
    try:
        return yf.Ticker(symbol).info
    except Exception:
        return {}


def load_yahoo_options(symbol):
    try:
        stock = yf.Ticker(symbol)
        expirations = list(stock.options)

        if not expirations:
            return pd.DataFrame(), []

        all_data = []

        for exp in expirations:
            try:
                chain = stock.option_chain(exp)

                calls = chain.calls.copy()
                calls["type"] = "call"
                calls["expiration"] = exp

                puts = chain.puts.copy()
                puts["type"] = "put"
                puts["expiration"] = exp

                all_data.append(calls)
                all_data.append(puts)
            except Exception:
                continue

        if not all_data:
            return pd.DataFrame(), expirations

        full_df = pd.concat(all_data, ignore_index=True)
        return full_df, expirations

    except Exception:
        return pd.DataFrame(), []


def load_polygon_options(symbol):
    try:
        api_key = st.secrets.get("POLYGON_API_KEY", "")

        if not api_key:
            return pd.DataFrame(), []

        url = f"https://api.polygon.io/v3/snapshot/options/{symbol.upper()}"
        params = {"apiKey": api_key, "limit": 250}

        response = requests.get(url, params=params, timeout=15)

        if response.status_code != 200:
            return pd.DataFrame(), []

        results = response.json().get("results", [])

        if not results:
            return pd.DataFrame(), []

        rows = []

        for opt in results:
            details = opt.get("details", {}) or {}
            greeks = opt.get("greeks", {}) or {}
            quote = opt.get("last_quote", {}) or {}
            trade = opt.get("last_trade", {}) or {}
            day = opt.get("day", {}) or {}

            rows.append({
                "contractSymbol": details.get("ticker"),
                "strike": details.get("strike_price"),
                "expiration": details.get("expiration_date"),
                "type": str(details.get("contract_type", "")).lower(),
                "bid": quote.get("bid"),
                "ask": quote.get("ask"),
                "lastPrice": trade.get("price"),
                "volume": day.get("volume"),
                "openInterest": opt.get("open_interest"),
                "impliedVolatility": opt.get("implied_volatility"),
                "Delta": greeks.get("delta"),
                "Gamma": greeks.get("gamma"),
                "Theta": greeks.get("theta"),
                "Vega": greeks.get("vega"),
                "dataSource": "Polygon"
            })

        df = pd.DataFrame(rows)

        numeric_cols = [
            "strike", "bid", "ask", "lastPrice", "volume", "openInterest",
            "impliedVolatility", "Delta", "Gamma", "Theta", "Vega"
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["volume"] = df["volume"].fillna(0)
        df["openInterest"] = df["openInterest"].fillna(0)
        df["lastPrice"] = df["lastPrice"].fillna(0)
        df["bid"] = df["bid"].fillna(0)
        df["ask"] = df["ask"].fillna(0)

        df = df.dropna(subset=["contractSymbol", "strike", "expiration", "type"])

        expirations = sorted(df["expiration"].dropna().unique().tolist())
        return df, expirations

    except Exception:
        return pd.DataFrame(), []


# =========================================================
# TECHNICAL INDICATORS
# =========================================================
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window, min_periods=5).mean()
    avg_loss = loss.rolling(window, min_periods=5).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()

    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    return macd_line, signal_line


def calculate_indicators(data):
    data = data.copy()

    data["RSI"] = calculate_rsi(data["Close"])
    data["MACD"], data["MACD_SIGNAL"] = calculate_macd(data["Close"])
    data["MA20"] = data["Close"].rolling(20, min_periods=5).mean()
    data["MA50"] = data["Close"].rolling(50, min_periods=10).mean()
    data["Volume_MA20"] = data["Volume"].rolling(20, min_periods=5).mean()

    return data


def calculate_signal(data):
    data = calculate_indicators(data)
    data = data.dropna(subset=["Close", "RSI", "MACD", "MACD_SIGNAL", "MA20", "MA50"])

    if data.empty:
        raise ValueError("Not enough price history to calculate indicators. Try a longer chart period.")

    latest = data.iloc[-1]

    price = float(latest["Close"])
    rsi = float(latest["RSI"])
    macd = float(latest["MACD"])
    macd_signal = float(latest["MACD_SIGNAL"])
    ma20 = float(latest["MA20"])
    ma50 = float(latest["MA50"])

    support = float(data["Low"].tail(30).min())
    resistance = float(data["High"].tail(30).max())

    ideal_entry_low = support * 1.005
    ideal_entry_high = support * 1.035
    bullish_exit = resistance * 0.985
    bearish_exit = support * 1.015

    score = 50

    if rsi < 30:
        score += 18
    elif 30 <= rsi <= 45:
        score += 12
    elif 45 < rsi <= 60:
        score += 5
    elif rsi > 70:
        score -= 18

    score += 18 if macd > macd_signal else -12
    score += 12 if price > ma20 else -8
    score += 10 if price > ma50 else -8

    if latest["Volume"] > latest["Volume_MA20"]:
        score += 8

    if support <= price <= resistance:
        score += 5

    score = max(0, min(100, int(score)))

    if score >= 78:
        signal = "STRONG BUY CALL"
    elif score >= 62:
        signal = "BUY CALL WATCH"
    elif score <= 22:
        signal = "STRONG BUY PUT"
    elif score <= 32:
        signal = "PUT WATCH"
    else:
        signal = "NEUTRAL"

    return data, {
        "price": price,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "ma20": ma20,
        "ma50": ma50,
        "support": support,
        "resistance": resistance,
        "ideal_entry_low": ideal_entry_low,
        "ideal_entry_high": ideal_entry_high,
        "bullish_exit": bullish_exit,
        "bearish_exit": bearish_exit,
        "score": score,
        "signal": signal
    }


# =========================================================
# CHART
# =========================================================
def make_chart(data, ticker, signal_data, selected_strike=None):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Price"
    ))

    fig.add_trace(go.Bar(
        x=data.index,
        y=data["Volume"],
        name="Volume",
        yaxis="y2",
        opacity=0.25
    ))

    fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="20-Day MA", mode="lines"))
    fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="50-Day MA", mode="lines"))

    fig.add_hline(y=signal_data["support"], line_dash="dash", annotation_text="Support")
    fig.add_hline(y=signal_data["resistance"], line_dash="dash", annotation_text="Resistance")

    fig.add_hrect(
        y0=signal_data["ideal_entry_low"],
        y1=signal_data["ideal_entry_high"],
        opacity=0.18,
        annotation_text="Ideal Entry Zone",
        annotation_position="top left"
    )

    fig.add_hline(y=signal_data["bullish_exit"], line_dash="dot", annotation_text="Bullish Exit Target")
    fig.add_hline(y=signal_data["bearish_exit"], line_dash="dot", annotation_text="Bearish Stop / Exit")

    if selected_strike:
        fig.add_hline(
            y=selected_strike,
            line_dash="solid",
            annotation_text=f"Selected Strike: {selected_strike}"
        )

    fig.update_layout(
        title=f"{ticker} Trading Chart",
        height=620,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#111827")
    )

    return fig


# =========================================================
# OPTIONS FUNCTIONS
# =========================================================
def ensure_option_columns(options_df):
    required_cols = [
        "contractSymbol", "strike", "lastPrice", "bid", "ask", "volume",
        "openInterest", "impliedVolatility", "type", "expiration"
    ]

    for col in required_cols:
        if col not in options_df.columns:
            options_df[col] = "N/A" if col in ["contractSymbol", "type", "expiration"] else np.nan

    numeric_cols = [
        "strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"
    ]

    for col in numeric_cols:
        options_df[col] = pd.to_numeric(options_df[col], errors="coerce")

    options_df["volume"] = options_df["volume"].fillna(0)
    options_df["openInterest"] = options_df["openInterest"].fillna(0)
    options_df["lastPrice"] = options_df["lastPrice"].fillna(0)
    options_df["bid"] = options_df["bid"].fillna(0)
    options_df["ask"] = options_df["ask"].fillna(0)
    options_df["impliedVolatility"] = options_df["impliedVolatility"].fillna(0)

    return options_df


def prepare_options(options_df, selected_expiration, option_type, stock_price, source):
    if options_df.empty:
        return pd.DataFrame()

    out = ensure_option_columns(options_df.copy())

    wanted_type = "call" if option_type == "CALL" else "put"

    out["type"] = out["type"].astype(str).str.lower()
    out = out[out["expiration"].astype(str) == str(selected_expiration)]
    out = out[out["type"] == wanted_type]

    if out.empty:
        return pd.DataFrame()

    days_to_expiration = max(
        (datetime.strptime(selected_expiration, "%Y-%m-%d").date() - date.today()).days,
        1
    )

    out["Clean IV"] = out["impliedVolatility"].apply(clean_iv_value)

    out["Spread"] = np.where(
        (out["ask"] > 0) & (out["bid"] > 0),
        out["ask"] - out["bid"],
        np.nan
    )

    for greek_col in ["Delta", "Gamma", "Theta", "Vega"]:
        if greek_col not in out.columns:
            out[greek_col] = np.nan

    missing_greeks = (
        out["Delta"].isna().all()
        or out["Gamma"].isna().all()
        or out["Theta"].isna().all()
        or out["Vega"].isna().all()
    )

    if missing_greeks:
        greeks = out.apply(
            lambda row: black_scholes_greeks(
                stock_price,
                row["strike"],
                days_to_expiration,
                row["Clean IV"],
                option_type
            ),
            axis=1
        )

        out["Delta"] = greeks.apply(lambda x: x[0])
        out["Gamma"] = greeks.apply(lambda x: x[1])
        out["Theta"] = greeks.apply(lambda x: x[2])
        out["Vega"] = greeks.apply(lambda x: x[3])

    out["Probability of Profit"] = out.apply(
        lambda row: estimate_probability_of_profit(
            stock_price,
            row["strike"],
            row["lastPrice"],
            days_to_expiration,
            row["Clean IV"],
            option_type
        ),
        axis=1
    )

    out["Total Contract Cost"] = out["lastPrice"] * 100
    out["IV Rank"] = calculate_iv_rank(out)
    out["dataSource"] = source

    return out


def score_option(row, stock_price, delta_min, delta_max, pop_min):
    score = 0

    volume = row.get("volume", 0) or 0
    oi = row.get("openInterest", 0) or 0
    last = row.get("lastPrice", 0) or 0
    strike = row.get("strike", 0) or 0
    delta = abs(row.get("Delta", np.nan))
    pop = row.get("Probability of Profit", np.nan)
    spread = row.get("Spread", np.nan)

    if volume >= 500:
        score += 18
    elif volume >= 100:
        score += 12
    elif volume >= 25:
        score += 6

    if oi >= 1000:
        score += 18
    elif oi >= 500:
        score += 12
    elif oi >= 100:
        score += 6

    if not pd.isna(spread):
        if spread <= 0.10:
            score += 18
        elif spread <= 0.30:
            score += 12
        elif spread <= 0.75:
            score += 6

    if stock_price > 0 and strike > 0:
        distance = abs(strike - stock_price) / stock_price

        if distance <= 0.03:
            score += 20
        elif distance <= 0.07:
            score += 12
        elif distance <= 0.12:
            score += 6

    if not pd.isna(delta) and delta_min <= delta <= delta_max:
        score += 20

    if not pd.isna(pop) and pop >= pop_min:
        score += 15

    if last > 0:
        score += 5

    return max(0, min(100, int(score)))


# =========================================================
# ALERTS / SCANNER
# =========================================================
def send_discord_alert(webhook_url, message):
    try:
        if webhook_url:
            requests.post(webhook_url, json={"content": message}, timeout=5)
    except Exception:
        pass


def send_telegram_alert(bot_token, chat_id, message):
    try:
        if bot_token and chat_id:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=5)
    except Exception:
        pass


def scan_market(tickers):
    rows = []

    for symbol in tickers:
        try:
            d = load_data(symbol, "3mo", "1d")

            if d.empty:
                continue

            _, sig = calculate_signal(d)

            rows.append({
                "Ticker": symbol,
                "Price": round(sig["price"], 2),
                "RSI": round(sig["rsi"], 1),
                "Support": round(sig["support"], 2),
                "Resistance": round(sig["resistance"], 2),
                "Entry Low": round(sig["ideal_entry_low"], 2),
                "Entry High": round(sig["ideal_entry_high"], 2),
                "Exit Target": round(sig["bullish_exit"], 2),
                "Stock Trade Score": sig["score"],
                "Signal": sig["signal"],
            })

        except Exception:
            pass

    return pd.DataFrame(rows)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings")

    auto_refresh = st.toggle("Auto-refresh", value=False)
    refresh_seconds = st.selectbox("Refresh every", [30, 60, 120, 300], index=1)

    st.divider()

    st.subheader("🔔 Alerts")

    alerts_enabled = st.toggle("Enable in-app alerts", value=True)
    discord_webhook = st.text_input("Discord webhook URL", type="password", placeholder="Optional")
    telegram_token = st.text_input("Telegram bot token", type="password", placeholder="Optional")
    telegram_chat_id = st.text_input("Telegram chat ID", placeholder="Optional")

    st.divider()

    st.subheader("📊 Option Filters")

    delta_min = st.slider("Minimum Delta", 0.00, 1.00, 0.25, 0.05)
    delta_max = st.slider("Maximum Delta", 0.00, 1.00, 0.85, 0.05)
    pop_min = st.slider("Minimum Probability of Profit", 0.00, 1.00, 0.20, 0.05)
    iv_min = st.slider("Minimum IV", 0.00, 2.00, 0.00, 0.05)
    iv_max = st.slider("Maximum IV", 0.00, 2.00, 1.50, 0.05)

if auto_refresh:
    st.markdown(f'<meta http-equiv="refresh" content="{refresh_seconds}">', unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="sticky-header">', unsafe_allow_html=True)

st.title("📈 TradeEdge Pro")

h1, h2, h3, h4 = st.columns([2, 2, 1, 1])

with h1:
    st.selectbox(
        "Autocomplete stock search",
        list(POPULAR_TICKERS.keys()),
        key="selected_company"
    )

with h2:
    st.text_input(
        "Manual ticker entry",
        placeholder="AAPL, TSLA, NVDA...",
        key="manual_ticker"
    )

with h3:
    period = st.selectbox("Chart period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

with h4:
    interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m"], index=0)

if st.session_state.manual_ticker.strip():
    ticker = st.session_state.manual_ticker.upper().strip()
else:
    ticker = POPULAR_TICKERS[st.session_state.selected_company]

st.session_state.ticker = ticker

st.markdown(f"### Active ticker: `{ticker}`")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="floating-reset">', unsafe_allow_html=True)
st.button("🔄 Reset", key="floating_reset", on_click=reset_app)
st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# MAIN APP
# =========================================================
try:
    raw_data = load_data(ticker, period, interval)

    if raw_data.empty:
        st.error("No data found. Try another ticker.")
        st.stop()

    data, signal_data = calculate_signal(raw_data)

    price = signal_data["price"]
    signal = signal_data["signal"]
    trade_score = signal_data["score"]

    if alerts_enabled:
        alert_message = f"{ticker}: {signal} | Stock Score {trade_score}/100 | Price ${price:,.2f}"

        if signal != st.session_state.last_alert:
            if "STRONG" in signal or "WATCH" in signal:
                st.toast(f"🚨 {alert_message}", icon="🚀")
                send_discord_alert(discord_webhook, alert_message)
                send_telegram_alert(telegram_token, telegram_chat_id, alert_message)

            st.session_state.last_alert = signal

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    m1.metric("Price", f"${price:,.2f}")
    m2.metric("RSI", f"{signal_data['rsi']:.1f}")
    m3.metric("Stock Score", f"{trade_score}/100")
    m4.metric("Support", f"${signal_data['support']:,.2f}")
    m5.metric("Resistance", f"${signal_data['resistance']:,.2f}")
    m6.metric("Entry Zone", f"${signal_data['ideal_entry_low']:,.2f} - ${signal_data['ideal_entry_high']:,.2f}")

    if signal == "STRONG BUY CALL":
        signal_color = "linear-gradient(135deg, #16a34a, #22c55e)"
    elif signal == "BUY CALL WATCH":
        signal_color = "linear-gradient(135deg, #2563eb, #38bdf8)"
    elif signal == "STRONG BUY PUT":
        signal_color = "linear-gradient(135deg, #dc2626, #ef4444)"
    elif signal == "PUT WATCH":
        signal_color = "linear-gradient(135deg, #ea580c, #f97316)"
    else:
        signal_color = "linear-gradient(135deg, #6b7280, #9ca3af)"

    st.markdown(
        f"""
        <div class="signal-box" style="background: {signal_color}; color: white;">
            <div class="signal-title">{signal}</div>
            <div class="signal-subtitle">Stock Trade Score: {trade_score}/100</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    tabs = st.tabs([
        "📊 Chart",
        "🧾 Options Chain",
        "📓 Trading Journal",
        "📉 Technical Breakdown",
        "🔍 Scanner",
        "📰 News",
        "🏢 Company Info",
        "❓ How To Use"
    ])

    # =====================================================
    # CHART TAB
    # =====================================================
    with tabs[0]:
        st.subheader("Chart with Volume, Support, Resistance, Entry Zone, Exit Lines")

        selected_strike = st.number_input(
            "Highlight chosen strike on chart",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

        chart = make_chart(
            data,
            ticker,
            signal_data,
            selected_strike if selected_strike > 0 else None
        )

        st.plotly_chart(chart, use_container_width=True)

    # =====================================================
    # OPTIONS TAB
    # =====================================================
    with tabs[1]:
        st.subheader("Options Chain with Polygon + Yahoo Fallback")

        polygon_df, polygon_expirations = load_polygon_options(ticker)

        if not polygon_df.empty and polygon_expirations:
            raw_options_df = polygon_df
            expirations = polygon_expirations
            data_source = "Polygon"
            st.success("Using Polygon options snapshot data.")
        else:
            yahoo_df, yahoo_expirations = load_yahoo_options(ticker)
            raw_options_df = yahoo_df
            expirations = yahoo_expirations
            data_source = "Yahoo fallback"
            st.warning("Polygon options unavailable or limited. Using Yahoo fallback.")

        if raw_options_df.empty or not expirations:
            st.error("No options data available for this ticker. Try refreshing, changing ticker, or running locally.")
        else:
            e1, e2 = st.columns([2, 1])

            with e1:
                selected_expiration = st.selectbox("Expiration", expirations)

            with e2:
                option_type = st.radio("Type", ["CALL", "PUT"], horizontal=True)

            options_df = prepare_options(
                raw_options_df,
                selected_expiration,
                option_type,
                price,
                data_source
            )

            if options_df.empty:
                st.warning("No contracts found for this expiration and option type.")
            else:
                days_to_expiration = max(
                    (datetime.strptime(selected_expiration, "%Y-%m-%d").date() - date.today()).days,
                    1
                )

                options_df["Option Score"] = options_df.apply(
                    lambda row: score_option(row, price, delta_min, delta_max, pop_min),
                    axis=1
                )

                filtered = options_df.copy()

                filtered = filtered[
                    filtered["Delta"].abs().between(delta_min, delta_max, inclusive="both")
                    | filtered["Delta"].isna()
                ]

                filtered = filtered[
                    (filtered["Probability of Profit"] >= pop_min)
                    | filtered["Probability of Profit"].isna()
                ]

                filtered = filtered[
                    filtered["Clean IV"].between(iv_min, iv_max, inclusive="both")
                ]

                filtered = filtered.sort_values("Option Score", ascending=False)

                if filtered.empty:
                    st.warning("No contracts matched your filters. Showing all contracts instead.")
                    filtered = options_df.copy().sort_values("Option Score", ascending=False)

                best_contract = choose_best_contract(filtered)

                g1, g2, g3, g4 = st.columns(4)

                g1.metric("IV Rank", safe_format(calculate_iv_rank(filtered), "{:.1f}%"))
                g2.metric("Days to Expiration", days_to_expiration)
                g3.metric("Contracts Shown", len(filtered))
                g4.metric("Data Source", data_source)

                if best_contract is not None:
                    st.success(
                        f"Best {option_type}: Strike ${safe_format(best_contract['strike'])} | "
                        f"Last ${safe_format(best_contract['lastPrice'])} | "
                        f"Cost ${safe_format(best_contract['Total Contract Cost'])} | "
                        f"Delta {safe_format(best_contract['Delta'])} | "
                        f"Gamma {safe_format(best_contract['Gamma'], '{:.4f}')} | "
                        f"Theta {safe_format(best_contract['Theta'], '{:.4f}')} | "
                        f"Vega {safe_format(best_contract['Vega'], '{:.4f}')} | "
                        f"POP {safe_format(best_contract['Probability of Profit'], '{:.1%}')} | "
                        f"Option Score {safe_format(best_contract['Option Score'], '{:.0f}')}/100"
                    )

                display_cols = [
                    "contractSymbol",
                    "strike",
                    "lastPrice",
                    "bid",
                    "ask",
                    "Spread",
                    "volume",
                    "openInterest",
                    "impliedVolatility",
                    "Clean IV",
                    "IV Rank",
                    "Delta",
                    "Gamma",
                    "Theta",
                    "Vega",
                    "Probability of Profit",
                    "Total Contract Cost",
                    "Option Score"
                ]

                for col in display_cols:
                    if col not in filtered.columns:
                        filtered[col] = np.nan

                st.dataframe(filtered[display_cols], use_container_width=True, height=560)

    # =====================================================
    # TRADING JOURNAL TAB
    # =====================================================
    with tabs[2]:
        st.subheader("📓 Trading Journal")

        journal_df = load_journal()

        with st.form("journal_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                journal_ticker = st.text_input("Ticker", value=ticker)
                trade_type = st.selectbox("Trade Type", ["CALL", "PUT", "SPREAD"])
                strike = st.number_input("Strike", min_value=0.0, step=1.0)

            with c2:
                expiration = st.date_input("Expiration")
                entry_price = st.number_input("Entry Price", min_value=0.0, step=0.01)
                exit_price = st.number_input("Exit Price", min_value=0.0, step=0.01)

            with c3:
                contracts = st.number_input("Contracts", min_value=1, value=1, step=1)
                submitted = st.form_submit_button("Add Trade")

            if submitted:
                profit_loss = calculate_pl(entry_price, exit_price, contracts)
                win_loss = "WIN" if profit_loss > 0 else "LOSS" if profit_loss < 0 else "BREAKEVEN"

                new_trade = pd.DataFrame([{
                    "Date/Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Ticker": journal_ticker.upper().strip(),
                    "Trade Type": trade_type,
                    "Strike": strike,
                    "Expiration": expiration.strftime("%Y-%m-%d"),
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "Contracts": contracts,
                    "Profit/Loss": profit_loss,
                    "Win/Loss": win_loss
                }])

                journal_df = pd.concat([journal_df, new_trade], ignore_index=True)
                save_journal(journal_df)
                st.success("Trade added to journal.")

        st.divider()

        if journal_df.empty:
            st.info("No journal entries yet.")
        else:
            journal_df["Profit/Loss"] = pd.to_numeric(journal_df["Profit/Loss"], errors="coerce").fillna(0)

            total_trades = len(journal_df)
            wins = len(journal_df[journal_df["Win/Loss"] == "WIN"])
            losses = len(journal_df[journal_df["Win/Loss"] == "LOSS"])
            breakevens = len(journal_df[journal_df["Win/Loss"] == "BREAKEVEN"])
            total_pl = journal_df["Profit/Loss"].sum()
            win_rate = (wins / total_trades) * 100 if total_trades else 0

            j1, j2, j3, j4, j5 = st.columns(5)

            j1.metric("Total Trades", total_trades)
            j2.metric("Wins", wins)
            j3.metric("Losses", losses)
            j4.metric("Breakevens", breakevens)
            j5.metric("Win Rate", f"{win_rate:.1f}%")

            st.metric("Total Profit / Loss", f"${total_pl:,.2f}")

            st.dataframe(journal_df, use_container_width=True, height=420)

            csv = journal_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Trading Journal CSV",
                data=csv,
                file_name="trade_journal.csv",
                mime="text/csv"
            )

            if st.button("Clear Journal"):
                empty_df = pd.DataFrame(columns=journal_df.columns)
                save_journal(empty_df)
                st.warning("Journal cleared. Refresh the app.")

    # =====================================================
    # TECHNICAL TAB
    # =====================================================
    with tabs[3]:
        st.subheader("Technical Breakdown")

        breakdown = pd.DataFrame({
            "Metric": [
                "Current Price",
                "RSI",
                "MACD",
                "MACD Signal",
                "20-Day Moving Average",
                "50-Day Moving Average",
                "Support",
                "Resistance",
                "Ideal Entry Low",
                "Ideal Entry High",
                "Bullish Exit Target",
                "Bearish Stop / Exit",
                "Stock Trade Score",
                "Signal"
            ],
            "Value": [
                f"${signal_data['price']:,.2f}",
                f"{signal_data['rsi']:.1f}",
                f"{signal_data['macd']:.4f}",
                f"{signal_data['macd_signal']:.4f}",
                f"${signal_data['ma20']:,.2f}",
                f"${signal_data['ma50']:,.2f}",
                f"${signal_data['support']:,.2f}",
                f"${signal_data['resistance']:,.2f}",
                f"${signal_data['ideal_entry_low']:,.2f}",
                f"${signal_data['ideal_entry_high']:,.2f}",
                f"${signal_data['bullish_exit']:,.2f}",
                f"${signal_data['bearish_exit']:,.2f}",
                f"{signal_data['score']}/100",
                signal_data["signal"]
            ]
        })

        st.dataframe(breakdown, use_container_width=True)

    # =====================================================
    # SCANNER TAB
    # =====================================================
    with tabs[4]:
        st.subheader("Multi-Stock Scanner")

        scan_input = st.text_area("Tickers to scan", value=", ".join(DEFAULT_SCAN_TICKERS), height=90)

        tickers_to_scan = [
            x.strip().upper()
            for x in scan_input.replace("\n", ",").split(",")
            if x.strip()
        ]

        if st.button("Run Scanner"):
            scanner_df = scan_market(tickers_to_scan)

            if scanner_df.empty:
                st.warning("Scanner did not return results.")
            else:
                scanner_df = scanner_df.sort_values("Stock Trade Score", ascending=False)
                st.dataframe(scanner_df, use_container_width=True, height=500)

    # =====================================================
    # NEWS TAB
    # =====================================================
    with tabs[5]:
        st.subheader("Clickable News Links")

        try:
            news = yf.Ticker(ticker).news or []
        except Exception:
            news = []

        if not news:
            st.warning("No news found.")
        else:
            for item in news[:12]:
                title = item.get("title") or item.get("content", {}).get("title", "No title")
                link = item.get("link") or item.get("content", {}).get("canonicalUrl", {}).get("url", "")
                publisher = item.get("publisher") or item.get("content", {}).get("provider", {}).get("displayName", "Unknown")
                publish_time = item.get("providerPublishTime") or item.get("content", {}).get("pubDate", None)

                if isinstance(publish_time, int):
                    news_date = datetime.fromtimestamp(publish_time).strftime("%Y-%m-%d")
                elif isinstance(publish_time, str):
                    news_date = publish_time[:10]
                else:
                    news_date = "N/A"

                if link:
                    st.markdown(
                        f"""
                        <div class="news-card">
                        <b><a href="{link}" target="_blank">{title}</a></b><br>
                        {publisher} | {news_date}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="news-card">
                        <b>{title}</b><br>
                        {publisher} | {news_date}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # =====================================================
    # COMPANY INFO TAB
    # =====================================================
    with tabs[6]:
        st.subheader("Company Info")

        info = load_info(ticker)

        name = info.get("longName", ticker)
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        market_cap = info.get("marketCap", None)
        summary = info.get("longBusinessSummary", "No company summary available.")

        st.write(f"**Company:** {name}")
        st.write(f"**Sector:** {sector}")
        st.write(f"**Industry:** {industry}")

        if market_cap:
            st.write(f"**Market Cap:** ${market_cap:,.0f}")

        st.write(summary)

    # =====================================================
    # HOW TO USE TAB
    # =====================================================
    with tabs[7]:
        st.subheader("How To Use TradeEdge Pro")

        st.markdown("""
        1. Choose a ticker or type one manually.
        2. Check the main stock signal and stock trade score.
        3. Review the chart, support, resistance, ideal entry zone, and exit target.
        4. Go to the options chain.
        5. Filter by delta, probability of profit, and IV.
        6. Track trades in the Trading Journal.
        7. Download your journal as a CSV.

        This is a decision-support tool, not financial advice.
        """)

    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

except Exception as e:
    st.error("Something went wrong.")
    st.write(e)
