# Pro Options Trade App - Corrected Full Version With Options Spreads

import os
import math
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange


st.set_page_config(
    page_title="Pro Options Trade App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Pro Options Trade Decision App")

st.warning(
    "Educational decision-support only. This app does not guarantee profit. "
    "Use stops, position sizing, and discipline."
)


# ============================================================
# MATH
# ============================================================

def bs_delta(S, K, T, r, sigma, option_type):
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return np.nan

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

        if option_type == "call":
            return norm.cdf(d1)

        return norm.cdf(d1) - 1
    except Exception:
        return np.nan


def prob_itm(S, K, T, r, sigma, option_type):
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return np.nan

        d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

        if option_type == "call":
            return norm.cdf(d2)

        return norm.cdf(-d2)
    except Exception:
        return np.nan


def prob_above_price(S, target_price, T, r, sigma):
    return prob_itm(S, target_price, T, r, sigma, "call")


def prob_below_price(S, target_price, T, r, sigma):
    return prob_itm(S, target_price, T, r, sigma, "put")


def dte(exp):
    try:
        return max((datetime.strptime(exp, "%Y-%m-%d").date() - date.today()).days, 0)
    except Exception:
        return 0


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=60)
def get_price_data(ticker, period, interval):
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df.replace([np.inf, -np.inf], np.nan)
        return df.dropna()

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_expirations(ticker):
    try:
        return list(yf.Ticker(ticker).options)
    except Exception:
        return []


@st.cache_data(ttl=60)
def get_chain(ticker, exp):
    try:
        chain = yf.Ticker(ticker).option_chain(exp)
        return chain.calls.copy(), chain.puts.copy()
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=300)
def get_news(ticker):
    try:
        tk = yf.Ticker(ticker)
        news = tk.news or []
        rows = []

        for item in news[:10]:
            title = item.get("title", "")
            publisher = item.get("publisher", "")
            link = item.get("link", "")
            published = item.get("providerPublishTime", None)
            published_text = ""

            if not title and isinstance(item.get("content"), dict):
                content = item.get("content", {})
                title = content.get("title", "")

                provider = content.get("provider", {})
                if isinstance(provider, dict):
                    publisher = provider.get("displayName", "")

                canonical = content.get("canonicalUrl", {})
                if isinstance(canonical, dict):
                    link = canonical.get("url", "")

                pub_date = content.get("pubDate", "")
                if pub_date:
                    try:
                        published_text = pd.to_datetime(pub_date).strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        published_text = ""

            if published:
                try:
                    published_text = datetime.fromtimestamp(published).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    published_text = ""

            if title:
                rows.append({
                    "Title": title,
                    "Publisher": publisher,
                    "Published": published_text,
                    "Link": link
                })

        return pd.DataFrame(rows, columns=["Title", "Publisher", "Published", "Link"])

    except Exception:
        return pd.DataFrame(columns=["Title", "Publisher", "Published", "Link"])


@st.cache_data(ttl=300)
def get_earnings_warning(ticker):
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        earnings_date = None

        if cal is None:
            return "Unknown", None, 0

        if isinstance(cal, dict):
            raw = cal.get("Earnings Date") or cal.get("EarningsDate")
            if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) > 0:
                raw = raw[0]
            if raw is not None:
                earnings_date = pd.to_datetime(raw).date()

        elif isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.index:
                raw = cal.loc["Earnings Date"].values[0]
                if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) > 0:
                    raw = raw[0]
                earnings_date = pd.to_datetime(raw).date()
            elif "Earnings Date" in cal.columns and not cal.empty:
                raw = cal["Earnings Date"].iloc[0]
                if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) > 0:
                    raw = raw[0]
                earnings_date = pd.to_datetime(raw).date()

        if earnings_date is None:
            return "Unknown", None, 0

        days = (earnings_date - date.today()).days

        if 0 <= days <= 3:
            return "High", earnings_date, days
        if 4 <= days <= 10:
            return "Medium", earnings_date, days
        if days > 10:
            return "Low", earnings_date, days

        return "Past/Unknown", earnings_date, days

    except Exception:
        return "Unknown", None, 0


def add_indicators(df):
    try:
        out = df.copy()

        if out.empty or len(out) < 30:
            return pd.DataFrame()

        out["RSI"] = RSIIndicator(out["Close"], window=14).rsi()

        macd = MACD(out["Close"])
        out["MACD"] = macd.macd()
        out["MACD_SIGNAL"] = macd.macd_signal()
        out["MACD_HIST"] = macd.macd_diff()

        out["EMA9"] = out["Close"].ewm(span=9, adjust=False).mean()
        out["EMA21"] = out["Close"].ewm(span=21, adjust=False).mean()
        out["EMA50"] = out["Close"].ewm(span=50, adjust=False).mean()
        out["EMA200"] = out["Close"].ewm(span=200, adjust=False).mean()

        atr = AverageTrueRange(out["High"], out["Low"], out["Close"], window=14)
        out["ATR"] = atr.average_true_range()

        out["VOL_AVG"] = out["Volume"].rolling(20).mean()
        out["VOL_RATIO"] = out["Volume"] / out["VOL_AVG"]

        out["RET"] = out["Close"].pct_change()
        out["REALIZED_VOL"] = out["RET"].rolling(20).std() * np.sqrt(252)

        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna()

        return out

    except Exception:
        return pd.DataFrame()


# ============================================================
# NEWS SENTIMENT
# ============================================================

POSITIVE_WORDS = [
    "beat", "beats", "upgrade", "upgraded", "raises", "raised", "strong",
    "growth", "surge", "record", "profit", "profits", "bullish", "outperform",
    "buy", "partnership", "contract", "approval", "launch", "expands",
    "higher", "positive", "momentum", "guidance raised"
]

NEGATIVE_WORDS = [
    "miss", "misses", "downgrade", "downgraded", "cuts", "cut", "weak",
    "decline", "falls", "fall", "drops", "drop", "lawsuit", "probe",
    "investigation", "sec", "bearish", "underperform", "sell", "layoffs",
    "recession", "warning", "guidance cut", "loss", "losses", "slumps",
    "recall", "delay", "delayed"
]


def score_news_sentiment(news_df):
    if news_df is None or news_df.empty:
        return 0, "No recent news"

    score = 0

    for title in news_df["Title"].fillna("").head(10):
        text = str(title).lower()
        pos_hits = [w for w in POSITIVE_WORDS if w in text]
        neg_hits = [w for w in NEGATIVE_WORDS if w in text]
        score += 10 * len(pos_hits)
        score -= 10 * len(neg_hits)

    score = int(max(-100, min(100, score)))

    if score >= 30:
        label = "Positive"
    elif score <= -30:
        label = "Negative"
    else:
        label = "Neutral"

    return score, label


# ============================================================
# MARKET FILTER
# ============================================================

def market_filter():
    try:
        spy = add_indicators(get_price_data("SPY", "6mo", "1d"))
        qqq = add_indicators(get_price_data("QQQ", "6mo", "1d"))
        vix = get_price_data("^VIX", "3mo", "1d")

        if spy.empty or qqq.empty:
            return {
                "bias": "Unknown",
                "call_ok": True,
                "put_ok": True,
                "vix": np.nan,
                "details": "Market data unavailable"
            }

        spy_last = spy.iloc[-1]
        qqq_last = qqq.iloc[-1]

        spy_bull = spy_last["Close"] > spy_last["EMA21"] > spy_last["EMA50"]
        qqq_bull = qqq_last["Close"] > qqq_last["EMA21"] > qqq_last["EMA50"]

        spy_bear = spy_last["Close"] < spy_last["EMA21"] < spy_last["EMA50"]
        qqq_bear = qqq_last["Close"] < qqq_last["EMA21"] < qqq_last["EMA50"]

        vix_value = np.nan
        if not vix.empty:
            vix_value = float(vix["Close"].iloc[-1])

        if spy_bull and qqq_bull:
            bias = "Bullish"
            call_ok = True
            put_ok = False
        elif spy_bear and qqq_bear:
            bias = "Bearish"
            call_ok = False
            put_ok = True
        else:
            bias = "Mixed"
            call_ok = True
            put_ok = True

        if not np.isnan(vix_value) and vix_value >= 25:
            bias = f"{bias} / High Volatility"

        return {
            "bias": bias,
            "call_ok": call_ok,
            "put_ok": put_ok,
            "vix": vix_value,
            "details": f"SPY bullish: {spy_bull}, QQQ bullish: {qqq_bull}"
        }

    except Exception:
        return {
            "bias": "Unknown",
            "call_ok": True,
            "put_ok": True,
            "vix": np.nan,
            "details": "Market filter failed"
        }


# ============================================================
# SETUP SCORING
# ============================================================

def support_resistance(df, lookback=60):
    recent = df.tail(min(lookback, len(df)))
    return float(recent["Low"].min()), float(recent["High"].max())


def score_setup(df):
    try:
        if df is None or df.empty or len(df) < 2:
            return None

        row = df.iloc[-1]
        prev = df.iloc[-2]

        price = float(row["Close"])
        support, resistance = support_resistance(df)
        atr = float(row["ATR"])
        rsi = float(row["RSI"])
        vol_ratio = float(row["VOL_RATIO"])

        call_score = 0
        put_score = 0
        call_reasons = []
        put_reasons = []

        if price > row["EMA21"] > row["EMA50"]:
            call_score += 25
            call_reasons.append("Bullish trend: price above EMA21 and EMA50")

        if price > row["EMA200"]:
            call_score += 10
            call_reasons.append("Price above EMA200")

        if price < row["EMA21"] < row["EMA50"]:
            put_score += 25
            put_reasons.append("Bearish trend: price below EMA21 and EMA50")

        if price < row["EMA200"]:
            put_score += 10
            put_reasons.append("Price below EMA200")

        if 30 <= rsi <= 45:
            call_score += 15
            call_reasons.append("RSI pullback zone")

        if rsi < 30:
            call_score += 10
            call_reasons.append("RSI oversold")

        if 55 <= rsi <= 70:
            put_score += 10
            put_reasons.append("RSI elevated")

        if rsi > 70:
            put_score += 15
            put_reasons.append("RSI overbought")

        bullish_cross = prev["MACD"] <= prev["MACD_SIGNAL"] and row["MACD"] > row["MACD_SIGNAL"]
        bearish_cross = prev["MACD"] >= prev["MACD_SIGNAL"] and row["MACD"] < row["MACD_SIGNAL"]

        if bullish_cross:
            call_score += 20
            call_reasons.append("Bullish MACD crossover")
        elif row["MACD"] > row["MACD_SIGNAL"]:
            call_score += 10
            call_reasons.append("MACD above signal")

        if bearish_cross:
            put_score += 20
            put_reasons.append("Bearish MACD crossover")
        elif row["MACD"] < row["MACD_SIGNAL"]:
            put_score += 10
            put_reasons.append("MACD below signal")

        if vol_ratio >= 1.5:
            call_score += 10
            put_score += 10
            call_reasons.append("Volume expansion")
            put_reasons.append("Volume expansion")

        if atr > 0:
            if abs(price - support) / atr <= 1.5:
                call_score += 15
                call_reasons.append("Near support")

            if abs(resistance - price) / atr <= 1.5:
                put_score += 15
                put_reasons.append("Near resistance")

        if price > 0 and abs(row["EMA21"] - row["EMA50"]) / price < 0.005:
            call_score -= 8
            put_score -= 8

        call_score = int(max(0, min(100, call_score)))
        put_score = int(max(0, min(100, put_score)))

        if call_score >= 75 and call_score > put_score:
            signal = "STRONG BUY CALL"
        elif put_score >= 75 and put_score > call_score:
            signal = "STRONG BUY PUT"
        elif call_score >= 60 and call_score > put_score:
            signal = "CALL WATCHLIST"
        elif put_score >= 60 and put_score > call_score:
            signal = "PUT WATCHLIST"
        else:
            signal = "NO TRADE / WAIT"

        if "CALL" in signal and atr > 0:
            entry_low = price - 0.75 * atr
            entry_high = price + 0.25 * atr
            stop = price - 1.25 * atr
            target1 = price + 1.5 * atr
            target2 = price + 2.5 * atr
        elif "PUT" in signal and atr > 0:
            entry_low = price - 0.25 * atr
            entry_high = price + 0.75 * atr
            stop = price + 1.25 * atr
            target1 = price - 1.5 * atr
            target2 = price - 2.5 * atr
        else:
            entry_low = np.nan
            entry_high = np.nan
            stop = np.nan
            target1 = np.nan
            target2 = np.nan

        return {
            "price": price,
            "support": support,
            "resistance": resistance,
            "atr": atr,
            "rsi": rsi,
            "vol_ratio": vol_ratio,
            "call_score": call_score,
            "put_score": put_score,
            "signal": signal,
            "entry_low": entry_low,
            "entry_high": entry_high,
            "stop": stop,
            "target1": target1,
            "target2": target2,
            "call_reasons": call_reasons,
            "put_reasons": put_reasons,
        }

    except Exception:
        return None


# ============================================================
# OPTIONS HELPERS
# ============================================================

def prepare_chain(chain):
    if chain is None or chain.empty:
        return pd.DataFrame()

    df = chain.copy()

    for col in ["bid", "ask", "lastPrice", "strike", "volume", "openInterest", "impliedVolatility", "contractSymbol"]:
        if col not in df.columns:
            df[col] = np.nan

    df["volume"] = df["volume"].fillna(0)
    df["openInterest"] = df["openInterest"].fillna(0)

    df = df[
        (df["bid"].fillna(0) > 0) &
        (df["ask"].fillna(0) > 0) &
        (df["ask"] >= df["bid"]) &
        (df["impliedVolatility"].fillna(0) > 0)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]
    df["spread_pct"] = np.where(df["mid"] > 0, df["spread"] / df["mid"] * 100, np.nan)

    df["cost_mid"] = df["mid"] * 100
    df["cost_ask"] = df["ask"] * 100

    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mid", "spread_pct", "cost_ask"])


def build_options_table(
    chain,
    option_type,
    stock_price,
    exp,
    setup_score,
    min_delta,
    max_delta,
    min_volume,
    min_oi,
    max_spread_pct,
    max_cost
):
    try:
        df = prepare_chain(chain)

        if df.empty:
            return pd.DataFrame()

        T = max(dte(exp) / 365, 1 / 365)
        r = 0.045

        df["delta"] = df.apply(
            lambda x: bs_delta(stock_price, x["strike"], T, r, x["impliedVolatility"], option_type),
            axis=1
        )

        df["abs_delta"] = df["delta"].abs()

        df["prob_itm"] = df.apply(
            lambda x: prob_itm(stock_price, x["strike"], T, r, x["impliedVolatility"], option_type),
            axis=1
        )

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["delta", "prob_itm"])

        if df.empty:
            return pd.DataFrame()

        df["liquidity_score"] = 0
        df.loc[df["volume"] >= min_volume, "liquidity_score"] += 20
        df.loc[df["openInterest"] >= min_oi, "liquidity_score"] += 20
        df.loc[df["spread_pct"].fillna(999) <= max_spread_pct, "liquidity_score"] += 20
        df.loc[df["abs_delta"].between(min_delta, max_delta), "liquidity_score"] += 20
        df.loc[df["cost_ask"].fillna(999999) <= max_cost, "liquidity_score"] += 20

        df["trade_score"] = (0.55 * setup_score + 0.45 * df["liquidity_score"]).round(0).astype(int)

        filtered = df[
            (df["abs_delta"].between(min_delta, max_delta)) &
            (df["volume"] >= min_volume) &
            (df["openInterest"] >= min_oi) &
            (df["spread_pct"].fillna(999) <= max_spread_pct) &
            (df["cost_ask"].fillna(999999) <= max_cost)
        ].copy()

        if filtered.empty:
            filtered = df.copy()

        cols = [
            "contractSymbol", "strike", "lastPrice", "bid", "ask", "mid",
            "cost_mid", "cost_ask", "spread_pct", "volume", "openInterest",
            "impliedVolatility", "delta", "prob_itm", "trade_score"
        ]

        return filtered[cols].sort_values("trade_score", ascending=False)

    except Exception:
        return pd.DataFrame()


# ============================================================
# OPTIONS SPREAD STRATEGIES
# ============================================================

def spread_liquidity_ok(a, b, min_volume=10, min_oi=25, max_spread_pct=35):
    try:
        avg_volume = (float(a["volume"]) + float(b["volume"])) / 2
        avg_oi = (float(a["openInterest"]) + float(b["openInterest"])) / 2
        avg_spread = (float(a["spread_pct"]) + float(b["spread_pct"])) / 2

        return (
            avg_volume >= min_volume and
            avg_oi >= min_oi and
            avg_spread <= max_spread_pct
        )
    except Exception:
        return False


def spread_liquidity_score(a, b):
    avg_volume = (float(a["volume"]) + float(b["volume"])) / 2
    avg_oi = (float(a["openInterest"]) + float(b["openInterest"])) / 2
    avg_spread = (float(a["spread_pct"]) + float(b["spread_pct"])) / 2

    volume_score = min(30, avg_volume / 5)
    oi_score = min(30, avg_oi / 20)
    spread_score = max(0, 40 - avg_spread)

    return min(100, volume_score + oi_score + spread_score)


def build_debit_call_spreads(calls, stock_price, exp, setup_score):
    df = prepare_chain(calls)
    if df.empty:
        return pd.DataFrame()

    rows = []
    df = df.sort_values("strike").reset_index(drop=True)
    T = max(dte(exp) / 365, 1 / 365)

    for i in range(len(df)):
        buy = df.iloc[i]

        for j in range(i + 1, min(i + 6, len(df))):
            sell = df.iloc[j]

            if not spread_liquidity_ok(buy, sell):
                continue

            width = sell["strike"] - buy["strike"]
            debit = buy["ask"] - sell["bid"]

            if width <= 0 or debit <= 0:
                continue

            max_profit = width - debit
            max_loss = debit
            breakeven = buy["strike"] + debit

            if max_profit <= 0 or max_loss <= 0:
                continue

            rr = max_profit / max_loss
            prob_est = prob_above_price(stock_price, breakeven, T, 0.045, buy["impliedVolatility"]) * 100
            liquidity = spread_liquidity_score(buy, sell)

            score = int(max(0, min(
                100,
                0.40 * setup_score +
                0.25 * min(100, rr * 25) +
                0.20 * liquidity +
                0.15 * prob_est
            )))

            rows.append({
                "Strategy": "Debit Call Spread",
                "Bias": "Bullish",
                "Buy Leg": buy["contractSymbol"],
                "Sell Leg": sell["contractSymbol"],
                "Buy Strike": buy["strike"],
                "Sell Strike": sell["strike"],
                "Width": width,
                "Debit/Credit": round(debit * 100, 2),
                "Max Profit": round(max_profit * 100, 2),
                "Max Loss": round(max_loss * 100, 2),
                "Breakeven": round(breakeven, 2),
                "Risk/Reward": round(rr, 2),
                "Probability Estimate %": round(prob_est, 1),
                "Spread Score": score
            })

    return pd.DataFrame(rows).sort_values("Spread Score", ascending=False) if rows else pd.DataFrame()


def build_debit_put_spreads(puts, stock_price, exp, setup_score):
    df = prepare_chain(puts)
    if df.empty:
        return pd.DataFrame()

    rows = []
    df = df.sort_values("strike", ascending=False).reset_index(drop=True)
    T = max(dte(exp) / 365, 1 / 365)

    for i in range(len(df)):
        buy = df.iloc[i]

        for j in range(i + 1, min(i + 6, len(df))):
            sell = df.iloc[j]

            if not spread_liquidity_ok(buy, sell):
                continue

            width = buy["strike"] - sell["strike"]
            debit = buy["ask"] - sell["bid"]

            if width <= 0 or debit <= 0:
                continue

            max_profit = width - debit
            max_loss = debit
            breakeven = buy["strike"] - debit

            if max_profit <= 0 or max_loss <= 0:
                continue

            rr = max_profit / max_loss
            prob_est = prob_below_price(stock_price, breakeven, T, 0.045, buy["impliedVolatility"]) * 100
            liquidity = spread_liquidity_score(buy, sell)

            score = int(max(0, min(
                100,
                0.40 * setup_score +
                0.25 * min(100, rr * 25) +
                0.20 * liquidity +
                0.15 * prob_est
            )))

            rows.append({
                "Strategy": "Debit Put Spread",
                "Bias": "Bearish",
                "Buy Leg": buy["contractSymbol"],
                "Sell Leg": sell["contractSymbol"],
                "Buy Strike": buy["strike"],
                "Sell Strike": sell["strike"],
                "Width": width,
                "Debit/Credit": round(debit * 100, 2),
                "Max Profit": round(max_profit * 100, 2),
                "Max Loss": round(max_loss * 100, 2),
                "Breakeven": round(breakeven, 2),
                "Risk/Reward": round(rr, 2),
                "Probability Estimate %": round(prob_est, 1),
                "Spread Score": score
            })

    return pd.DataFrame(rows).sort_values("Spread Score", ascending=False) if rows else pd.DataFrame()


def build_credit_put_spreads(puts, stock_price, exp, setup_score):
    df = prepare_chain(puts)
    if df.empty:
        return pd.DataFrame()

    rows = []
    df = df.sort_values("strike", ascending=False).reset_index(drop=True)
    T = max(dte(exp) / 365, 1 / 365)

    for i in range(len(df)):
        sell = df.iloc[i]

        for j in range(i + 1, min(i + 6, len(df))):
            buy = df.iloc[j]

            if not spread_liquidity_ok(sell, buy):
                continue

            width = sell["strike"] - buy["strike"]
            credit = sell["bid"] - buy["ask"]

            if width <= 0 or credit <= 0:
                continue

            max_profit = credit
            max_loss = width - credit
            breakeven = sell["strike"] - credit

            if max_profit <= 0 or max_loss <= 0:
                continue

            rr = max_profit / max_loss
            prob_est = prob_above_price(stock_price, breakeven, T, 0.045, sell["impliedVolatility"]) * 100
            liquidity = spread_liquidity_score(sell, buy)

            score = int(max(0, min(
                100,
                0.35 * setup_score +
                0.30 * prob_est +
                0.20 * liquidity +
                0.15 * min(100, rr * 100)
            )))

            rows.append({
                "Strategy": "Credit Put Spread",
                "Bias": "Bullish / Neutral",
                "Sell Leg": sell["contractSymbol"],
                "Buy Leg": buy["contractSymbol"],
                "Sell Strike": sell["strike"],
                "Buy Strike": buy["strike"],
                "Width": width,
                "Debit/Credit": round(credit * 100, 2),
                "Max Profit": round(max_profit * 100, 2),
                "Max Loss": round(max_loss * 100, 2),
                "Breakeven": round(breakeven, 2),
                "Risk/Reward": round(rr, 2),
                "Probability Estimate %": round(prob_est, 1),
                "Spread Score": score
            })

    return pd.DataFrame(rows).sort_values("Spread Score", ascending=False) if rows else pd.DataFrame()


def build_credit_call_spreads(calls, stock_price, exp, setup_score):
    df = prepare_chain(calls)
    if df.empty:
        return pd.DataFrame()

    rows = []
    df = df.sort_values("strike").reset_index(drop=True)
    T = max(dte(exp) / 365, 1 / 365)

    for i in range(len(df)):
        sell = df.iloc[i]

        for j in range(i + 1, min(i + 6, len(df))):
            buy = df.iloc[j]

            if not spread_liquidity_ok(sell, buy):
                continue

            width = buy["strike"] - sell["strike"]
            credit = sell["bid"] - buy["ask"]

            if width <= 0 or credit <= 0:
                continue

            max_profit = credit
            max_loss = width - credit
            breakeven = sell["strike"] + credit

            if max_profit <= 0 or max_loss <= 0:
                continue

            rr = max_profit / max_loss
            prob_est = prob_below_price(stock_price, breakeven, T, 0.045, sell["impliedVolatility"]) * 100
            liquidity = spread_liquidity_score(sell, buy)

            score = int(max(0, min(
                100,
                0.35 * setup_score +
                0.30 * prob_est +
                0.20 * liquidity +
                0.15 * min(100, rr * 100)
            )))

            rows.append({
                "Strategy": "Credit Call Spread",
                "Bias": "Bearish / Neutral",
                "Sell Leg": sell["contractSymbol"],
                "Buy Leg": buy["contractSymbol"],
                "Sell Strike": sell["strike"],
                "Buy Strike": buy["strike"],
                "Width": width,
                "Debit/Credit": round(credit * 100, 2),
                "Max Profit": round(max_profit * 100, 2),
                "Max Loss": round(max_loss * 100, 2),
                "Breakeven": round(breakeven, 2),
                "Risk/Reward": round(rr, 2),
                "Probability Estimate %": round(prob_est, 1),
                "Spread Score": score
            })

    return pd.DataFrame(rows).sort_values("Spread Score", ascending=False) if rows else pd.DataFrame()


def build_all_spreads(calls, puts, stock_price, exp, call_score, put_score):
    debit_calls = build_debit_call_spreads(calls, stock_price, exp, call_score)
    debit_puts = build_debit_put_spreads(puts, stock_price, exp, put_score)
    credit_puts = build_credit_put_spreads(puts, stock_price, exp, call_score)
    credit_calls = build_credit_call_spreads(calls, stock_price, exp, put_score)

    frames = [x for x in [debit_calls, debit_puts, credit_puts, credit_calls] if not x.empty]

    if not frames:
        return pd.DataFrame(), debit_calls, debit_puts, credit_puts, credit_calls

    all_spreads = pd.concat(frames, ignore_index=True)
    return all_spreads.sort_values("Spread Score", ascending=False), debit_calls, debit_puts, credit_puts, credit_calls


# ============================================================
# CONFIDENCE
# ============================================================

def final_confidence(setup, option_row, side, news_score, earnings_risk, market):
    technical_score = setup["call_score"] if side == "call" else setup["put_score"]
    option_score = option_row["trade_score"] if option_row is not None else 0
    news_component = (news_score + 100) / 2

    final = 0.50 * technical_score + 0.30 * option_score + 0.20 * news_component

    penalties = []
    do_not_trade = False

    if setup["signal"] == "NO TRADE / WAIT":
        final -= 20
        penalties.append("Stock setup says WAIT")
        do_not_trade = True

    if side == "call" and not market["call_ok"]:
        final -= 15
        penalties.append("Market trend does not favor calls")

    if side == "put" and not market["put_ok"]:
        final -= 15
        penalties.append("Market trend does not favor puts")

    if earnings_risk == "High":
        final -= 25
        penalties.append("Earnings risk is high")
        do_not_trade = True
    elif earnings_risk == "Medium":
        final -= 10
        penalties.append("Earnings risk is medium")

    if option_row is not None:
        if option_row["spread_pct"] > 20:
            final -= 20
            penalties.append("Spread is too wide")
            do_not_trade = True

        if option_row["volume"] < 20:
            final -= 10
            penalties.append("Low option volume")

        if option_row["openInterest"] < 50:
            final -= 10
            penalties.append("Low open interest")

        if option_row["impliedVolatility"] > 0.90:
            final -= 15
            penalties.append("Very high IV / expensive option")

    final = int(max(0, min(100, final)))

    if do_not_trade:
        label = "DO NOT TRADE"
    elif final >= 80:
        label = "A+ SETUP"
    elif final >= 70:
        label = "HIGH CONFIDENCE"
    elif final >= 60:
        label = "WATCHLIST"
    else:
        label = "LOW CONFIDENCE / WAIT"

    return final, label, penalties


# ============================================================
# CHART
# ============================================================

def make_chart(df, setup, ticker):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.20, 0.22],
        subplot_titles=("Price Action", "Volume", "RSI / MACD")
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candles"
        ),
        row=1,
        col=1
    )

    for ema in ["EMA9", "EMA21", "EMA50", "EMA200"]:
        fig.add_trace(go.Scatter(x=df.index, y=df[ema], mode="lines", name=ema), row=1, col=1)

    fig.add_hline(y=setup["support"], line_dash="dash", annotation_text="Support", row=1, col=1)
    fig.add_hline(y=setup["resistance"], line_dash="dash", annotation_text="Resistance", row=1, col=1)

    if not np.isnan(setup["entry_low"]) and not np.isnan(setup["entry_high"]):
        fig.add_hrect(
            y0=setup["entry_low"],
            y1=setup["entry_high"],
            opacity=0.18,
            annotation_text="Entry Zone",
            row=1,
            col=1
        )

    if not np.isnan(setup["stop"]):
        fig.add_hline(y=setup["stop"], line_dash="dot", annotation_text="Stop", row=1, col=1)

    if not np.isnan(setup["target1"]):
        fig.add_hline(y=setup["target1"], line_dash="dot", annotation_text="Target 1", row=1, col=1)

    if not np.isnan(setup["target2"]):
        fig.add_hline(y=setup["target2"], line_dash="dot", annotation_text="Target 2", row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=3, col=1)

    fig.add_hline(y=70, line_dash="dash", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_HIST"], name="MACD Hist"), row=3, col=1)

    fig.update_layout(
        title=ticker.upper(),
        height=800,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h")
    )

    return fig


# ============================================================
# SCANNER
# ============================================================

def run_scanner(tickers, period, interval):
    rows = []
    market = market_filter()

    for t in tickers:
        t = t.strip().upper()

        if not t:
            continue

        try:
            df = add_indicators(get_price_data(t, period, interval))

            if df.empty or len(df) < 2:
                continue

            setup = score_setup(df)

            if setup is None:
                continue

            news_df = get_news(t)
            news_score, news_label = score_news_sentiment(news_df)
            earnings_risk, earnings_date, earnings_days = get_earnings_warning(t)

            base_best = max(setup["call_score"], setup["put_score"])
            adjusted = base_best + (news_score * 0.20)

            if earnings_risk == "High":
                adjusted -= 25

            if "CALL" in setup["signal"] and not market["call_ok"]:
                adjusted -= 15

            if "PUT" in setup["signal"] and not market["put_ok"]:
                adjusted -= 15

            adjusted = int(max(0, min(100, adjusted)))

            rows.append({
                "Ticker": t,
                "Price": round(setup["price"], 2),
                "Signal": setup["signal"],
                "Call Score": setup["call_score"],
                "Put Score": setup["put_score"],
                "News": news_label,
                "News Score": news_score,
                "Earnings Risk": earnings_risk,
                "Market Bias": market["bias"],
                "Adjusted Score": adjusted,
                "RSI": round(setup["rsi"], 1),
                "Support": round(setup["support"], 2),
                "Resistance": round(setup["resistance"], 2),
                "Entry Low": None if np.isnan(setup["entry_low"]) else round(setup["entry_low"], 2),
                "Entry High": None if np.isnan(setup["entry_high"]) else round(setup["entry_high"], 2),
                "Stop": None if np.isnan(setup["stop"]) else round(setup["stop"], 2),
                "Target 1": None if np.isnan(setup["target1"]) else round(setup["target1"], 2),
                "Target 2": None if np.isnan(setup["target2"]) else round(setup["target2"], 2),
            })

        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("Adjusted Score", ascending=False)


# ============================================================
# TRADE TRACKER
# ============================================================

TRADE_FILE = "trade_journal.csv"


def load_trades():
    if os.path.exists(TRADE_FILE):
        try:
            return pd.read_csv(TRADE_FILE)
        except Exception:
            pass

    return pd.DataFrame(columns=[
        "Date", "Ticker", "Trade Type", "Signal", "Contract",
        "Entry Price", "Exit Price", "Contracts",
        "Entry Value", "Exit Value", "P&L", "Return %",
        "Result", "Notes"
    ])


def save_trade(trade):
    df = load_trades()
    df = pd.concat([df, pd.DataFrame([trade])], ignore_index=True)
    df.to_csv(TRADE_FILE, index=False)


def calculate_trade_stats(df):
    if df.empty:
        return {
            "Total Trades": 0,
            "Win Rate": 0,
            "Total P&L": 0,
            "Average Win": 0,
            "Average Loss": 0,
            "Average Return": 0,
        }

    df = df.copy()
    df["P&L"] = pd.to_numeric(df["P&L"], errors="coerce").fillna(0)
    df["Return %"] = pd.to_numeric(df["Return %"], errors="coerce").fillna(0)

    total_trades = len(df)
    wins = df[df["P&L"] > 0]
    losses = df[df["P&L"] < 0]

    return {
        "Total Trades": total_trades,
        "Win Rate": len(wins) / total_trades * 100 if total_trades else 0,
        "Total P&L": df["P&L"].sum(),
        "Average Win": wins["P&L"].mean() if not wins.empty else 0,
        "Average Loss": losses["P&L"].mean() if not losses.empty else 0,
        "Average Return": df["Return %"].mean() if total_trades else 0
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Controls")

mode = st.sidebar.radio(
    "Mode",
    ["Single Ticker", "Multi-Stock Scanner", "Trade Tracker"]
)

ticker = st.sidebar.text_input("Ticker", "MSFT").upper().strip()

period = st.sidebar.selectbox(
    "Chart Period",
    ["1mo", "3mo", "6mo", "1y", "2y"],
    index=2
)

interval = st.sidebar.selectbox(
    "Chart Interval",
    ["1d", "1h", "30m", "15m"],
    index=0
)

if interval in ["15m", "30m"] and period != "1mo":
    st.sidebar.warning("15m/30m data works best with 1mo. Switching period to 1mo.")
    period = "1mo"

elif interval == "1h" and period in ["1y", "2y"]:
    st.sidebar.warning("1h data works best with 6mo or less. Switching period to 6mo.")
    period = "6mo"

st.sidebar.markdown("---")
st.sidebar.subheader("Options Filters")

min_delta = st.sidebar.slider("Min Delta", 0.05, 0.95, 0.30, 0.05)
max_delta = st.sidebar.slider("Max Delta", 0.05, 0.95, 0.60, 0.05)

if min_delta > max_delta:
    st.sidebar.error("Min Delta cannot be greater than Max Delta.")
    st.stop()

min_volume = st.sidebar.number_input("Min Option Volume", min_value=0, value=50, step=10)
min_oi = st.sidebar.number_input("Min Open Interest", min_value=0, value=100, step=25)
max_spread_pct = st.sidebar.slider("Max Spread %", 1.0, 50.0, 15.0, 1.0)
max_cost = st.sidebar.number_input("Max Contract Cost", min_value=1, value=1500, step=50)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk Rules")

risk_per_trade = st.sidebar.number_input("Max $ Risk Per Trade", min_value=10, value=250, step=25)
stop_loss_pct = st.sidebar.slider("Option Stop Loss %", 5, 100, 25, 5)
profit_target_pct = st.sidebar.slider("Option Profit Target %", 10, 200, 50, 5)


# ============================================================
# TRADE TRACKER MODE
# ============================================================

if mode == "Trade Tracker":
    st.subheader("Trade Tracker")

    st.info(
        "Note: on Streamlit Cloud, CSV storage may reset when the app restarts. "
        "For permanent tracking, use Google Sheets or a database later."
    )

    with st.form("trade_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            trade_date = st.date_input("Trade Date", value=date.today())
            trade_ticker = st.text_input("Trade Ticker", "MSFT").upper()

            trade_type = st.selectbox(
                "Trade Type",
                [
                    "Long Call / Long Put",
                    "Debit Spread",
                    "Credit Spread"
                ]
            )

            signal = st.selectbox(
                "Signal",
                [
                    "STRONG BUY CALL",
                    "CALL WATCHLIST",
                    "STRONG BUY PUT",
                    "PUT WATCHLIST",
                    "SPREAD STRATEGY",
                    "NO TRADE / MANUAL"
                ]
            )

        with col2:
            contract = st.text_input("Contract / Spread Legs", "")
            entry_price = st.number_input("Entry Price / Net Debit or Credit", min_value=0.00, value=1.00, step=0.01)
            exit_price = st.number_input("Exit Price / Closing Value", min_value=0.00, value=0.50, step=0.01)
            contracts = st.number_input("Number of Contracts", min_value=1, value=1, step=1)

        with col3:
            notes = st.text_area("Notes", "")

        submitted = st.form_submit_button("Save Trade")

        if submitted:
            entry_value = entry_price * 100 * contracts
            exit_value = exit_price * 100 * contracts

            if trade_type == "Credit Spread":
                pnl = entry_value - exit_value
                return_pct = (pnl / entry_value * 100) if entry_value > 0 else 0
            else:
                pnl = exit_value - entry_value
                return_pct = (pnl / entry_value * 100) if entry_value > 0 else 0

            result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"

            save_trade({
                "Date": trade_date,
                "Ticker": trade_ticker,
                "Trade Type": trade_type,
                "Signal": signal,
                "Contract": contract,
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "Contracts": contracts,
                "Entry Value": entry_value,
                "Exit Value": exit_value,
                "P&L": pnl,
                "Return %": return_pct,
                "Result": result,
                "Notes": notes
            })

            st.success(f"Trade saved: {result} | P&L: ${pnl:.2f} | Return: {return_pct:.1f}%")

    trades = load_trades()

    if trades.empty:
        st.info("No trades logged yet.")
    else:
        stats = calculate_trade_stats(trades)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Trades", stats["Total Trades"])
        c2.metric("Win Rate", f"{stats['Win Rate']:.1f}%")
        c3.metric("Total P&L", f"${stats['Total P&L']:.2f}")
        c4.metric("Avg Win", f"${stats['Average Win']:.2f}")
        c5.metric("Avg Loss", f"${stats['Average Loss']:.2f}")

        st.metric("Average Return", f"{stats['Average Return']:.1f}%")
        st.dataframe(trades, use_container_width=True, height=500)

        st.download_button(
            "Download Trade Journal CSV",
            trades.to_csv(index=False),
            file_name="trade_journal.csv",
            mime="text/csv"
        )

    st.stop()


# ============================================================
# SCANNER MODE
# ============================================================

if mode == "Multi-Stock Scanner":
    st.subheader("Multi-Stock Scanner")

    default_list = "AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, AMD, NFLX, SPY, QQQ"
    tickers_text = st.text_area("Tickers", default_list, height=110)

    if st.button("Run Scanner"):
        tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]

        with st.spinner("Scanning stocks with market/news/earnings filters..."):
            scan = run_scanner(tickers, period, interval)

        if scan.empty:
            st.warning("No scanner results found. Try 6mo/1d or 1y/1d.")
        else:
            st.dataframe(scan, use_container_width=True, height=560)

    st.stop()


# ============================================================
# SINGLE TICKER MODE
# ============================================================

if not ticker:
    st.warning("Enter a ticker.")
    st.stop()

df = add_indicators(get_price_data(ticker, period, interval))

if df.empty or len(df) < 2:
    st.error("Not enough usable data. Try 6mo or 1y with 1d interval.")
    st.stop()

setup = score_setup(df)

if setup is None:
    st.error("Not enough data to generate a signal.")
    st.stop()

market = market_filter()
news_df = get_news(ticker)
news_score, news_label = score_news_sentiment(news_df)
earnings_risk, earnings_date, earnings_days = get_earnings_warning(ticker)


if "CALL" in setup["signal"]:
    st.success(f"{setup['signal']} | Call Score: {setup['call_score']} | Put Score: {setup['put_score']}")
elif "PUT" in setup["signal"]:
    st.error(f"{setup['signal']} | Call Score: {setup['call_score']} | Put Score: {setup['put_score']}")
else:
    st.info(f"{setup['signal']} | Call Score: {setup['call_score']} | Put Score: {setup['put_score']}")


c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", f"${setup['price']:.2f}")
c2.metric("RSI", f"{setup['rsi']:.1f}")
c3.metric("Support", f"${setup['support']:.2f}")
c4.metric("Resistance", f"${setup['resistance']:.2f}")
c5.metric("Volume Ratio", f"{setup['vol_ratio']:.2f}x")

c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("Entry Low", "-" if np.isnan(setup["entry_low"]) else f"${setup['entry_low']:.2f}")
c7.metric("Entry High", "-" if np.isnan(setup["entry_high"]) else f"${setup['entry_high']:.2f}")
c8.metric("Stop", "-" if np.isnan(setup["stop"]) else f"${setup['stop']:.2f}")
c9.metric("Target 1", "-" if np.isnan(setup["target1"]) else f"${setup['target1']:.2f}")
c10.metric("Target 2", "-" if np.isnan(setup["target2"]) else f"${setup['target2']:.2f}")

st.subheader("Market / News / Earnings Filters")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Market Bias", market["bias"])
m2.metric("VIX", "-" if np.isnan(market["vix"]) else f"{market['vix']:.2f}")
m3.metric("News Sentiment", news_label)
m4.metric("News Score", news_score)

if earnings_date:
    st.write(f"**Earnings Risk:** {earnings_risk} | Earnings Date: {earnings_date} | Days Away: {earnings_days}")
else:
    st.write(f"**Earnings Risk:** {earnings_risk}")

if earnings_risk == "High":
    st.error("High earnings risk. Avoid unless intentionally trading earnings.")
elif earnings_risk == "Medium":
    st.warning("Medium earnings risk. Be careful with IV crush.")

with st.expander("Why this signal?"):
    left, right = st.columns(2)

    with left:
        st.markdown("### Call Reasons")
        if setup["call_reasons"]:
            for reason in setup["call_reasons"]:
                st.write(f"- {reason}")
        else:
            st.write("No strong call reasons.")

    with right:
        st.markdown("### Put Reasons")
        if setup["put_reasons"]:
            for reason in setup["put_reasons"]:
                st.write(f"- {reason}")
        else:
            st.write("No strong put reasons.")

st.plotly_chart(make_chart(df, setup, ticker), use_container_width=True)


# ============================================================
# OPTIONS CHAIN + SPREADS
# ============================================================

st.subheader("Options Chain")

try:
    expirations = get_expirations(ticker)

    if not expirations:
        st.warning("No options found for this ticker.")
        st.stop()

    exp = st.selectbox("Expiration Date", expirations)

    suggested_side = "call" if setup["call_score"] >= setup["put_score"] else "put"

    side = st.radio(
        "Single-Leg Option Side",
        ["call", "put"],
        index=0 if suggested_side == "call" else 1
    )

    calls, puts = get_chain(ticker, exp)
    chain = calls if side == "call" else puts
    setup_score = setup["call_score"] if side == "call" else setup["put_score"]

    option_df = build_options_table(
        chain=chain,
        option_type=side,
        stock_price=setup["price"],
        exp=exp,
        setup_score=setup_score,
        min_delta=min_delta,
        max_delta=max_delta,
        min_volume=min_volume,
        min_oi=min_oi,
        max_spread_pct=max_spread_pct,
        max_cost=max_cost
    )

    if option_df.empty:
        st.warning("No usable option contracts found after filtering bad bid/ask data.")
    else:
        best = option_df.iloc[0]

        confidence_score, confidence_label, penalties = final_confidence(
            setup=setup,
            option_row=best,
            side=side,
            news_score=news_score,
            earnings_risk=earnings_risk,
            market=market
        )

        st.subheader("Final Single-Leg Trade Confidence")

        if confidence_label == "DO NOT TRADE":
            st.error(f"{confidence_label} | Confidence Score: {confidence_score}")
        elif confidence_score >= 70:
            st.success(f"{confidence_label} | Confidence Score: {confidence_score}")
        else:
            st.warning(f"{confidence_label} | Confidence Score: {confidence_score}")

        if penalties:
            st.write("**Warnings / Penalties:**")
            for p in penalties:
                st.write(f"- {p}")

        checklist = {
            "Signal score above 60": setup_score >= 60,
            "Delta within range": min_delta <= abs(best["delta"]) <= max_delta,
            "Spread acceptable": best["spread_pct"] <= max_spread_pct,
            "Volume acceptable": best["volume"] >= min_volume,
            "Open interest acceptable": best["openInterest"] >= min_oi,
            "Contract cost acceptable": best["cost_ask"] <= max_cost,
            "Earnings risk not high": earnings_risk != "High",
            "Market does not strongly conflict": not (
                (side == "call" and not market["call_ok"]) or
                (side == "put" and not market["put_ok"])
            )
        }

        with st.expander("Trade Checklist"):
            for item, passed in checklist.items():
                st.write(("✅ " if passed else "❌ ") + item)

        display = option_df.copy()

        for col in ["lastPrice", "bid", "ask", "mid"]:
            display[col] = display[col].round(2)

        display["cost_mid"] = display["cost_mid"].round(0)
        display["cost_ask"] = display["cost_ask"].round(0)
        display["spread_pct"] = display["spread_pct"].round(1)
        display["impliedVolatility"] = (display["impliedVolatility"] * 100).round(1)
        display["delta"] = display["delta"].round(2)
        display["prob_itm"] = (display["prob_itm"] * 100).round(1)

        display = display.rename(columns={
            "contractSymbol": "Contract",
            "strike": "Strike",
            "lastPrice": "Last",
            "bid": "Bid",
            "ask": "Ask",
            "mid": "Mid",
            "cost_mid": "Cost Mid",
            "cost_ask": "Cost Ask",
            "spread_pct": "Spread %",
            "volume": "Volume",
            "openInterest": "Open Interest",
            "impliedVolatility": "IV %",
            "delta": "Delta",
            "prob_itm": "Prob ITM %",
            "trade_score": "Trade Score"
        })

        st.dataframe(display, use_container_width=True, height=420)

        st.markdown("### Best Single Contract Based on Your Filters")

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Strike", f"${best['strike']:.2f}")
        b2.metric("Ask Cost", f"${best['cost_ask']:.0f}")
        b3.metric("Delta", f"{best['delta']:.2f}")
        b4.metric("Prob ITM", f"{best['prob_itm'] * 100:.1f}%")
        b5.metric("Trade Score", f"{best['trade_score']}")

        risk_per_contract = best["cost_ask"] * (stop_loss_pct / 100)
        max_contracts = int(risk_per_trade // max(risk_per_contract, 1))

        st.markdown("### Single-Leg Trade Plan")
        st.write(f"**Contract:** {best['contractSymbol']}")
        st.write(f"**Expiration:** {exp}")
        st.write(f"**Side:** {side.upper()}")
        st.write(f"**Ask cost per contract:** ${best['cost_ask']:.0f}")
        st.write(f"**Estimated risk per contract at {stop_loss_pct}% stop:** ${risk_per_contract:.0f}")
        st.write(f"**Max contracts based on ${risk_per_trade:.0f} risk:** {max_contracts}")
        st.write(f"**Profit target:** +{profit_target_pct}% option gain")
        st.write(f"**Stop loss:** -{stop_loss_pct}% option loss or break of stock technical stop")
        st.write(f"**Days to expiration:** {dte(exp)}")

    st.subheader("Options Spread Strategies")

    with st.spinner("Building spread strategies..."):
        all_spreads, debit_calls, debit_puts, credit_puts, credit_calls = build_all_spreads(
            calls=calls,
            puts=puts,
            stock_price=setup["price"],
            exp=exp,
            call_score=setup["call_score"],
            put_score=setup["put_score"]
        )

    if all_spreads.empty:
        st.warning("No valid spread strategies found for this expiration.")
    else:
        best_spread = all_spreads.iloc[0]

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Best Strategy", best_spread["Strategy"])
        s2.metric("Spread Score", f"{best_spread['Spread Score']}")
        s3.metric("Max Profit", f"${best_spread['Max Profit']:.0f}")
        s4.metric("Max Loss", f"${best_spread['Max Loss']:.0f}")
        s5.metric("Breakeven", f"${best_spread['Breakeven']:.2f}")

        st.markdown("### Best Overall Spread")
        st.dataframe(pd.DataFrame([best_spread]), use_container_width=True)

        strategy_filter = st.selectbox(
            "View Spread Strategy",
            [
                "All Spreads",
                "Debit Call Spread",
                "Debit Put Spread",
                "Credit Put Spread",
                "Credit Call Spread"
            ]
        )

        if strategy_filter == "All Spreads":
            spread_display = all_spreads
        elif strategy_filter == "Debit Call Spread":
            spread_display = debit_calls
        elif strategy_filter == "Debit Put Spread":
            spread_display = debit_puts
        elif strategy_filter == "Credit Put Spread":
            spread_display = credit_puts
        else:
            spread_display = credit_calls

        if spread_display.empty:
            st.info("No spreads available for this strategy.")
        else:
            st.dataframe(spread_display.head(25), use_container_width=True, height=500)

        st.markdown("### Spread Strategy Notes")
        st.write("- **Debit Call Spread:** bullish, defined risk, cheaper than buying a call.")
        st.write("- **Debit Put Spread:** bearish, defined risk, cheaper than buying a put.")
        st.write("- **Credit Put Spread:** bullish/neutral, collects premium, wins if price stays above breakeven.")
        st.write("- **Credit Call Spread:** bearish/neutral, collects premium, wins if price stays below breakeven.")
        st.write("- Probability estimates are simplified and should be treated as directional risk estimates, not guarantees.")

except Exception as e:
    st.error(f"Options error: {e}")


st.subheader("Recent News")

if news_df.empty:
    st.info("No recent news found.")
else:
    st.dataframe(news_df, use_container_width=True, height=300)

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data source: yfinance")