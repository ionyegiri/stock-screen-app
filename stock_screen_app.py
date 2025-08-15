# app.py
import math
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

# ---------- Streamlit Page Setup ----------
st.set_page_config(page_title="Graham Stock Screener & Comparator", layout="wide")

st.title("Graham-Inspired Stock Screener & Comparator")
st.caption("Screens stocks using a practical, modern adaptation of Benjamin Graham's defensive-investor criteria. "
           "For education only — not financial advice.")

# ---------- Sidebar Controls ----------
st.sidebar.header("Screen Settings")
min_market_cap = st.sidebar.number_input("Min Market Cap (USD)", min_value=0, value=2_000_000_000, step=100_000_000, help="Adequate size threshold")
min_current_ratio = st.sidebar.number_input("Min Current Ratio", min_value=0.0, value=2.0, step=0.1)
max_pe = st.sidebar.number_input("Max P/E (Trailing)", min_value=0.0, value=15.0, step=0.5)
max_pb = st.sidebar.number_input("Max P/B", min_value=0.0, value=1.5, step=0.1)
max_pe_pb_product = st.sidebar.number_input("Max P/E × P/B", min_value=0.0, value=22.5, step=0.5)
years_earnings_stability = st.sidebar.slider("Years of earnings stability (available range)", min_value=3, max_value=4, value=4)
years_dividend = st.sidebar.slider("Years of dividend continuity", min_value=3, max_value=10, value=5)
growth_pct_required = st.sidebar.slider("Net Income growth threshold (%)", min_value=0, max_value=200, value=33, step=1)

st.sidebar.divider()
st.sidebar.caption("Optional: Use an API key (e.g., FinancialModelingPrep) via Streamlit Secrets for deeper history.")
use_fmp = st.sidebar.checkbox("Use FinancialModelingPrep (requires st.secrets['FMP_API_KEY'])", value=False)

# ---------- Helpers ----------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yf(ticker: str):
    """Fetches market data and financial statements via yfinance."""
    t = yf.Ticker(ticker)
    info = getattr(t, "fast_info", {}) or {}
    # Some yfinance installs use .get_fast_info(); we’ll try a fallback
    if not info:
        try:
            info = t.get_fast_info()
        except Exception:
            info = {}

    hist = t.history(period="10y")  # for dividend history and price
    bs = t.balance_sheet  # annual
    is_ = t.financials    # annual income statement
    dividends = t.dividends
    shares = None
    try:
        # Newer yfinance has get_shares_full or get_info fields
        shares = t.get_shares_full(start="2010-01-01")  # series by date
        if isinstance(shares, pd.Series) and not shares.empty:
            shares = shares.iloc[-1]
        else:
            shares = None
    except Exception:
        shares = None

    # Some info fields
    trailing_pe = info.get("trailing_pe", None)
    price_to_book = info.get("price_to_book", None)
    market_cap = info.get("market_cap", None)
    last_price = info.get("last_price", None) or (hist["Close"].iloc[-1] if not hist.empty else None)

    return {
        "ticker": ticker.upper(),
        "info": info,
        "hist": hist,
        "bs": bs,
        "is": is_,
        "dividends": dividends,
        "shares_out": shares,
        "pe": trailing_pe,
        "pb": price_to_book,
        "market_cap": market_cap,
        "price": last_price,
    }

def safe_num(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.number)):
            return float(x)
        if isinstance(x, pd.Series):
            return float(x.iloc[0])
        return float(x)
    except Exception:
        return None

def latest_series_value(df: pd.DataFrame, label: str):
    """Return the latest value from a yfinance financials/balance_sheet DataFrame by row label."""
    try:
        if df is None or df.empty:
            return None
        # yfinance frames are labeled with account names as index; columns are periods
        if label not in df.index:
            return None
        s = df.loc[label]
        s = s.dropna()
        if s.empty:
            return None
        return float(s.iloc[0])  # Most recent column is first
    except Exception:
        return None

def series_by_years(df: pd.DataFrame, label: str, years: int):
    """Get up to `years` most recent annual values for a given label from yfinance frames."""
    try:
        if df is None or df.empty or label not in df.index:
            return []
        s = df.loc[label].dropna()
        vals = list(map(float, s.iloc[:years]))
        return vals
    except Exception:
        return []

def net_income_series(is_df: pd.DataFrame, years: int):
    return series_by_years(is_df, "Net Income", years)

def current_assets(bs_df: pd.DataFrame):
    return latest_series_value(bs_df, "Total Current Assets")

def current_liabilities(bs_df: pd.DataFrame):
    return latest_series_value(bs_df, "Total Current Liabilities")

def long_term_debt(bs_df: pd.DataFrame):
    # Try common variants
    for label in ["Long Term Debt", "Long-Term Debt", "Long Term Debt And Capital Lease Obligation"]:
        v = latest_series_value(bs_df, label)
        if v is not None:
            return v
    return None

def total_equity(bs_df: pd.DataFrame):
    for label in ["Total Stockholder Equity", "Total Shareholder Equity", "Stockholders Equity"]:
        v = latest_series_value(bs_df, label)
        if v is not None:
            return v
    return None

def annual_dividends_by_year(dividends: pd.Series, years: int):
    """Return dict of year->sum(dividends) for last `years` completed calendar years."""
    if dividends is None or dividends.empty:
        return {}
    end_year = datetime.utcnow().year - 1  # completed years
    start_year = end_year - years + 1
    div = dividends.copy()
    div.index = pd.to_datetime(div.index)
    yearly = div.groupby(div.index.year).sum()
    out = {}
    for y in range(start_year, end_year + 1):
        out[y] = float(yearly.get(y, 0.0))
    return out

def compute_book_value_per_share(bs_df: pd.DataFrame, shares_out, price):
    eq = total_equity(bs_df)
    if eq is None or not shares_out or shares_out == 0:
        return None, None
    bvps = eq / shares_out
    pb = None
    if price and bvps:
        pb = price / bvps if bvps != 0 else None
    return bvps, pb

def graham_screen(data: dict,
                  min_market_cap: float,
                  min_current_ratio: float,
                  max_pe: float,
                  max_pb: float,
                  max_pe_pb_product: float,
                  years_earnings_stability: int,
                  years_dividend: int,
                  growth_pct_required: float):
    """Evaluate a single ticker against the Graham-inspired criteria."""
    notes = []
    pe = safe_num(data.get("pe"))
    pb = safe_num(data.get("pb"))
    market_cap = safe_num(data.get("market_cap"))
    price = safe_num(data.get("price"))
    bs = data.get("bs")
    is_df = data.get("is")
    dividends = data.get("dividends")
    shares_out = safe_num(data.get("shares_out"))

    # Recompute P/B from book value if missing
    bvps, pb_calc = compute_book_value_per_share(bs, shares_out, price)
    if pb is None and pb_calc is not None:
        pb = pb_calc
        notes.append("P/B computed from book value per share.")

    # Current ratio and net current assets
    ca = current_assets(bs)
    cl = current_liabilities(bs)
    ltd = long_term_debt(bs)
    current_ratio = (ca / cl) if ca is not None and cl not in (None, 0) else None
    net_current_assets = (ca - cl) if ca is not None and cl is not None else None

    # Earnings stability (net income > 0 for N years)
    ni_series = net_income_series(is_df, years_earnings_stability)
    earnings_stable = (len(ni_series) == years_earnings_stability) and all(v > 0 for v in ni_series)

    # Earnings growth (approx): compare oldest vs latest of the grabbed series
    earnings_growth_ok = None
    growth_pct = None
    if len(ni_series) >= 2 and ni_series[-1] != 0:
        oldest = ni_series[-1]
        latest = ni_series[0]
        if oldest is not None and latest is not None and oldest != 0:
            growth_pct = ((latest - oldest) / abs(oldest)) * 100.0
            earnings_growth_ok = growth_pct >= growth_pct_required

    # Dividend record for last N completed years
    div_years = annual_dividends_by_year(dividends, years_dividend)
    dividend_continuous = (len(div_years) == years_dividend) and all(v > 0 for v in div_years.values())

    # Criteria evaluations
    crit = {}
    crit["Adequate Size (Market Cap)"] = (market_cap is not None) and (market_cap >= min_market_cap)
    crit["Current Ratio ≥ threshold"] = (current_ratio is not None) and (current_ratio >= min_current_ratio)
    crit["LT Debt ≤ Net Current Assets"] = (ltd is not None and net_current_assets is not None) and (ltd <= net_current_assets)
    crit[f"Earnings Stability ({years_earnings_stability}y positive NI)"] = earnings_stable
    crit[f"Dividend Record ({years_dividend}y continuous)"] = dividend_continuous
    crit[f"Earnings Growth ≥ {growth_pct_required}%"] = bool(earnings_growth_ok) if earnings_growth_ok is not None else False
    # PE/PB
    pe_ok = (pe is not None) and (pe <= max_pe)
    pb_ok = (pb is not None) and (pb <= max_pb)
    pe_pb_ok = (pe is not None and pb is not None) and ((pe * pb) <= max_pe_pb_product)
    crit["P/E ≤ max OR (P/E×P/B ≤ limit)"] = pe_ok or pe_pb_ok
    crit["P/B ≤ max"] = pb_ok
    # Summaries
    details = {
        "market_cap": market_cap,
        "current_ratio": current_ratio,
        "long_term_debt": ltd,
        "net_current_assets": net_current_assets,
        "pe": pe,
        "pb": pb,
        "bvps": bvps,
        "earnings_series": ni_series,
        "earnings_growth_pct": growth_pct,
        "dividends_by_year": div_years,
        "price": price,
        "notes": notes
    }

    passed = sum(1 for v in crit.values() if v)
    total = len(crit)
    score = (passed, total)
    return crit, details, score

def format_currency(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)

def format_ratio(x, d=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{d}f}"

def normalize_metrics_for_radar(rows):
    # Normalize comparable metrics for radar: market_cap, current_ratio, pe (inverse), pb (inverse), earnings_growth_pct
    # Return dict of ticker->metrics
    cols = ["market_cap", "current_ratio", "pe", "pb", "earnings_growth_pct"]
    df = pd.DataFrame(rows).set_index("ticker")[cols].copy()
    # For PE and PB lower is better; invert them before scaling
    for col in ["pe", "pb"]:
        df[col] = df[col].apply(lambda v: (1 / v) if v and v > 0 else np.nan)
    # Min-max scale each column to [0,1]
    for col in df.columns:
        v = df[col].astype(float)
        mn, mx = np.nanmin(v), np.nanmax(v)
        if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
            df[col] = 0.5  # neutral if cannot scale
        else:
            df[col] = (v - mn) / (mx - mn)
    return df
# ---------- UI: Input ----------
ticker_input = st.text_input("Enter ticker(s)", help="Example: AAPL or AAPL, MSFT, GOOG")
tickers = []
if ticker_input:
    tickers = [t.strip().upper() for t in ticker_input.replace(";", ",").replace(" ", ",").split(",") if t.strip()]

if not tickers:
    st.info("Enter 1–3 tickers to begin. Example: **XOM, CVX, BP**")
    st.stop()

if len(tickers) > 3:
    st.warning("Please enter up to 3 tickers for the comparison view. Only the first 3 will be used.")
    tickers = tickers[:3]

# ---------- Data Fetch & Evaluation ----------
records = []
crit_maps = {}
detail_maps = {}
score_maps = {}

with st.spinner("Fetching data and evaluating..."):
    for t in tickers:
        data = fetch_yf(t)
        crit, details, score = graham_screen(
            data,
            min_market_cap=min_market_cap,
            min_current_ratio=min_current_ratio,
            max_pe=max_pe,
            max_pb=max_pb,
            max_pe_pb_product=max_pe_pb_product,
            years_earnings_stability=years_earnings_stability,
            years_dividend=years_dividend,
            growth_pct_required=growth_pct_required,
        )
        records.append({
            "ticker": t,
            "market_cap": details["market_cap"],
            "current_ratio": details["current_ratio"],
            "pe": details["pe"],
            "pb": details["pb"],
            "earnings_growth_pct": details["earnings_growth_pct"],
            "score_passed": score[0],
            "score_total": score[1]
        })
        crit_maps[t] = crit
        detail_maps[t] = details
        score_maps[t] = score

# ---------- Views ----------
if len(tickers) == 1:
    t = tickers[0]
    st.subheader(f"Screening Result: {t}")
    passed, total = score_maps[t]
    st.metric("Graham Score", f"{passed}/{total}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Market Cap**:", format_currency(detail_maps[t]["market_cap"]))
        st.write("**Price**:", format_currency(detail_maps[t]["price"]))
        st.write("**P/E**:", format_ratio(detail_maps[t]["pe"]))
        st.write("**P/B**:", format_ratio(detail_maps[t]["pb"]))
    with col2:
        st.write("**Current Ratio**:", format_ratio(detail_maps[t]["current_ratio"]))
        st.write("**LT Debt**:", format_currency(detail_maps[t]["long_term_debt"]))
        st.write("**Net Current Assets**:", format_currency(detail_maps[t]["net_current_assets"]))
        st.write("**BVPS**:", format_ratio(detail_maps[t]["bvps"]))
    with col3:
        st.write("**Earnings (latest → older)**:", detail_maps[t]["earnings_series"])
        st.write("**Earnings Growth %**:", format_ratio(detail_maps[t]["earnings_growth_pct"]))
        st.write("**Dividends by Year**:", detail_maps[t]["dividends_by_year"])
        if detail_maps[t]["notes"]:
            st.info("Notes: " + "; ".join(detail_maps[t]["notes"]))

    st.markdown("### Criteria Check")
    for k, v in crit_maps[t].items():
        emoji = "✅" if v else "❌"
        st.write(f"{emoji} {k}")

else:
    st.subheader("Comparison")
    # Score table
    df_scores = pd.DataFrame({
        "Ticker": tickers,
        "Score": [f"{score_maps[t][0]}/{score_maps[t][1]}" for t in tickers],
        "P/E": [format_ratio(detail_maps[t]["pe"]) for t in tickers],
        "P/B": [format_ratio(detail_maps[t]["pb"]) for t in tickers],
        "Current Ratio": [format_ratio(detail_maps[t]["current_ratio"]) for t in tickers],
        "Market Cap": [format_currency(detail_maps[t]["market_cap"]) for t in tickers],
        "Earnings Growth %": [format_ratio(detail_maps[t]["earnings_growth_pct"]) for t in tickers],
    })
    st.dataframe(df_scores, use_container_width=True)

    # Radar chart
    try:
        import plotly.graph_objects as go
        norm = normalize_metrics_for_radar(records)
        categories = ["Market Cap", "Current Ratio", "Inverse P/E", "Inverse P/B", "Earnings Growth %"]
        mapped_cols = {
            "market_cap": "Market Cap",
            "current_ratio": "Current Ratio",
            "pe": "Inverse P/E",
            "pb": "Inverse P/B",
            "earnings_growth_pct": "Earnings Growth %"
        }
        fig = go.Figure()
        for t in norm.index:
            fig.add_trace(go.Scatterpolar(
                r=norm.loc[t].tolist(),
                theta=[mapped_cols[c] for c in norm.columns],
                fill='toself',
                name=t
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Normalized Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render radar chart: {e}")

    # Expanders with per-ticker details
    st.markdown("### Detailed Criteria by Ticker")
    for t in tickers:
        with st.expander(f"Details: {t}", expanded=False):
            passed, total = score_maps[t]
            st.write(f"**Graham Score:** {passed}/{total}")
            for k, v in crit_maps[t].items():
                emoji = "✅" if v else "❌"
                st.write(f"{emoji} {k}")
            d = detail_maps[t]
            st.write("**Market Cap:**", format_currency(d["market_cap"]))
            st.write("**Price:**", format_currency(d["price"]))
            st.write("**P/E:**", format_ratio(d["pe"]))
            st.write("**P/B:**", format_ratio(d["pb"]))
            st.write("**BVPS:**", format_ratio(d["bvps"]))
            st.write("**Current Ratio:**", format_ratio(d["current_ratio"]))
            st.write("**LT Debt:**", format_currency(d["long_term_debt"]))
            st.write("**Net Current Assets:**", format_currency(d["net_current_assets"]))
            st.write("**Earnings (latest → older):**", d["earnings_series"])
            st.write("**Earnings Growth %:**", format_ratio(d["earnings_growth_pct"]))
            st.write("**Dividends by Year:**", d["dividends_by_year"])
            if d["notes"]:
                st.info("Notes: " + "; ".join(d["notes"]))

st.divider()
st.caption(
    "Data via yfinance/Yahoo Finance. Some companies may have incomplete statements, which can affect criteria. "
    "This tool provides an approximation of Graham's framework for a modern data context."
)
