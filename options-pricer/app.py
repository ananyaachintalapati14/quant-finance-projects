# app.py — run with: streamlit run app.py
import os
import uuid
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime

SNAPSHOT_PATH = "data/snapshot.parquet"

st.set_page_config(page_title="BS vs Market — US", layout="wide")


# ============================================================
# Design system
# ============================================================

C_MARKET = "#1f77b4"   # blue   — market (truth)
C_BS     = "#d62728"   # red    — European benchmark
C_CRR    = "#2ca02c"   # green  — American correction
C_INTR   = "#9e9e9e"   # gray   — intrinsic floor
C_ATM    = "#bbbbbb"   # light  — ATM vertical
C_TV     = "#ff7f0e"   # orange — time value

PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

LEGEND_H = dict(orientation="h", yanchor="bottom", y=1.06,
                xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)")


# ============================================================
# Universe — US only
# ============================================================

USA_50 = {
    "NVDA": "NVIDIA", "AAPL": "Apple", "GOOGL": "Alphabet (A)", "MSFT": "Microsoft",
    "AMZN": "Amazon", "AVGO": "Broadcom", "META": "Meta Platforms", "TSLA": "Tesla",
    "BRK-B": "Berkshire Hathaway", "LLY": "Eli Lilly", "JPM": "JPMorgan Chase",
    "WMT": "Walmart", "AMD": "Advanced Micro Devices", "XOM": "ExxonMobil", "V": "Visa",
    "JNJ": "Johnson & Johnson", "INTC": "Intel", "MA": "Mastercard", "ORCL": "Oracle",
    "ABBV": "AbbVie", "CSCO": "Cisco Systems", "BAC": "Bank of America", "CVX": "Chevron",
    "PLTR": "Palantir", "COST": "Costco", "KO": "Coca-Cola", "CAT": "Caterpillar",
    "LRCX": "Lam Research", "AMAT": "Applied Materials", "DELL": "Dell Technologies",
    "MRK": "Merck", "UNH": "UnitedHealth", "PG": "Procter & Gamble", "MS": "Morgan Stanley",
    "GE": "GE Aerospace", "NFLX": "Netflix", "GS": "Goldman Sachs", "HD": "Home Depot",
    "PM": "Philip Morris", "WFC": "Wells Fargo", "PANW": "Palo Alto Networks",
    "RTX": "RTX Corp", "GEV": "GE Vernova", "ANET": "Arista Networks",
    "TXN": "Texas Instruments", "SNDK": "Sandisk", "KLAC": "KLA Corp",
    "C": "Citigroup", "TMUS": "T-Mobile US", "IBM": "IBM",
}

ALL_TICKERS = USA_50


def ticker_label(t: str) -> str:
    name = ALL_TICKERS.get(t)
    return f"{t} — {name}" if name else t


# ============================================================
# Pricing
# ============================================================

def bs_price(S, K, T, r, q, sigma, kind):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if kind == "call" else max(0.0, K - S)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def crr_price(S, K, T, r, q, sigma, kind, N=150, american=True):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if kind == "call" else max(0.0, K - S)
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(N + 1)
    ST = S * u**(N - j) * d**j
    V = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            j = np.arange(i + 1)
            Si = S * u**(i - j) * d**j
            intrinsic = np.maximum(Si - K, 0.0) if kind == "call" else np.maximum(K - Si, 0.0)
            V = np.maximum(V, intrinsic)
    return float(V[0])


def implied_vol(market_mid, S, K, T, r, q, kind, model="bs"):
    pricer = bs_price if model == "bs" else crr_price
    intrinsic = max(0.0, S - K) if kind == "call" else max(0.0, K - S)
    if not np.isfinite(market_mid) or market_mid <= intrinsic or T <= 0:
        return np.nan
    f = lambda s: pricer(S, K, T, r, q, s, kind) - market_mid
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-6, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan


# ============================================================
# Data — live + snapshot fallback
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def get_ticker_info(sym):
    tk = yf.Ticker(sym)
    info = tk.info
    return {
        "spot": info.get("currentPrice") or info.get("regularMarketPrice"),
        "div_yield": info.get("dividendYield") or 0.0,
        "expiries": list(tk.options),
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_chain(sym, expiry):
    ch = yf.Ticker(sym).option_chain(expiry)
    return pd.concat([ch.calls.assign(type="call"), ch.puts.assign(type="put")],
                     ignore_index=True)


@st.cache_data(show_spinner=False)
def load_snapshot():
    """Load cached snapshot. Returns empty DataFrame if none exists."""
    if not os.path.exists(SNAPSHOT_PATH):
        return pd.DataFrame()
    try:
        return pd.read_parquet(SNAPSHOT_PATH)
    except Exception:
        return pd.DataFrame()


def has_snapshot() -> bool:
    return os.path.exists(SNAPSHOT_PATH)


def get_ticker_info_safe(sym):
    """Try live Yahoo; fall back to snapshot. Returns (info, source)."""
    try:
        return get_ticker_info(sym), "live"
    except Exception:
        snap = load_snapshot()
        if snap.empty:
            raise RuntimeError("live fetch failed and no snapshot available")
        sub = snap[snap["symbol"] == sym]
        if sub.empty:
            raise RuntimeError(f"{sym} not in snapshot")
        return {
            "spot": float(sub["spot"].iloc[0]),
            "div_yield": float(sub["div_yield"].iloc[0]),
            "expiries": sorted(sub["expiry"].unique().tolist()),
        }, "snapshot"


def get_chain_safe(sym, expiry):
    """Try live Yahoo; fall back to snapshot. Returns (df, source)."""
    try:
        return get_chain(sym, expiry), "live"
    except Exception:
        snap = load_snapshot()
        if snap.empty:
            raise RuntimeError("live fetch failed and no snapshot available")
        sub = snap[(snap["symbol"] == sym) & (snap["expiry"] == expiry)]
        if sub.empty:
            raise RuntimeError(f"{sym} {expiry} not in snapshot")
        return sub.copy(), "snapshot"


def years_to_expiry(expiry_str: str) -> float:
    expiry = datetime.strptime(expiry_str, "%Y-%m-%d").replace(hour=16)
    seconds = (expiry - datetime.now()).total_seconds()
    return max(seconds / (365.0 * 86400.0), 1.0 / (365.0 * 24))


# ============================================================
# Analytics
# ============================================================

def filter_liquid(df, max_spread_pct=0.25, min_oi=10):
    df = df.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"].replace(0, np.nan)
    mask = (
        (df["bid"] > 0) & (df["ask"] > 0)
        & (df["ask"] >= df["bid"])
        & (df["spread_pct"] <= max_spread_pct)
        & (df["openInterest"].fillna(0) >= min_oi)
    )
    return df.loc[mask].reset_index(drop=True)


def enrich(df, spot, T, r, q, assumed_vol, model="bs"):
    if df.empty:
        return df
    df = df.copy()
    df["bs"] = [bs_price(spot, K, T, r, q, assumed_vol, k) for K, k in zip(df.strike, df.type)]
    df["crr"] = [crr_price(spot, K, T, r, q, assumed_vol, k) for K, k in zip(df.strike, df.type)]
    df["iv_market"] = [implied_vol(m, spot, K, T, r, q, k, model=model)
                       for m, K, k in zip(df.mid, df.strike, df.type)]
    df["intrinsic"] = np.where(
        df["type"] == "call",
        np.maximum(spot - df["strike"], 0.0),
        np.maximum(df["strike"] - spot, 0.0),
    )
    df["tv_market"] = df["mid"] - df["intrinsic"]
    df["tv_bs"]     = df["bs"]  - df["intrinsic"]
    df["tv_crr"]    = df["crr"] - df["intrinsic"]
    df["mispricing_$"] = df["mid"] - df["bs"]
    df["mispricing_vol"] = df["iv_market"] - assumed_vol
    df["american_premium"] = df["crr"] - df["bs"]
    df["moneyness"] = df["strike"] / spot
    return df


def accuracy_metrics(df, spot):
    d = df.dropna(subset=["iv_market"])
    if len(d) < 5:
        return {}
    iv = d["iv_market"].to_numpy()
    mid = d["mid"].to_numpy()
    bs = d["bs"].to_numpy()
    safe = mid > 0
    mape = float(np.mean(np.abs(mid[safe] - bs[safe]) / mid[safe]) * 100) if safe.any() else np.nan
    return {
        "IV std (pts)":     float(iv.std() * 100),
        "IV range (pts)":   float((iv.max() - iv.min()) * 100),
        "MAPE (%)":         mape,
        "Mean |err| (pts)": float(np.mean(np.abs(iv - iv.mean())) * 100),
        "Mean IV (%)":      float(iv.mean() * 100),
        "N":                len(d),
    }


def load_one(sym, r, sigma, model, max_spread, min_oi, expiry_idx=0):
    try:
        info, src1 = get_ticker_info_safe(sym)
        spot, q, expiries = info["spot"], info["div_yield"], info["expiries"]
    except Exception as e:
        return {"error": f"fetch failed — {e}"}

    if not spot:
        return {"error": "no spot returned"}
    if not expiries:
        return {"error": "no listed options"}

    expiry = expiries[min(expiry_idx, len(expiries) - 1)]
    T = years_to_expiry(expiry)

    try:
        raw, src2 = get_chain_safe(sym, expiry)
    except Exception as e:
        return {"error": f"chain failed — {e}"}

    df = filter_liquid(raw, max_spread_pct=max_spread, min_oi=int(min_oi))
    df = enrich(df, spot, T, r, q, sigma, model=model)
    if df.empty:
        return {"error": "no liquid contracts"}

    return {
        "df": df, "spot": spot, "q": q, "expiry": expiry, "T": T,
        "expiries": expiries, "metrics": accuracy_metrics(df, spot),
        "source": "snapshot" if "snapshot" in (src1, src2) else "live",
    }


# ============================================================
# Chart builders
# ============================================================

def _atm_line(fig, row=None, col=None):
    fig.add_vline(x=1.0, line_dash="dot", line_color=C_ATM, line_width=1,
                  row=row, col=col)


def _moneyness_zones(fig, row=None, col=None):
    for x0, x1, c in [(0.0, 0.97, "rgba(200,200,200,0.06)"),
                      (0.97, 1.03, "rgba(255,220,120,0.08)"),
                      (1.03, 2.0, "rgba(200,200,200,0.06)")]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=c, line_width=0,
                      layer="below", row=row, col=col)


def price_traces(d, spot, kind, show_crr=False):
    sub = d[d["type"] == kind].sort_values("strike")
    if sub.empty:
        return []
    x = sub["strike"] / spot
    y_market = sub["mid"] / spot * 100
    y_bs     = sub["bs"]  / spot * 100
    traces = [
        (go.Scatter(
            x=pd.concat([x, x[::-1]]),
            y=pd.concat([y_market, y_bs[::-1]]),
            fill="toself", fillcolor="rgba(214,39,40,0.12)",
            line=dict(width=0), hoverinfo="skip",
            showlegend=False, name="Gap",
        ), "Gap"),
        (go.Scatter(
            x=x, y=y_market, mode="markers",
            name="Market",
            marker=dict(symbol="circle", size=9, color=C_MARKET,
                        line=dict(width=1, color="white")),
            hovertemplate="K/S=%{x:.3f}<br>market=%{y:.3f}%<extra></extra>",
        ), "Market"),
        (go.Scatter(
            x=x, y=y_bs, mode="lines",
            name="BS",
            line=dict(color=C_BS, width=2.5),
            hovertemplate="K/S=%{x:.3f}<br>BS=%{y:.3f}%<extra></extra>",
        ), "BS"),
    ]
    if show_crr:
        y_crr = sub["crr"] / spot * 100
        traces.append((go.Scatter(
            x=x, y=y_crr, mode="lines",
            name="CRR",
            line=dict(color=C_CRR, width=1.8, dash="dash"),
            hovertemplate="K/S=%{x:.3f}<br>CRR=%{y:.3f}%<extra></extra>",
        ), "CRR"))
    return traces


def time_value_traces(d, spot, kind):
    sub = d[d["type"] == kind].sort_values("strike")
    if sub.empty:
        return []
    x = sub["strike"] / spot
    return [
        (go.Scatter(
            x=x, y=sub["tv_market"] / spot * 100, mode="markers",
            name="Market time value",
            marker=dict(symbol="circle", size=9, color=C_MARKET,
                        line=dict(width=1, color="white")),
            hovertemplate="K/S=%{x:.3f}<br>TV=%{y:.3f}%<extra></extra>",
        ), "Market"),
        (go.Scatter(
            x=x, y=sub["tv_bs"] / spot * 100, mode="lines",
            name="BS time value",
            line=dict(color=C_BS, width=2.5),
            hovertemplate="K/S=%{x:.3f}<br>BS TV=%{y:.3f}%<extra></extra>",
        ), "BS"),
        (go.Scatter(
            x=x, y=sub["tv_crr"] / spot * 100, mode="lines",
            name="CRR time value",
            line=dict(color=C_CRR, width=1.8, dash="dash"),
            hovertemplate="K/S=%{x:.3f}<br>CRR TV=%{y:.3f}%<extra></extra>",
        ), "CRR"),
    ]


def residual_traces(d, spot, kind):
    sub = d[d["type"] == kind].sort_values("strike")
    if sub.empty:
        return []
    x = sub["strike"] / spot
    err_bs = (sub["mid"] - sub["bs"]) / spot * 100
    err_crr = (sub["mid"] - sub["crr"]) / spot * 100
    return [
        (go.Scatter(
            x=x, y=err_bs, mode="markers", name="Market − BS",
            marker=dict(symbol="circle", size=9, color=C_BS,
                        line=dict(width=1, color="white")),
            hovertemplate="K/S=%{x:.3f}<br>err=%{y:+.3f}%<extra></extra>",
        ), "Market − BS"),
        (go.Scatter(
            x=x, y=err_crr, mode="markers", name="Market − CRR",
            marker=dict(symbol="diamond", size=8, color=C_CRR,
                        line=dict(width=1, color="white")),
            hovertemplate="K/S=%{x:.3f}<br>err=%{y:+.3f}%<extra></extra>",
        ), "Market − CRR"),
    ]


def iv_traces(d, spot):
    d = d.dropna(subset=["iv_market"])
    if d.empty:
        return []
    out = []
    for kind, color, dash, sym in [("call", C_MARKET, "solid", "circle"),
                                   ("put", "#9467bd", "dot", "diamond")]:
        sub = d[d["type"] == kind].sort_values("strike")
        if sub.empty:
            continue
        out.append((go.Scatter(
            x=sub["strike"] / spot, y=sub["iv_market"],
            mode="markers+lines",
            name=f"{kind.capitalize()} IV",
            line=dict(color=color, width=1.8, dash=dash),
            marker=dict(size=8, color=color, line=dict(width=1, color="white")),
            hovertemplate=f"{kind}<br>K/S=%{{x:.3f}}<br>IV=%{{y:.1%}}<extra></extra>",
        ), f"{kind.capitalize()} IV"))
    return out


# ============================================================
# Session state
# ============================================================

def _new_row(sym: str = "") -> dict:
    return {"id": str(uuid.uuid4()), "sym": sym}


def _default_tickers():
    return [_new_row("AAPL"), _new_row("MSFT"), _new_row("NVDA")]


if "tickers" not in st.session_state:
    st.session_state.tickers = _default_tickers()


def add_ticker():
    used = {r["sym"] for r in st.session_state.tickers}
    for t in ALL_TICKERS:
        if t not in used:
            st.session_state.tickers.append(_new_row(t))
            return
    st.session_state.tickers.append(_new_row(""))


def remove_ticker(tid):
    st.session_state.tickers = [t for t in st.session_state.tickers if t["id"] != tid]


def reset_to_default():
    st.session_state.tickers = _default_tickers()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("Global parameters")
r = st.sidebar.number_input("Risk-free (%)", value=4.5, step=0.1) / 100
sigma = st.sidebar.number_input("Assumed vol (%)", value=30.0, step=1.0) / 100
model_choice = st.sidebar.radio("Pricing model", ["European (BS)", "American (CRR)"],
                                help="US equity options are American. Use CRR for apples-to-apples.")
model = "bs" if model_choice.startswith("European") else "crr"
max_spread = st.sidebar.slider("Max spread (% of mid)", 0.05, 0.50, 0.25, 0.05)
min_oi = st.sidebar.number_input("Min open interest", value=10, step=5)

st.sidebar.divider()
st.sidebar.subheader("Tickers")

for row in st.session_state.tickers:
    cols = st.sidebar.columns([6, 1])
    options = list(ALL_TICKERS.keys())
    current = st.session_state.get(f"sym_{row['id']}", row["sym"])
    if current and current not in options:
        options = [current] + options
    idx = options.index(current) if current in options else 0
    cols[0].selectbox(
        "Ticker", options=options, index=idx,
        format_func=ticker_label,
        key=f"sym_{row['id']}", label_visibility="collapsed",
    )
    cols[1].button("✕", key=f"rm_{row['id']}", on_click=remove_ticker, args=(row["id"],))

col_a, col_b = st.sidebar.columns(2)
col_a.button("➕ Add", on_click=add_ticker, width="stretch")
col_b.button("↺ Reset", on_click=reset_to_default, width="stretch")

with st.sidebar.expander("➕ Add custom ticker"):
    with st.form("custom_form", clear_on_submit=True):
        custom = st.text_input("Symbol", placeholder="e.g. BRK-A or ASML")
        if st.form_submit_button("Add custom") and custom.strip():
            st.session_state.tickers.append(_new_row(custom.strip().upper()))
            st.rerun()


# ============================================================
# Compute sidebar tickers
# ============================================================

symbols = []
seen = set()
for row in st.session_state.tickers:
    s = st.session_state.get(f"sym_{row['id']}", row["sym"]).upper().strip()
    if s and s not in seen:
        seen.add(s)
        symbols.append(s)

st.title("Black-Scholes vs Market — US")

if not symbols:
    st.info("Add at least one ticker in the sidebar to begin.")
    st.stop()

# Data-source status banner
if has_snapshot():
    st.caption("🟡 Fallback snapshot available — if Yahoo blocks the cloud IP, "
               "cached data will be used automatically.")
else:
    st.caption("⚠️ No fallback snapshot found. Run `python build_snapshot.py` "
               "locally, commit `data/snapshot.parquet`, and redeploy.")

results = {}
used_snapshot = False

for sym in symbols:
    try:
        info, src1 = get_ticker_info_safe(sym)
        spot, q, expiries = info["spot"], info["div_yield"], info["expiries"]
    except Exception as e:
        results[sym] = {"error": f"fetch failed — {e}"}
        continue
    if not spot or not expiries:
        results[sym] = {"error": "no spot or listed options"}
        continue

    key = f"expiry_{sym}"
    if key not in st.session_state or st.session_state[key] not in expiries:
        st.session_state[key] = expiries[0]
    expiry = st.session_state[key]
    T = years_to_expiry(expiry)

    try:
        raw, src2 = get_chain_safe(sym, expiry)
    except Exception as e:
        results[sym] = {"error": f"chain fetch failed — {e}"}
        continue

    df = filter_liquid(raw, max_spread_pct=max_spread, min_oi=int(min_oi))
    df = enrich(df, spot, T, r, q, sigma, model=model)

    if df.empty:
        results[sym] = {"error": "no contracts passed the liquidity filter"}
        continue

    source = "snapshot" if "snapshot" in (src1, src2) else "live"
    if source == "snapshot":
        used_snapshot = True

    results[sym] = {
        "df": df, "spot": spot, "q": q, "expiry": expiry, "T": T,
        "expiries": expiries, "metrics": accuracy_metrics(df, spot),
        "source": source,
    }

if used_snapshot:
    st.warning(
        "🟡 **Live Yahoo data unavailable** — showing cached snapshot. "
        "Data is from the last time `build_snapshot.py` was run locally, "
        "not current market prices."
    )


# ============================================================
# Tabs
# ============================================================

tab_cmp, tab_ind = st.tabs(["🔬 Compare", "📊 Individual"])


# ============================================================
# COMPARE
# ============================================================

with tab_cmp:
    st.markdown("### Compare at a glance")
    st.caption(
        "**Option prices vs strike are nearly linear** — that's the math, not a bug. "
        "ITM → slope −1, OTM → flat. The curved structure lives in *time value* "
        "and *implied volatility*, not in raw prices.  \n"
        "**X-axis: K/S = Strike ÷ Spot.** 1.00 = ATM. < 1 = ITM calls. > 1 = OTM calls."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        universe_choice = st.radio(
            "Universe", ["Sidebar tickers", "US top 6", "US top 12", "US top 20"],
            key="cmp_universe",
        )
    with c2:
        option_type = st.radio("Option type", ["Calls", "Puts"], key="cmp_type")
    with c3:
        chart_mode = st.radio(
            "Chart mode",
            ["Time value", "IV smile", "Residuals", "Prices"],
            key="cmp_mode",
            help="Time value = price − intrinsic (the curved part). "
                 "IV smile = implied volatility per strike. "
                 "Residuals = market − model. "
                 "Prices = raw levels (will look linear — that's correct).",
        )
    with c4:
        show_crr = st.checkbox("Show CRR overlay", value=False,
                               help="CRR ≈ BS for most options; only differs on ITM puts.")
        zoom_choice = st.radio(
            "Zoom",
            ["ATM (0.95–1.05)", "Normal (0.85–1.15)", "Wide (0.7–1.3)"],
            key="cmp_zoom",
        )

    zoom_ranges = {
        "ATM (0.95–1.05)":   [0.95, 1.05],
        "Normal (0.85–1.15)": [0.85, 1.15],
        "Wide (0.7–1.3)":    [0.70, 1.30],
    }
    x_range = zoom_ranges[zoom_choice]

    if universe_choice == "Sidebar tickers":
        cmp_tickers = list(symbols)
    elif universe_choice == "US top 6":
        cmp_tickers = list(USA_50.keys())[:6]
    elif universe_choice == "US top 12":
        cmp_tickers = list(USA_50.keys())[:12]
    else:
        cmp_tickers = list(USA_50.keys())[:20]

    if not cmp_tickers:
        st.info("No tickers selected.")
    else:
        with st.spinner(f"Loading {len(cmp_tickers)} tickers…"):
            cmp_data = {s: load_one(s, r, sigma, model, max_spread, int(min_oi))
                        for s in cmp_tickers}

        ok_cmp = {s: d for s, d in cmp_data.items() if "error" not in d}
        failed = {s: d["error"] for s, d in cmp_data.items() if "error" in d}

        if failed:
            with st.expander(f"⚠️ {len(failed)} tickers failed to load"):
                for s, err in failed.items():
                    st.write(f"**{s}** — {err}")

        if len(ok_cmp) < 1:
            st.warning("No tickers loaded successfully.")
        else:
            n_cols = 3 if len(ok_cmp) <= 12 else 4
            n_rows = (len(ok_cmp) + n_cols - 1) // n_cols
            kind = "call" if option_type == "Calls" else "put"

            titles = []
            for s, d in ok_cmp.items():
                m = d["metrics"]
                if m and chart_mode != "IV smile":
                    titles.append(
                        f"<b>{s}</b><br>"
                        f"<span style='font-size:10px;color:#888'>"
                        f"IV σ={m['IV std (pts)']:.1f}pt · "
                        f"MAPE={m['MAPE (%)']:.0f}%</span>"
                    )
                else:
                    titles.append(f"<b>{s}</b>")

            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=titles,
                horizontal_spacing=0.05, vertical_spacing=0.18,
            )

            for i, (sym, d) in enumerate(ok_cmp.items()):
                row, col = i // n_cols + 1, i % n_cols + 1

                if chart_mode == "Prices":
                    traces = price_traces(d["df"], d["spot"], kind,
                                          show_crr=show_crr)
                elif chart_mode == "Time value":
                    traces = time_value_traces(d["df"], d["spot"], kind)
                elif chart_mode == "Residuals":
                    traces = residual_traces(d["df"], d["spot"], kind)
                else:
                    traces = iv_traces(d["df"], d["spot"])

                for trace, group in traces:
                    trace.legendgroup = group
                    trace.showlegend = (i == 0)
                    fig.add_trace(trace, row=row, col=col)

                if chart_mode == "Residuals":
                    fig.add_hline(y=0, line_dash="dash",
                                  line_color=C_ATM, line_width=1,
                                  row=row, col=col)
                elif chart_mode == "Prices":
                    _moneyness_zones(fig, row=row, col=col)
                    _atm_line(fig, row=row, col=col)
                else:
                    _atm_line(fig, row=row, col=col)

            fig.update_xaxes(range=x_range, title_text="K/S", tickformat=".2f")
            if chart_mode == "Prices":
                fig.update_yaxes(title_text="% of spot")
            elif chart_mode == "Time value":
                fig.update_yaxes(title_text="Time value (% of spot)")
            elif chart_mode == "Residuals":
                fig.update_yaxes(title_text="Error (% of spot)")
            else:
                fig.update_yaxes(title_text="Implied vol", tickformat=".0%")

            fig.update_layout(
                height=320 * n_rows,
                hovermode="closest",
                margin=dict(t=90, b=50, l=60, r=20),
                legend=LEGEND_H,
            )
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

            st.info(
                "**Reading these charts.**  \n"
                "• **Time value** is where the interesting curvature lives — "
                "raw prices are dominated by the linear intrinsic component.  \n"
                "• **IV smile** is the direct evidence BS is wrong: it should be "
                "a flat horizontal line if BS held.  \n"
                "• **Residuals** shows the actual error. Points near zero = good fit.  \n"
                "• Zoom to **ATM (0.95–1.05)** to see the smile's curvature clearly."
            )

            # ---------- Metrics table ----------
            st.subheader("Black-Scholes accuracy")
            st.caption(
                "**IV std** = how much implied vol varies across strikes "
                "(higher = BS more wrong). "
                "**MAPE** = actual average % error of BS vs market."
            )
            rows = [{
                "Ticker": s,
                "Name": ALL_TICKERS.get(s, "—"),
                "Source": d.get("source", "live"),
                "Spot": round(d["spot"], 2),
                "N": d["metrics"].get("N", 0),
                "IV std (pts)": round(d["metrics"].get("IV std (pts)", np.nan), 2),
                "MAPE (%)": round(d["metrics"].get("MAPE (%)", np.nan), 1),
                "Mean |err| (pts)": round(d["metrics"].get("Mean |err| (pts)", np.nan), 2),
                "Mean IV (%)": round(d["metrics"].get("Mean IV (%)", np.nan), 2),
            } for s, d in ok_cmp.items() if d["metrics"]]
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            # ---------- Ranking bars ----------
            st.subheader("Ranked by how wrong BS is")
            bar_df = pd.DataFrame([
                {"Ticker": s,
                 "IV std (pts)": d["metrics"].get("IV std (pts)", np.nan),
                 "MAPE (%)": d["metrics"].get("MAPE (%)", np.nan),
                 "Mean IV (%)": d["metrics"].get("Mean IV (%)", np.nan)}
                for s, d in ok_cmp.items() if d["metrics"]
            ])
            if not bar_df.empty:
                bc1, bc2, bc3 = st.columns(3)
                bc1.plotly_chart(
                    px.bar(bar_df.sort_values("IV std (pts)", ascending=False),
                           x="Ticker", y="IV std (pts)",
                           title="IV std (vol points) — BS violation",
                           color_discrete_sequence=[C_BS]),
                    width="stretch", config=PLOTLY_CONFIG)
                bc2.plotly_chart(
                    px.bar(bar_df.sort_values("MAPE (%)", ascending=False),
                           x="Ticker", y="MAPE (%)",
                           title="MAPE — average % error of BS",
                           color_discrete_sequence=[C_MARKET]),
                    width="stretch", config=PLOTLY_CONFIG)
                bc3.plotly_chart(
                    px.bar(bar_df.sort_values("Mean IV (%)", ascending=False),
                           x="Ticker", y="Mean IV (%)",
                           title="Mean implied vol",
                           color_discrete_sequence=[C_CRR]),
                    width="stretch", config=PLOTLY_CONFIG)


# ============================================================
# INDIVIDUAL
# ============================================================

with tab_ind:
    summary_rows = []
    for sym, res in results.items():
        if "error" in res:
            st.error(f"{sym}: {res['error']}")
            continue

        name = ALL_TICKERS.get(sym, "")
        header = f"📈 {sym} — {name}" if name else f"📈 {sym}"
        with st.expander(header, expanded=(len(results) == 1)):
            expiry = st.selectbox(f"Expiry for {sym}", res["expiries"],
                                  key=f"expiry_{sym}")
            df = res["df"]
            spot, q, T = res["spot"], res["q"], res["T"]

            src_tag = "🟡 snapshot" if res.get("source") == "snapshot" else "🟢 live"
            st.caption(f"Spot ${spot:.2f} · Div {q*100:.2f}% · Expiry {expiry} "
                       f"({T*365:.1f}d) · {len(df)} liquid contracts · "
                       f"{src_tag} · delayed ~15 min")

            m = res["metrics"]
            if m:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("IV std (pts)", f"{m['IV std (pts)']:.2f}",
                          help="How much BS is violated — higher is worse")
                c2.metric("MAPE (%)", f"{m['MAPE (%)']:.1f}",
                          help="Average % error of BS vs market")
                c3.metric("Mean IV", f"{m['Mean IV (%)']:.1f}%")
                c4.metric("N contracts", f"{m['N']}")
                summary_rows.append({
                    "Ticker": sym, "Name": name, "Expiry": expiry,
                    "Spot": round(spot, 2), "N": m["N"],
                    "IV std (pts)": round(m["IV std (pts)"], 2),
                    "MAPE (%)": round(m["MAPE (%)"], 1),
                    "Mean IV (%)": round(m["Mean IV (%)"], 2),
                })

            t_tv, t_iv, t_err, t_price, t_table = st.tabs(
                ["Time value", "IV smile", "Error by strike", "Raw prices", "Data"]
            )

            with t_tv:
                st.caption(
                    "**Time value = price − intrinsic.** The linear part is removed, "
                    "so the convexity becomes visible. Market = dots, BS = red line, "
                    "CRR = green dashed."
                )
                d2 = df.copy()
                calls = d2[d2["type"] == "call"].sort_values("strike")
                puts  = d2[d2["type"] == "put"].sort_values("strike")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=calls["strike"] / spot, y=calls["tv_market"],
                    mode="markers", name="Market call TV",
                    marker=dict(symbol="circle", size=11, color=C_MARKET,
                                line=dict(width=1.5, color="white")),
                    hovertemplate="K/S=%{x:.3f}<br>TV=$%{y:.2f}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=puts["strike"] / spot, y=puts["tv_market"],
                    mode="markers", name="Market put TV",
                    marker=dict(symbol="diamond", size=11, color="#9467bd",
                                line=dict(width=1.5, color="white")),
                    hovertemplate="K/S=%{x:.3f}<br>TV=$%{y:.2f}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=calls["strike"] / spot, y=calls["tv_bs"],
                    mode="lines", name="BS call TV",
                    line=dict(color=C_BS, width=2.5),
                ))
                fig.add_trace(go.Scatter(
                    x=puts["strike"] / spot, y=puts["tv_bs"],
                    mode="lines", name="BS put TV",
                    line=dict(color=C_BS, width=2.5, dash="dash"),
                ))
                _atm_line(fig)
                _moneyness_zones(fig)
                fig.update_layout(
                    xaxis_title="Moneyness (K/S)",
                    yaxis_title="Time value ($)",
                    height=460, legend=LEGEND_H,
                    margin=dict(t=80, b=50, l=60, r=20),
                )
                st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

            with t_iv:
                st.caption(
                    "Implied vol per strike. **A flat line would mean BS is correct.** "
                    "The curve is the smile — direct evidence BS is wrong."
                )
                iv = df.dropna(subset=["iv_market"]).copy()
                iv["K_over_S"] = iv["strike"] / spot
                fig = go.Figure()
                for k_name, color, sym_m in [("call", C_MARKET, "circle"),
                                             ("put", "#9467bd", "diamond")]:
                    sub = iv[iv["type"] == k_name].sort_values("strike")
                    if sub.empty:
                        continue
                    fig.add_trace(go.Scatter(
                        x=sub["K_over_S"], y=sub["iv_market"],
                        mode="markers+lines", name=f"{k_name.capitalize()} IV",
                        marker=dict(size=10, color=color,
                                    line=dict(width=1, color="white")),
                        line=dict(color=color, width=1.8),
                        hovertemplate="K/S=%{x:.3f}<br>IV=%{y:.1%}<extra></extra>",
                    ))
                fig.add_hline(y=sigma, line_dash="dash", line_color=C_BS,
                              annotation_text=f"Assumed {sigma*100:.0f}%",
                              annotation_position="top right")
                _atm_line(fig)
                fig.update_layout(
                    xaxis_title="Moneyness (K/S)",
                    yaxis_title="Implied volatility",
                    yaxis_tickformat=".0%",
                    height=440, legend=LEGEND_H,
                    margin=dict(t=80, b=50, l=60, r=20),
                )
                st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

            with t_err:
                st.caption(
                    "Market − model, in % of spot. "
                    "Blue = market richer than model. Red = market cheaper. "
                    "Zero line = perfect fit."
                )
                d2 = df.copy()
                d2["err_bs"] = (d2["mid"] - d2["bs"]) / spot * 100
                d2["err_crr"] = (d2["mid"] - d2["crr"]) / spot * 100
                d2["K_over_S"] = d2["strike"] / spot

                fig = go.Figure()
                fig.add_hline(y=0, line_dash="dash", line_color=C_ATM, line_width=1)
                _moneyness_zones(fig)
                for col_name, label, color in [
                    ("err_bs", "Market − BS", C_BS),
                    ("err_crr", "Market − CRR", C_CRR),
                ]:
                    for k_name, sym_m in [("call", "circle"), ("put", "diamond")]:
                        sub = d2[d2["type"] == k_name]
                        if sub.empty:
                            continue
                        fig.add_trace(go.Scatter(
                            x=sub["K_over_S"], y=sub[col_name],
                            mode="markers",
                            name=f"{label} ({k_name})",
                            marker=dict(
                                symbol=sym_m, size=11,
                                color=np.where(sub[col_name] > 0, C_MARKET, C_BS),
                                line=dict(width=1, color="white"),
                            ),
                            hovertemplate=f"K/S=%{{x:.3f}}<br>{label}=%{{y:+.3f}}%<extra></extra>",
                        ))
                _atm_line(fig)
                fig.update_layout(
                    xaxis_title="Moneyness (K/S)",
                    yaxis_title="Error (% of spot)",
                    height=460, legend=LEGEND_H,
                    margin=dict(t=80, b=50, l=60, r=20),
                )
                st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

            with t_price:
                st.caption(
                    "**These will look nearly linear.** That's expected: "
                    "ITM prices track S − K, OTM prices go to zero. "
                    "The interesting structure is in the Time value tab."
                )
                fig = go.Figure()
                calls = df[df["type"] == "call"].sort_values("strike")
                puts  = df[df["type"] == "put"].sort_values("strike")
                fig.add_trace(go.Scatter(
                    x=calls["strike"] / spot, y=calls["mid"], mode="markers",
                    name="Market call",
                    marker=dict(symbol="circle", size=11, color=C_MARKET,
                                line=dict(width=1.5, color="white")),
                ))
                fig.add_trace(go.Scatter(
                    x=puts["strike"] / spot, y=puts["mid"], mode="markers",
                    name="Market put",
                    marker=dict(symbol="diamond", size=11, color="#9467bd",
                                line=dict(width=1.5, color="white")),
                ))
                fig.add_trace(go.Scatter(
                    x=calls["strike"] / spot, y=calls["bs"], mode="lines",
                    name="BS call", line=dict(color=C_BS, width=2.5),
                ))
                fig.add_trace(go.Scatter(
                    x=puts["strike"] / spot, y=puts["bs"], mode="lines",
                    name="BS put", line=dict(color=C_BS, width=2.5, dash="dash"),
                ))
                _atm_line(fig)
                fig.update_layout(
                    xaxis_title="Moneyness (K/S)",
                    yaxis_title="Option price ($)",
                    height=460, legend=LEGEND_H,
                    margin=dict(t=80, b=50, l=60, r=20),
                )
                st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

            with t_table:
                show = df[["type", "strike", "bid", "ask", "mid", "spread_pct",
                           "bs", "crr", "iv_market", "mispricing_$",
                           "mispricing_vol", "american_premium"]]
                show = show.sort_values(["type", "strike"])
                st.dataframe(
                    show.style.format({
                        "bid": "{:.2f}", "ask": "{:.2f}", "mid": "{:.2f}",
                        "bs": "{:.2f}", "crr": "{:.2f}",
                        "spread_pct": "{:.1%}", "iv_market": "{:.1%}",
                        "mispricing_$": "{:+.2f}", "mispricing_vol": "{:+.2%}",
                        "american_premium": "{:.3f}",
                    }),
                    width="stretch", height=400,
                )

    if len(summary_rows) > 1:
        st.divider()
        st.subheader("Cross-ticker summary")
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)