# app.py — Earnings Call Sentiment Engine
# HuggingFace Spaces (Streamlit SDK) — must live at repo root
# built by Dhaval Vibhakar

from __future__ import annotations  # enables X | Y union syntax on Python 3.9

import os
import io
import re
import time

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import yfinance as yf

# torch/transformers are heavy — wrap the import so the rest of the app
# still loads even if someone runs this locally without GPU dependencies
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_FINBERT = True
except ImportError:
    HAS_FINBERT = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the very first st.* call, before any other output
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Earnings Sentiment Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── dark mode CSS ────────────────────────────────────────────────────────────
# streamlit's built-in dark mode is env-dependent; forcing it with CSS gives
# consistent rendering across HuggingFace Spaces, local dev, and Streamlit Cloud
st.markdown("""
<style>
    /* ── base backgrounds ── */
    .stApp, .main .block-container { background-color: #0e1117; color: #f0f6fc; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] * { color: #c9d1d9; }

    /* ── metric cards ── */
    [data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 13px; }
    [data-testid="stMetricValue"] { color: #f0f6fc !important; font-size: 1.4rem; }

    /* ── buttons ── */
    .stButton > button {
        background-color: #238636;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    .stButton > button:hover { background-color: #2ea043; }

    /* ── text inputs ── */
    .stTextInput > div > div > input {
        background-color: #161b22;
        color: #f0f6fc;
        border-color: #30363d;
        border-radius: 6px;
    }

    /* ── headers ── */
    h1, h2, h3, h4 { color: #f0f6fc !important; }

    /* ── divider ── */
    hr { border-color: #30363d; }

    /* ── radio buttons ── */
    .stRadio > div { gap: 8px; }

    /* ── dataframe ── */
    [data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }

    /* ── info/warning boxes ── */
    [data-testid="stAlert"] { border-radius: 8px; }

    /* ── finding cards (custom HTML) ── */
    .finding-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 12px;
        border-left: 4px solid #3fb950;
    }
    .finding-card.warning  { border-left-color: #d29922; }
    .finding-card.danger   { border-left-color: #f85149; }
    .finding-card.info     { border-left-color: #1f6feb; }
    .finding-card.neutral  { border-left-color: #8b949e; }
    .finding-card h4 { color: #f0f6fc; margin: 0 0 8px 0; font-size: 15px; }
    .finding-card p  { color: #c9d1d9; margin: 0; font-size: 14px; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SECTOR_MAP = {
    "AAPL": "Tech",   "MSFT": "Tech",    "AMZN": "Tech",  "GOOGL": "Tech",
    "META": "Tech",   "NVDA": "Tech",    "TSLA": "Tech",
    "JPM":  "Finance","BAC":  "Finance", "GS":   "Finance",
    "MS":   "Finance","WFC":  "Finance", "V":    "Finance",
    "JNJ":  "Healthcare","PFE":"Healthcare","UNH":"Healthcare",
    "CVS":  "Healthcare","ABT":"Healthcare",
    "WMT":  "Consumer","KO":  "Consumer",
}

SECTOR_COLORS = {
    "Tech":       "#1f6feb",
    "Finance":    "#3fb950",
    "Healthcare": "#a371f7",
    "Consumer":   "#d29922",
}

# app/ lives one level below the repo root; data/ is at the repo root
# so we go up one directory to find price_changes.csv etc.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Merges the three output CSVs into one master dataframe.
    cache_data prevents re-reading on every Streamlit rerun — the script
    re-executes top-to-bottom on every user click, which would be slow.
    """
    # debug - show us exactly what path is being used
    st.write(f"BASE_DIR: {BASE_DIR}")
    st.write(f"Files in BASE_DIR: {os.listdir(BASE_DIR)}")
    st.write(f"Data folder exists: {os.path.exists(os.path.join(BASE_DIR, 'data'))}")

    price     = pd.read_csv(os.path.join(BASE_DIR, "data", "price_changes.csv"))
    sentiment = pd.read_csv(os.path.join(BASE_DIR, "data", "sentiment_results.csv"))
    metadata  = pd.read_csv(os.path.join(BASE_DIR, "data", "filings_metadata.csv"))

    df = price.merge(sentiment, on="ticker").merge(metadata, on="ticker")
    df["sector"] = df["ticker"].map(SECTOR_MAP)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FINBERT MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_finbert():
    """
    Loads FinBERT once and pins it in memory for the whole session.
    cache_resource (not cache_data) because model objects aren't pickle-serializable.
    On HuggingFace Spaces, the model downloads from the hub on first load (~420MB)
    and is then cached to the container's model cache — so restarts are fast.
    """
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()  # we're doing inference only, not fine-tuning
    return tokenizer, model


def run_finbert(text: str, tokenizer, model) -> dict:
    """
    Chunks text into 400-word segments (FinBERT max context is 512 tokens;
    400 words ≈ 480 tokens on average, keeping us under the limit).
    Returns averaged softmax scores across all chunks.

    Same chunking strategy as notebook 02 — single pass truncates 90% of a
    typical earnings call filing.
    """
    words  = text.split()
    chunks = [" ".join(words[i:i + 400]) for i in range(0, len(words), 400) if words[i:i + 400]]
    chunks = chunks[:20]  # cap at 20 chunks (~8000 words) so CPU doesn't time out

    all_probs = []
    with torch.no_grad():
        for chunk in chunks:
            inputs  = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512, padding=True)
            outputs = model(**inputs)
            probs   = F.softmax(outputs.logits, dim=-1).squeeze().numpy()
            all_probs.append(probs)

    if not all_probs:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "sentiment_ratio": 0.0, "n_chunks": 0}

    avg = np.mean(all_probs, axis=0)
    # label order from ProsusAI/finbert model card: 0=positive, 1=negative, 2=neutral
    labels = {model.config.id2label[i].lower(): float(avg[i]) for i in range(3)}
    labels["sentiment_ratio"] = labels["positive"] - labels["negative"]
    labels["n_chunks"] = len(chunks)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# SEC EDGAR UTILITIES  (modern REST API — data.sec.gov, not cgi-bin/browse-edgar)
# ─────────────────────────────────────────────────────────────────────────────
# SEC policy: User-Agent must include a real name + email so they can contact you
# browse-edgar (CGI) aggressively 403s automated requests; data.sec.gov doesn't
EDGAR_HEADERS = {
    "User-Agent": "EarningsSentimentEngine portfolio@dhavalvibhakar.dev",
    "Accept":     "application/json, text/html, */*",
}


@st.cache_data(ttl=3600)
def _edgar_company_data() -> tuple:
    """
    Fetches SEC's official company list once per hour (~500 KB JSON).
    Returns two things from the same HTTP call so we don't hit the endpoint twice:
      ticker_map — {"AAPL": 320193, "C": 831001, ...}  used by fetch_latest_8k
      options    — sorted ["AAPL — Apple Inc.", ...]     used by the search dropdown
    """
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=EDGAR_HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    data       = list(r.json().values())
    ticker_map = {v["ticker"].upper(): v["cik_str"] for v in data}
    options    = sorted(f"{v['ticker']} — {v['title']}" for v in data)
    return ticker_map, options


def fetch_latest_8k(ticker: str) -> tuple:
    """
    Pulls the most recent 8-K filing using EDGAR's modern REST API.

    Old flow (broken): browse-edgar CGI → atom XML → index HTML → document
    New flow: company_tickers.json → submissions JSON → document URL directly

    The submissions endpoint returns structured JSON — no HTML parsing needed
    until we get to the actual filing document itself.
    """
    # ── step 1: ticker → CIK via the official mapping ─────────────────────────
    ticker_map, _ = _edgar_company_data()
    cik = ticker_map.get(ticker.upper())
    if not cik:
        raise ValueError(
            f"**{ticker}** wasn't found in SEC EDGAR's company list. "
            "Check the ticker is correct: use **C** for Citigroup, **BRK-B** for Berkshire, "
            "**META** for Meta Platforms. International listings (e.g. TSM) may not have US 8-Ks."
        )

    # ── step 2: get the company's recent filing list as JSON ──────────────────
    cik_padded = str(cik).zfill(10)
    time.sleep(0.15)  # stay well under EDGAR's 10 req/sec limit
    sub_r = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik_padded}.json",
        headers=EDGAR_HEADERS,
        timeout=20,
    )
    sub_r.raise_for_status()

    recent = sub_r.json()["filings"]["recent"]
    forms        = recent["form"]
    dates        = recent["filingDate"]
    accessions   = recent["accessionNumber"]
    primary_docs = recent["primaryDocument"]

    # walk the list (already sorted newest-first) and grab the first 8-K
    doc_url      = None
    filing_date  = "Unknown"
    for i, form in enumerate(forms):
        if form in ("8-K", "8-K/A"):
            filing_date  = dates[i]
            accession_id = accessions[i].replace("-", "")
            doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                f"{accession_id}/{primary_docs[i]}"
            )
            break

    if not doc_url:
        raise ValueError(f"No 8-K filings found in EDGAR's recent submissions for **{ticker}**.")

    # ── step 3: fetch the document and strip to plain text ────────────────────
    time.sleep(0.15)
    doc_r = requests.get(doc_url, headers=EDGAR_HEADERS, timeout=30)
    doc_r.raise_for_status()
    raw_text = BeautifulSoup(doc_r.text, "html.parser").get_text(separator=" ")
    return re.sub(r"\s+", " ", raw_text).strip(), filing_date


def get_current_price(ticker: str) -> float | None:
    """Quick yfinance lookup — returns None silently if the ticker doesn't exist."""
    try:
        hist = yf.Ticker(ticker).history(period="1d")
        return float(hist["Close"].iloc[-1]) if not hist.empty else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df = load_data()


# ─────────────────────────────────────────────────────────────────────────────
# ══ HEADER ══
# ─────────────────────────────────────────────────────────────────────────────
col_title, col_meta = st.columns([3, 1])

with col_title:
    st.markdown("# 📊 Earnings Call Sentiment Engine")
    st.markdown(
        "**Q1 2026 · 20 S&P 500 companies · FinBERT NLP analysis** — "
        "did what CEOs *said* predict what the stock *did*?"
    )

with col_meta:
    st.markdown("""
    <div style="text-align:right; padding-top:18px; color:#8b949e; font-size:14px; line-height:1.8;">
        <strong style="color:#f0f6fc; font-size:15px;">Dhaval Vibhakar</strong><br>
        Data Science Portfolio<br>
        <a href="https://github.com/wtfdhaval"
           style="color:#1f6feb; text-decoration:none;">
            🔗 GitHub ↗
        </a>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ══ SIDEBAR — sector filter ══
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Filter")
    st.markdown("---")

    sector_choice = st.radio(
        "Sector",
        options=["All", "Tech", "Finance", "Healthcare", "Consumer"],
        index=0,
        help="Filters all charts and metrics below"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:12px; color:#8b949e; line-height:1.8;">
        <strong style="color:#c9d1d9;">Data sources</strong><br>
        📄 SEC EDGAR — 8-K filings<br>
        🤖 ProsusAI/FinBERT — NLP model<br>
        📈 yfinance — price data<br><br>
        <strong style="color:#c9d1d9;">Time period</strong><br>
        Q1 2026 (Apr – May 2026)<br><br>
        <em>20 companies, 20 earnings calls,<br>one counterintuitive finding.</em>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#6e7681;">
        Built with Streamlit + Plotly<br>
        Deployed on HuggingFace Spaces
    </div>
    """, unsafe_allow_html=True)

# apply sector filter — "All" means use the full dataset
filtered = df if sector_choice == "All" else df[df["sector"] == sector_choice]


# ─────────────────────────────────────────────────────────────────────────────
# ══ KPI METRICS ROW ══
# ─────────────────────────────────────────────────────────────────────────────
n = len(filtered)
sector_label = f"({sector_choice})" if sector_choice != "All" else "(all sectors)"
st.markdown(f"### Key Metrics  <span style='font-size:14px; color:#8b949e; font-weight:400;'>{sector_label}</span>", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Companies", f"{n}", help="Number of companies in the current filter")

with m2:
    avg_ratio = filtered["sentiment_ratio"].mean()
    # positive = bullish language, but remember our key finding:
    # this correlates NEGATIVELY with returns (cheap talk hypothesis)
    st.metric(
        "Avg Sentiment Ratio",
        f"{avg_ratio:+.3f}",
        help="Mean(positive − negative score). Higher = more bullish language.",
    )

with m3:
    pct_pos = (filtered["change_1d_pct"] > 0).mean() * 100
    st.metric(
        "Positive Next-Day Return",
        f"{pct_pos:.0f}%",
        help="Fraction of companies with a positive stock return the day after earnings",
    )

with m4:
    best = filtered.loc[filtered["change_1d_pct"].idxmax()]
    st.metric(
        "Best 1-Day Return",
        f"{best['ticker']}  {best['change_1d_pct']:+.2f}%",
        help=f"{best['company']} · {best['sector']}",
    )

with m5:
    worst = filtered.loc[filtered["change_1d_pct"].idxmin()]
    st.metric(
        "Worst 1-Day Return",
        f"{worst['ticker']}  {worst['change_1d_pct']:+.2f}%",
        help=f"{worst['company']} · {worst['sector']}",
    )

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# ══ SECTOR SENTIMENT CHART ══
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Average Sentiment Ratio by Sector")
st.caption(
    "Sentiment ratio = avg(positive) − avg(negative) across all earnings calls in a sector. "
    "Higher = more bullish language. Selected sector is highlighted."
)

sector_avg = (
    df.groupby("sector")["sentiment_ratio"]
    .mean()
    .reset_index()
    .rename(columns={"sentiment_ratio": "avg_ratio"})
)

fig_bar = go.Figure()
for _, row in sector_avg.iterrows():
    is_selected = sector_choice == "All" or row["sector"] == sector_choice
    fig_bar.add_trace(go.Bar(
        x=[row["sector"]],
        y=[row["avg_ratio"]],
        name=row["sector"],
        marker_color=SECTOR_COLORS.get(row["sector"], "#8b949e"),
        opacity=1.0 if is_selected else 0.25,
        text=[f"{row['avg_ratio']:.3f}"],
        textposition="outside",
        textfont=dict(color="#f0f6fc", size=13),
        hovertemplate=(
            f"<b>{row['sector']}</b><br>"
            f"Avg sentiment ratio: {row['avg_ratio']:.4f}<extra></extra>"
        ),
        showlegend=True,
    ))

fig_bar.add_hline(
    y=0, line_dash="dash", line_color="#8b949e", opacity=0.5,
    annotation_text="neutral", annotation_font_color="#8b949e",
)
fig_bar.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#161b22",
    height=350,
    margin=dict(t=30, b=20, l=20, r=20),
    bargap=0.35,
    yaxis=dict(title="Avg Sentiment Ratio", gridcolor="#21262d"),
    xaxis=dict(title="Sector"),
    legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
    showlegend=True,
)
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# ══ SCATTER PLOTS — sentiment vs returns ══
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Sentiment Ratio vs. Stock Returns")
st.caption(
    "Each dot = one company. Red dashed line = linear trend. "
    "If the hypothesis held, the slope would be positive — look what actually happens."
)

fig_scatter = make_subplots(
    rows=1, cols=2,
    subplot_titles=["vs. 1-Day Return", "vs. 1-Week Return"],
    horizontal_spacing=0.10,
)

sectors_in_filter = filtered["sector"].unique()

for sector in ["Tech", "Finance", "Healthcare", "Consumer"]:
    s_data = filtered[filtered["sector"] == sector]
    if s_data.empty:
        continue
    color = SECTOR_COLORS[sector]

    common = dict(
        mode="markers+text",
        name=sector,
        text=s_data["ticker"],
        textposition="top center",
        textfont=dict(size=9, color="#c9d1d9"),
        marker=dict(color=color, size=11, opacity=0.9, line=dict(width=1, color="#0e1117")),
        legendgroup=sector,
    )

    fig_scatter.add_trace(
        go.Scatter(
            x=s_data["sentiment_ratio"],
            y=s_data["change_1d_pct"],
            hovertemplate="<b>%{text}</b><br>Sentiment: %{x:.3f}<br>1D Return: %{y:.2f}%<extra></extra>",
            showlegend=True,
            **common,
        ),
        row=1, col=1,
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=s_data["sentiment_ratio"],
            y=s_data["change_1w_pct"],
            hovertemplate="<b>%{text}</b><br>Sentiment: %{x:.3f}<br>1W Return: %{y:.2f}%<extra></extra>",
            showlegend=False,
            **common,
        ),
        row=1, col=2,
    )

# add trend lines via least-squares fit
for col_idx, ret_col in enumerate(["change_1d_pct", "change_1w_pct"], start=1):
    valid = filtered.dropna(subset=["sentiment_ratio", ret_col])
    if len(valid) >= 3:
        z = np.polyfit(valid["sentiment_ratio"], valid[ret_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid["sentiment_ratio"].min(), valid["sentiment_ratio"].max(), 60)
        fig_scatter.add_trace(
            go.Scatter(
                x=x_line, y=p(x_line),
                mode="lines",
                name=f"Trend ({ret_col})",
                line=dict(color="#f85149", dash="dash", width=2),
                showlegend=(col_idx == 1),
                legendgroup="trend",
                hoverinfo="skip",
            ),
            row=1, col=col_idx,
        )

fig_scatter.update_xaxes(title_text="Sentiment Ratio", gridcolor="#21262d", zeroline=False)
fig_scatter.update_yaxes(title_text="Return (%)", gridcolor="#21262d", zeroline=True, zerolinecolor="#30363d")
fig_scatter.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#161b22",
    height=500,
    margin=dict(t=50, b=30, l=20, r=20),
    legend=dict(orientation="h", y=1.10, x=0.5, xanchor="center"),
    showlegend=True,
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# ══ COMPANY TABLE ══
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Company Details")
st.caption("Sorted by 1-day return by default. Click column headers to re-sort.")

table_df = (
    filtered[["ticker", "company", "sector", "sentiment_ratio", "change_1d_pct", "change_1w_pct", "positive", "negative"]]
    .copy()
    .rename(columns={
        "ticker":           "Ticker",
        "company":          "Company",
        "sector":           "Sector",
        "sentiment_ratio":  "Sentiment Ratio",
        "change_1d_pct":    "1D Return %",
        "change_1w_pct":    "1W Return %",
        "positive":         "Positive Score",
        "negative":         "Negative Score",
    })
    .sort_values("1D Return %", ascending=False)
    .reset_index(drop=True)
)


def _color_returns(series: pd.Series) -> list:
    """Applied column-wise via Styler.apply — green for gains, red for losses."""
    return [
        "color: #3fb950; font-weight: 600" if v > 0
        else "color: #f85149; font-weight: 600" if v < 0
        else "color: #8b949e"
        for v in series
    ]


styled = (
    table_df.style
    .apply(_color_returns, subset=["1D Return %", "1W Return %"])
    .format({
        "Sentiment Ratio":  "{:+.3f}",
        "1D Return %":      "{:+.2f}%",
        "1W Return %":      "{:+.2f}%",
        "Positive Score":   "{:.3f}",
        "Negative Score":   "{:.3f}",
    })
    .set_properties(**{
        "background-color": "#161b22",
        "color":            "#f0f6fc",
        "border":           "1px solid #30363d",
    })
)

st.dataframe(styled, use_container_width=True, height=420)

csv_buf = io.StringIO()
table_df.to_csv(csv_buf, index=False)
st.download_button(
    label="⬇ Download CSV",
    data=csv_buf.getvalue(),
    file_name="earnings_sentiment_q1_2026.csv",
    mime="text/csv",
)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# ══ LIVE ANALYSIS ══
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🔴 Live Sentiment Analysis")
st.markdown(
    "Enter **any** S&P 500 ticker. Already in our dataset of 20? Results show "
    "instantly from pre-computed data. New ticker? The app pulls the latest 8-K "
    "from SEC EDGAR, runs FinBERT, and fetches live price data."
)

# ── helper functions scoped to this section ───────────────────────────────────

def _price_chart(ticker: str) -> go.Figure | None:
    """
    Fetches 3 months of closing prices from yfinance and returns a Plotly figure.
    We do this for both the fast path (pre-computed) and the live path so the
    chart is always showing real-time data, not the Q1 snapshot.
    """
    try:
        hist = yf.Ticker(ticker).history(period="3mo")
        if hist.empty:
            return None

        first_close = hist["Close"].iloc[0]
        last_close  = hist["Close"].iloc[-1]
        pct_3m      = (last_close - first_close) / first_close * 100
        trend_color = "#3fb950" if pct_3m >= 0 else "#f85149"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist["Close"],
            mode="lines",
            line=dict(color=trend_color, width=2),
            fill="tozeroy",
            fillcolor=f"rgba({'63,185,80' if pct_3m >= 0 else '248,81,73'}, 0.07)",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>$%{y:.2f}<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#161b22",
            height=240,
            margin=dict(t=8, b=8, l=8, r=8),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(gridcolor="#21262d", title="Price (USD)"),
            showlegend=False,
        )
        return fig, pct_3m
    except Exception:
        return None, None


def _show_sentiment_metrics(pos: float, neg: float, neu: float, ratio: float, label_suffix: str = ""):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Positive", f"{pos:.1%}")
    c2.metric("Negative", f"{neg:.1%}")
    c3.metric("Neutral",  f"{neu:.1%}")
    c4.metric(f"Sentiment Ratio{label_suffix}", f"{ratio:+.3f}")


def _show_price_section(ticker: str, current_price: float | None):
    """Current price metric on the left, 3-month chart on the right."""
    pc, cc = st.columns([1, 3])
    with pc:
        if current_price:
            st.metric("Current Price", f"${current_price:.2f}")
        else:
            st.caption("Live price unavailable")
    with cc:
        result = _price_chart(ticker)
        fig, pct_3m = result if result else (None, None)
        if fig:
            pct_label = f"  {pct_3m:+.2f}% (3 mo)" if pct_3m is not None else ""
            st.caption(f"**{ticker}** 3-month price history{pct_label}")
            st.plotly_chart(fig, use_container_width=True)


def _show_sector_comparison(ticker: str, ratio: float, source: str = "this filing"):
    sector = SECTOR_MAP.get(ticker)
    if sector:
        sector_mean = df[df["sector"] == sector]["sentiment_ratio"].mean()
        delta     = ratio - sector_mean
        direction = "above" if delta > 0 else "below"
        tone = "More bullish language than sector peers." if delta > 0 else "More cautious language than sector peers."
        st.markdown(
            f"**Sector comparison ({sector}):** {source.capitalize()} sentiment ratio is "
            f"**{abs(delta):.3f} {direction}** the {sector} average "
            f"({sector_mean:.3f} from Q1 2026 dataset). {tone}"
        )
    else:
        st.markdown(f"**{ticker}** is not in our Q1 2026 dataset — no sector benchmark to compare against.")


# ── input row ─────────────────────────────────────────────────────────────────
# load the company list for the dropdown — same cached call fetch_latest_8k uses,
# so no extra HTTP request. ~13,000 companies; selectbox filters as you type.
_, company_options = _edgar_company_data()

live_col1, live_col2 = st.columns([3, 1])
with live_col1:
    selection = st.selectbox(
        "Company",
        options=company_options,
        index=None,
        placeholder="Search by ticker or name — e.g. 'C', 'Citigroup', 'COST', 'Netflix'…",
        label_visibility="collapsed",
    )
    live_ticker = selection.split(" — ")[0].strip() if selection else ""
with live_col2:
    run_btn = st.button("Analyze", type="primary", use_container_width=True)

# ── logic ─────────────────────────────────────────────────────────────────────
if run_btn and live_ticker:

    known_tickers = set(df["ticker"].tolist())

    if live_ticker in known_tickers:
        # ── FAST PATH — pre-computed, no model needed ─────────────────────────
        row = df[df["ticker"] == live_ticker].iloc[0]
        st.success(
            f"**{live_ticker} ({row['company']})** — already in our Q1 2026 dataset · "
            f"filing date: **{row['date']}** · showing pre-computed results instantly"
        )

        _show_sentiment_metrics(
            pos=row["positive"], neg=row["negative"],
            neu=row["neutral"],  ratio=row["sentiment_ratio"],
            label_suffix=f" ({int(row['chunks'])} chunks)",
        )

        # show the actual Q1 returns — useful context alongside the sentiment scores
        ra, rb = st.columns(2)
        ra.metric("1-Day Return (Q1 2026 actual)",  f"{row['change_1d_pct']:+.2f}%",
                  delta=None, help="Return the day after the Q1 2026 earnings release")
        rb.metric("1-Week Return (Q1 2026 actual)", f"{row['change_1w_pct']:+.2f}%",
                  delta=None, help="Return one week after the Q1 2026 earnings release")

        st.divider()
        current_price = get_current_price(live_ticker)
        _show_price_section(live_ticker, current_price)
        _show_sector_comparison(live_ticker, row["sentiment_ratio"], source="the Q1 2026 earnings call")

    elif not HAS_FINBERT:
        st.error(
            f"**{live_ticker}** isn't in our Q1 2026 dataset, and FinBERT isn't installed "
            "in this environment — live analysis unavailable. "
            "Try one of the 20 tickers from the table above for instant results."
        )

    else:
        # ── LIVE PATH — new ticker: EDGAR → FinBERT → yfinance ───────────────
        st.info(
            "**Heads up:** First-time FinBERT load downloads ~420 MB — takes 1–2 minutes "
            "on HuggingFace Spaces. After that it stays cached for the session.",
            icon="ℹ️",
        )
        progress = st.progress(0, text="Initializing…")
        try:
            with st.spinner("Loading FinBERT model (first run: 1–2 min) …"):
                tokenizer, model = load_finbert()
            progress.progress(20, text="Model ready. Fetching 8-K from SEC EDGAR…")

            raw_text, filing_date = fetch_latest_8k(live_ticker)
            n_est = len(raw_text.split()) // 400 + 1
            progress.progress(55, text=f"Got filing ({filing_date}). Running FinBERT across ~{n_est} chunks…")

            if len(raw_text.split()) < 50:
                progress.empty()
                st.error(f"Filing for {live_ticker} has almost no text — might be a binary or XBRL-only document.")
            else:
                results = run_finbert(raw_text, tokenizer, model)
                progress.progress(90, text="Fetching current stock price and 3-month chart…")
                current_price = get_current_price(live_ticker)
                progress.progress(100, text="Done!")
                time.sleep(0.4)
                progress.empty()

                st.success(
                    f"**{live_ticker}** — most recent 8-K filed **{filing_date}** · "
                    f"analyzed {results['n_chunks']} text chunks"
                )

                _show_sentiment_metrics(
                    pos=results["positive"], neg=results["negative"],
                    neu=results["neutral"],  ratio=results["sentiment_ratio"],
                    label_suffix=f" ({results['n_chunks']} chunks)",
                )

                st.divider()
                _show_price_section(live_ticker, current_price)
                _show_sector_comparison(live_ticker, results["sentiment_ratio"])

                with st.expander("View raw text sample (first 600 chars)"):
                    st.code(raw_text[:600] + " …", language=None)

        except requests.exceptions.HTTPError as e:
            progress.empty()
            st.error(f"SEC EDGAR returned an HTTP error for {live_ticker}: {e}")
        except ValueError as e:
            progress.empty()
            # ValueError messages from fetch_latest_8k contain markdown formatting
            st.error(str(e), icon="🚫")
        except Exception as e:
            progress.empty()
            st.error(f"Unexpected error: {e}")
            st.info("Make sure the ticker is a real SEC-registered US company (e.g. AAPL, not AAPL.O).")

elif run_btn and not live_ticker:
    st.warning("Enter a ticker symbol first.")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# ══ KEY FINDINGS ══
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Key Findings")
st.caption("What did 20 earnings calls, 3 NLP models, and real market data reveal?")

# compute sector performance dynamically so cards stay accurate if data changes
sector_1d = df.groupby("sector")["change_1d_pct"].mean()
best_sector  = sector_1d.idxmax()
worst_sector = sector_1d.idxmin()
best_val     = sector_1d.max()
worst_val    = sector_1d.min()

fc1, fc2 = st.columns(2)

with fc1:
    st.markdown("""
    <div class="finding-card">
      <h4>🗣️ The Cheap Talk Hypothesis — Confirmed</h4>
      <p>Higher positive sentiment in earnings calls <strong>correlated with worse
      subsequent stock performance</strong>. CEOs who sounded most optimistic tended
      to disappoint the market the next day. Investors seem to discount bullish
      language, or it peaks exactly when the underlying numbers are weakest.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="finding-card info">
      <h4>🚀 Top Outlier: GOOGL +9.96%</h4>
      <p>Alphabet bucked the trend entirely — moderate sentiment ratio (0.079) but
      the market rewarded them with a nearly 10% single-day gain. Cloud + AI revenue
      surprise drove the move; the earnings <em>call</em> was almost incidental.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="finding-card">
      <h4>🏆 Best Sector: {best_sector} (+{best_val:.2f}% avg 1D return)</h4>
      <p>The {best_sector} sector delivered the strongest average next-day return
      across all earnings calls. Conservative, metrics-focused language paired with
      solid fundamentals — a combination that markets reliably reward.</p>
    </div>
    """, unsafe_allow_html=True)

with fc2:
    st.markdown("""
    <div class="finding-card warning">
      <h4>📉 Worst Outlier: META −8.55%</h4>
      <p>Meta had a moderate sentiment ratio (0.110) but fell sharply — investors
      focused on advertising revenue guidance and operating cost concerns, not the
      overall tone. One more data point against using sentiment as a trading signal
      in isolation.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="finding-card danger">
      <h4>⚠️ Worst Sector: {worst_sector} ({worst_val:+.2f}% avg 1D return)</h4>
      <p>The {worst_sector} sector had the weakest average next-day return — a mix
      of macro headwinds, high analyst expectations, and elevated pre-earnings
      sentiment ratios that left little room for upside surprise.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="finding-card neutral">
      <h4>🤖 FinBERT Limitation Worth Knowing</h4>
      <p>Every single company got labeled <em>Neutral</em> by FinBERT's headline
      class — boilerplate earnings call language is too hedged for a binary
      positive/negative split. The continuous <strong>sentiment ratio</strong>
      (positive − negative score) is where the real signal lives.</p>
    </div>
    """, unsafe_allow_html=True)

# ─── footer ───────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.caption(
    "Data: SEC EDGAR (Q1 2026 8-K filings)  ·  "
    "NLP: ProsusAI/FinBERT  ·  "
    "Prices: yfinance  ·  "
    "Built by **Dhaval Vibhakar**  ·  "
    "Source: [github.com/wtfdhaval](https://github.com/wtfdhaval)"
)
