---
title: Earnings Sentiment Engine
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.32.0"
python_version: "3.9"
app_file: app.py
pinned: false
---

# Earnings Call Sentiment Engine

Analyzes Q1 2026 earnings call transcripts for 20 S&P 500 companies
using FinBERT — a BERT model fine-tuned on financial text.

## Key Finding
Higher positive sentiment in earnings calls correlated with WORSE 
stock performance — the "cheap talk" hypothesis confirmed.

## Tech Stack
Python · FinBERT · SEC EDGAR API · yfinance · Streamlit · Plotly

## Data
- 20 S&P 500 companies across Tech, Finance, Healthcare, Consumer
- Real 8-K filings from SEC EDGAR
- Q1 2026 earnings releases

Built by Dhaval Vibhakar
