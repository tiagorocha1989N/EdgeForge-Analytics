# EdgeForge Analytics
### Statistical Advantage Matrix for Bitcoin Trading Strategies

**Author:** Tiago Rocha  
**Ticker:** INDEX:BTCUSD  
**Backtest Period:** 01/01/2018 → 01/03/2026 (2,982 days)  

---

## 📊 About This Study

EdgeForge Analytics is a quantitative research project that evaluates the **statistical robustness** of 16 trading indicators applied to Bitcoin (INDEX:BTCUSD) through exhaustive combinatorial backtesting.

### Methodology

- **Total Combinations Tested:** 65,535 (all non-zero 16-bit binary masks)
- **Entry Condition:** `longPercentage > 50`
- **Exit Condition:** `longPercentage < 50`
- **Capital:** $1,000,000 (100% equity per trade)
- **Commission:** 0%
- **Slippage:** 1 tick

### Indicators Tested
16 trading indicators (Indicator_1 through Indicator_16), each represented as a binary bit in the combination mask.

### Performance Classification (AdvancedMetrics Library)

| Label | Score | Criteria |
|-------|-------|----------|
| ⚡ SSSlapper | ≥ 6 | Top-tier performance |
| 🤨 Mid | 2–5 | Acceptable performance |
| 💩 Shit | < 2 | Poor performance |

### Metrics Tracked
- Equity Max DD, Sharpe Ratio, Sortino Ratio, Omega Ratio
- Profit Factor, Profitable %, Half Kelly %, Trades, Net Profit %
- Calmar Ratio, Recovery Factor, Profit per Trade

---

## 🚀 Live Dashboard

[👉 Open EdgeForge Analytics Dashboard](https://edgeforge.streamlit.app)

---

## 📁 Repository Structure

```
EdgeForge-Analytics/
├── app.py                    # Main Streamlit dashboard
├── data/
│   └── results_tradingview.csv  # Full backtest results (65,535 combinations)
├── requirements.txt
└── README.md
```

---

## ⚠️ Disclaimer

This study is for **educational and research purposes only**.  
It does **not** constitute financial advice or investment recommendations.  
Past performance does not guarantee future results.

---

*© 2026 Tiago Rocha — EdgeForge Analytics*
