"""
EdgeForge Analytics — Statistical Advantage Matrix
Author: Tiago Rocha
Ticker: INDEX:BTCUSD | 2018-2026
⚠️ Not financial advice
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EdgeForge Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0a0a1a; }
    .stApp { background-color: #0a0a1a; }
    h1, h2, h3 { color: #00d4ff; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #00d4ff33;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .ssslapper { color: #00ff88; font-weight: bold; }
    .mid { color: #ff8c00; font-weight: bold; }
    .shit { color: #ff4444; font-weight: bold; }
    .header-box {
        background: linear-gradient(135deg, #0f3460, #16213e);
        border-left: 4px solid #00d4ff;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("data/results_tradingview.csv", on_bad_lines='skip')

    def decode_mask(mask_str):
        mask = str(mask_str).zfill(16)
        return [i+1 for i, b in enumerate(mask) if b == '1']

    df['mask']         = df['combination'].astype(str).str.zfill(16)
    df['active_inds']  = df['mask'].apply(decode_mask)
    df['n_indicators'] = df['active_inds'].apply(len)
    df['active_str']   = df['active_inds'].apply(
        lambda x: ' + '.join(f'Ind_{i}' for i in x))

    # Additional metrics
    df['calmar']           = (df['net_profit'] / df['equity_max_dd'].abs()).round(2)
    df['recovery_factor']  = (df['net_profit'] / df['equity_max_dd'].abs()).round(2)
    df['profit_per_trade'] = (df['net_profit'] / df['trades']).round(2)

    # Slap Score
    def slap_score(r):
        s = 0
        s += 1 if r['equity_max_dd'] >= -25 else (-1 if r['equity_max_dd'] <= -45 else 0)
        s += 1 if r['sharpe']        >= 2.0  else (-1 if r['sharpe']        <= 1.0  else 0)
        s += 1 if r['sortino']       >= 3.0  else (-1 if r['sortino']       <= 2.0  else 0)
        s += 1 if r['omega']         >= 1.35 else (-1 if r['omega']         <= 1.0  else 0)
        s += 1 if r['profit_factor'] >= 4.0  else (-1 if r['profit_factor'] <= 2.0  else 0)
        s += 1 if r['profitable']    >= 55   else (-1 if r['profitable']    <= 40   else 0)
        s += 1 if r['half_kelly']    >= 15   else (-1 if r['half_kelly']    <= 5    else 0)
        s += 1 if 45 <= r['trades'] <= 105   else -1
        return s

    df['slap_score'] = df.apply(slap_score, axis=1)
    df['label']      = df['slap_score'].apply(
        lambda s: 'SSSlapper' if s >= 6 else ('Mid' if s >= 2 else 'Shit'))

    return df

# ══════════════════════════════════════════════════════════════════════════════
# STATS HELPER
# ══════════════════════════════════════════════════════════════════════════════
def calc_stats(series):
    s = series.dropna()
    if len(s) == 0:
        return {}
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr    = q3 - q1
    sigma  = s.std()
    mean   = s.mean()
    outliers = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
    return {
        'Mean': round(mean, 4),
        'Median': round(s.median(), 4),
        'Min': round(s.min(), 4),
        'Max': round(s.max(), 4),
        'Range': round(s.max() - s.min(), 4),
        'Std (σ)': round(sigma, 4),
        '2σ': round(2*sigma, 4),
        '3σ': round(3*sigma, 4),
        'IQR': round(iqr, 4),
        'Kurtosis': round(s.kurtosis(), 4),
        'Skewness': round(s.skew(), 4),
        'CV (%)': round(sigma / abs(mean) * 100, 2) if mean != 0 else None,
        'Outliers N': int(outliers),
        'Outliers %': round(outliers / len(s) * 100, 2),
    }

def stats_table(df_sub, metrics):
    rows = []
    for m in metrics:
        if m not in df_sub.columns:
            continue
        s = calc_stats(df_sub[m])
        s['Metric'] = m.replace('_', ' ').title()
        rows.append(s)
    return pd.DataFrame(rows).set_index('Metric')

# ══════════════════════════════════════════════════════════════════════════════
# BELL CURVE PLOT
# ══════════════════════════════════════════════════════════════════════════════
def bell_curve_plot(df_sub, metric, title):
    s = df_sub[metric].dropna()
    if len(s) < 10:
        return go.Figure()

    counts, bins = np.histogram(s, bins=40, density=True)
    bin_centers  = (bins[:-1] + bins[1:]) / 2

    # Normal fit
    mu, sigma = s.mean(), s.std()
    x_norm    = np.linspace(s.min(), s.max(), 200)
    y_norm    = stats.norm.pdf(x_norm, mu, sigma)

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Bar(
        x=bin_centers, y=counts,
        name='Distribution',
        marker_color='rgba(0, 212, 255, 0.6)',
        marker_line_color='rgba(0, 212, 255, 1)',
        marker_line_width=1,
    ))

    # Normal curve
    fig.add_trace(go.Scatter(
        x=x_norm, y=y_norm,
        name='Normal Fit',
        line=dict(color='#ff6b35', width=2),
    ))

    # Mean line
    fig.add_vline(x=mu, line_dash="dash",
                  line_color="#00ff88",
                  annotation_text=f"μ={mu:.2f}",
                  annotation_position="top right")

    # ±1σ
    fig.add_vline(x=mu+sigma, line_dash="dot", line_color="#ffdd00", line_width=1)
    fig.add_vline(x=mu-sigma, line_dash="dot", line_color="#ffdd00", line_width=1)

    fig.update_layout(
        title=title,
        paper_bgcolor='#0a0a1a',
        plot_bgcolor='#0f1923',
        font=dict(color='#cccccc'),
        legend=dict(bgcolor='#1a1a2e'),
        height=350,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(gridcolor='#1a1a2e'),
        yaxis=dict(gridcolor='#1a1a2e'),
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/nolan/64/lightning-bolt.png", width=60)
        st.title("⚡ EdgeForge")
        st.caption("Statistical Advantage Matrix")
        st.divider()

        page = st.radio("Navigation", [
            "🏠 Overview",
            "📊 All Combinations",
            "🔢 By N Indicators",
            "✅ Inclusion Analysis",
            "❌ Exclusion Analysis",
            "🔔 Bell Curves",
            "🏆 Top Results",
        ])

        st.divider()
        st.caption("**Study Details**")
        st.caption("📈 INDEX:BTCUSD")
        st.caption("📅 2018-01-01 → 2026-03-01")
        st.caption("⏱️ 2,982 days")
        st.caption("🔢 65,535 combinations")
        st.divider()
        st.caption("**Entry:** longPercentage > 50")
        st.caption("**Exit:** longPercentage < 50")
        st.divider()
        st.caption("*© 2026 Tiago Rocha*")
        st.caption("*Not financial advice*")

    return page

# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════
def overview_comparison_table(df, rows_config, title, col_label='Type'):
    METRICS = ['equity_max_dd','sharpe','sortino','omega',
               'profit_factor','profitable','half_kelly','trades','net_profit']
    METRIC_COLS = ['Equity Max DD','Sharpe','Sortino','Omega',
                   'Profit Factor','Profitable','Half Kelly','Trades','Net Profit']
    rows = []
    for lbl, subset in rows_config:
        row = {col_label: lbl}
        for m, col in zip(METRICS, METRIC_COLS):
            row[col] = round(subset[m].mean(), 3) if m in subset.columns and len(subset) > 0 else None
        rows.append(row)
    table_df = pd.DataFrame(rows).set_index(col_label)
    st.subheader(title)
    st.dataframe(table_df, use_container_width=True)


def page_overview(df):
    st.markdown("""
    <div class="header-box">
        <h1>⚡ EdgeForge Analytics</h1>
        <h3>Statistical Advantage Matrix for Bitcoin Trading Strategies</h3>
        <p>Author: <strong>Tiago Rocha</strong> | Ticker: <strong>INDEX:BTCUSD</strong> | Period: <strong>2018–2026</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    n_slap = len(df[df.label=='SSSlapper'])
    n_mid  = len(df[df.label=='Mid'])
    n_shit = len(df[df.label=='Shit'])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Combinations", f"{len(df):,}")
    c2.metric("⚡ SSSlapper", f"{n_slap:,}", f"{n_slap/len(df)*100:.1f}%")
    c3.metric("🤨 Mid", f"{n_mid:,}", f"{n_mid/len(df)*100:.1f}%")
    c4.metric("💩 Shit", f"{n_shit:,}", f"{n_shit/len(df)*100:.1f}%")
    c5.metric("Max Slap Score", f"{df.slap_score.max()}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Pie(
            labels=['SSSlapper', 'Mid', 'Shit'],
            values=[n_slap, n_mid, n_shit],
            marker_colors=['#00ff88', '#ff8c00', '#ff4444'], hole=0.4,
        ))
        fig.update_layout(title="Performance Distribution",
                          paper_bgcolor='#0a0a1a', font=dict(color='#cccccc'), height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        n_counts = df.groupby('n_indicators').size().reset_index(name='count')
        fig2 = px.bar(n_counts, x='n_indicators', y='count',
                      title="Combinations by Number of Active Indicators",
                      color='count', color_continuous_scale='Blues')
        fig2.update_layout(paper_bgcolor='#0a0a1a', plot_bgcolor='#0f1923',
                           font=dict(color='#cccccc'), height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Stats overview
    st.subheader("📊 Statistical Summary — All Combinations")
    metrics = ['equity_max_dd','sharpe','sortino','omega',
               'profit_factor','profitable','half_kelly','trades','net_profit']
    st.dataframe(stats_table(df, metrics), use_container_width=True)

    st.divider()

    # ── Bell Curve All Combinations ──────────────────────────────────
    st.subheader("🔔 Bell Curve — All Combinations")
    bell_all = st.selectbox("Select Metric", metrics,
        format_func=lambda x: x.replace('_',' ').title(), key="bell_all")
    st.plotly_chart(bell_curve_plot(df, bell_all,
        f"Distribution — {bell_all.replace('_',' ').title()} (All Combinations)"),
        use_container_width=True)

    st.divider()

    # ── Table 1: By N Combinations ───────────────────────────────────
    rows_n = [("All Combinations", df)] + \
             [(f"Combination {n}", df[df['n_indicators'] == n]) for n in range(1, 17)]
    overview_comparison_table(df, rows_n,
        "📊 Performance by N Active Indicators (Mean Values)", "Type Combination")

    st.subheader("🔔 Bell Curve — By N Active Indicators")
    col1, col2 = st.columns(2)
    with col1:
        bell_n_m = st.selectbox("Select Metric", metrics,
            format_func=lambda x: x.replace('_',' ').title(), key="bell_n_m")
    with col2:
        bell_n_n = st.selectbox("N Indicators", list(range(1, 17)), key="bell_n_n")
    st.plotly_chart(bell_curve_plot(df[df['n_indicators'] == bell_n_n], bell_n_m,
        f"Distribution — {bell_n_m.replace('_',' ').title()} | N={bell_n_n}"),
        use_container_width=True)

    st.divider()

    # ── Table 2: Inclusion Analysis ──────────────────────────────────
    rows_inc = [("All Combinations", df)] + \
               [(f"Indicator {i}", df[df['active_inds'].apply(lambda x: i in x)])
                for i in range(1, 17)]
    overview_comparison_table(df, rows_inc,
        "📊 Inclusion Analysis — Mean Performance per Indicator", "Indicator")

    st.subheader("🔔 Bell Curve — Inclusion Analysis")
    col1, col2 = st.columns(2)
    with col1:
        bell_inc_m = st.selectbox("Select Metric", metrics,
            format_func=lambda x: x.replace('_',' ').title(), key="bell_inc_m")
    with col2:
        bell_inc_i = st.selectbox("Indicator", list(range(1, 17)), key="bell_inc_i")
    df_inc = df[df['active_inds'].apply(lambda x: bell_inc_i in x)]
    st.plotly_chart(bell_curve_plot(df_inc, bell_inc_m,
        f"Distribution — {bell_inc_m.replace('_',' ').title()} | Indicator {bell_inc_i} Included"),
        use_container_width=True)

    st.divider()

    # ── Table 3: Exclusion Analysis ──────────────────────────────────
    rows_exc = [("All Combinations", df)] + \
               [(f"Exclusion Analysis Indicator {i}",
                 df[df['active_inds'].apply(lambda x: i not in x)])
                for i in range(1, 17)]
    overview_comparison_table(df, rows_exc,
        "📊 Exclusion Analysis — Mean Performance per Indicator", "Indicator")

    st.subheader("🔔 Bell Curve — Exclusion Analysis")
    col1, col2 = st.columns(2)
    with col1:
        bell_exc_m = st.selectbox("Select Metric", metrics,
            format_func=lambda x: x.replace('_',' ').title(), key="bell_exc_m")
    with col2:
        bell_exc_i = st.selectbox("Indicator", list(range(1, 17)), key="bell_exc_i")
    df_exc = df[df['active_inds'].apply(lambda x: bell_exc_i not in x)]
    st.plotly_chart(bell_curve_plot(df_exc, bell_exc_m,
        f"Distribution — {bell_exc_m.replace('_',' ').title()} | Indicator {bell_exc_i} Excluded"),
        use_container_width=True)

    st.divider()

    with st.expander("📖 Study Methodology"):
        st.markdown("""
        ### EdgeForge Analytics — Methodology

        **Objective:** Identify statistically robust trading strategies through multiple performance lenses.

        **Approach:** Exhaustive combinatorial backtesting of all 65,535 non-zero 16-bit binary combinations of 16 trading indicators.

        **Combination Encoding:**
        - Each combination is represented as a 16-bit binary mask
        - `1000000000000000` = Only Indicator_1 active
        - `1100000000000000` = Indicators 1 and 2 active
        - `1111111111111111` = All 16 indicators active

        **Performance Classification (AdvancedMetrics):**

        | Metric | SSSlapper (+1) | Shit (-1) |
        |--------|---------------|-----------|
        | Equity Max DD | ≥ -25% | ≤ -45% |
        | Sharpe Ratio | ≥ 2.0 | ≤ 1.0 |
        | Sortino Ratio | ≥ 3.0 | ≤ 2.0 |
        | Omega Ratio | ≥ 1.35 | ≤ 1.0 |
        | Profit Factor | ≥ 4.0 | ≤ 2.0 |
        | Profitable % | ≥ 55% | ≤ 40% |
        | Half Kelly % | ≥ 15% | ≤ 5% |
        | Trades | 45–105 | Outside range |

        **Final Score:** SSSlapper ≥ 6 | Mid 2–5 | Shit < 2

        ⚠️ *This study is for educational purposes only and does not constitute financial advice.*
        """)


def page_all_combinations(df):
    st.title("📊 All Combinations")
    st.caption(f"{len(df):,} total combinations")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        label_filter = st.multiselect("Label", ['SSSlapper','Mid','Shit'],
                                       default=['SSSlapper','Mid','Shit'])
    with col2:
        n_filter = st.slider("N Indicators", 1, 16, (1, 16))
    with col3:
        sharpe_filter = st.slider("Min Sharpe", 0.0, 3.0, 0.0, 0.1)

    df_f = df[
        (df.label.isin(label_filter)) &
        (df.n_indicators >= n_filter[0]) &
        (df.n_indicators <= n_filter[1]) &
        (df.sharpe >= sharpe_filter)
    ]

    st.caption(f"Showing {len(df_f):,} combinations")

    display_cols = ['mask','active_str','n_indicators','equity_max_dd','sharpe',
                    'sortino','omega','profit_factor','profitable','half_kelly',
                    'trades','net_profit','calmar','slap_score','label']

    st.dataframe(df_f[display_cols], use_container_width=True, height=500)

    # Stats
    st.subheader("📊 Statistical Summary")
    metrics = ['equity_max_dd','sharpe','sortino','omega',
               'profit_factor','profitable','half_kelly','trades','net_profit']
    st.dataframe(stats_table(df_f, metrics), use_container_width=True)


def page_by_n(df):
    st.title("🔢 Analysis by Number of Indicators")

    n_sel = st.selectbox("Select N Indicators", list(range(1, 17)))
    df_n  = df[df.n_indicators == n_sel]

    st.caption(f"**N={n_sel}:** {len(df_n):,} combinations | "
               f"SSSlapper: {len(df_n[df_n.label=='SSSlapper'])} | "
               f"Mid: {len(df_n[df_n.label=='Mid'])} | "
               f"Shit: {len(df_n[df_n.label=='Shit'])}")

    # Distribution
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Pie(
            labels=['SSSlapper','Mid','Shit'],
            values=[len(df_n[df_n.label=='SSSlapper']),
                    len(df_n[df_n.label=='Mid']),
                    len(df_n[df_n.label=='Shit'])],
            marker_colors=['#00ff88','#ff8c00','#ff4444'],
            hole=0.4,
        ))
        fig.update_layout(title=f"N={n_sel} Label Distribution",
                          paper_bgcolor='#0a0a1a', font=dict(color='#cccccc'), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.histogram(df_n, x='sharpe', color='label',
                            color_discrete_map={'SSSlapper':'#00ff88','Mid':'#ff8c00','Shit':'#ff4444'},
                            title=f"N={n_sel} Sharpe Distribution", nbins=30)
        fig2.update_layout(paper_bgcolor='#0a0a1a', plot_bgcolor='#0f1923',
                           font=dict(color='#cccccc'), height=300)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📊 Statistical Summary")
    metrics = ['equity_max_dd','sharpe','sortino','omega',
               'profit_factor','profitable','half_kelly','trades','net_profit']
    st.dataframe(stats_table(df_n, metrics), use_container_width=True)

    st.subheader("📋 All Combinations")
    display_cols = ['mask','active_str','equity_max_dd','sharpe','sortino',
                    'omega','profit_factor','profitable','trades','net_profit','slap_score','label']
    st.dataframe(df_n[display_cols], use_container_width=True, height=400)


def page_inclusion(df):
    st.title("✅ Inclusion Analysis")
    st.caption("Performance of all combinations where Indicator X is active")

    ind_sel = st.selectbox("Select Indicator", [f"Indicator_{i}" for i in range(1, 17)])
    ind_num = int(ind_sel.split('_')[1])

    df_inc = df[df['active_inds'].apply(lambda x: ind_num in x)]

    st.caption(f"**{ind_sel} included:** {len(df_inc):,} combinations | "
               f"SSSlapper: {len(df_inc[df_inc.label=='SSSlapper'])} | "
               f"Mid: {len(df_inc[df_inc.label=='Mid'])} | "
               f"Shit: {len(df_inc[df_inc.label=='Shit'])}")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Pie(
            labels=['SSSlapper','Mid','Shit'],
            values=[len(df_inc[df_inc.label=='SSSlapper']),
                    len(df_inc[df_inc.label=='Mid']),
                    len(df_inc[df_inc.label=='Shit'])],
            marker_colors=['#00ff88','#ff8c00','#ff4444'], hole=0.4,
        ))
        fig.update_layout(title=f"{ind_sel} Included — Label Distribution",
                          paper_bgcolor='#0a0a1a', font=dict(color='#cccccc'), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Compare metrics with/without indicator
        df_exc = df[df['active_inds'].apply(lambda x: ind_num not in x)]
        compare_metrics = ['sharpe','sortino','profit_factor','profitable','net_profit']
        comp_data = pd.DataFrame({
            f'{ind_sel} Active': [df_inc[m].mean().round(3) for m in compare_metrics],
            f'{ind_sel} Inactive': [df_exc[m].mean().round(3) for m in compare_metrics],
        }, index=[m.replace('_',' ').title() for m in compare_metrics])

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name=f'{ind_sel} Active',
                              x=comp_data.index, y=comp_data[f'{ind_sel} Active'],
                              marker_color='#00d4ff'))
        fig2.add_trace(go.Bar(name=f'{ind_sel} Inactive',
                              x=comp_data.index, y=comp_data[f'{ind_sel} Inactive'],
                              marker_color='#ff6b35'))
        fig2.update_layout(title="Active vs Inactive — Mean Metrics",
                           paper_bgcolor='#0a0a1a', plot_bgcolor='#0f1923',
                           font=dict(color='#cccccc'), height=300, barmode='group')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📊 Statistical Summary")
    metrics = ['equity_max_dd','sharpe','sortino','omega',
               'profit_factor','profitable','half_kelly','trades','net_profit']
    st.dataframe(stats_table(df_inc, metrics), use_container_width=True)

    st.subheader("📋 Combinations")
    display_cols = ['mask','active_str','n_indicators','equity_max_dd','sharpe',
                    'sortino','profit_factor','profitable','trades','net_profit','slap_score','label']
    st.dataframe(df_inc[display_cols], use_container_width=True, height=400)


def page_exclusion(df):
    st.title("❌ Exclusion Analysis")
    st.caption("Performance of all combinations where Indicator X is NOT active")

    ind_sel = st.selectbox("Select Indicator", [f"Indicator_{i}" for i in range(1, 17)])
    ind_num = int(ind_sel.split('_')[1])

    df_exc = df[df['active_inds'].apply(lambda x: ind_num not in x)]

    st.caption(f"**{ind_sel} excluded:** {len(df_exc):,} combinations | "
               f"SSSlapper: {len(df_exc[df_exc.label=='SSSlapper'])} | "
               f"Mid: {len(df_exc[df_exc.label=='Mid'])} | "
               f"Shit: {len(df_exc[df_exc.label=='Shit'])}")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Pie(
            labels=['SSSlapper','Mid','Shit'],
            values=[len(df_exc[df_exc.label=='SSSlapper']),
                    len(df_exc[df_exc.label=='Mid']),
                    len(df_exc[df_exc.label=='Shit'])],
            marker_colors=['#00ff88','#ff8c00','#ff4444'], hole=0.4,
        ))
        fig.update_layout(title=f"{ind_sel} Excluded — Label Distribution",
                          paper_bgcolor='#0a0a1a', font=dict(color='#cccccc'), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        df_inc = df[df['active_inds'].apply(lambda x: ind_num in x)]
        compare_metrics = ['sharpe','sortino','profit_factor','profitable','net_profit']
        comp_data = pd.DataFrame({
            f'{ind_sel} Active': [df_inc[m].mean().round(3) for m in compare_metrics],
            f'{ind_sel} Inactive': [df_exc[m].mean().round(3) for m in compare_metrics],
        }, index=[m.replace('_',' ').title() for m in compare_metrics])

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name=f'{ind_sel} Active',
                              x=comp_data.index, y=comp_data[f'{ind_sel} Active'],
                              marker_color='#00d4ff'))
        fig2.add_trace(go.Bar(name=f'{ind_sel} Inactive',
                              x=comp_data.index, y=comp_data[f'{ind_sel} Inactive'],
                              marker_color='#ff6b35'))
        fig2.update_layout(title="Active vs Inactive — Mean Metrics",
                           paper_bgcolor='#0a0a1a', plot_bgcolor='#0f1923',
                           font=dict(color='#cccccc'), height=300, barmode='group')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📊 Statistical Summary")
    metrics = ['equity_max_dd','sharpe','sortino','omega',
               'profit_factor','profitable','half_kelly','trades','net_profit']
    st.dataframe(stats_table(df_exc, metrics), use_container_width=True)

    st.subheader("📋 Combinations")
    display_cols = ['mask','active_str','n_indicators','equity_max_dd','sharpe',
                    'sortino','profit_factor','profitable','trades','net_profit','slap_score','label']
    st.dataframe(df_exc[display_cols], use_container_width=True, height=400)


def page_bell_curves(df):
    st.title("🔔 Bell Curves — Metric Distributions")

    # Filter by label
    label_filter = st.multiselect("Filter by Label",
                                   ['SSSlapper','Mid','Shit'],
                                   default=['SSSlapper','Mid','Shit'])
    df_f = df[df.label.isin(label_filter)]

    metrics = {
        'sharpe':       'Sharpe Ratio',
        'sortino':      'Sortino Ratio',
        'omega':        'Omega Ratio',
        'equity_max_dd':'Equity Max DD (%)',
        'net_profit':   'Net Profit (%)',
        'profitable':   'Profitable (%)',
        'profit_factor':'Profit Factor',
        'trades':       'Trades',
        'half_kelly':   'Half Kelly (%)',
    }

    # 3 colunas de gráficos
    items = list(metrics.items())
    for i in range(0, len(items), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(items):
                metric, label = items[i+j]
                with col:
                    fig = bell_curve_plot(df_f, metric, label)
                    st.plotly_chart(fig, use_container_width=True)


def page_top_results(df):
    st.title("🏆 Top Results")

    col1, col2 = st.columns(2)
    with col1:
        sort_metric = st.selectbox("Sort by", [
            'slap_score','sharpe','sortino','net_profit',
            'profit_factor','profitable','omega'])
    with col2:
        top_n = st.slider("Show Top N", 10, 200, 50)

    df_top = df.nlargest(top_n, sort_metric)

    # Heatmap das métricas
    metrics_heat = ['sharpe','sortino','omega','profit_factor',
                    'profitable','equity_max_dd','net_profit']
    heat_data = df_top[metrics_heat].values
    heat_norm  = (heat_data - heat_data.min(axis=0)) / (heat_data.max(axis=0) - heat_data.min(axis=0) + 1e-9)

    fig = go.Figure(go.Heatmap(
        z=heat_norm.T,
        x=[f"#{i+1}" for i in range(len(df_top))],
        y=[m.replace('_',' ').title() for m in metrics_heat],
        colorscale='RdYlGn',
        colorbar=dict(title='Normalized'),
    ))
    fig.update_layout(
        title=f"Top {top_n} — Metric Heatmap (Normalized)",
        paper_bgcolor='#0a0a1a',
        font=dict(color='#cccccc'),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabela
    display_cols = ['mask','active_str','n_indicators','equity_max_dd','sharpe',
                    'sortino','omega','profit_factor','profitable','half_kelly',
                    'trades','net_profit','calmar','slap_score','label']

    st.dataframe(df_top[display_cols], use_container_width=True, height=500)

    # Download
    csv = df_top[display_cols].to_csv(index=False)
    st.download_button(
        "⬇️ Download Top Results CSV",
        data=csv,
        file_name=f"edgeforge_top{top_n}.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    df   = load_data()
    page = sidebar()

    if page == "🏠 Overview":
        page_overview(df)
    elif page == "📊 All Combinations":
        page_all_combinations(df)
    elif page == "🔢 By N Indicators":
        page_by_n(df)
    elif page == "✅ Inclusion Analysis":
        page_inclusion(df)
    elif page == "❌ Exclusion Analysis":
        page_exclusion(df)
    elif page == "🔔 Bell Curves":
        page_bell_curves(df)
    elif page == "🏆 Top Results":
        page_top_results(df)

if __name__ == "__main__":
    main()
