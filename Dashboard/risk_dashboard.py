"""
╔══════════════════════════════════════════════════════════════╗
║   MARKET RISK DASHBOARD  ·  Streamlit                       ║
║   Upload an Excel with daily price time series              ║
║   Expected format: column "Date" + one column per ticker    ║
╚══════════════════════════════════════════════════════════════╝

Run:   streamlit run streamlit_risk_dashboard.py
"""

import io, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  :root {
    --bg:    #0a0c12; --panel: #0f1118; --border: #1c2035;
    --acc:   #00d4a0; --acc2:  #f0a500; --acc3:   #e8445a;
    --acc4:  #5b8dee; --text:  #e2e8f8; --dim:    #7a8aaa;
  }
  html, body, [data-testid="stAppViewContainer"],
  [data-testid="stMain"], .main { background: var(--bg) !important; color: var(--text); }
  [data-testid="stSidebar"] { background: var(--panel) !important; border-right: 1px solid var(--border); }
  .stTabs [data-baseweb="tab-list"] { background: var(--panel); border-bottom: 1px solid var(--border); gap: 4px; }
  .stTabs [data-baseweb="tab"] {
    background: transparent; color: var(--dim); border: 1px solid var(--border);
    border-radius: 6px 6px 0 0; font-family: monospace; font-size: 11px;
    letter-spacing: .1em; text-transform: uppercase;
  }
  .stTabs [aria-selected="true"] { background: rgba(0,212,160,.1) !important; color: var(--acc) !important; border-color: var(--acc) !important; }
  [data-testid="metric-container"] {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px 18px;
  }
  [data-testid="stMetricValue"]  { color: var(--acc); font-family: monospace; font-size: 1.5rem !important; }
  [data-testid="stMetricLabel"]  { color: var(--dim); font-size: .7rem; font-family: monospace; text-transform: uppercase; letter-spacing: .1em; }
  [data-testid="stMetricDelta"]  { font-family: monospace; font-size: .8rem; }
  section[data-testid="stFileUploadDropzone"] {
    background: var(--panel); border: 2px dashed var(--border); border-radius: 10px; color: var(--dim);
  }
  .stSlider > div > div { color: var(--dim); }
  .stDataFrame { background: var(--panel) !important; }
  h1,h2,h3,h4 { color: var(--text) !important; }
  .stAlert { background: var(--panel) !important; border-color: var(--border) !important; }
  div[data-testid="stSelectbox"] > div { background: var(--panel); border-color: var(--border); color: var(--text); }
  .kpi-danger [data-testid="stMetricValue"] { color: var(--acc3) !important; }
  .kpi-warn   [data-testid="stMetricValue"] { color: var(--acc2) !important; }
  .kpi-info   [data-testid="stMetricValue"] { color: var(--acc4) !important; }
  div[data-testid="stHorizontalBlock"] { gap: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark template ───────────────────────────────────────────────────────
TMPL = dict(
    layout=go.Layout(
        paper_bgcolor="#0a0c12", plot_bgcolor="#0f1118",
        font=dict(family="monospace", color="#e2e8f8", size=11),
        xaxis=dict(gridcolor="#1c2035", zerolinecolor="#1c2035", linecolor="#1c2035"),
        yaxis=dict(gridcolor="#1c2035", zerolinecolor="#1c2035", linecolor="#1c2035"),
        legend=dict(bgcolor="#0f1118", bordercolor="#1c2035", borderwidth=1),
        colorway=["#00d4a0","#f0a500","#e8445a","#5b8dee","#a78bfa","#fb923c","#34d399","#f472b6"],
        margin=dict(l=50, r=20, t=40, b=40),
    )
)
COLORS = dict(acc="#00d4a0", acc2="#f0a500", acc3="#e8445a", acc4="#5b8dee", dim="#7a8aaa", text="#e2e8f8")

# ── Risk functions ─────────────────────────────────────────────────────────────
def hist_var(pnl, alpha): return np.percentile(pnl, (1-alpha)*100)
def hist_es(pnl, alpha):
    v = hist_var(pnl, alpha)
    tail = pnl[pnl <= v]
    return tail.mean() if len(tail) else v
def param_var(pnl, alpha):
    return pnl.mean() + pnl.std() * stats.norm.ppf(1-alpha)
def max_drawdown(cum): return ((cum / cum.cummax()) - 1).min()
def kupiec_pof(n_obs, n_exc, alpha):
    p, ph = 1-alpha, n_exc/n_obs
    if ph == 0:   lr = -2*n_obs*np.log(1-p)
    elif ph == 1: lr = -2*n_obs*np.log(p)
    else:
        lr = -2*(n_exc*np.log(p/ph) + (n_obs-n_exc)*np.log((1-p)/(1-ph)))
    return lr, 1-stats.chi2.cdf(lr, 1), ph

def compute_all_var(pnl_series, cov_ann, weights, alpha=0.99, n_mc=20_000):
    h_var = hist_var(pnl_series, alpha)
    h_es  = hist_es(pnl_series, alpha)
    p_var = param_var(pnl_series, alpha)
    Z_mc  = np.random.multivariate_normal(np.zeros(len(weights)), cov_ann/252, n_mc)
    mc_p  = (Z_mc @ weights) * 1e7
    m_var = hist_var(mc_p, alpha)
    m_es  = hist_es(mc_p, alpha)
    return dict(h_var=h_var, h_es=h_es, p_var=p_var, m_var=m_var, m_es=m_es, mc_pnl=mc_p)

def optimize_portfolio(mu_ann, cov_ann, n):
    w0  = np.ones(n)/n
    bnd = [(0.02, 0.40)]*n
    con = [{"type":"eq","fun":lambda w: w.sum()-1}]
    def neg_sr(w): r=w@mu_ann; v=np.sqrt(w@cov_ann@w); return -r/v if v>0 else 0
    def port_var(w): return w@cov_ann@w
    ms = minimize(neg_sr,  w0, method="SLSQP", bounds=bnd, constraints=con, options={"ftol":1e-12,"maxiter":1000})
    mv = minimize(port_var, w0, method="SLSQP", bounds=bnd, constraints=con, options={"ftol":1e-12,"maxiter":1000})
    return ms.x if ms.success else w0, mv.x if mv.success else w0

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<span style='color:#00d4a0;font-family:monospace;font-size:13px;letter-spacing:.15em'>◈ MARKET RISK ENGINE</span>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload Price Excel (.xlsx)",
        type=["xlsx","xls","csv"],
        help="Expected: Date column + one column per ticker. Sheet name: 'Prices'"
    )
    st.markdown("---")
    st.markdown("<span style='color:#7a8aaa;font-size:11px;font-family:monospace'>PARAMETERS</span>", unsafe_allow_html=True)
    confidence  = st.selectbox("VaR Confidence", [0.99, 0.95, 0.90], format_func=lambda x: f"{x:.0%}")
    nav         = st.number_input("Portfolio NAV ($)", value=10_000_000, step=500_000, format="%d")
    port_method = st.selectbox("Portfolio Construction", ["Max Sharpe", "Min Variance", "Equal Weight"])
    bt_window   = st.slider("Backtest Window (days)", 126, 504, 252, 63)
    shock_pct   = st.slider("Stress Shock (%)", -50, 30, -20)
    st.markdown("---")
    st.markdown("<span style='color:#7a8aaa;font-size:10px;font-family:monospace'>Models: Hist Sim · Parametric · Monte Carlo<br>Backtest: Kupiec POF · Basel Traffic Light</span>", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_prices(file_bytes, fname):
    if fname.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes), index_col=0, parse_dates=True)
    else:
        try:    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Prices", index_col=0, parse_dates=True)
        except: df = pd.read_excel(io.BytesIO(file_bytes), index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all").ffill()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

# ── Default sample data (from simulation) ────────────────────────────────────
@st.cache_data
def make_sample():
    np.random.seed(42)
    ASSETS = {"AAPL":(0.22,0.28,150),"MSFT":(0.20,0.24,300),"GOOGL":(0.18,0.27,120),
              "AMZN":(0.17,0.30,100),"JPM":(0.14,0.30,130),"GS":(0.13,0.33,350),
              "XOM":(0.10,0.31,60),"JNJ":(0.09,0.17,165),"PG":(0.08,0.15,140),
              "BRK.B":(0.11,0.20,300),"NVDA":(0.35,0.52,450),"META":(0.19,0.38,180)}
    tickers = list(ASSETS.keys())
    mus = np.array([v[0] for v in ASSETS.values()])
    sigs = np.array([v[1] for v in ASSETS.values()])
    s0s  = np.array([v[2] for v in ASSETS.values()], dtype=float)
    n = len(tickers); T = 1260; dt = 1/252
    rho = np.eye(n)
    grp = {"tech":[0,1,2,3,10,11],"fin":[4,5,9],"oth":[6,7,8]}
    for g in grp.values():
        for i in g:
            for j in g:
                if i!=j: rho[i,j] = 0.70 if g==grp["tech"] else 0.65
    for i in range(n):
        for j in range(n):
            if rho[i,j]==0 and i!=j: rho[i,j]=0.25
    np.fill_diagonal(rho,1)
    cov = np.diag(sigs)@rho@np.diag(sigs)
    L   = np.linalg.cholesky(cov)
    dW  = np.random.standard_normal((T,n))@L.T
    lr  = (mus-0.5*sigs**2)*dt + np.sqrt(dt)*dW
    arr = s0s*np.exp(np.cumsum(lr,axis=0))
    arr = np.vstack([s0s,arr])
    dates = pd.bdate_range("2019-01-02", periods=T+1)
    return pd.DataFrame(arr, index=dates, columns=tickers).round(2)

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<h1 style='margin:0;font-size:1.8rem;letter-spacing:-.02em'>Portfolio Market Risk Dashboard</h1>
<p style='color:#7a8aaa;font-family:monospace;font-size:12px;margin-top:4px'>
  Quant Risk Analytics · VaR · CVaR · Backtesting · Sensitivity · {pd.Timestamp.today().strftime('%d %b %Y')}
</p>
""", unsafe_allow_html=True)

if uploaded:
    prices = load_prices(uploaded.read(), uploaded.name)
    st.success(f"✓ Loaded **{uploaded.name}** — {len(prices):,} rows × {len(prices.columns)} assets  ({prices.index[0].date()} → {prices.index[-1].date()})")
else:
    prices = make_sample()
    st.info("📂 Using built-in sample data (12 assets, 5 years). Upload your own Excel to analyse a real portfolio.")

tickers   = list(prices.columns)
n_assets  = len(tickers)
returns   = prices.pct_change().dropna()
mu_ann    = returns.mean() * 252
cov_ann   = returns.cov()  * 252

# Asset selector
st.markdown("**Select assets for portfolio:**")
selected = st.multiselect("", tickers, default=tickers, label_visibility="collapsed")
if len(selected) < 2:
    st.warning("Select at least 2 assets."); st.stop()

ret_sel   = returns[selected]
mu_sel    = mu_ann[selected].values
cov_sel   = ret_sel.cov().values * 252
n_sel     = len(selected)

# Compute weights
w_ms, w_mv = optimize_portfolio(mu_sel, cov_sel, n_sel)
w_ew       = np.ones(n_sel) / n_sel
weight_map = {"Max Sharpe": w_ms, "Min Variance": w_mv, "Equal Weight": w_ew}
weights    = weight_map[port_method]

port_rets  = (ret_sel * weights).sum(axis=1)
port_cum   = (1 + port_rets).cumprod()
pnl_series = port_rets.values * nav

port_ret_ann = weights @ mu_sel
port_vol_ann = np.sqrt(weights @ cov_sel @ weights)
sharpe       = port_ret_ann / port_vol_ann if port_vol_ann > 0 else 0
mdd          = max_drawdown(port_cum)

# VaR
alpha  = confidence
var_r  = compute_all_var(pnl_series, pd.DataFrame(cov_sel, index=selected, columns=selected), weights, alpha)
var99  = var_r["h_var"]
es99   = var_r["h_es"]

# Backtest
if len(pnl_series) > bt_window:
    rv99, rv95, bt_pnl, bt_dates = [], [], [], []
    for i in range(bt_window, len(pnl_series)):
        w = pnl_series[i-bt_window:i]
        rv99.append(hist_var(w, 0.99)); rv95.append(hist_var(w, 0.95))
    rv99 = np.array(rv99); rv95 = np.array(rv95)
    bt_pnl   = pnl_series[bt_window:]
    bt_dates = port_rets.index[bt_window:]
    br99 = (bt_pnl < rv99).sum(); br95 = (bt_pnl < rv95).sum()
    n_bt = len(bt_pnl)
    lr99, pv99, ph99 = kupiec_pof(n_bt, br99, 0.99)
    kupiec_pass = pv99 > 0.05
    tl = "🟢 GREEN" if br99 <= 4 else ("🟡 YELLOW" if br99 <= 9 else "🔴 RED")
else:
    bt_pnl = rv99 = rv95 = br99 = pv99 = lr99 = None; kupiec_pass = None; tl = "N/A"; n_bt = 0

# ── TABS ───────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📊 Overview", "📉 VaR Analysis", "🔍 Backtesting", "🎲 Monte Carlo", "⚡ Sensitivity"])

# ═══ TAB 1: OVERVIEW ══════════════════════════════════════════════════════════
with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Portfolio Return", f"{port_ret_ann:.2%}", delta=f"vs EW {(w_ew@mu_sel - port_ret_ann):+.2%}")
    c2.metric("Portfolio Vol",    f"{port_vol_ann:.2%}")
    c3.metric("Sharpe Ratio",     f"{sharpe:.3f}")
    c4.metric(f"VaR {alpha:.0%} (1-day)", f"${abs(var99):,.0f}")
    c5.metric("Max Drawdown",     f"{mdd:.2%}")

    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure(template=TMPL)
        fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values,
            fill="tozeroy", line=dict(color=COLORS["acc"], width=1.5),
            fillcolor="rgba(0,212,160,0.08)", name=f"{port_method}"))
        ew_cum = (1 + (ret_sel * w_ew).sum(axis=1)).cumprod()
        fig.add_trace(go.Scatter(x=ew_cum.index, y=ew_cum.values,
            line=dict(color=COLORS["dim"], width=1, dash="dot"), name="Equal Weight"))
        fig.update_layout(title="Cumulative Return", yaxis_title="NAV (base=1)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig_w = go.Figure(template=TMPL)
        fig_w.add_trace(go.Bar(x=selected, y=weights*100,
            marker_color=[COLORS["acc"] if w > 0.15 else COLORS["acc4"] if w > 0.08 else COLORS["dim"] for w in weights],
            text=[f"{w:.1%}" for w in weights], textposition="outside",
            textfont=dict(size=9, color=COLORS["text"])))
        fig_w.update_layout(title="Portfolio Weights (%)", height=300,
                            yaxis_title="Weight (%)", xaxis_tickangle=-30)
        st.plotly_chart(fig_w, use_container_width=True)

    # Risk attribution table
    st.markdown("**Risk Attribution**")
    ann_vols = np.sqrt(np.diag(cov_sel))
    var_contribs = weights * ann_vols / (weights @ ann_vols)
    df_attr = pd.DataFrame({
        "Ticker":       selected,
        "Weight":       [f"{w:.2%}" for w in weights],
        "Ann. Vol":     [f"{v:.2%}" for v in ann_vols],
        "VaR Contrib":  [f"{v:.2%}" for v in var_contribs],
        "Marginal VaR": [f"${abs(hist_var(pnl_series * w, alpha)):,.0f}" for w in weights],
    }).set_index("Ticker")
    st.dataframe(df_attr, use_container_width=True)

    # Correlation heatmap
    corr = ret_sel.corr()
    fig_c = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn",
                      zmin=-1, zmax=1, aspect="auto", template="plotly_dark")
    fig_c.update_layout(title="Asset Correlation Matrix", height=380,
                        paper_bgcolor="#0a0c12", plot_bgcolor="#0f1118",
                        font=dict(family="monospace", color="#e2e8f8"))
    st.plotly_chart(fig_c, use_container_width=True)

# ═══ TAB 2: VAR ANALYSIS ══════════════════════════════════════════════════════
with tabs[1]:
    r99  = {a: compute_all_var(pnl_series, pd.DataFrame(cov_sel, index=selected, columns=selected), weights, a) for a in [0.90, 0.95, 0.99]}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Hist VaR {alpha:.0%}",  f"${abs(r99[alpha]['h_var']):,.0f}")
    c2.metric(f"CVaR/ES {alpha:.0%}",   f"${abs(r99[alpha]['h_es']):,.0f}")
    c3.metric(f"MC VaR {alpha:.0%}",    f"${abs(r99[alpha]['m_var']):,.0f}")
    c4.metric("10-Day VaR",             f"${abs(r99[alpha]['h_var'])*np.sqrt(10):,.0f}")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        fig_h = go.Figure(template=TMPL)
        pnl_k = pnl_series / 1e3
        v99k  = r99[0.99]["h_var"] / 1e3
        v95k  = r99[0.95]["h_var"] / 1e3
        counts, edges = np.histogram(pnl_k, bins=55)
        bar_colors = [COLORS["acc3"] if e < v99k else (COLORS["acc2"]+"cc" if e < 0 else COLORS["acc4"]+"99")
                      for e in edges[:-1]]
        fig_h.add_trace(go.Bar(x=edges[:-1], y=counts, marker_color=bar_colors,
                               marker_line_width=0, name="P&L frequency"))
        fig_h.add_vline(x=v99k, line_color=COLORS["acc3"], line_dash="dash",
                        annotation_text=f"VaR 99%", annotation_font_color=COLORS["acc3"])
        fig_h.add_vline(x=v95k, line_color=COLORS["acc2"], line_dash="dot",
                        annotation_text=f"VaR 95%", annotation_font_color=COLORS["acc2"])
        fig_h.update_layout(title="Daily P&L Distribution (Historical)", xaxis_title="P&L ($K)", height=320)
        st.plotly_chart(fig_h, use_container_width=True)

    with col2:
        # VaR method comparison
        methods = ["Hist VaR", "Param VaR", "MC VaR", "CVaR"]
        fig_comp = go.Figure(template=TMPL)
        for ai, (a_label, a_val) in enumerate([(f"90%",0.90),(f"95%",0.95),(f"99%",0.99)]):
            rd = r99[a_val]
            vals = [abs(rd["h_var"]), abs(rd["p_var"]), abs(rd["m_var"]), abs(rd["h_es"])]
            fig_comp.add_trace(go.Bar(name=a_label, x=methods, y=[v/1e3 for v in vals],
                                      marker_color=[COLORS["acc"], COLORS["acc2"], COLORS["acc3"]][ai]))
        fig_comp.update_layout(barmode="group", title="VaR Method Comparison ($K)",
                               yaxis_title="Loss ($K)", height=320)
        st.plotly_chart(fig_comp, use_container_width=True)

    # Horizon scaling
    horizons = [1, 2, 3, 5, 10, 15, 20]
    fig_hor = go.Figure(template=TMPL)
    for a_val, col in [(0.99, COLORS["acc3"]), (0.95, COLORS["acc2"])]:
        v = abs(r99[a_val]["h_var"])
        e = abs(r99[a_val]["h_es"])
        fig_hor.add_trace(go.Scatter(x=horizons, y=[v*np.sqrt(h)/1e3 for h in horizons],
            mode="lines+markers", name=f"VaR {a_val:.0%}", line=dict(color=col, width=2)))
        fig_hor.add_trace(go.Scatter(x=horizons, y=[e*np.sqrt(h)/1e3 for h in horizons],
            mode="lines+markers", name=f"CVaR {a_val:.0%}",
            line=dict(color=col, width=1.5, dash="dot")))
    fig_hor.update_layout(title="VaR / CVaR Scaling by Horizon (√t rule)",
                          xaxis_title="Horizon (days)", yaxis_title="Loss ($K)", height=300)
    st.plotly_chart(fig_hor, use_container_width=True)

    # Rolling VaR
    if bt_pnl is not None:
        fig_rv = go.Figure(template=TMPL)
        fig_rv.add_trace(go.Scatter(x=bt_dates, y=rv99/1e3, name="Rolling VaR 99%",
                                    line=dict(color=COLORS["acc3"], width=1.5)))
        fig_rv.add_trace(go.Scatter(x=bt_dates, y=rv95/1e3, name="Rolling VaR 95%",
                                    line=dict(color=COLORS["acc2"], width=1.2, dash="dot")))
        fig_rv.update_layout(title=f"Rolling {bt_window}-Day VaR",
                             yaxis_title="VaR ($K)", height=280)
        st.plotly_chart(fig_rv, use_container_width=True)

# ═══ TAB 3: BACKTESTING ═══════════════════════════════════════════════════════
with tabs[2]:
    if bt_pnl is None:
        st.warning(f"Need at least {bt_window+1} observations for backtesting.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Breaches (99%)",    str(br99), delta=f"Expected ~{n_bt*0.01:.1f}")
        c2.metric("Breach Rate",       f"{ph99:.4f}", delta=f"Model: {1-0.99:.4f}")
        c3.metric("Kupiec LR Stat",    f"{lr99:.4f}")
        c4.metric("Kupiec p-value",    f"{pv99:.4f}", delta="PASS ✓" if kupiec_pass else "FAIL ✗")
        st.markdown(f"**Basel Traffic Light: {tl}**")
        st.markdown("---")

        # P&L vs VaR
        breach_mask = bt_pnl < rv99
        fig_bt = go.Figure(template=TMPL)
        bar_cols = [COLORS["acc3"] if b else (COLORS["acc"]+"bb" if p >= 0 else COLORS["acc2"]+"88")
                    for b, p in zip(breach_mask, bt_pnl)]
        fig_bt.add_trace(go.Bar(x=bt_dates, y=bt_pnl/1e3, marker_color=bar_cols,
                                 marker_line_width=0, name="Daily P&L"))
        fig_bt.add_trace(go.Scatter(x=bt_dates, y=rv99/1e3, name="Rolling VaR 99%",
                                     line=dict(color=COLORS["acc3"], dash="dash", width=1.5)))
        breach_dates = bt_dates[breach_mask]
        fig_bt.add_trace(go.Scatter(x=breach_dates, y=bt_pnl[breach_mask]/1e3,
                                     mode="markers", marker=dict(color=COLORS["acc3"], size=7, symbol="x"),
                                     name=f"Breaches ({br99})"))
        fig_bt.update_layout(title=f"P&L vs Rolling VaR Limit — {bt_window}-Day Window",
                              yaxis_title="P&L ($K)", height=360)
        st.plotly_chart(fig_bt, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            # Kupiec χ² chart
            x_lr = np.linspace(0, 12, 400)
            fig_k = go.Figure(template=TMPL)
            fig_k.add_trace(go.Scatter(x=x_lr, y=stats.chi2.pdf(x_lr, 1),
                                        line=dict(color=COLORS["text"], width=1.5), name="χ²(1)"))
            fig_k.add_vrect(x0=3.841, x1=12, fillcolor=COLORS["acc3"], opacity=0.15,
                             annotation_text="Rejection region", annotation_font_color=COLORS["acc3"])
            fig_k.add_vline(x=3.841, line_color=COLORS["acc3"], line_dash="dash",
                             annotation_text="CV=3.841")
            fig_k.add_vline(x=lr99, line_color=COLORS["acc2"], line_width=2,
                             annotation_text=f"LR={lr99:.3f}", annotation_font_color=COLORS["acc2"])
            fig_k.update_layout(title="Kupiec POF Test (χ² distribution, df=1)", height=300)
            st.plotly_chart(fig_k, use_container_width=True)

        with col2:
            st.markdown("**Traffic Light Summary**")
            for zone, rng, color, desc in [
                ("🟢 Green", "0–4 exceptions", "#00d4a0", "Model statistically adequate"),
                ("🟡 Yellow", "5–9 exceptions", "#f0a500", "Investigate model assumptions"),
                ("🔴 Red", "10+ exceptions", "#e8445a", "Model rejected, recalibrate"),
            ]:
                active = (br99 <= 4 and "Green" in zone) or (5<=br99<=9 and "Yellow" in zone) or (br99>=10 and "Red" in zone)
                bg = f"rgba({','.join(str(int(color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.15)" if active else "transparent"
                st.markdown(f"""<div style='background:{bg};border:1px solid {"#333" if not active else color};
                    border-radius:8px;padding:10px 14px;margin-bottom:8px'>
                    <span style='color:{color};font-weight:bold'>{zone}: {rng}</span><br>
                    <span style='color:#7a8aaa;font-size:12px'>{desc}</span>
                    {"<br><span style='color:"+color+";font-size:11px'>← Current: "+str(br99)+" breaches</span>" if active else ""}
                    </div>""", unsafe_allow_html=True)

# ═══ TAB 4: MONTE CARLO ═══════════════════════════════════════════════════════
with tabs[3]:
    mc_pnl = var_r["mc_pnl"]
    mc_var = var_r["m_var"]; mc_es = var_r["m_es"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MC VaR 99%",    f"${abs(mc_var):,.0f}")
    c2.metric("MC CVaR 99%",   f"${abs(mc_es):,.0f}")
    c3.metric("MC P1 (worst)", f"${abs(np.percentile(mc_pnl, 1)):,.0f}")
    c4.metric("Simulations",   "20,000")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        counts_mc, edges_mc = np.histogram(mc_pnl/1e3, bins=70)
        bar_c = [COLORS["acc3"] if e < mc_var/1e3 else (COLORS["acc4"]+"99" if e < 0 else COLORS["acc"]+"88")
                 for e in edges_mc[:-1]]
        fig_mc = go.Figure(template=TMPL)
        fig_mc.add_trace(go.Bar(x=edges_mc[:-1], y=counts_mc, marker_color=bar_c, marker_line_width=0))
        fig_mc.add_vline(x=mc_var/1e3, line_color=COLORS["acc3"], line_dash="dash",
                          annotation_text=f"MC VaR 99%")
        fig_mc.add_vline(x=mc_es/1e3,  line_color="#ff4500", line_dash="dot",
                          annotation_text="MC CVaR")
        fig_mc.update_layout(title="Monte Carlo P&L Distribution (20K paths)",
                              xaxis_title="P&L ($K)", height=340)
        st.plotly_chart(fig_mc, use_container_width=True)

    with col2:
        pcts = [0.5, 1, 2.5, 5, 10, 25, 50, 75, 90]
        vals = [abs(np.percentile(mc_pnl, p)) for p in pcts]
        fig_pct = go.Figure(template=TMPL)
        fig_pct.add_trace(go.Bar(
            x=[f"P{p}" for p in pcts], y=[v/1e3 for v in vals],
            marker_color=[COLORS["acc3"] if p<=1 else (COLORS["acc2"] if p<=5 else COLORS["acc4"]) for p in pcts],
            text=[f"${v/1e3:.0f}K" for v in vals], textposition="outside",
            textfont=dict(size=9, color=COLORS["text"])
        ))
        fig_pct.update_layout(title="P&L Percentile Ladder (MC)", yaxis_title="Loss ($K)", height=340)
        st.plotly_chart(fig_pct, use_container_width=True)

# ═══ TAB 5: SENSITIVITY ════════════════════════════════════════════════════════
with tabs[4]:
    # Estimate betas from returns vs equal-weight "market"
    mkt = (ret_sel * w_ew).sum(axis=1)
    betas = np.array([np.cov(ret_sel[t], mkt)[0,1] / np.var(mkt) for t in selected])
    port_beta = weights @ betas
    shock = shock_pct / 100

    port_pnl_shock = weights @ (betas * shock) * nav
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"P&L at {shock_pct:+.0f}%", f"${port_pnl_shock:+,.0f}")
    c2.metric("% of NAV",   f"{port_pnl_shock/nav:+.2%}")
    c3.metric("Portfolio β", f"{port_beta:.3f}")
    c4.metric("GFC (-50%) Loss", f"${weights@(betas*-0.50)*nav:+,.0f}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        shk = np.linspace(-0.50, 0.30, 200)
        sweep = np.array([weights @ (betas * s) * nav / 1e3 for s in shk])
        fig_sw = go.Figure(template=TMPL)
        fig_sw.add_traces([
            go.Scatter(x=shk[sweep<0]*100, y=sweep[sweep<0], fill="tozeroy",
                       fillcolor="rgba(232,68,90,0.12)", line=dict(color="rgba(0,0,0,0)"), showlegend=False),
            go.Scatter(x=shk[sweep>=0]*100, y=sweep[sweep>=0], fill="tozeroy",
                       fillcolor="rgba(0,212,160,0.12)", line=dict(color="rgba(0,0,0,0)"), showlegend=False),
            go.Scatter(x=shk*100, y=sweep, line=dict(color=COLORS["text"], width=2), name="Portfolio P&L"),
        ])
        for nm, s in [("COVID",-0.34),("GFC",-0.50),("Bull",0.15)]:
            p = (weights @ (betas * s) * nav) / 1e3
            fig_sw.add_trace(go.Scatter(x=[s*100], y=[p], mode="markers+text",
                text=[nm], textposition="top right", textfont=dict(size=9, color=COLORS["dim"]),
                marker=dict(size=9, color=COLORS["acc3"] if s<0 else COLORS["acc"])))
        fig_sw.update_layout(title="Portfolio Sensitivity to Market Shock",
                              xaxis_title="Shock (%)", yaxis_title="P&L ($K)", height=340)
        st.plotly_chart(fig_sw, use_container_width=True)

    with col2:
        asset_impacts = weights * betas * shock * nav / 1e3
        fig_ai = go.Figure(template=TMPL)
        fig_ai.add_trace(go.Bar(
            x=selected, y=asset_impacts,
            marker_color=[COLORS["acc3"] if v < 0 else COLORS["acc"] for v in asset_impacts],
            text=[f"${v:.1f}K" for v in asset_impacts], textposition="outside",
            textfont=dict(size=9)
        ))
        fig_ai.update_layout(title=f"Per-Asset P&L at {shock_pct:+.0f}% Shock",
                              yaxis_title="P&L ($K)", height=340, xaxis_tickangle=-30)
        st.plotly_chart(fig_ai, use_container_width=True)

    # Stress scenario table
    st.markdown("**Named Stress Scenarios**")
    scenarios = [
        ("GFC 2008 (-50%)", -0.50, "CRITICAL"),
        ("COVID Crash (-34%)", -0.34, "CRITICAL"),
        ("Black Monday (-22%)", -0.22, "SEVERE"),
        ("2022 Bear (-20%)", -0.20, "SEVERE"),
        ("Tech Selloff (-15%)", -0.15, "SEVERE"),
        ("Mild Correction (-10%)", -0.10, "MODERATE"),
        ("Rate Shock (-8%)", -0.08, "MODERATE"),
        ("Flat (0%)", 0.00, "NEUTRAL"),
        ("Bull Rally (+15%)", 0.15, "POSITIVE"),
        ("Strong Bull (+20%)", 0.20, "POSITIVE"),
    ]
    rows = []
    for name, s, sev in scenarios:
        p = weights @ (betas * s) * nav
        rows.append({"Scenario": name, "Market Move": f"{s:+.0%}", "Portfolio P&L": f"${p:+,.0f}",
                     "% of NAV": f"{p/nav:+.2%}", "Severity": sev})
    df_scen = pd.DataFrame(rows).set_index("Scenario")
    st.dataframe(df_scen, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<p style='color:#4a5270;font-size:10px;font-family:monospace;text-align:center'>
  ◈ MARKET RISK ANALYTICS · INTERNAL USE ONLY · NOT FOR DISTRIBUTION<br>
  Models: Historical Simulation · Parametric (Normal) · Monte Carlo (Correlated GBM) · Kupiec POF Backtest · Basel Traffic Light
</p>""", unsafe_allow_html=True)