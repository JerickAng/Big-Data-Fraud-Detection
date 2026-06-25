import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudLens",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixed Theme — muted warm neutrals
# ──────────────────────────────────────────────────────────────────────────────
T = {
    "bg":             "#F5F3EF",   # warm off-white page
    "sidebar_bg":     "#EDEAE3",   # slightly darker warm grey sidebar
    "card_bg":        "#FAFAF8",   # near-white cards
    "border":         "#D8D3C8",   # warm light border
    "accent":         "#3D6B5E",   # muted forest teal
    "text":           "#1E2822",   # near-black for body text
    "text_secondary": "#4A5568",   # medium grey for secondary text
    "muted":          "#6E7A74",   # subdued label colour
    "fraud":          "#A83232",   # deep muted red
    "safe":           "#3A7048",   # deep muted green
    "warning":        "#8C6320",   # deep muted amber
    "grid":           "#E2DDD6",   # very light grid lines
    "nav_active":     "#3D6B5E",   # active nav text (accent)
    "nav_hover":      "#E4E0D8",   # subtle hover background
}


# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────
def inject_custom_css():
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
.stApp {{
    background-color: {T['bg']};
    color: {T['text']};
}}

/* ── Sidebar shell ── */
[data-testid="stSidebar"] {{
    background-color: {T['sidebar_bg']};
    border-right: 1px solid {T['border']};
}}
[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0;
}}

/* ── Sidebar nav buttons ── */
[data-testid="stSidebar"] .nav-btn button {{
    width: 100%;
    background: transparent;
    border: none;
    border-radius: 0;
    text-align: left;
    padding: 0.55rem 1.25rem;
    font-size: 0.875rem;
    font-weight: 400;
    color: {T['muted']};
    cursor: pointer;
    transition: background 0.12s ease, color 0.12s ease;
    box-shadow: none;
}}
[data-testid="stSidebar"] .nav-btn button:hover {{
    background: {T['nav_hover']};
    color: {T['text']};
    border: none;
    box-shadow: none;
}}
[data-testid="stSidebar"] .nav-btn-active button {{
    color: {T['nav_active']} !important;
    font-weight: 600 !important;
    border-left: 2px solid {T['accent']} !important;
    padding-left: calc(1.25rem - 2px) !important;
    background: transparent !important;
    box-shadow: none !important;
}}
[data-testid="stSidebar"] .nav-btn-active button:hover {{
    background: transparent !important;
}}

/* ── Sidebar toggle ── */
[data-testid="stSidebar"] .stToggle label {{
    color: {T['text_secondary']} !important;
    font-size: 0.85rem;
}}

/* ── Metric cards ── */
[data-testid="metric-container"] {{
    background: {T['card_bg']};
    border: 1px solid {T['border']};
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}}
[data-testid="metric-container"] label,
[data-testid="metric-container"] label p {{
    color: {T['muted']} !important;
    font-size: 0.74rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="metric-container"] [data-testid="stMetricValue"] > div,
[data-testid="metric-container"] [data-testid="stMetricValue"] p {{
    color: #1E2822 !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    font-family: 'Roboto Mono', monospace !important;
}}
[data-testid="metric-container"] [data-testid="stMetricDelta"],
[data-testid="metric-container"] [data-testid="stMetricDelta"] p {{
    color: {T['muted']} !important;
    font-size: 0.8rem !important;
}}

/* ── Headings ── */
h1 {{
    color: {T['text']} !important;
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}}
h2 {{ color: {T['text']} !important; font-size: 1.3rem !important; font-weight: 600 !important; }}
h3 {{ color: {T['text']} !important; font-size: 1.05rem !important; font-weight: 600 !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: transparent;
    border-bottom: 1px solid {T['border']};
    gap: 0;
    padding: 0;
}}
.stTabs [data-baseweb="tab"] {{
    color: {T['muted']} !important;
    font-size: 0.875rem !important;
    font-weight: 500;
    padding: 0.6rem 1.1rem !important;
    background: transparent !important;
}}
.stTabs [aria-selected="true"] {{
    color: {T['accent']} !important;
    border-bottom: 2px solid {T['accent']} !important;
    font-weight: 600 !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {T['border']};
    border-radius: 8px;
    overflow: hidden;
}}

/* ── Alerts ── */
[data-testid="stAlert"] {{
    background-color: {T['card_bg']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
    color: {T['text_secondary']} !important;
    font-size: 0.875rem;
}}

/* ── File uploader ── */
.stFileUploader {{
    background-color: {T['card_bg']} !important;
    border: 1px dashed {T['border']} !important;
    border-radius: 8px;
}}

/* ── Divider ── */
hr {{ border-color: {T['border']}; margin: 1rem 0; opacity: 1; }}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {{ color: {T['accent']} !important; }}

/* ── fl-card ── */
.fl-card {{
    background: {T['card_bg']};
    border: 1px solid {T['border']};
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}}
.fl-card-title {{
    color: {T['text']};
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 0.45rem;
}}
.fl-card-desc {{
    color: {T['text_secondary']};
    font-size: 0.84rem;
    margin-bottom: 0.8rem;
    line-height: 1.55;
}}

/* ── Insight cards ── */
.insight-card {{
    background: {T['card_bg']};
    border-left: 2px solid {T['accent']};
    border-top: 1px solid {T['border']};
    border-right: 1px solid {T['border']};
    border-bottom: 1px solid {T['border']};
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
}}
.insight-card-title {{
    color: {T['text']};
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}}
.insight-card-body {{
    color: {T['text_secondary']};
    font-size: 0.82rem;
    line-height: 1.55;
}}

/* ── Page header ── */
.page-header {{
    padding: 0.25rem 0 0.9rem 0;
    border-bottom: 1px solid {T['border']};
    margin-bottom: 1.5rem;
}}
.page-header h1 {{ margin: 0; }}

/* ── Model summary card ── */
.model-best-card {{
    background: {T['card_bg']};
    border: 1px solid {T['border']};
    border-top: 2px solid {T['accent']};
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.model-best-label {{
    color: {T['muted']};
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-family: 'Roboto Mono', monospace;
    font-weight: 600;
}}
.model-best-value {{
    color: {T['text']};
    font-size: 1.2rem;
    font-weight: 700;
    margin-top: 0.25rem;
}}

/* ── Chart caption ── */
.chart-caption {{
    color: {T['muted']} !important;
    font-size: 0.8rem;
    font-style: italic;
    margin-top: 0.4rem;
    line-height: 1.5;
}}

/* ── Sidebar logo ── */
.sidebar-logo {{
    padding: 1.5rem 1.25rem 1rem 1.25rem;
    margin-bottom: 0.25rem;
}}
.sidebar-logo-title {{
    color: {T['text']};
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}}

/* ── Sidebar section label ── */
.sidebar-section {{
    color: {T['muted']};
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
    font-family: 'Roboto Mono', monospace;
    padding: 1rem 1.25rem 0.4rem 1.25rem;
}}

/* ── Sidebar divider ── */
.sidebar-divider {{
    border: none;
    border-top: 1px solid {T['border']};
    margin: 0.75rem 1.25rem;
}}

/* ── Data source section in sidebar ── */
[data-testid="stSidebar"] .stToggle {{
    padding: 0 1.25rem;
}}
[data-testid="stSidebar"] .stFileUploader {{
    margin: 0 1.25rem;
}}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Chart theme helper
# ──────────────────────────────────────────────────────────────────────────────
def apply_chart_theme(ax, fig):
    fig.patch.set_facecolor(T["card_bg"])
    ax.set_facecolor(T["card_bg"])
    ax.tick_params(colors=T["muted"], labelsize=9)
    ax.xaxis.label.set_color(T["muted"])
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_color(T["muted"])
    ax.yaxis.label.set_fontsize(10)
    ax.title.set_color(T["text"])
    ax.title.set_fontsize(11)
    ax.title.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_edgecolor(T["border"])
    ax.yaxis.grid(True, color=T["grid"], linewidth=0.5, linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)


# ──────────────────────────────────────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────────────────────────────────────
def render_page_header(title: str):
    st.markdown(f"""
<div class="page-header"><h1>{title}</h1></div>
""", unsafe_allow_html=True)


def render_chart_card(title: str, description: str = ""):
    desc_html = f"<div class='fl-card-desc'>{description}</div>" if description else ""
    st.markdown(f"""
<div class="fl-card" style="padding-bottom:0.5rem;">
    <div class="fl-card-title">{title}</div>
    {desc_html}
</div>
""", unsafe_allow_html=True)


def render_insight_card(title: str, body: str):
    st.markdown(f"""
<div class="insight-card">
    <div class="insight-card-title">{title}</div>
    <div class="insight-card-body">{body}</div>
</div>
""", unsafe_allow_html=True)


def render_model_best_card(label: str, value: str):
    st.markdown(f"""
<div class="model-best-card">
    <div class="model-best-label">{label}</div>
    <div class="model-best-value">{value}</div>
</div>
""", unsafe_allow_html=True)


def render_info_card(title: str, items: list):
    bullets = "".join(
        f"<li style='margin-bottom:0.35rem;color:{T['text_secondary']};font-size:0.84rem;'>{item}</li>"
        for item in items
    )
    st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">{title}</div>
    <ul style="margin:0;padding-left:1.2rem;">{bullets}</ul>
</div>
""", unsafe_allow_html=True)


def chart_caption(text: str):
    st.markdown(f"<p class='chart-caption'>{text}</p>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple:
    with st.sidebar:
        # Title
        st.markdown("""
<div class="sidebar-logo">
    <div class="sidebar-logo-title">FraudLens</div>
</div>
<hr class="sidebar-divider">
""", unsafe_allow_html=True)

        # Navigation — plain buttons, no radio chrome
        st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
        pages = ["Dashboard", "Data Analysis", "Model Performance"]
        if "current_page" not in st.session_state:
            st.session_state.current_page = "Dashboard"

        for p in pages:
            is_active = st.session_state.current_page == p
            css_class = "nav-btn nav-btn-active" if is_active else "nav-btn"
            st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
            if st.button(p, key=f"nav_{p}", use_container_width=True):
                st.session_state.current_page = p
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        page = st.session_state.current_page

        # Data Source
        st.markdown('<hr class="sidebar-divider"><div class="sidebar-section">Data Source</div>', unsafe_allow_html=True)
        use_demo = st.toggle("Use Demonstration Data", value=True, key="demo_toggle")
        t_file = i_file = None
        if not use_demo:
            t_file = st.file_uploader("Transaction CSV", type="csv", key="trans")
            i_file = st.file_uploader("Identity CSV", type="csv", key="iden")

    return page, use_demo, t_file, i_file


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_uploaded_data(transaction_bytes, identity_bytes):
    import io
    transaction = pd.read_csv(io.BytesIO(transaction_bytes))
    identity    = pd.read_csv(io.BytesIO(identity_bytes))
    return transaction.merge(identity, on="TransactionID", how="left")


@st.cache_data
def generate_demo_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    fraud_mask = rng.random(n) < 0.035
    data = {
        "TransactionID": np.arange(1, n + 1),
        "isFraud":        fraud_mask.astype(int),
        "TransactionAmt": np.where(fraud_mask, rng.exponential(800, n), rng.exponential(150, n)),
        "card1":          rng.integers(1000, 18000, n),
        "card2":          rng.choice([100, 111, 117, 121, 150, 200, 300, 360, 490, 555], n),
        "addr1":          rng.integers(100, 500, n),
        "addr2":          rng.integers(10, 100, n),
        "dist1":          np.where(fraud_mask, rng.exponential(200, n), rng.exponential(50, n)),
        "dist2":          rng.exponential(70, n),
        "C1":             rng.integers(0, 15, n),
        "C2":             rng.integers(0, 10, n),
        "C13":            rng.integers(0, 40, n),
        "V258":           rng.standard_normal(n) + np.where(fraud_mask, 1.2, 0),
        "V294":           rng.standard_normal(n) + np.where(fraud_mask, 0.8, 0),
        "V201":           rng.standard_normal(n),
        "D1":             np.clip(rng.exponential(100, n), 0, 500),
        "D4":             np.clip(rng.exponential(30,  n), 0, 300),
        "ProductCD":      rng.choice(["W", "H", "C", "S", "R"], n),
        "P_emaildomain":  rng.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com"], n),
        "DeviceType":     rng.choice(["desktop", "mobile", None], n, p=[0.45, 0.40, 0.15]),
    }
    return pd.DataFrame(data)


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ──────────────────────────────────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"

inject_custom_css()
page, use_demo, t_file, i_file = render_sidebar()
page = st.session_state.current_page

# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
df = None
data_ready = False

if use_demo:
    df = generate_demo_data()
    data_ready = True
elif t_file and i_file:
    with st.spinner("Merging datasets…"):
        df = load_uploaded_data(t_file.read(), i_file.read())
    data_ready = True
else:
    st.info("Upload both CSV files in the sidebar to get started, or enable demonstration mode.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":

    render_page_header("Dashboard")

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Transactions", "590,540")
    with c2:
        st.metric("Fraud Cases", "20,663", delta="-5.2% vs prev period", delta_color="normal")
    with c3:
        st.metric("Non-Fraud Cases", "569,877", delta="+2.1%", delta_color="normal")
    with c4:
        st.metric("Fraud Rate", "3.50%")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        # Project overview
        list_items = [
            ("Dataset Scale", "590,540 transactions with 434 engineered features"),
            ("Imbalance Handling", "SMOTE applied to address 3.5% fraud minority"),
            ("Model Portfolio", "Logistic Regression, Random Forest, Neural Network"),
            ("Metrics", "Accuracy, Precision, Recall, F1-Score, AUC-ROC"),
            ("Business Goal", "Detect fraud while minimising false positives"),
        ]
        rows = "".join(
            f"<li style='margin-bottom:0.5rem;font-size:0.86rem;color:{T['text_secondary']};'>"
            f"<span style='color:{T['text']};font-weight:600;'>{k}</span> — {v}"
            f"</li>"
            for k, v in list_items
        )
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Project Overview</div>
    <div class="fl-card-desc">
        A fraud detection system built on the IEEE-CIS Fraud Detection dataset, applying
        machine learning to identify fraudulent financial transactions.
    </div>
    <ul style="margin:0;padding-left:1.1rem;line-height:1.9;">{rows}</ul>
</div>
""", unsafe_allow_html=True)

    with col_right:
        # Pie chart
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Class Distribution</div>
    <div class="fl-card-desc">Proportion of legitimate vs fraudulent transactions.</div>
""", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(4, 3))
        sizes  = [96.5, 3.5]
        colors = [T["safe"], T["fraud"]]
        _, _, autotexts = ax.pie(
            sizes, colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.72,
            wedgeprops=dict(width=0.52, edgecolor=T["card_bg"], linewidth=2),
        )
        for at in autotexts:
            at.set_color("#FFFFFF"); at.set_fontsize(9); at.set_fontweight("bold")
        ax.legend(["Non-Fraud (96.5%)", "Fraud (3.5%)"],
                  loc="lower center", frameon=False,
                  labelcolor=T["text_secondary"], fontsize=8)
        ax.set_title("Transaction Split", color=T["text"], fontsize=10, fontweight="bold", pad=6)
        apply_chart_theme(ax, fig)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Risk summary
        summary_rows = [
            ("Best Model", "Random Forest (AUC 0.81)"),
            ("Recall", "43% of all fraud detected"),
            ("Precision", "12% of flagged are truly fraud"),
            ("SMOTE", "Balances class ratio for training"),
            ("Threshold", "Tunable 0.3–0.7 range"),
        ]
        rows2 = "".join(
            f"<li style='margin-bottom:0.4rem;font-size:0.84rem;color:{T['text_secondary']};'>"
            f"<span style='color:{T['text']};font-weight:600;'>{k}</span> — {v}"
            f"</li>"
            for k, v in summary_rows
        )
        st.markdown(f"""
<div class="fl-card" style="margin-top:0.75rem;">
    <div class="fl-card-title">Fraud Risk Summary</div>
    <ul style="margin:0;padding-left:1.1rem;line-height:1.8;">{rows2}</ul>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Analysis":

    render_page_header("Data Analysis")

    if not data_ready:
        st.stop()

    fraud_count = int(df["isFraud"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Rows",  f"{df.shape[0]:,}")
    c2.metric("Features",      f"{df.shape[1]:,}")
    c3.metric("Fraud Cases",   f"{fraud_count:,}")
    c4.metric("Fraud Rate",    f"{fraud_count/len(df)*100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribution", "Patterns", "Features", "Sample Data",
    ])

    # ── Tab 1: Distribution
    with tab1:
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            render_chart_card(
                "Class Balance",
                "Bar comparison of legitimate vs fraudulent transaction counts.",
            )
            fraud_counts = df["isFraud"].value_counts().sort_index()
            labels = ["Non-Fraud", "Fraud"]
            colors = [T["safe"], T["fraud"]]
            fig, ax = plt.subplots(figsize=(5, 3.8))
            bars = ax.bar(labels, fraud_counts.values, color=colors, width=0.45,
                          edgecolor=T["card_bg"], linewidth=1.5)
            for bar, val in zip(bars, fraud_counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(fraud_counts) * 0.02,
                        f"{val:,}\n({val/len(df)*100:.1f}%)",
                        ha="center", va="bottom", color=T["muted"],
                        fontsize=8, fontfamily="monospace")
            ax.set_ylabel("Count", fontsize=10)
            ax.set_ylim(0, max(fraud_counts) * 1.18)
            ax.set_title("Class Balance", fontsize=11)
            apply_chart_theme(ax, fig)
            st.pyplot(fig, use_container_width=True)
            chart_caption("Severe class imbalance: 96.5% legitimate. SMOTE is applied during model training.")

        with col_b:
            render_chart_card(
                "Transaction Amount Distribution",
                "Overlapping density curves comparing amount ranges for fraud vs legitimate.",
            )
            fig2, ax2 = plt.subplots(figsize=(5, 3.8))
            fraud_amt = df[df["isFraud"] == 1]["TransactionAmt"].clip(upper=5000)
            legit_amt = df[df["isFraud"] == 0]["TransactionAmt"].clip(upper=5000)
            ax2.hist(legit_amt, bins=60, color=T["safe"],  alpha=0.6, label="Non-Fraud", density=True)
            ax2.hist(fraud_amt, bins=60, color=T["fraud"], alpha=0.6, label="Fraud",     density=True)
            ax2.set_xlabel("Transaction Amount (USD)")
            ax2.set_ylabel("Density")
            ax2.set_title("Amount Comparison")
            ax2.legend(frameon=False, labelcolor=T["text_secondary"], fontsize=9)
            apply_chart_theme(ax2, fig2)
            st.pyplot(fig2, use_container_width=True)
            chart_caption("Fraudulent transactions show bimodal behaviour — micro-amounts and high-value spikes.")

    # ── Tab 2: Patterns
    with tab2:
        render_chart_card(
            "Top Features by Fraud Correlation",
            "Absolute Pearson correlation with the isFraud label (numeric columns only, top 10 shown).",
        )
        num_cols = df.select_dtypes(include=np.number).columns.tolist()[:50]
        corr_series = (
            df[num_cols].corr()["isFraud"]
            .abs()
            .drop("isFraud", errors="ignore")
            .sort_values(ascending=False)
            .head(10)
        )
        palette = [
            T["accent"]  if v >= corr_series.iloc[0] * 0.7 else
            T["warning"] if v >= corr_series.iloc[0] * 0.4 else
            T["muted"]
            for v in corr_series.values
        ]
        fig3, ax3 = plt.subplots(figsize=(10, 4.5))
        bars3 = ax3.barh(corr_series.index[::-1], corr_series.values[::-1],
                         color=palette[::-1], edgecolor=T["card_bg"], linewidth=0.8)
        ax3.set_xlabel("Absolute Correlation")
        ax3.set_title("Top 10 Features Associated with Fraud")
        for bar, val in zip(bars3, corr_series.values[::-1]):
            ax3.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", color=T["muted"],
                     fontsize=8, fontfamily="monospace")
        apply_chart_theme(ax3, fig3)
        st.pyplot(fig3, use_container_width=True)
        chart_caption("Teal = high correlation, amber = moderate, grey = lower.")

    # ── Tab 3: Features
    with tab3:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Feature Overview</div>
    <div class="fl-card-desc">Data type, missing percentage, and unique value count for every column.</div>
</div>
""", unsafe_allow_html=True)
        feature_info = pd.DataFrame({
            "Column":        df.columns,
            "Data Type":     df.dtypes.values,
            "Missing %":     (df.isnull().sum() / len(df) * 100).round(2).values,
            "Unique Values": [df[col].nunique() for col in df.columns],
        })
        st.dataframe(feature_info, use_container_width=True, hide_index=True)
        chart_caption("Columns above 50% missing may need to be dropped during preprocessing.")

    # ── Tab 4: Sample Data
    with tab4:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Raw Data Sample</div>
    <div class="fl-card-desc">Fraud rows highlighted in red; legitimate rows in green.</div>
</div>
""", unsafe_allow_html=True)
        n_show = st.slider("Rows to display", 5, 50, 10)
        sample = df.head(n_show).copy()

        def highlight_fraud(val):
            if val == 1:   return "background-color:#f5e0e0; color:#7A2020"
            elif val == 0: return "background-color:#e0f0e4; color:#1F5230"
            return ""

        st.dataframe(
            sample.style.applymap(highlight_fraud, subset=["isFraud"]),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":

    render_page_header("Model Performance")

    perf = pd.DataFrame({
        "Model":              ["Logistic Regression", "Random Forest", "Neural Network"],
        "Accuracy":           [0.83, 0.87, 0.87],
        "Precision (Fraud)":  [0.09, 0.12, 0.12],
        "Recall (Fraud)":     [0.44, 0.43, 0.44],
        "F1-Score (Fraud)":   [0.16, 0.19, 0.19],
        "AUC-ROC":            [0.72, 0.81, 0.80],
    })

    c1, c2, c3 = st.columns(3)
    with c1:
        render_model_best_card("Best Accuracy", "Random Forest — 87%")
    with c2:
        render_model_best_card("Best AUC-ROC", "Random Forest — 0.81")
    with c3:
        render_model_best_card("Best F1-Score", "RF & NN — 19%")

    st.markdown("<br>", unsafe_allow_html=True)

    tab_summary, tab_charts, tab_confusion, tab_insights = st.tabs([
        "Metrics Summary", "Performance Charts", "Confusion Matrices", "Key Insights",
    ])

    # ── Tab 1: Metrics Summary
    with tab_summary:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Model Comparison Table</div>
    <div class="fl-card-desc">Evaluated on held-out test set with SMOTE-balanced training. Best values per column are highlighted.</div>
</div>
""", unsafe_allow_html=True)

        def color_best(col):
            styles = [""] * len(col)
            best_idx = col.idxmax()
            styles[best_idx] = f"color:{T['accent']}; font-weight:700; background:#E8F2F0"
            return styles

        styled = (
            perf.style
            .apply(color_best, subset=["Accuracy","Precision (Fraud)","Recall (Fraud)","F1-Score (Fraud)","AUC-ROC"])
            .format({c: "{:.2%}" for c in ["Accuracy","Precision (Fraud)","Recall (Fraud)","F1-Score (Fraud)","AUC-ROC"]})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
        chart_caption("Highlighted cells indicate the best-performing model for each metric.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{T['muted']};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;font-family:Roboto Mono,monospace;margin-bottom:0.5rem;'>Metric Definitions</p>", unsafe_allow_html=True)

        metrics_info = {
            "Accuracy":  "Overall correctness: (TP + TN) / Total",
            "Precision": "Of all flagged transactions, how many were truly fraudulent: TP / (TP + FP)",
            "Recall":    "Of all actual fraud cases, fraction successfully caught: TP / (TP + FN)",
            "F1-Score":  "Harmonic mean of Precision and Recall",
            "AUC-ROC":   "Area under the ROC curve — discrimination ability across all thresholds",
        }
        for metric, definition in metrics_info.items():
            st.markdown(
                f"<p style='font-size:0.84rem;color:{T['text_secondary']};margin-bottom:0.3rem;'>"
                f"<b style='color:{T['text']};'>{metric}</b>: {definition}</p>",
                unsafe_allow_html=True,
            )

    # ── Tab 2: Performance Charts
    with tab_charts:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Per-Metric Bar Charts</div>
    <div class="fl-card-desc">Side-by-side comparison of all three models across every evaluation metric.</div>
</div>
""", unsafe_allow_html=True)

        metrics_to_plot = ["Accuracy","Precision (Fraud)","Recall (Fraud)","F1-Score (Fraud)","AUC-ROC"]
        bar_colors = [T["accent"], T["warning"], T["safe"]]

        row1_cols = st.columns(3, gap="medium")
        row2_cols = st.columns(2, gap="medium")
        all_cols  = row1_cols + row2_cols

        for col_ui, metric in zip(all_cols, metrics_to_plot):
            with col_ui:
                fig, ax = plt.subplots(figsize=(3.2, 3))
                bars = ax.bar(["LR", "RF", "NN"], perf[metric],
                              color=bar_colors, edgecolor=T["card_bg"], linewidth=1, width=0.5)
                ax.set_ylim(0, min(perf[metric].max() * 1.4, 1.0))
                ax.set_title(metric.replace(" (Fraud)", ""), fontsize=10)
                for b, v in zip(bars, perf[metric]):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                            f"{v:.0%}", ha="center", va="bottom",
                            color=T["muted"], fontsize=8, fontfamily="monospace")
                ax.set_ylabel("Score", fontsize=9)
                apply_chart_theme(ax, fig)
                st.pyplot(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{T['muted']};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;font-family:Roboto Mono,monospace;margin-bottom:0.75rem;'>Comparative Analysis</p>", unsafe_allow_html=True)

        cs1, cs2, cs3 = st.columns(3)
        with cs1:
            render_info_card("Random Forest", [
                "Highest accuracy (87%)",
                "Best AUC-ROC (0.81)",
                "Best precision and F1-score",
                "Robust to outliers",
                "Recommended for deployment",
            ])
        with cs2:
            render_info_card("Logistic Regression", [
                "Highest recall (44%)",
                "Fully interpretable coefficients",
                "Fastest inference time",
                "Lowest memory footprint",
                "Best baseline model",
            ])
        with cs3:
            render_info_card("Neural Network", [
                "Matches RF on accuracy (87%)",
                "Strong AUC-ROC (0.80)",
                "Handles complex feature interactions",
                "Requires more compute",
                "Good recall performance",
            ])

    # ── Tab 3: Confusion Matrices
    with tab_confusion:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Confusion Matrices</div>
    <div class="fl-card-desc">Estimated on a 10,000-sample test set with 3.5% fraud rate, derived from reported Precision/Recall metrics.</div>
</div>
""", unsafe_allow_html=True)

        n_test  = 10_000
        n_fraud = int(n_test * 0.035)
        n_legit = n_test - n_fraud

        model_cms = []
        for _, row in perf.iterrows():
            tp = int(row["Recall (Fraud)"] * n_fraud)
            fn = n_fraud - tp
            fp = int(tp / row["Precision (Fraud)"]) - tp if row["Precision (Fraud)"] > 0 else 0
            tn = n_legit - fp
            model_cms.append(np.array([[tn, fp], [fn, tp]]))

        cm_cols = st.columns(3, gap="large")
        for col_ui, cm, name in zip(cm_cols, model_cms, perf["Model"]):
            with col_ui:
                st.markdown(f"""
<div class="fl-card" style="text-align:center;padding-bottom:0.5rem;">
    <div class="fl-card-title">{name}</div>
""", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(3.5, 3))
                fig.patch.set_facecolor(T["card_bg"])
                ax.set_facecolor(T["card_bg"])
                ax.imshow(cm, cmap="Oranges", vmin=0, vmax=n_legit)
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                                color=T["text"] if cm[i,j] < cm.max() * 0.6 else "#FFFFFF",
                                fontsize=11, fontweight="bold", fontfamily="monospace")
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["Pred Legit", "Pred Fraud"], color=T["muted"], fontsize=8)
                ax.set_yticklabels(["Actual Legit", "Actual Fraud"], color=T["muted"], fontsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor(T["border"])
                st.pyplot(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{T['muted']};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;font-family:Roboto Mono,monospace;margin-bottom:0.5rem;'>Matrix Interpretation</p>", unsafe_allow_html=True)

        interp = [
            ("TN — True Negatives",  "Legitimate transactions correctly identified as safe (top-left)."),
            ("FP — False Positives", "Legitimate transactions incorrectly flagged as fraud (top-right). Increases customer friction."),
            ("FN — False Negatives", "Missed fraud cases — the most costly error in financial contexts (bottom-left)."),
            ("TP — True Positives",  "Correctly identified fraudulent transactions (bottom-right). Maximise this."),
        ]
        for label, desc in interp:
            st.markdown(
                f"<p style='font-size:0.84rem;color:{T['text_secondary']};margin-bottom:0.3rem;'>"
                f"<b style='color:{T['text']};'>{label}</b>: {desc}</p>",
                unsafe_allow_html=True,
            )

    # ── Tab 4: Key Insights
    with tab_insights:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Key Findings and Recommendations</div>
    <div class="fl-card-desc">Actionable insights derived from the model evaluation for production deployment.</div>
</div>
""", unsafe_allow_html=True)

        i1, i2, i3 = st.columns(3, gap="medium")
        with i1:
            render_insight_card(
                "Best Overall Model",
                "Random Forest and Neural Network tie at <b>87% accuracy</b>. "
                "Random Forest is recommended for deployment due to superior AUC-ROC (0.81) and interpretability.",
            )
        with i2:
            render_insight_card(
                "Precision-Recall Tradeoff",
                "At 12% precision, expect <b>approximately 8 false positives per true fraud caught</b>. "
                "High false alarm rates degrade customer experience and create operational burden.",
            )
        with i3:
            render_insight_card(
                "Threshold Optimisation",
                "Lowering the decision threshold from 0.5 to 0.3 raises recall to approximately <b>60%</b>. "
                "Tune this value based on business cost-benefit analysis.",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        render_info_card(
            "Deployment Strategy",
            [
                "<b>Primary Model</b>: Deploy Random Forest for production fraud detection",
                "<b>Threshold Tuning</b>: Adjust based on false-positive tolerance and fraud-loss budget",
                "<b>Ensemble Approach</b>: Combine multiple model predictions for high-value flagging decisions",
                "<b>Continuous Retraining</b>: Monthly retraining schedule to adapt to evolving fraud patterns",
                "<b>Feature Engineering</b>: Collect additional temporal and behavioural features to enrich inputs",
            ]
        )
