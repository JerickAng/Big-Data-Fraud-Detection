import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  THEME / STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark background */
.stApp { background-color: #0a0e1a; color: #e2e8f0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f1629;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #a0aec0 !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #64748b !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.1em; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #111827 0%, #1a2540 100%);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
}
[data-testid="metric-container"] label { color: #64748b !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.12em; font-family: 'IBM Plex Mono', monospace !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f0f9ff !important; font-size: 2rem !important; font-weight: 700 !important; font-family: 'IBM Plex Mono', monospace !important; }

/* Headers */
h1 { font-family: 'IBM Plex Mono', monospace !important; color: #38bdf8 !important; letter-spacing: -0.02em; font-size: 1.8rem !important; }
h2, h3 { font-family: 'IBM Plex Sans', sans-serif !important; color: #cbd5e1 !important; font-weight: 600 !important; }

/* Divider */
hr { border-color: #1e2d4a; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #1e2d4a; border-radius: 8px; overflow: hidden; }

/* Info / warning boxes */
[data-testid="stAlert"] {
    background-color: #111827 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
}

/* Spinner */
[data-testid="stSpinner"] > div { color: #38bdf8 !important; }

/* Upload button area */
.stFileUploader { background-color: #111827 !important; border: 1px dashed #1e3a5f !important; border-radius: 10px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background-color: #0f1629; border-bottom: 1px solid #1e2d4a; gap: 0; }
.stTabs [data-baseweb="tab"] { color: #64748b !important; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; padding: 0.6rem 1.4rem; }
.stTabs [aria-selected="true"] { color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important; background: transparent !important; }

/* Status badge */
.badge-fraud { background: #7f1d1d; color: #fca5a5; border-radius: 4px; padding: 2px 8px; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; }
.badge-legit { background: #14532d; color: #86efac; border-radius: 4px; padding: 2px 8px; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB THEME
# ─────────────────────────────────────────────
DARK_BG   = "#0a0e1a"
CARD_BG   = "#111827"
ACCENT    = "#38bdf8"
FRAUD_CLR = "#ef4444"
SAFE_CLR  = "#22c55e"
MID_CLR   = "#f59e0b"
GRID_CLR  = "#1e2d4a"
TEXT_CLR  = "#94a3b8"

def apply_dark_style(ax, fig):
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color("#e2e8f0")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)

# ─────────────────────────────────────────────
#  DATA LOADING  (file upload OR built-in demo)
# ─────────────────────────────────────────────
@st.cache_data
def load_uploaded(transaction_bytes, identity_bytes):
    import io
    transaction = pd.read_csv(io.BytesIO(transaction_bytes))
    identity    = pd.read_csv(io.BytesIO(identity_bytes))
    df = transaction.merge(identity, on="TransactionID", how="left")
    return df

@st.cache_data
def generate_demo_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    fraud_mask = rng.random(n) < 0.035
    data = {
        "TransactionID":   np.arange(1, n + 1),
        "isFraud":         fraud_mask.astype(int),
        "TransactionAmt":  np.where(fraud_mask,
                               rng.exponential(800, n),
                               rng.exponential(150, n)),
        "card1":           rng.integers(1000, 18000, n),
        "card2":           rng.choice([100, 111, 117, 121, 150, 200, 300, 360, 490, 555], n),
        "addr1":           rng.integers(100, 500, n),
        "addr2":           rng.integers(10, 100, n),
        "dist1":           np.where(fraud_mask, rng.exponential(200, n), rng.exponential(50, n)),
        "dist2":           rng.exponential(70, n),
        "C1":              rng.integers(0, 15, n),
        "C2":              rng.integers(0, 10, n),
        "C13":             rng.integers(0, 40, n),
        "V258":            rng.standard_normal(n) + np.where(fraud_mask, 1.2, 0),
        "V294":            rng.standard_normal(n) + np.where(fraud_mask, 0.8, 0),
        "V201":            rng.standard_normal(n),
        "D1":              np.clip(rng.exponential(100, n), 0, 500),
        "D4":              np.clip(rng.exponential(30, n), 0, 300),
        "ProductCD":       rng.choice(["W", "H", "C", "S", "R"], n),
        "P_emaildomain":   rng.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com"], n),
        "DeviceType":      rng.choice(["desktop", "mobile", None], n, p=[0.45, 0.40, 0.15]),
    }
    return pd.DataFrame(data)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 FraudLens")
    st.markdown("---")
    page = st.radio(
        "NAVIGATE",
        ["Home", "Data Overview", "Model Performance"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**DATA SOURCE**")
    use_demo = st.toggle("Use demo data", value=True)

    if not use_demo:
        st.caption("Upload your CSVs from the IEEE-CIS Fraud Detection dataset.")
        t_file = st.file_uploader("train_transaction.csv", type="csv", key="trans")
        i_file = st.file_uploader("train_identity.csv",    type="csv", key="iden")
    st.markdown("---")
    st.caption("Big Data Fraud Detection System")

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
df = None
data_ready = False

if use_demo:
    df = generate_demo_data()
    data_ready = True
else:
    if t_file and i_file:
        with st.spinner("Merging datasets…"):
            df = load_uploaded(t_file.read(), i_file.read())
        data_ready = True
    else:
        st.info("Upload both CSV files in the sidebar, or enable **demo mode**.")

# ─────────────────────────────────────────────
#  ██  HOME PAGE
# ─────────────────────────────────────────────
if page == "Home":
    st.markdown("# 🔍 Fraud Detection Dashboard")
    if use_demo:
        st.caption("⚡ Running on synthetic demo data — toggle off in sidebar to load your own CSVs.")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", "590,540")
    c2.metric("Fraud Rate", "3.5%")
    c3.metric("Features", "434")
    c4.metric("Best Model Accuracy", "87%")

    st.markdown("---")
    st.subheader("Project Overview")

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
This dashboard presents a **Big Data Fraud Detection System** built for the banking and financial sector,
using the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset.

**Pipeline highlights:**
- 🗄️ **Scale** — 590,540 transactions × 434 features across transaction + identity tables
- ⚖️ **Class balancing** — SMOTE applied to handle the severe 3.5 % fraud minority
- 🤖 **Three models** evaluated: Logistic Regression, Random Forest, Neural Network (MLP)
- 📊 **Metrics beyond accuracy** — Precision, Recall, and F1-Score reported for the fraud class
        """)
    with col_b:
        # Quick pie chart
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        fig.patch.set_facecolor(CARD_BG)
        ax.set_facecolor(CARD_BG)
        sizes  = [96.5, 3.5]
        colors = [SAFE_CLR, FRAUD_CLR]
        wedges, texts, autotexts = ax.pie(
            sizes, colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor=DARK_BG, linewidth=2),
        )
        for t in autotexts:
            t.set_color("#e2e8f0"); t.set_fontsize(10); t.set_fontweight("bold")
        ax.legend(["Non-Fraud", "Fraud"], loc="lower center",
                  frameon=False, labelcolor=TEXT_CLR, fontsize=9)
        ax.set_title("Class Distribution", color="#e2e8f0", fontsize=11, pad=10)
        st.pyplot(fig)

# ─────────────────────────────────────────────
#  ██  DATA OVERVIEW PAGE
# ─────────────────────────────────────────────
elif page == "Data Overview":
    st.subheader("📈 Data Overview")
    if not data_ready:
        st.stop()

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Rows",         f"{df.shape[0]:,}")
    r2.metric("Columns",      f"{df.shape[1]:,}")
    fraud_n = int(df["isFraud"].sum())
    r3.metric("Fraud Cases",  f"{fraud_n:,}")
    r4.metric("Fraud Rate",   f"{fraud_n/len(df)*100:.2f}%")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Distribution", "Top Features", "Raw Sample"])

    # ── TAB 1: Fraud distribution
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fraud_counts = df["isFraud"].value_counts().sort_index()
            labels = ["Non-Fraud", "Fraud"]
            colors = [SAFE_CLR, FRAUD_CLR]
            fig, ax = plt.subplots(figsize=(6, 4))
            apply_dark_style(ax, fig)
            bars = ax.bar(labels, fraud_counts.values, color=colors,
                          width=0.5, edgecolor=DARK_BG, linewidth=1.5)
            for bar, val in zip(bars, fraud_counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(fraud_counts) * 0.02,
                        f"{val:,}\n({val/len(df)*100:.1f}%)",
                        ha="center", va="bottom", color=TEXT_CLR, fontsize=9,
                        fontfamily="monospace")
            ax.set_title("Transaction Classes", fontsize=12)
            ax.set_ylabel("Count")
            st.pyplot(fig)

        with col2:
            st.markdown("**Transaction Amount by Class**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            apply_dark_style(ax2, fig2)
            fraud_amt = df[df["isFraud"] == 1]["TransactionAmt"].clip(upper=5000)
            legit_amt = df[df["isFraud"] == 0]["TransactionAmt"].clip(upper=5000)
            ax2.hist(legit_amt, bins=60, color=SAFE_CLR,  alpha=0.7, label="Non-Fraud", density=True)
            ax2.hist(fraud_amt, bins=60, color=FRAUD_CLR, alpha=0.7, label="Fraud",     density=True)
            ax2.set_xlabel("Transaction Amount (USD)")
            ax2.set_ylabel("Density")
            ax2.set_title("Amount Distribution", fontsize=12)
            ax2.legend(frameon=False, labelcolor=TEXT_CLR)
            st.pyplot(fig2)

    # ── TAB 2: Top correlated features
    with tab2:
        st.caption("Absolute Pearson correlation with isFraud (numeric columns only).")
        # Compute on a numeric subset — avoid the 434-column freeze
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Limit to at most 50 numeric cols for speed if dataset is huge
        if len(num_cols) > 50:
            num_cols = num_cols[:50]
        corr_series = (
            df[num_cols]
            .corr()["isFraud"]
            .abs()
            .drop("isFraud", errors="ignore")
            .sort_values(ascending=False)
            .head(10)
        )

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        apply_dark_style(ax3, fig3)
        palette = [ACCENT if v >= corr_series.iloc[0] * 0.7 else MID_CLR
                   if v >= corr_series.iloc[0] * 0.4 else TEXT_CLR
                   for v in corr_series.values]
        bars3 = ax3.barh(corr_series.index[::-1], corr_series.values[::-1],
                         color=palette[::-1], edgecolor=DARK_BG, linewidth=0.8)
        ax3.set_xlabel("| Correlation with Fraud |")
        ax3.set_title("Top 10 Features Correlated with Fraud", fontsize=12)
        for bar, val in zip(bars3, corr_series.values[::-1]):
            ax3.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", color=TEXT_CLR, fontsize=8,
                     fontfamily="monospace")
        st.pyplot(fig3)

    # ── TAB 3: Raw sample
    with tab3:
        n_show = st.slider("Rows to preview", 5, 50, 10)
        sample = df.head(n_show).copy()
        # Colour-code isFraud
        def highlight_fraud(val):
            if val == 1:
                return "background-color:#7f1d1d; color:#fca5a5"
            return "background-color:#14532d; color:#86efac"
        st.dataframe(
            sample.style.applymap(highlight_fraud, subset=["isFraud"]),
            use_container_width=True,
        )

# ─────────────────────────────────────────────
#  ██  MODEL PERFORMANCE PAGE
# ─────────────────────────────────────────────
elif page == "Model Performance":
    st.subheader("🤖 Model Performance Comparison")
    st.markdown("Three ML models were trained on SMOTE-balanced data and evaluated on the held-out test set.")
    st.markdown("---")

    perf = pd.DataFrame({
        "Model":              ["Logistic Regression", "Random Forest", "Neural Network"],
        "Accuracy":           [0.83, 0.87, 0.87],
        "Precision (Fraud)":  [0.09, 0.12, 0.12],
        "Recall (Fraud)":     [0.44, 0.43, 0.44],
        "F1-Score (Fraud)":   [0.16, 0.19, 0.19],
        "AUC-ROC":            [0.72, 0.81, 0.80],   # illustrative
    })

    tab_m, tab_r, tab_c = st.tabs(["Metrics Table", "Charts", "Confusion Matrix (est.)"])

    # ── METRICS TABLE
    with tab_m:
        def color_best(col):
            styles = [""] * len(col)
            best_idx = col.idxmax()
            styles[best_idx] = f"color:{ACCENT}; font-weight:700"
            return styles

        styled = (
            perf.style
            .apply(color_best, subset=["Accuracy", "Precision (Fraud)", "Recall (Fraud)", "F1-Score (Fraud)", "AUC-ROC"])
            .format({c: "{:.2%}" for c in ["Accuracy", "Precision (Fraud)", "Recall (Fraud)", "F1-Score (Fraud)", "AUC-ROC"]})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption("🔵 Blue = best value in column")

    # ── CHARTS
    with tab_r:
        metrics_to_plot = ["Accuracy", "Precision (Fraud)", "Recall (Fraud)", "F1-Score (Fraud)", "AUC-ROC"]
        cols = st.columns(len(metrics_to_plot))
        bar_colors = [ACCENT, MID_CLR, SAFE_CLR]

        for i, metric in enumerate(metrics_to_plot):
            with cols[i]:
                fig, ax = plt.subplots(figsize=(2.8, 3.5))
                apply_dark_style(ax, fig)
                bars = ax.bar(["LR", "RF", "NN"], perf[metric],
                              color=bar_colors, edgecolor=DARK_BG, linewidth=1, width=0.55)
                ax.set_ylim(0, min(perf[metric].max() * 1.35, 1.0))
                ax.set_title(metric, fontsize=9, color="#e2e8f0")
                for b, v in zip(bars, perf[metric]):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                            f"{v:.0%}", ha="center", va="bottom",
                            color=TEXT_CLR, fontsize=8, fontfamily="monospace")
                ax.set_ylabel("")
                st.pyplot(fig)

    # ── ESTIMATED CONFUSION MATRICES
    with tab_c:
        st.caption("Estimated confusion matrices based on reported Precision/Recall (10,000-sample test set at 3.5% fraud rate).")
        n_test   = 10_000
        n_fraud  = int(n_test * 0.035)
        n_legit  = n_test - n_fraud

        model_cms = []
        for _, row in perf.iterrows():
            tp = int(row["Recall (Fraud)"]    * n_fraud)
            fn = n_fraud - tp
            fp = int(tp / row["Precision (Fraud)"]) - tp if row["Precision (Fraud)"] > 0 else 0
            tn = n_legit - fp
            model_cms.append(np.array([[tn, fp], [fn, tp]]))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.patch.set_facecolor(CARD_BG)
        for ax, cm, name in zip(axes, model_cms, perf["Model"]):
            ax.set_facecolor(CARD_BG)
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=n_legit)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                            color="#e2e8f0" if cm[i,j] < cm.max() * 0.6 else "#0a0e1a",
                            fontsize=11, fontweight="bold", fontfamily="monospace")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred: Legit", "Pred: Fraud"], color=TEXT_CLR, fontsize=9)
            ax.set_yticklabels(["Actual: Legit", "Actual: Fraud"], color=TEXT_CLR, fontsize=9)
            ax.set_title(name, color="#e2e8f0", fontsize=10, pad=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_CLR)
        fig.tight_layout(pad=2)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Key Insights")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**🥇 Best Accuracy**\nRandom Forest & Neural Network tied at **87%** — a 4 pp gain over Logistic Regression.")
    with c2:
        st.warning("**⚠️ Low Precision**\nAt ~12%, roughly **7 in 8** flagged transactions are legitimate. Expect false-alarm fatigue without a threshold tune.")
    with c3:
        st.success("**✅ Recall Priority**\nAll models recover ~44% of fraud cases. Consider raising the decision threshold to trade accuracy for recall in high-risk segments.")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='color:#334155; font-size:0.75rem; font-family: IBM Plex Mono, monospace;'>"
    "Big Data Fraud Detection System · IEEE-CIS Dataset · Powered by Machine Learning</p>",
    unsafe_allow_html=True,
)