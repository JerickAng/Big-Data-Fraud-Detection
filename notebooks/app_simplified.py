import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FraudLens",
    layout="wide",
)

THEME = {
    "bg": "#FAF9F6",
    "sidebar_bg": "#EEE9DF",
    "card_bg": "#FFFFFF",
    "card_bg2": "#F7F5F0",
    "border": "#DDD6C8",
    "accent": "#2F6B5F",
    "text": "#2F3A35",
    "muted": "#6B7280",
    "fraud": "#C94C4C",
    "safe": "#4E8B57",
    "warning": "#C98A2E",
    "grid": "#E7E0D6",
    "nav_selected": "#E6F1EA",
    "nav_text": "#1F4337",
}


def inject_custom_css(t: dict):
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: {t['bg']};
    color: {t['text']};
}}
.stApp {{
    background: {t['bg']};
}}

[data-testid="stSidebar"] {{
    background: {t['sidebar_bg']};
    border-right: 1px solid {t['border']};
    padding: 1rem 0.75rem 1.5rem 0.75rem;
}}

.sidebar-logo {{
    padding-bottom: 1rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid {t['border']};
}}
.sidebar-logo-title {{
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: {t['text']};
    margin: 0;
}}
.sidebar-section {{
    color: {t['muted']};
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}}

[data-testid="stSidebar"] .stButton>button {{
    width: 100%;
    text-align: left;
    padding: 0.85rem 1rem;
    border-radius: 0.9rem;
    border: 1px solid transparent;
    background-color: transparent;
    color: {t['text']};
    font-size: 0.95rem;
    font-weight: 600;
    transition: background-color 0.15s ease, border-color 0.15s ease;
    margin-bottom: 0.35rem;
}}
[data-testid="stSidebar"] .stButton>button:hover {{
    background-color: #F4EEE4;
    border-color: {t['border']};
}}

.nav-item-selected {{
    background-color: {t['nav_selected']};
    color: {t['nav_text']};
    border-left: 4px solid {t['accent']};
    border-radius: 0.9rem;
    padding: 0.85rem 1rem;
    margin-bottom: 0.65rem;
    font-weight: 600;
}}

[data-testid="metric-container"] {{
    background: {t['card_bg']};
    border: 1px solid {t['border']};
    border-radius: 1rem;
    padding: 1.25rem 1.3rem;
    box-shadow: 0 10px 20px rgba(47, 58, 53, 0.06);
}}

[data-testid="metric-container"] label {{
    color: {t['muted']} !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em;
}}

[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {t['text']} !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}}

.fl-card {{
    background: {t['card_bg']};
    border: 1px solid {t['border']};
    border-radius: 1rem;
    padding: 1.4rem 1.4rem;
    box-shadow: 0 10px 20px rgba(47, 58, 53, 0.05);
    margin-bottom: 1rem;
}}

.fl-card-title {{
    color: {t['accent']};
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}}

.fl-card-desc {{
    color: {t['muted']};
    font-size: 0.92rem;
    line-height: 1.7;
}}

.page-header {{
    padding-bottom: 1rem;
    border-bottom: 1px solid {t['border']};
    margin-bottom: 1.25rem;
}}

.page-header h1 {{
    color: {t['text']};
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}}

.chart-caption {{
    color: {t['muted']};
    font-size: 0.86rem;
    margin-top: 0.65rem;
    line-height: 1.6;
}}

[data-testid="stDataFrame"] {{
    border: 1px solid {t['border']};
    border-radius: 0.9rem;
    overflow: hidden;
}}

.stFileUploader {{
    background-color: {t['card_bg']};
    border: 1px dashed {t['border']} !important;
    border-radius: 0.9rem;
}}

.stTabs [data-baseweb="tab-list"] {{
    background-color: {t['card_bg']};
    border-bottom: 1px solid {t['border']};
    border-radius: 1rem 1rem 0 0;
    margin-bottom: 1rem;
}}

.stTabs [data-baseweb="tab"] {{
    color: {t['muted']} !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 1rem !important;
}}

.stTabs [aria-selected="true"] {{
    color: {t['accent']} !important;
    border-bottom: 3px solid {t['accent']} !important;
    font-weight: 700 !important;
}}
</style>
""", unsafe_allow_html=True)


def apply_chart_style(ax, fig, t: dict):
    fig.patch.set_facecolor(t["card_bg"])
    ax.set_facecolor(t["card_bg"])
    ax.tick_params(colors=t["muted"], labelsize=9)
    ax.xaxis.label.set_color(t["muted"])
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_color(t["muted"])
    ax.yaxis.label.set_fontsize(10)
    ax.title.set_color(t["text"])
    ax.title.set_fontsize(12)
    ax.title.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_edgecolor(t["border"])
    ax.grid(True, color=t["grid"], linewidth=0.7, linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)


def render_page_header(title: str):
    st.markdown(f"""
<div class="page-header">
    <h1>{title}</h1>
</div>
""", unsafe_allow_html=True)


def render_chart_card(title: str, description: str = ""):
    st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">{title}</div>
    {f'<div class="fl-card-desc">{description}</div>' if description else ''}
</div>
""", unsafe_allow_html=True)


def render_info_card(title: str, items: list):
    bullets = "".join(
        f"<li style='margin-bottom:0.35rem;color:{THEME['muted']};font-size:0.88rem;'>{item}</li>"
        for item in items
    )
    st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">{title}</div>
    <ul style="margin:0;padding-left:1.2rem;">{bullets}</ul>
</div>
""", unsafe_allow_html=True)


def render_sidebar() -> tuple:
    with st.sidebar:
        st.markdown(
            """
<div class="sidebar-logo">
    <p class="sidebar-logo-title">FraudLens</p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
        pages = ["Dashboard", "Data Analysis", "Model Performance"]
        for page_name in pages:
            if st.session_state.current_page == page_name:
                st.markdown(
                    f'<div class="nav-item-selected">{page_name}</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(page_name, key=f"nav_{page_name}"):
                    st.session_state.current_page = page_name

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">Data Source</div>', unsafe_allow_html=True)
        use_demo = st.checkbox("Use demonstration data", value=True, key="demo_toggle")
        t_file = i_file = None
        if not use_demo:
            t_file = st.file_uploader("Transaction CSV", type="csv", key="trans")
            i_file = st.file_uploader("Identity CSV", type="csv", key="iden")

    return use_demo, t_file, i_file


@st.cache_data
def load_uploaded_data(transaction_bytes, identity_bytes):
    import io
    transaction = pd.read_csv(io.BytesIO(transaction_bytes))
    identity = pd.read_csv(io.BytesIO(identity_bytes))
    return transaction.merge(identity, on="TransactionID", how="left")


@st.cache_data
def generate_demo_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    fraud_mask = rng.random(n) < 0.035
    data = {
        "TransactionID": np.arange(1, n + 1),
        "isFraud": fraud_mask.astype(int),
        "TransactionAmt": np.where(fraud_mask, rng.exponential(800, n), rng.exponential(150, n)),
        "card1": rng.integers(1000, 18000, n),
        "card2": rng.choice([100, 111, 117, 121, 150, 200, 300, 360, 490, 555], n),
        "addr1": rng.integers(100, 500, n),
        "addr2": rng.integers(10, 100, n),
        "dist1": np.where(fraud_mask, rng.exponential(200, n), rng.exponential(50, n)),
        "dist2": np.random.exponential(70, n),
        "C1": rng.integers(0, 15, n),
        "C2": rng.integers(0, 10, n),
        "C13": rng.integers(0, 40, n),
        "V258": rng.standard_normal(n) + np.where(fraud_mask, 1.2, 0),
        "V294": rng.standard_normal(n) + np.where(fraud_mask, 0.8, 0),
        "V201": rng.standard_normal(n),
        "D1": np.clip(rng.exponential(100, n), 0, 500),
        "D4": np.clip(rng.exponential(30, n), 0, 300),
        "ProductCD": rng.choice(["W", "H", "C", "S", "R"], n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com"], n),
        "DeviceType": rng.choice(["desktop", "mobile", None], n, p=[0.45, 0.40, 0.15]),
    }
    return pd.DataFrame(data)


if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"

use_demo, t_file, i_file = render_sidebar()

inject_custom_css(THEME)


df = None

data_ready = False

if use_demo:
    df = generate_demo_data()
    data_ready = True
elif t_file and i_file:
    with st.spinner("Merging datasets..."):
        df = load_uploaded_data(t_file.read(), i_file.read())
    data_ready = True
else:
    st.info("Upload both CSV files in the sidebar to get started, or enable demonstration data.")

page = st.session_state.current_page

if page == "Dashboard":
    render_page_header("Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Transactions", "590,540")
    with c2:
        st.metric("Fraud Cases", "20,663")
    with c3:
        st.metric("Non-Fraud Cases", "569,877")
    with c4:
        st.metric("Fraud Rate", "3.50%")

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Overview</div>
    <div class="fl-card-desc">
        This dashboard presents fraud detection metrics, transaction patterns, and model performance for the loaded dataset.
    </div>
    <ul style="margin:0;padding-left:1.2rem;line-height:1.8;">
        <li style="color:{THEME['muted']};font-size:0.88rem;">Dataset scale: 590,540 transactions with 434 engineered features</li>
        <li style="color:{THEME['muted']};font-size:0.88rem;">Imbalance handling: SMOTE applied during model training</li>
        <li style="color:{THEME['muted']};font-size:0.88rem;">Models: Logistic Regression, Random Forest, Neural Network</li>
        <li style="color:{THEME['muted']};font-size:0.88rem;">Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC</li>
    </ul>
</div>
""", unsafe_allow_html=True)

    with col_right:
        render_chart_card(
            "Class Distribution",
            "Proportion of legitimate and fraudulent transactions.",
        )
        fig, ax = plt.subplots(figsize=(4, 3.2))
        sizes = [96.5, 3.5]
        colors = [THEME['safe'], THEME['fraud']]
        wedges, texts, autotexts = ax.pie(
            sizes,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.72,
            wedgeprops=dict(width=0.52, edgecolor=THEME['bg'], linewidth=2),
        )
        for at in autotexts:
            at.set_color(THEME['text'])
            at.set_fontsize(10)
            at.set_fontweight("bold")
        ax.legend(["Non-Fraud", "Fraud"], loc="lower center", frameon=False, labelcolor=THEME['muted'], fontsize=9)
        ax.set_title("Transaction Split", color=THEME['text'], fontsize=11, fontweight="bold", pad=8)
        apply_chart_style(ax, fig, THEME)
        st.pyplot(fig, use_container_width=True)

        render_info_card(
            "Fraud Risk Summary",
            [
                "Best model: Random Forest (AUC 0.81)",
                "Recall: 43% of fraud cases detected",
                "Precision: 12% of flagged cases are fraud",
                "SMOTE: balances training class ratios",
                "Threshold: tuned between 0.3 and 0.7",
            ],
        )

elif page == "Data Analysis":
    render_page_header("Data Analysis")

    if not data_ready:
        st.stop()

    fraud_count = int(df["isFraud"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Rows", f"{df.shape[0]:,}")
    c2.metric("Features", f"{df.shape[1]:,}")
    c3.metric("Fraud Cases", f"{fraud_count:,}")
    c4.metric("Fraud Rate", f"{fraud_count/len(df)*100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Patterns", "Features", "Sample Data"])

    with tab1:
        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            render_chart_card(
                "Class Balance",
                "Comparison of legitimate and fraudulent transaction counts.",
            )
            fraud_counts = df["isFraud"].value_counts().sort_index()
            labels = ["Non-Fraud", "Fraud"]
            colors = [THEME['safe'], THEME['fraud']]

            fig, ax = plt.subplots(figsize=(5, 3.8))
            bars = ax.bar(labels, fraud_counts.values, color=colors, width=0.45, edgecolor=THEME['bg'], linewidth=1.5)
            for bar, val in zip(bars, fraud_counts.values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(fraud_counts) * 0.02,
                    f"{val:,}\n({val/len(df)*100:.1f}%)",
                    ha="center",
                    va="bottom",
                    color=THEME['muted'],
                    fontsize=8,
                    fontfamily="monospace",
                )
            ax.set_ylabel("Count", fontsize=10)
            ax.set_ylim(0, max(fraud_counts) * 1.15)
            ax.set_title("Class Balance", fontsize=12)
            apply_chart_style(ax, fig, THEME)
            st.pyplot(fig, use_container_width=True)
            st.markdown(
                "<p class='chart-caption'>Severe class imbalance: 96.5% legitimate transactions. SMOTE is applied during model training to handle this.</p>",
                unsafe_allow_html=True,
            )

        with col_b:
            render_chart_card(
                "Transaction Amount Distribution",
                "Amount densities for fraud and legitimate transactions.",
            )
            fig2, ax2 = plt.subplots(figsize=(5, 3.8))
            fraud_amt = df[df["isFraud"] == 1]["TransactionAmt"].clip(upper=5000)
            legit_amt = df[df["isFraud"] == 0]["TransactionAmt"].clip(upper=5000)

            ax2.hist(legit_amt, bins=60, color=THEME['safe'], alpha=0.65, label="Non-Fraud", density=True)
            ax2.hist(fraud_amt, bins=60, color=THEME['fraud'], alpha=0.65, label="Fraud", density=True)
            ax2.set_xlabel("Transaction Amount (USD)")
            ax2.set_ylabel("Density")
            ax2.set_title("Amount Comparison")
            ax2.legend(frameon=False, labelcolor=THEME['muted'], fontsize=9)
            apply_chart_style(ax2, fig2, THEME)
            st.pyplot(fig2, use_container_width=True)
            st.markdown(
                "<p class='chart-caption'>Fraudulent transactions show bimodal behaviour — small test amounts and occasional large spikes.</p>",
                unsafe_allow_html=True,
            )

    with tab2:
        render_chart_card(
            "Top Features by Fraud Correlation",
            "Absolute Pearson correlation with the fraud label for numeric features.",
        )
        num_cols = df.select_dtypes(include=np.number).columns.tolist()[:50]
        corr_series = (
            df[num_cols]
            .corr()["isFraud"]
            .abs()
            .drop("isFraud", errors="ignore")
            .sort_values(ascending=False)
            .head(10)
        )

        palette = [
            THEME['accent'] if v >= corr_series.iloc[0] * 0.7 else
            THEME['warning'] if v >= corr_series.iloc[0] * 0.4 else
            THEME['muted']
            for v in corr_series.values
        ]

        fig3, ax3 = plt.subplots(figsize=(10, 4.5))
        bars3 = ax3.barh(corr_series.index[::-1], corr_series.values[::-1], color=palette[::-1], edgecolor=THEME['bg'], linewidth=0.8)
        ax3.set_xlabel("Absolute Correlation")
        ax3.set_title("Top 10 Features Associated with Fraud")
        for bar, val in zip(bars3, corr_series.values[::-1]):
            ax3.text(
                val + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                color=THEME['muted'],
                fontsize=8,
                fontfamily="monospace",
            )
        apply_chart_style(ax3, fig3, THEME)
        st.pyplot(fig3, use_container_width=True)
        st.markdown(
            "<p class='chart-caption'>Features are shown by strength of association with fraud.</p>",
            unsafe_allow_html=True,
        )

    with tab3:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Feature Overview</div>
    <div class="fl-card-desc">Data type, missing percentage, and unique counts for each dataset column.</div>
</div>
""", unsafe_allow_html=True)

        feature_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.values,
            "Missing %": (df.isnull().sum() / len(df) * 100).round(2).values,
            "Unique Values": [df[col].nunique() for col in df.columns],
        })
        st.dataframe(feature_info, use_container_width=True, hide_index=True)
        st.markdown(
            "<p class='chart-caption'>Review missing values and unique counts to guide preprocessing decisions.</p>",
            unsafe_allow_html=True,
        )

    with tab4:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Raw Data Sample</div>
    <div class="fl-card-desc">Preview a sample of the loaded transactions.</div>
</div>
""", unsafe_allow_html=True)

        n_show = st.slider("Rows to display", 5, 50, 10)
        sample = df.head(n_show).copy()

        def highlight_fraud(val):
            if val == 1:
                return "background-color:#FCE8E8; color:#7C2626"
            if val == 0:
                return "background-color:#E8F2EA; color:#2F513F"
            return ""

        st.dataframe(sample.style.applymap(highlight_fraud, subset=["isFraud"]), use_container_width=True)

elif page == "Model Performance":
    render_page_header("Model Performance")

    perf = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "Neural Network"],
        "Accuracy": [0.83, 0.87, 0.87],
        "Precision (Fraud)": [0.09, 0.12, 0.12],
        "Recall (Fraud)": [0.44, 0.43, 0.44],
        "F1-Score (Fraud)": [0.16, 0.19, 0.19],
        "AUC-ROC": [0.72, 0.81, 0.80],
    })

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Best Accuracy</div>
    <div style="font-size:1.5rem;font-weight:700;color:{THEME['text']};">Random Forest 87%</div>
</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Best AUC-ROC</div>
    <div style="font-size:1.5rem;font-weight:700;color:{THEME['text']};">Random Forest 0.81</div>
</div>
""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Best F1-Score</div>
    <div style="font-size:1.5rem;font-weight:700;color:{THEME['text']};">RF & NN 19%</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab_summary, tab_charts, tab_confusion, tab_insights = st.tabs([
        "Metrics Summary",
        "Performance Charts",
        "Confusion Matrices",
        "Key Insights",
    ])

    with tab_summary:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Model Comparison</div>
    <div class="fl-card-desc">Comparison of key model metrics on the evaluation set.</div>
</div>
""", unsafe_allow_html=True)

        def color_best(col):
            styles = [""] * len(col)
            best_idx = col.idxmax()
            styles[best_idx] = f"color:{THEME['accent']}; font-weight:700; background: {THEME['nav_selected']}"
            return styles

        styled = (
            perf.style
            .apply(color_best, subset=["Accuracy", "Precision (Fraud)", "Recall (Fraud)", "F1-Score (Fraud)", "AUC-ROC"])
            .format({c: "{:.2%}" for c in ["Accuracy", "Precision (Fraud)", "Recall (Fraud)", "F1-Score (Fraud)", "AUC-ROC"]})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.markdown(
            "<p class='chart-caption'>Highlighted values show the best-performing model for each metric.</p>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        metrics_info = {
            "Accuracy": "Overall correctness: (TP + TN) / Total",
            "Precision": "Of flagged transactions, the share that is truly fraud: TP / (TP + FP)",
            "Recall": "Of actual fraud cases, the share detected: TP / (TP + FN)",
            "F1-Score": "Harmonic mean of precision and recall.",
            "AUC-ROC": "Discrimination ability across thresholds.",
        }
        for metric, definition in metrics_info.items():
            st.markdown(
                f"<p style='font-size:0.88rem;color:{THEME['muted']};margin-bottom:0.4rem;'><b style='color:{THEME['text']};'>{metric}</b>: {definition}</p>",
                unsafe_allow_html=True,
            )

    with tab_charts:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Performance Charts</div>
    <div class="fl-card-desc">Visual comparison of model performance metrics.</div>
</div>
""", unsafe_allow_html=True)

        metrics_to_plot = ["Accuracy", "Precision (Fraud)", "Recall (Fraud)", "F1-Score (Fraud)", "AUC-ROC"]
        bar_colors = [THEME['accent'], THEME['warning'], THEME['safe']]

        row1_cols = st.columns(3, gap="medium")
        row2_cols = st.columns(2, gap="medium")
        all_cols = row1_cols + row2_cols

        for col_ui, metric in zip(all_cols, metrics_to_plot):
            with col_ui:
                fig, ax = plt.subplots(figsize=(3.2, 3))
                bars = ax.bar(["LR", "RF", "NN"], perf[metric], color=bar_colors, edgecolor=THEME['bg'], linewidth=1, width=0.5)
                ax.set_ylim(0, min(perf[metric].max() * 1.4, 1.0))
                ax.set_title(metric.replace(" (Fraud)", ""), fontsize=10)
                for b, v in zip(bars, perf[metric]):
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        b.get_height() + 0.01,
                        f"{v:.0%}",
                        ha="center",
                        va="bottom",
                        color=THEME['muted'],
                        fontsize=8,
                        fontfamily="monospace",
                    )
                ax.set_ylabel("Score", fontsize=9)
                apply_chart_style(ax, fig, THEME)
                st.pyplot(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:{THEME['muted']};font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.75rem;'>Comparative analysis</div>",
            unsafe_allow_html=True,
        )

        cs1, cs2, cs3 = st.columns(3)
        with cs1:
            render_info_card("Random Forest", [
                "Highest accuracy (87%)",
                "Best AUC-ROC (0.81)",
                "Strong precision and F1-score",
                "Robust to outliers",
            ])
        with cs2:
            render_info_card("Logistic Regression", [
                "Highest recall (44%)",
                "Interpretable coefficients",
                "Fast inference performance",
                "Low memory footprint",
            ])
        with cs3:
            render_info_card("Neural Network", [
                "Matching accuracy with RF",
                "Strong AUC-ROC (0.80)",
                "Captures complex interactions",
                "Requires more compute",
            ])

    with tab_confusion:
        render_chart_card(
            "Confusion Matrices",
            "Estimated confusion matrix counts based on reported metrics.",
        )

        n_test = 10_000
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
                fig.patch.set_facecolor(THEME['card_bg'])
                ax.set_facecolor(THEME['card_bg'])
                ax.imshow(cm, cmap="Blues", vmin=0, vmax=n_legit)
                for i in range(2):
                    for j in range(2):
                        ax.text(
                            j,
                            i,
                            f"{cm[i,j]:,}",
                            ha="center",
                            va="center",
                            color=THEME['text'] if cm[i,j] < cm.max() * 0.6 else THEME['bg'],
                            fontsize=11,
                            fontweight="bold",
                            fontfamily="monospace",
                        )
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["Pred Legit", "Pred Fraud"], color=THEME['muted'], fontsize=8)
                ax.set_yticklabels(["Actual Legit", "Actual Fraud"], color=THEME['muted'], fontsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor(THEME['border'])
                st.pyplot(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        for label, desc in [
            ("TN — True Negatives", "Legitimate transactions correctly identified."),
            ("FP — False Positives", "Legitimate transactions incorrectly flagged as fraud."),
            ("FN — False Negatives", "Missed fraud cases."),
            ("TP — True Positives", "Correctly identified fraudulent transactions."),
        ]:
            st.markdown(
                f"<p style='font-size:0.88rem;color:{THEME['muted']};margin-bottom:0.35rem;'><b style='color:{THEME['text']};'>{label}</b>: {desc}</p>",
                unsafe_allow_html=True,
            )

    with tab_insights:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Key Findings</div>
    <div class="fl-card-desc">Practical conclusions from the evaluation and model comparison.</div>
</div>
""", unsafe_allow_html=True)

        i1, i2, i3 = st.columns(3, gap="medium")
        with i1:
            render_info_card("Best Overall Model", [
                "Random Forest and Neural Network achieve 87% accuracy.",
                "Random Forest shows the strongest AUC-ROC.",
                "RF is recommended for production deployment.",
            ])
        with i2:
            render_info_card("Precision-Recall", [
                "Precision is 12% with the current threshold.",
                "Expect several false positives per true fraud case.",
                "Threshold tuning is required for deployment decisions.",
            ])
        with i3:
            render_info_card("Deployment Notes", [
                "Use a tuned threshold based on business cost tradeoffs.",
                "Consider ensemble signals for high-value transactions.",
                "Retrain regularly to adapt to changing fraud patterns.",
            ])
