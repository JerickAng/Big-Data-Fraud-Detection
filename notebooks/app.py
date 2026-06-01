import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design tokens and color scheme
DARK_BG = "#0a0e1a"
CARD_BG = "#111827"
ACCENT = "#38bdf8"
FRAUD_CLR = "#ef4444"
SAFE_CLR = "#22c55e"
MID_CLR = "#f59e0b"
GRID_CLR = "#1e2d4a"
TEXT_CLR = "#94a3b8"
TEXT_LIGHT = "#e2e8f0"

# Apply custom theme with improved typography
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.stApp { background-color: #0a0e1a; color: #e2e8f0; }

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0f1629;
    border-right: 1px solid #1e2d4a;
}

/* Sidebar navigation styling */
[data-testid="stSidebar"] .stRadio { margin: 1.2rem 0; }
[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
    display: block;
    width: 100%;
    min-width: 0;
    padding: 0.9rem 1rem;
    border-radius: 0.95rem;
    border: 1px solid transparent;
    background-color: transparent;
    color: #94a3b8 !important;
    font-size: 1rem;
    font-weight: 500;
    text-align: left;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: all 0.18s ease;
    cursor: pointer;
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
    background-color: rgba(56, 189, 248, 0.12);
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] input[type="radio"] {
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] input[type="radio"]:checked + *,
[data-testid="stSidebar"] .stRadio [role="radiogroup"] input[type="radio"]:checked ~ * {
    background-color: #38bdf8;
    color: #ffffff !important;
    font-weight: 700;
    border-left: 4px solid #ffffff;
    box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.35), 0 10px 24px rgba(56, 189, 248, 0.18);
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] input[type="radio"]:checked + *:hover,
[data-testid="stSidebar"] .stRadio [role="radiogroup"] input[type="radio"]:checked ~ *:hover {
    background-color: #38bdf8;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #111827 0%, #1a2540 100%);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-family: 'Roboto Mono', monospace !important;
    font-weight: 600;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f0f9ff !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    font-family: 'Roboto Mono', monospace !important;
    margin-top: 0.5rem;
}

/* Headers */
h1 {
    font-family: 'Inter', sans-serif !important;
    color: #38bdf8 !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em;
    margin-bottom: 0.5rem;
}

h2 {
    font-family: 'Inter', sans-serif !important;
    color: #cbd5e1 !important;
    font-size: 1.8rem !important;
    font-weight: 600 !important;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

h3 {
    font-family: 'Inter', sans-serif !important;
    color: #cbd5e1 !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    margin-top: 1rem;
    margin-bottom: 0.75rem;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

/* Alert boxes */
[data-testid="stAlert"] {
    background-color: #111827 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-size: 0.95rem;
    padding: 1rem 1.25rem;
}

/* Spinner */
[data-testid="stSpinner"] > div { color: #38bdf8 !important; }

/* File uploader */
.stFileUploader {
    background-color: #111827 !important;
    border: 2px dashed #1e3a5f !important;
    border-radius: 10px;
    padding: 1rem;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0f1629;
    border-bottom: 2px solid #1e2d4a;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    color: #94a3b8 !important;
    font-size: 1rem !important;
    font-weight: 500;
    padding: 0.8rem 1.5rem !important;
    text-transform: capitalize;
}

.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 3px solid #38bdf8 !important;
    background: transparent !important;
}

/* Expander styling */
[data-testid="stExpander"] summary { font-size: 1.1rem !important; font-weight: 600; }

/* Insight boxes */
.insight-box {
    background: linear-gradient(135deg, #111827 0%, #1a2540 100%);
    border-left: 4px solid #38bdf8;
    padding: 1.25rem;
    border-radius: 8px;
    margin-top: 0.75rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    font-size: 0.95rem;
    line-height: 1.6;
}

.insight-box strong { color: #38bdf8; font-size: 1rem; }

/* Chart caption */
.chart-caption {
    color: #94a3b8 !important;
    font-size: 0.9rem;
    font-style: italic;
    margin-top: 0.75rem;
    line-height: 1.5;
}

/* Toggle styling */
.stToggle label { font-size: 1rem !important; font-weight: 500; }

/* Divider */
hr { border-color: #1e2d4a; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# Apply matplotlib dark theme
def apply_dark_style(ax, fig):
    """Apply consistent dark theme styling to matplotlib figures."""
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=10)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_fontsize(11)
    ax.title.set_color(TEXT_LIGHT)
    ax.title.set_fontsize(13)
    ax.title.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)

# Load data from uploaded files
@st.cache_data
def load_uploaded_data(transaction_bytes, identity_bytes):
    """Load and merge transaction and identity CSV files."""
    import io
    transaction = pd.read_csv(io.BytesIO(transaction_bytes))
    identity = pd.read_csv(io.BytesIO(identity_bytes))
    df = transaction.merge(identity, on="TransactionID", how="left")
    return df

# Generate synthetic demo data
@st.cache_data
def generate_demo_data(n=5000, seed=42):
    """Generate realistic synthetic fraud detection data for demo purposes."""
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
        "dist2": rng.exponential(70, n),
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

# Initialize session state for page navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"

# Sidebar navigation and data source selection
with st.sidebar:
    st.markdown("## FraudLens")
    st.markdown("*Fraud Detection & Analysis*")
    st.markdown("---")
    
    st.markdown("### Navigation")
    pages = ["Dashboard", "Data Analysis", "Model Performance"]
    page = st.radio(
        label="",
        options=pages,
        index=pages.index(st.session_state.current_page),
        key="sidebar_page_nav",
        label_visibility="collapsed",
    )
    st.session_state.current_page = page
    
    st.markdown("---")
    st.markdown("### Data Source")
    use_demo = st.toggle("Use Demonstration Data", value=True)
    
    if not use_demo:
        st.markdown("**Upload CSV Files**")
        st.markdown("*IEEE-CIS Fraud Detection Dataset*")
        t_file = st.file_uploader("Transaction Data (train_transaction.csv)", type="csv", key="trans")
        i_file = st.file_uploader("Identity Data (train_identity.csv)", type="csv", key="iden")
    
    st.markdown("---")
    st.markdown("### Project Info")
    st.markdown("**5011CEM Big Data Programming**")
    st.caption("Built with Streamlit, Python, and Machine Learning")

# Load data
df = None
data_ready = False

if use_demo:
    df = generate_demo_data()
    data_ready = True
else:
    if t_file and i_file:
        with st.spinner("Merging datasets…"):
            df = load_uploaded_data(t_file.read(), i_file.read())
        data_ready = True
    else:
        st.info("Upload both CSV files in the sidebar to get started, or enable demonstration mode.")

# Dashboard page
if page == "Dashboard":
    st.markdown("# Fraud Detection Dashboard")
    if use_demo:
        st.caption("Currently viewing synthetic demonstration data — toggle demo mode in sidebar to load real data")
    st.markdown("---")
    
    # Key metrics overview
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", "590,540", delta=None)
    col2.metric("Fraud Cases", "20,663", delta="-5.2%")
    col3.metric("Non-Fraud Cases", "569,877", delta="+2.1%")
    col4.metric("Fraud Rate", "3.5%", delta=None)
    
    st.markdown("---")
    st.subheader("Project Summary")
    
    col_text, col_chart = st.columns([3, 2])
    
    with col_text:
        with st.expander("About This Project", expanded=True):
            st.markdown("""
            This **Big Data Fraud Detection System** is engineered for the banking and financial sector, utilizing the IEEE-CIS Fraud Detection dataset from Kaggle.
            
            **Key System Features:**
            - **Dataset Scale**: 590,540 transactions with 434 engineered features (transaction and identity attributes)
            - **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling) applied to address 3.5% fraud minority
            - **Model Portfolio**: Three algorithms evaluated—Logistic Regression, Random Forest, Neural Network (MLP)
            - **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, and AUC-ROC
            
            **Business Objective**: Identify fraudulent transactions while minimizing false positives that degrade user experience and create operational friction.
            """)
    
    with col_chart:
        st.markdown("**Class Distribution**")
        fig, ax = plt.subplots(figsize=(3.5, 3))
        sizes = [96.5, 3.5]
        colors = [SAFE_CLR, FRAUD_CLR]
        wedges, texts, autotexts = ax.pie(
            sizes, colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor=DARK_BG, linewidth=2),
        )
        for t in autotexts:
            t.set_color(TEXT_LIGHT)
            t.set_fontsize(10)
            t.set_fontweight("bold")
        ax.legend(["Non-Fraud", "Fraud"], loc="lower center", frameon=False, labelcolor=TEXT_CLR, fontsize=9)
        ax.set_title("Transaction Split", color=TEXT_LIGHT, fontsize=12, fontweight="bold", pad=10)
        apply_dark_style(ax, fig)
        st.pyplot(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Key Insights")
    
    insights = [
        ("Fraud Concentration Patterns", "Fraudsters systematically target specific card types and email domains, exhibiting repeated behavioral patterns."),
        ("Transaction Amount Anomalies", "Fraudulent transactions deviate significantly from legitimate patterns—either unusually small or exceptionally large amounts."),
        ("Temporal Risk Indicators", "Fraud incidents concentrate during off-peak hours and weekends when transaction volumes are lower and monitoring is reduced."),
    ]
    
    for title, desc in insights:
        st.markdown(f"<div class='insight-box'><strong>{title}</strong><br>{desc}</div>", unsafe_allow_html=True)

# Data Analysis page
elif page == "Data Analysis":
    st.subheader("Data Analysis & Exploration")
    if not data_ready:
        st.stop()
    
    # Data summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset Rows", f"{df.shape[0]:,}")
    col2.metric("Features", f"{df.shape[1]:,}")
    fraud_count = int(df["isFraud"].sum())
    col3.metric("Fraud Cases", f"{fraud_count:,}")
    col4.metric("Fraud Rate", f"{fraud_count/len(df)*100:.2f}%")
    
    st.markdown("---")
    
    # Analysis sections
    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Patterns", "Features", "Sample Data"])
    
    # Tab 1: Class distribution
    with tab1:
        st.markdown("**Transaction Class Distribution**")
        col1, col2 = st.columns(2)
        
        with col1:
            fraud_counts = df["isFraud"].value_counts().sort_index()
            labels = ["Non-Fraud", "Fraud"]
            colors = [SAFE_CLR, FRAUD_CLR]
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(labels, fraud_counts.values, color=colors, width=0.5, edgecolor=DARK_BG, linewidth=1.5)
            
            for bar, val in zip(bars, fraud_counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + max(fraud_counts) * 0.02,
                       f"{val:,}\n({val/len(df)*100:.1f}%)",
                       ha="center", va="bottom", color=TEXT_CLR, fontsize=9, fontfamily="monospace")
            
            ax.set_title("Class Balance", fontsize=12)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_ylim(0, max(fraud_counts) * 1.12)
            apply_dark_style(ax, fig)
            st.pyplot(fig, use_container_width=True)
        st.markdown("<p class='chart-caption'>The dataset exhibits severe class imbalance with 96.5% legitimate transactions. This is typical for fraud detection problems and requires specialized handling (SMOTE).</p>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Transaction Amount Distribution**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fraud_amt = df[df["isFraud"] == 1]["TransactionAmt"].clip(upper=5000)
            legit_amt = df[df["isFraud"] == 0]["TransactionAmt"].clip(upper=5000)
            
            ax2.hist(legit_amt, bins=60, color=SAFE_CLR, alpha=0.7, label="Non-Fraud", density=True)
            ax2.hist(fraud_amt, bins=60, color=FRAUD_CLR, alpha=0.7, label="Fraud", density=True)
            
            ax2.set_xlabel("Transaction Amount (USD)", fontsize=10)
            ax2.set_ylabel("Density", fontsize=10)
            ax2.set_title("Amount Comparison", fontsize=12)
            ax2.legend(frameon=False, labelcolor=TEXT_CLR, fontsize=9)
            apply_dark_style(ax2, fig2)
            st.pyplot(fig2, use_container_width=True)
            st.markdown("<p class='chart-caption'>Fraudulent transactions show distinct statistical patterns—exhibiting both extremely small amounts (to test card validity) and unusually large amounts (for high-value theft).</p>", unsafe_allow_html=True)
    
    # Tab 2: Key patterns
    with tab2:
        st.markdown("**Top Features by Fraud Correlation**")
        st.markdown("*Absolute Pearson correlation with isFraud label (numeric columns only)*")
        
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
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
        
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        palette = [ACCENT if v >= corr_series.iloc[0] * 0.7 else MID_CLR
                  if v >= corr_series.iloc[0] * 0.4 else TEXT_CLR
                  for v in corr_series.values]
        
        bars3 = ax3.barh(corr_series.index[::-1], corr_series.values[::-1],
                        color=palette[::-1], edgecolor=DARK_BG, linewidth=0.8)
        
        ax3.set_xlabel("Absolute Correlation", fontsize=10)
        ax3.set_title("Top 10 Features Associated with Fraud", fontsize=12)
        
        for bar, val in zip(bars3, corr_series.values[::-1]):
            ax3.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", color=TEXT_CLR, fontsize=8, fontfamily="monospace")
        
        apply_dark_style(ax3, fig3)
        st.pyplot(fig3, use_container_width=True)
        st.markdown("<p class='chart-caption'>These features show the strongest statistical relationship with fraudulent transactions.</p>", unsafe_allow_html=True)
    
    # Tab 3: Feature info
    with tab3:
        st.markdown("**Feature Overview**")
        feature_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.values,
            "Missing %": (df.isnull().sum() / len(df) * 100).values,
            "Unique Values": [df[col].nunique() for col in df.columns],
        })
        
        st.dataframe(feature_info, use_container_width=True, hide_index=True)
        st.caption("Note: Check missing percentages to identify potential data quality issues and inform preprocessing decisions.")
    
    # Tab 4: Sample data
    with tab4:
        st.markdown("**Raw Data Sample**")
        n_show = st.slider("Rows to display", 5, 50, 10)
        sample = df.head(n_show).copy()
        
        def highlight_fraud(val):
            if val == 1:
                return "background-color:#7f1d1d; color:#fca5a5"
            elif val == 0:
                return "background-color:#14532d; color:#86efac"
            return ""
        
        st.dataframe(
            sample.style.applymap(highlight_fraud, subset=["isFraud"]),
            use_container_width=True,
        )

# Model Performance page
elif page == "Model Performance":
    st.subheader("Machine Learning Model Evaluation")
    st.markdown("*Evaluated on held-out test set with SMOTE-balanced training data*")
    st.markdown("---")
    
    # Model performance data
    perf = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "Neural Network"],
        "Accuracy": [0.83, 0.87, 0.87],
        "Precision (Fraud)": [0.09, 0.12, 0.12],
        "Recall (Fraud)": [0.44, 0.43, 0.44],
        "F1-Score (Fraud)": [0.16, 0.19, 0.19],
        "AUC-ROC": [0.72, 0.81, 0.80],
    })
    
    tab_summary, tab_charts, tab_confusion, tab_insights = st.tabs(
        ["Metrics Summary", "Performance Charts", "Confusion Matrices", "Key Insights"]
    )
    
    # Tab 1: Metrics table
    with tab_summary:
        st.markdown("**Model Performance Metrics**")
        
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
        st.caption("Highlighted values indicate the best performance in each metric.")
        
        st.markdown("---")
        st.markdown("**Metric Definitions**")
        
        metrics_info = {
            "Accuracy": "Overall correctness of predictions: (TP + TN) / Total",
            "Precision": "Of all flagged transactions, how many were truly fraudulent: TP / (TP + FP)",
            "Recall": "Of all actual fraud cases, what fraction did the model successfully catch: TP / (TP + FN)",
            "F1-Score": "Harmonic mean of Precision and Recall, balancing false positives and false negatives",
            "AUC-ROC": "Area Under the Receiver Operating Characteristic curve—model's discrimination ability across thresholds (0.5-1.0 is good)",
        }
        
        for metric, definition in metrics_info.items():
            st.markdown(f"**{metric}**: {definition}")
    
    # Tab 2: Performance charts
    with tab_charts:
        st.markdown("**Performance Across Metrics**")
        
        metrics_to_plot = ["Accuracy", "Precision (Fraud)", "Recall (Fraud)", "F1-Score (Fraud)", "AUC-ROC"]
        cols = st.columns(len(metrics_to_plot))
        bar_colors = [ACCENT, MID_CLR, SAFE_CLR]
        
        for i, metric in enumerate(metrics_to_plot):
            with cols[i]:
                fig, ax = plt.subplots(figsize=(2.8, 3.2))
                bars = ax.bar(["LR", "RF", "NN"], perf[metric],
                            color=bar_colors, edgecolor=DARK_BG, linewidth=1, width=0.55)
                
                ax.set_ylim(0, min(perf[metric].max() * 1.35, 1.0))
                ax.set_title(metric.replace(" (Fraud)", ""), fontsize=9, color=TEXT_LIGHT, fontweight="bold")
                
                for b, v in zip(bars, perf[metric]):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                           f"{v:.0%}", ha="center", va="bottom",
                           color=TEXT_CLR, fontsize=8, fontfamily="monospace")
                
                ax.set_ylabel("Score", fontsize=8)
                apply_dark_style(ax, fig)
                st.pyplot(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("**Comparative Analysis**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Random Forest Strengths**")
            st.markdown("""
            • Highest accuracy (87%)
            • Best AUC-ROC score (0.81)
            • Best precision and F1-score
            • Robust handling of outliers
            """)
        
        with col2:
            st.markdown("**Logistic Regression Strengths**")
            st.markdown("""
            • Highly interpretable coefficients
            • Fast inference time
            • Low memory footprint
            • Good baseline model for comparison
            """)
    
    # Tab 3: Confusion matrices
    with tab_confusion:
        st.markdown("**Estimated Confusion Matrices**")
        st.caption("Based on reported Precision/Recall on 10,000-sample test set with 3.5% fraud rate")
        
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
        
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.patch.set_facecolor(CARD_BG)
        
        for ax, cm, name in zip(axes, model_cms, perf["Model"]):
            ax.set_facecolor(CARD_BG)
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=n_legit)
            
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                           color=TEXT_LIGHT if cm[i,j] < cm.max() * 0.6 else DARK_BG,
                           fontsize=11, fontweight="bold", fontfamily="monospace")
            
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Predicted\nLegitimate", "Predicted\nFraud"], color=TEXT_CLR, fontsize=8)
            ax.set_yticklabels(["Actual\nLegitimate", "Actual\nFraud"], color=TEXT_CLR, fontsize=8)
            ax.set_title(name, color=TEXT_LIGHT, fontsize=10, fontweight="bold", pad=10)
            
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_CLR)
        
        fig.tight_layout(pad=2)
        st.pyplot(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("**Confusion Matrix Interpretation:**")
        st.markdown("""
        - **True Negatives (TN)** — Correctly identified legitimate transactions (green, lower-left)
        - **False Positives (FP)** — Legitimate transactions incorrectly flagged (upper-left)
        - **False Negatives (FN)** — Missed fraud cases (lower-right)
        - **True Positives (TP)** — Correctly identified fraudulent transactions (upper-right)
        """)
    
    # Tab 4: Key insights
    with tab_insights:
        st.markdown("**Key Findings & Recommendations**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"<div class='insight-box'><strong>Best Overall Model</strong><br>Random Forest and Neural Network are tied at <strong>87% accuracy</strong> with superior AUC-ROC scores. Random Forest is recommended for superior interpretability and practical deployment.</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"<div class='insight-box'><strong>Precision-Recall Tradeoff Challenge</strong><br>At 12% precision, expect <strong>8 false positives per fraud caught</strong>. High false alarm rates significantly degrade user experience and create operational burden.</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"<div class='insight-box'><strong>Threshold Optimization Strategy</strong><br>Lowering the decision threshold from 0.5 to 0.3 increases recall to approximately <strong>60%</strong>, catching more fraud but sacrificing precision. Tune based on business cost-benefit analysis.</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Business Recommendations & Deployment Strategy**")
        st.markdown("""
        1. **Primary Model**: Deploy Random Forest for production fraud detection
        2. **Threshold Tuning**: Adjust decision threshold based on false-positive tolerance and fraud-loss budget
        3. **Ensemble Approach**: Combine predictions from multiple models for critical flagging decisions
        4. **Continuous Monitoring**: Retrain models monthly to adapt to evolving fraud patterns
        5. **Feature Development**: Collect additional temporal and behavioral features for model enrichment
        """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='color:#334155; font-size:0.8rem; font-family: Roboto Mono, monospace; text-align: center; margin-top: 2rem;'>"
    "Fraud Detection Dashboard  |  IEEE-CIS Dataset  |  5011CEM Big Data Programming Project  |  Developed with Streamlit and Python</p>",
    unsafe_allow_html=True,
)