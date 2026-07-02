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
    page_title="Fraud Detection Supermodel",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixed Theme — muted warm neutrals
# ──────────────────────────────────────────────────────────────────────────────
T = {
    "bg":             "#F5F3EF",
    "sidebar_bg":     "#EDEAE3",
    "card_bg":        "#FAFAF8",
    "border":         "#D8D3C8",
    "accent":         "#3D6B5E",
    "text":           "#1E2822",
    "text_secondary": "#4A5568",
    "muted":          "#6E7A74",
    "fraud":          "#A83232",
    "safe":           "#3A7048",
    "warning":        "#8C6320",
    "grid":           "#E2DDD6",
    "nav_active":     "#3D6B5E",
    "nav_hover":      "#E4E0D8",
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
[data-testid="stSidebar"] .nav-item-selected {{
    width: 100%;
    padding: 0.7rem 1.25rem;
    margin-bottom: 0.35rem;
    background: {T['nav_hover']};
    color: {T['nav_active']};
    font-weight: 600;
    border-left: 4px solid {T['accent']};
    border-radius: 0 8px 8px 0;
    text-align: left;
}}
[data-testid="stSidebar"] .nav-item-button button {{
    width: 100%;
    background: transparent;
    border: none;
    border-radius: 0;
    text-align: left;
    padding: 0.7rem 1.25rem;
    font-size: 0.95rem;
    font-weight: 500;
    color: {T['muted']};
    cursor: pointer;
    transition: background 0.12s ease, color 0.12s ease;
}}
[data-testid="stSidebar"] .nav-item-button button:hover {{
    background: {T['nav_hover']};
    color: {T['text']};
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
[data-testid="metric-container"] [data-testid="stMetricValue"] *,
[data-testid="metric-container"] [data-testid="stMetricValue"] > div,
[data-testid="metric-container"] [data-testid="stMetricValue"] p,
[data-testid="metric-container"] [data-testid="stMetricValue"] span {{
    color: #1E2822 !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    font-family: 'Roboto Mono', monospace !important;
}}
[data-testid="metric-container"] [data-testid="stMetricDelta"],
[data-testid="metric-container"] [data-testid="stMetricDelta"] *,
[data-testid="metric-container"] [data-testid="stMetricDelta"] p,
[data-testid="metric-container"] [data-testid="stMetricDelta"] span {{
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


def render_kpi_card(label: str, value: str, delta: str = None):
    delta_html = f"<div class='kpi-delta'>{delta}</div>" if delta else ""
    st.markdown(f"""
<div class="model-best-card" style="padding:1rem 1.25rem;">
    <div class="model-best-label">{label}</div>
    <div class="model-best-value">{value}</div>
    {delta_html}
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

def render_sidebar(pg, pages_by_title):
    with st.sidebar:
        st.markdown(
            """
<div class="sidebar-logo">
    <p class="sidebar-logo-title">Fraud Detection Supermodel</p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
        for page_name, page_obj in pages_by_title.items():
            if pg == page_obj:
                st.markdown(
                    f'<div class="nav-item-selected">{page_name}</div>',
                    unsafe_allow_html=True,
                )
            else:
                with st.container():
                    if st.button(page_name, key=f"nav_{page_name}", use_container_width=True):
                        st.switch_page(page_obj)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def render_dashboard():


    render_page_header("Dashboard")

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Total Transactions", "590,540")
    with c2:
        render_kpi_card("Fraud Cases", "20,663")
    with c3:
        render_kpi_card("Non-Fraud Cases", "569,877")
    with c4:
        render_kpi_card("Fraud Rate", "3.50%")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        # Project overview
        list_items = [
            ("Dataset Scale", "590,540 transactions with 434 engineered features"),
            ("Imbalance Handling", "SMOTE applied to address 3.5% fraud minority"),
            ("Model Portfolio", "Logistic Regression, Random Forest"),
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
            ("Best Model", "Random Forest (AUC 0.67)"),
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
def render_data_analysis():

    render_page_header("Data Analysis")

    # Hardcoded from the real IEEE-CIS dataset
    TOTAL       = 590_540
    FRAUD_COUNT = 20_663
    LEGIT_COUNT = 569_877
    FRAUD_RATE  = 3.50
    N_FEATURES  = 434

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Dataset Rows",  f"{TOTAL:,}")
    with c2:
        render_kpi_card("Features",      f"{N_FEATURES:,}")
    with c3:
        render_kpi_card("Fraud Cases",   f"{FRAUD_COUNT:,}")
    with c4:
        render_kpi_card("Fraud Rate",    f"{FRAUD_RATE:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Distribution", "Patterns", "Features"])

    # ── Tab 1: Distribution
    with tab1:
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            render_chart_card(
                "Class Balance",
                "Bar comparison of legitimate vs fraudulent transaction counts.",
            )
            labels = ["Non-Fraud", "Fraud"]
            counts = [LEGIT_COUNT, FRAUD_COUNT]
            colors = [T["safe"], T["fraud"]]
            fig, ax = plt.subplots(figsize=(5, 3.8))
            bars = ax.bar(labels, counts, color=colors, width=0.45,
                          edgecolor=T["card_bg"], linewidth=1.5)
            for bar, val in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + LEGIT_COUNT * 0.02,
                        f"{val:,}\n({val/TOTAL*100:.1f}%)",
                        ha="center", va="bottom", color=T["muted"],
                        fontsize=8, fontfamily="monospace")
            ax.set_ylabel("Number of Transactions", fontsize=10)
            ax.set_xlabel("Transaction Class", fontsize=10)
            ax.set_ylim(0, LEGIT_COUNT * 1.25)
            ax.set_title("Class Balance", fontsize=11)
            apply_chart_theme(ax, fig)
            st.pyplot(fig, use_container_width=True)
            chart_caption("Severe class imbalance: 96.5% legitimate. SMOTE is applied during model training.")

        with col_b:
            render_chart_card(
                "Class Balance Before vs After SMOTE",
                "How SMOTE rebalances the training set to give the model equal exposure to fraud and non-fraud.",
            )
            before_counts = [455497, 16935]   # Non-Fraud, Fraud before SMOTE
            after_counts  = [455497, 455497]  # Balanced after SMOTE
            x = np.arange(2)
            width = 0.35
            labels = ["Non-Fraud", "Fraud"]
            fig2, ax2 = plt.subplots(figsize=(5, 3.8))
            b1 = ax2.bar(x - width/2, before_counts, width, label="Before SMOTE",
                         color=T["warning"], edgecolor=T["card_bg"], linewidth=1)
            b2 = ax2.bar(x + width/2, after_counts,  width, label="After SMOTE",
                         color=T["accent"],  edgecolor=T["card_bg"], linewidth=1)
            for bar in list(b1) + list(b2):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 8000,
                         f"{int(bar.get_height()):,}",
                         ha="center", va="bottom", color=T["muted"],
                         fontsize=7.5, fontfamily="monospace")
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels)
            ax2.set_ylabel("Number of Transactions", fontsize=10)
            ax2.set_xlabel("Transaction Class", fontsize=10)
            ax2.set_ylim(0, 455497 * 1.35)
            ax2.set_title("SMOTE Rebalancing")
            ax2.legend(frameon=False, labelcolor=T["text_secondary"], fontsize=9,
                       loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
            apply_chart_theme(ax2, fig2)
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            chart_caption("SMOTE generates synthetic fraud samples so the model learns from equal numbers of both classes.")

    # ── Tab 2: Patterns
    with tab2:
        render_chart_card(
            "Random Forest Feature Importance",
            "Top 10 features ranked by how much they contributed to the Random Forest model's fraud detection decisions.",
        )
        feature_importance = {
            "V79":  0.041,
            "V40":  0.058,
            "V45":  0.060,
            "V87":  0.061,
            "V38":  0.068,
            "V51":  0.073,
            "V52":  0.098,
            "V74":  0.103,
            "V33":  0.104,
            "V94":  0.112,
        }
        fi_series = pd.Series(feature_importance)
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        bars3 = ax3.barh(fi_series.index, fi_series.values,
                         color=T["accent"], edgecolor=T["card_bg"], linewidth=0.8)
        ax3.set_xlabel("Feature Importance Score")
        ax3.set_ylabel("Feature Names")
        ax3.set_title("Figure 3.4: Random Forest Feature Importance")
        for bar, val in zip(bars3, fi_series.values):
            ax3.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", color=T["muted"],
                     fontsize=8, fontfamily="monospace")
        apply_chart_theme(ax3, fig3)
        st.pyplot(fig3, use_container_width=True)
        chart_caption("Feature importance measures how much each feature contributed to correct splits across all 100 decision trees.")

    # ── Tab 3: Features
    with tab3:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Top 10 Features by Importance</div>
    <div class="fl-card-desc">Ranked by Random Forest feature importance score. These are Vesta-engineered anonymised features from the IEEE-CIS dataset.</div>
</div>
""", unsafe_allow_html=True)
        feature_info = pd.DataFrame({
            "Rank":               list(range(1, 11)),
            "Feature":            list(reversed(list(feature_importance.keys()))),
            "Importance Score":   [round(v, 3) for v in reversed(list(feature_importance.values()))],
            "Type":               ["Vesta Engineered (V-feature)"] * 10,
        })
        st.dataframe(feature_info, use_container_width=True, hide_index=True)
        chart_caption("V-features are anonymised by Vesta — their exact meaning is proprietary, but their predictive power is real.")




# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
def render_model_performance():

    render_page_header("Model Performance")

    perf = pd.DataFrame({
        "Model":              ["Logistic Regression", "Random Forest", "Random Forest (Threshold 0.3)"],
        "Accuracy":           [0.83, 0.87, 0.08],
        "Precision (Fraud)":  [0.09, 0.12, 0.04],
        "Recall (Fraud)":     [0.44, 0.43, 0.97],
        "F1-Score (Fraud)":   [0.16, 0.19, 0.07],
        "AUC-ROC":            [0.6748, 0.6741, np.nan],
    })

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_model_best_card("Best Accuracy", "Random Forest — 87%")
    with c2:
        render_model_best_card("Best AUC-ROC", "Logistic Regression — 0.67")
    with c3:
        render_model_best_card("Best F1-Score", "Random Forest — 19%")
    with c4:
        render_model_best_card("Best Recall", "RF @ Threshold 0.3 — 97%")

    st.markdown("<br>", unsafe_allow_html=True)

    tab_summary, tab_charts, tab_confusion = st.tabs([
        "Metrics Summary", "Performance Charts", "Confusion Matrices",
    ])

    # ── Tab 1: Metrics Summary
    with tab_summary:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Model Comparison Table</div>
    <div class="fl-card-desc">Evaluated on held-out test set with SMOTE-balanced training. Includes a Random Forest variant with its decision threshold lowered to 0.3 to trade precision for recall. Best values per column are highlighted; AUC-ROC is threshold-independent, so it's N/A for the tuned variant.</div>
</div>
""", unsafe_allow_html=True)

        def color_best(col):
            styles = [""] * len(col)
            best_idx = col.idxmax(skipna=True)
            styles[best_idx] = f"color:{T['accent']}; font-weight:700; background:#E8F2F0"
            return styles

        def fmt_auc(v):
            return "N/A" if pd.isna(v) else f"{v:.2%}"

        styled = (
            perf.style
            .apply(color_best, subset=["Accuracy","Precision (Fraud)","Recall (Fraud)","F1-Score (Fraud)","AUC-ROC"])
            .format({c: "{:.2%}" for c in ["Accuracy","Precision (Fraud)","Recall (Fraud)","F1-Score (Fraud)"]})
            .format({"AUC-ROC": fmt_auc})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
        chart_caption("Highlighted cells indicate the best-performing model for each metric.")

        st.markdown("<br>", unsafe_allow_html=True)
       


    # ── Tab 2: Performance Charts
    with tab_charts:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Per-Metric Bar Charts</div>
    <div class="fl-card-desc">Comparison of both models across every evaluation metric, including a threshold-tuned Random Forest variant (threshold 0.3).</div>
</div>
""", unsafe_allow_html=True)

        metrics_to_plot = ["Accuracy","Precision (Fraud)","Recall (Fraud)","F1-Score (Fraud)","AUC-ROC"]
        bar_colors = [T["accent"], T["warning"], T["safe"]]
        bar_labels = ["LR", "RF", "RF@0.3"]

        row1_cols = st.columns(3, gap="medium")
        row2_cols = st.columns(2, gap="medium")
        all_cols  = row1_cols + row2_cols

        for col_ui, metric in zip(all_cols, metrics_to_plot):
            with col_ui:
                fig, ax = plt.subplots(figsize=(3.2, 3))
                values = perf[metric].fillna(0)
                bars = ax.bar(bar_labels, values,
                              color=bar_colors, edgecolor=T["card_bg"], linewidth=1, width=0.55)
                ax.set_ylim(0, min(values.max() * 1.4, 1.0))
                ax.set_title(metric.replace(" (Fraud)", ""), fontsize=10)
                for b, v, raw in zip(bars, values, perf[metric]):
                    label = "N/A" if pd.isna(raw) else f"{v:.0%}"
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                            label, ha="center", va="bottom",
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
                "Best precision and F1-score",
                "Comparable AUC-ROC to LR (0.67)",
                "Robust to outliers",
                "Recommended for deployment",
            ])
        with cs2:
            render_info_card("Logistic Regression", [
                "Highest recall (44%)",
                "Best AUC-ROC (0.67)",
                "Fully interpretable coefficients",
                "Fastest inference time",
                "Lowest memory footprint",
            ])
        with cs3:
            render_info_card("RF @ Threshold 0.3", [
                "Catches 97% of fraud (recall)",
                "Precision drops to 4%",
                "Accuracy falls to 8%",
                "Flags most transactions as fraud",
                "Useful only if missed fraud is far costlier than false alarms",
            ])

    # ── Tab 3: Confusion Matrices
    with tab_confusion:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">Confusion Matrices</div>
    <div class="fl-card-desc">Evaluated on a 118,108-sample held-out test set (20% of full dataset). Values taken directly from notebook output (Figure 3.2).</div>
</div>
""", unsafe_allow_html=True)


        model_cms = [
            np.array([[96375, 17600], [2301, 1832]]),   # Logistic Regression
            np.array([[100487, 12513], [2375, 1758]]),  # Random Forest
        ]
        base_model_names = ["Logistic Regression", "Random Forest"]
        max_val = 100487  # for colour scale

        cm_cols = st.columns(2, gap="large")
        for col_ui, cm, name in zip(cm_cols, model_cms, base_model_names):
            with col_ui:
                st.markdown(f"""
<div class="fl-card" style="text-align:center;padding-bottom:0.5rem;">
    <div class="fl-card-title">{name}</div>
""", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(3.5, 3))
                fig.patch.set_facecolor(T["card_bg"])
                ax.set_facecolor(T["card_bg"])
                ax.imshow(cm, cmap="Oranges", vmin=0, vmax=max_val)
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
        st.markdown(
            f"<p style='font-size:0.8rem;color:{T['muted']};font-style:italic;'>"
            f"Note: the notebook only reports aggregate metrics (accuracy, precision, recall, F1) for the "
            f"threshold-tuned Random Forest (threshold 0.3) — no confusion matrix was computed for it, so it's "
            f"not shown here.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        


inject_custom_css()

pages_by_title = {
    "Dashboard":         st.Page(render_dashboard,         title="Dashboard",         default=True),
    "Data Analysis":      st.Page(render_data_analysis,      title="Data Analysis"),
    "Model Performance":  st.Page(render_model_performance,  title="Model Performance"),
}

pg = st.navigation(list(pages_by_title.values()), position="hidden")
render_sidebar(pg, pages_by_title)
pg.run()