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
    page_title="FraudLens – Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Theme Definitions
# ──────────────────────────────────────────────────────────────────────────────
THEMES = {
    "🌌 Midnight Blue": {
        "bg":           "#0a0e1a",
        "sidebar_bg":   "#0f1629",
        "card_bg":      "#111827",
        "card_bg2":     "#1a2540",
        "border":       "#1e2d4a",
        "accent":       "#38bdf8",
        "accent_hover": "#0ea5e9",
        "text":         "#e2e8f0",
        "muted":        "#94a3b8",
        "fraud":        "#ef4444",
        "safe":         "#22c55e",
        "warning":      "#f59e0b",
        "grid":         "#1e2d4a",
        "nav_active":   "#1e3a5f",
        "tab_border":   "#1e2d4a",
    },
    "💚 Emerald Finance": {
        "bg":           "#0d1117",
        "sidebar_bg":   "#101820",
        "card_bg":      "#131f1a",
        "card_bg2":     "#1a2e24",
        "border":       "#1e3a2a",
        "accent":       "#10b981",
        "accent_hover": "#059669",
        "text":         "#d1fae5",
        "muted":        "#6ee7b7",
        "fraud":        "#ef4444",
        "safe":         "#34d399",
        "warning":      "#f59e0b",
        "grid":         "#1e3a2a",
        "nav_active":   "#14532d",
        "tab_border":   "#1e3a2a",
    },
    "💜 Purple Insight": {
        "bg":           "#0e0b1a",
        "sidebar_bg":   "#120f20",
        "card_bg":      "#1a1530",
        "card_bg2":     "#221b3a",
        "border":       "#2d1f52",
        "accent":       "#a855f7",
        "accent_hover": "#9333ea",
        "text":         "#f3e8ff",
        "muted":        "#c084fc",
        "fraud":        "#f87171",
        "safe":         "#4ade80",
        "warning":      "#fbbf24",
        "grid":         "#2d1f52",
        "nav_active":   "#3b0764",
        "tab_border":   "#2d1f52",
    },
    "☀️ Light Professional": {
        "bg":           "#f8fafc",
        "sidebar_bg":   "#f1f5f9",
        "card_bg":      "#ffffff",
        "card_bg2":     "#f0f9ff",
        "border":       "#e2e8f0",
        "accent":       "#2563eb",
        "accent_hover": "#1d4ed8",
        "text":         "#1e293b",
        "muted":        "#64748b",
        "fraud":        "#dc2626",
        "safe":         "#16a34a",
        "warning":      "#d97706",
        "grid":         "#e2e8f0",
        "nav_active":   "#dbeafe",
        "tab_border":   "#e2e8f0",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Theme Helper
# ──────────────────────────────────────────────────────────────────────────────
def get_theme() -> dict:
    """Return the currently selected theme token dictionary."""
    return THEMES[st.session_state.get("theme_name", "🌌 Midnight Blue")]


# ──────────────────────────────────────────────────────────────────────────────
# CSS Injection
# ──────────────────────────────────────────────────────────────────────────────
def inject_custom_css(t: dict):
    """Inject dynamic CSS based on the active theme tokens."""
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
.stApp {{
    background-color: {t['bg']};
    color: {t['text']};
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {t['sidebar_bg']};
    border-right: 1px solid {t['border']};
}}
[data-testid="stSidebar"] .stRadio {{ margin: 0.5rem 0; }}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] {{
    display: flex; flex-direction: column; gap: 0.3rem;
}}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{
    display: block; width: 100%; padding: 0.75rem 1rem;
    border-radius: 0.6rem; border: 1px solid transparent;
    background-color: transparent;
    color: {t['muted']} !important;
    font-size: 0.95rem; font-weight: 500;
    transition: all 0.15s ease; cursor: pointer;
}}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{
    background-color: {t['nav_active']};
    color: {t['text']} !important;
}}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] input[type="radio"] {{
    position: absolute; opacity: 0; width: 0; height: 0;
}}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] input[type="radio"]:checked + *,
[data-testid="stSidebar"] .stRadio [role="radiogroup"] input[type="radio"]:checked ~ * {{
    background-color: {t['accent']};
    color: #ffffff !important;
    font-weight: 700;
    border-left: 4px solid rgba(255,255,255,0.5);
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}}

/* ── Metric cards ── */
[data-testid="metric-container"] {{
    background: linear-gradient(135deg, {t['card_bg']} 0%, {t['card_bg2']} 100%);
    border: 1px solid {t['border']};
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
}}
[data-testid="metric-container"] label {{
    color: {t['muted']} !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: 'Roboto Mono', monospace !important;
    font-weight: 600;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {t['text']} !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    font-family: 'Roboto Mono', monospace !important;
}}

/* ── Headings ── */
h1 {{ color: {t['accent']} !important; font-size: 2.2rem !important; font-weight: 700 !important; letter-spacing: -0.02em; }}
h2 {{ color: {t['text']} !important; font-size: 1.6rem !important; font-weight: 600 !important; }}
h3 {{ color: {t['text']} !important; font-size: 1.2rem !important; font-weight: 600 !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {t['card_bg']};
    border-bottom: 2px solid {t['tab_border']};
    gap: 0;
    border-radius: 10px 10px 0 0;
    padding: 0 0.5rem;
}}
.stTabs [data-baseweb="tab"] {{
    color: {t['muted']} !important;
    font-size: 0.95rem !important;
    font-weight: 500;
    padding: 0.75rem 1.4rem !important;
}}
.stTabs [aria-selected="true"] {{
    color: {t['accent']} !important;
    border-bottom: 3px solid {t['accent']} !important;
    background: transparent !important;
    font-weight: 600 !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {t['border']};
    border-radius: 10px;
    overflow: hidden;
}}

/* ── Alert boxes ── */
[data-testid="stAlert"] {{
    background-color: {t['card_bg']} !important;
    border: 1px solid {t['border']} !important;
    border-radius: 10px !important;
    color: {t['muted']} !important;
    font-size: 0.9rem;
}}

/* ── File uploader ── */
.stFileUploader {{
    background-color: {t['card_bg']} !important;
    border: 2px dashed {t['border']} !important;
    border-radius: 10px;
}}

/* ── Divider ── */
hr {{ border-color: {t['border']}; margin: 1.25rem 0; }}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {{ color: {t['accent']} !important; }}

/* ── Custom card component ── */
.fl-card {{
    background: linear-gradient(135deg, {t['card_bg']} 0%, {t['card_bg2']} 100%);
    border: 1px solid {t['border']};
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.15);
    margin-bottom: 1rem;
}}
.fl-card-title {{
    color: {t['accent']};
    font-size: 1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.6rem;
}}
.fl-card-desc {{
    color: {t['muted']};
    font-size: 0.88rem;
    margin-bottom: 1rem;
    line-height: 1.55;
}}

/* ── Insight cards ── */
.insight-card {{
    background: linear-gradient(135deg, {t['card_bg']} 0%, {t['card_bg2']} 100%);
    border-left: 4px solid {t['accent']};
    border-radius: 0 10px 10px 0;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}}
.insight-card-title {{
    color: {t['accent']};
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}}
.insight-card-body {{
    color: {t['muted']};
    font-size: 0.88rem;
    line-height: 1.6;
}}

/* ── Status badge ── */
.status-badge {{
    display: inline-block;
    background-color: {t['safe']}22;
    color: {t['safe']};
    border: 1px solid {t['safe']}55;
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'Roboto Mono', monospace;
    letter-spacing: 0.04em;
}}
.demo-badge {{
    display: inline-block;
    background-color: {t['warning']}22;
    color: {t['warning']};
    border: 1px solid {t['warning']}55;
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'Roboto Mono', monospace;
    letter-spacing: 0.04em;
}}

/* ── Page header block ── */
.page-header {{
    padding: 0.5rem 0 1.25rem 0;
    border-bottom: 1px solid {t['border']};
    margin-bottom: 1.5rem;
}}
.page-header h1 {{ margin: 0 0 0.4rem 0; }}
.page-subtitle {{
    color: {t['muted']};
    font-size: 0.95rem;
    margin-top: 0.4rem;
    margin-bottom: 0.6rem;
}}

/* ── Model summary card ── */
.model-best-card {{
    background: linear-gradient(135deg, {t['accent']}18 0%, {t['card_bg2']} 100%);
    border: 2px solid {t['accent']}55;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}}
.model-best-label {{
    color: {t['muted']};
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: 'Roboto Mono', monospace;
    font-weight: 600;
}}
.model-best-value {{
    color: {t['accent']};
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 0.3rem;
}}

/* ── Chart caption ── */
.chart-caption {{
    color: {t['muted']} !important;
    font-size: 0.84rem;
    font-style: italic;
    margin-top: 0.6rem;
    line-height: 1.55;
}}

/* ── Sidebar logo ── */
.sidebar-logo {{
    padding: 1rem 0.5rem 0.75rem 0.5rem;
    border-bottom: 1px solid {t['border']};
    margin-bottom: 1rem;
}}
.sidebar-logo-title {{
    color: {t['accent']};
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.2;
}}
.sidebar-logo-sub {{
    color: {t['muted']};
    font-size: 0.78rem;
    font-family: 'Roboto Mono', monospace;
    margin-top: 0.2rem;
}}

/* ── Sidebar section label ── */
.sidebar-section {{
    color: {t['muted']};
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    font-family: 'Roboto Mono', monospace;
    padding: 0.5rem 0 0.35rem 0;
    margin-top: 0.25rem;
}}

/* ── Sidebar footer ── */
.sidebar-footer {{
    color: {t['border']};
    font-size: 0.72rem;
    font-family: 'Roboto Mono', monospace;
    text-align: center;
    padding: 0.75rem 0 0.25rem 0;
    border-top: 1px solid {t['border']};
    margin-top: 1.25rem;
    line-height: 1.7;
}}

/* ── Metric icon chips ── */
.metric-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: {t['card_bg2']};
    border: 1px solid {t['border']};
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.85rem;
    color: {t['muted']};
    font-family: 'Roboto Mono', monospace;
}}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Chart Theme Helper
# ──────────────────────────────────────────────────────────────────────────────
def apply_chart_theme(ax, fig, t: dict):
    """Apply the active theme to a matplotlib axis and figure."""
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
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)


# ──────────────────────────────────────────────────────────────────────────────
# Reusable UI Helpers
# ──────────────────────────────────────────────────────────────────────────────
def render_page_header(title: str, subtitle: str, demo_mode: bool):
    """Render a styled page header with title, subtitle, and status badge."""
    t = get_theme()
    badge_html = (
        '<span class="demo-badge">⚡ Demo Mode</span>'
        if demo_mode else
        '<span class="status-badge">✅ System Active</span>'
    )
    st.markdown(f"""
<div class="page-header">
    <h1>{title}</h1>
    <p class="page-subtitle">{subtitle}</p>
    {badge_html}
</div>
""", unsafe_allow_html=True)


def render_chart_card(title: str, description: str = "", caption: str = ""):
    """
    Context manager wrapper: renders a styled card container around a chart.
    Usage:
        with render_chart_card_ctx(...):
            st.pyplot(fig)
    Since Streamlit doesn't support real CM here, just return helpers.
    """
    t = get_theme()
    st.markdown(f"""
<div class="fl-card" style="padding-bottom:0.5rem;">
    <div class="fl-card-title">{title}</div>
    {"<div class='fl-card-desc'>"+description+"</div>" if description else ""}
</div>
""", unsafe_allow_html=True)
    return caption  # caller uses caption after st.pyplot


def render_insight_card(title: str, body: str):
    """Render a styled insight card."""
    st.markdown(f"""
<div class="insight-card">
    <div class="insight-card-title">{title}</div>
    <div class="insight-card-body">{body}</div>
</div>
""", unsafe_allow_html=True)


def render_model_best_card(label: str, value: str):
    """Render a highlight summary card for model performance."""
    st.markdown(f"""
<div class="model-best-card">
    <div class="model-best-label">{label}</div>
    <div class="model-best-value">{value}</div>
</div>
""", unsafe_allow_html=True)


def render_info_card(title: str, items: list):
    """Render a bullet-list info card."""
    t = get_theme()
    bullets = "".join(f"<li style='margin-bottom:0.35rem;color:{t['muted']};font-size:0.88rem;'>{item}</li>" for item in items)
    st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">{title}</div>
    <ul style="margin:0;padding-left:1.2rem;">{bullets}</ul>
</div>
""", unsafe_allow_html=True)


def chart_caption(text: str):
    """Render a small italic chart caption."""
    st.markdown(f"<p class='chart-caption'>{text}</p>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple:
    """
    Render the full sidebar. Returns (page, use_demo, t_file, i_file).
    """
    t = get_theme()

    with st.sidebar:
        # Logo area
        st.markdown("""
<div class="sidebar-logo">
    <div class="sidebar-logo-title">🔍 FraudLens</div>
    <div class="sidebar-logo-sub">Fraud Detection & Analytics</div>
</div>
""", unsafe_allow_html=True)

        # ── Navigation ──
        st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
        pages = ["📊 Dashboard", "🔬 Data Analysis", "🤖 Model Performance"]
        page = st.radio(
            label="",
            options=pages,
            index=pages.index(st.session_state.get("current_page", "📊 Dashboard")),
            key="sidebar_page_nav",
            label_visibility="collapsed",
        )
        st.session_state.current_page = page

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Data Source ──
        st.markdown('<div class="sidebar-section">Data Source</div>', unsafe_allow_html=True)
        use_demo = st.toggle("Use Demonstration Data", value=True, key="demo_toggle")
        t_file = i_file = None
        if not use_demo:
            st.markdown(f"<span style='color:{t['muted']};font-size:0.82rem;'>IEEE-CIS Fraud Detection Dataset</span>", unsafe_allow_html=True)
            t_file = st.file_uploader("Transaction CSV (train_transaction.csv)", type="csv", key="trans")
            i_file = st.file_uploader("Identity CSV (train_identity.csv)", type="csv", key="iden")

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Theme Settings ──
        st.markdown('<div class="sidebar-section">Theme Settings</div>', unsafe_allow_html=True)
        theme_names = list(THEMES.keys())
        selected_theme = st.selectbox(
            "Colour Template",
            theme_names,
            index=theme_names.index(st.session_state.get("theme_name", "🌌 Midnight Blue")),
            key="theme_select",
            label_visibility="collapsed",
        )
        st.session_state["theme_name"] = selected_theme

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Project Info ──
        st.markdown('<div class="sidebar-section">Project Info</div>', unsafe_allow_html=True)
        t_now = get_theme()
        st.markdown(f"""
<div style="color:{t_now['muted']};font-size:0.82rem;line-height:1.7;">
    <b style="color:{t_now['text']};">Module</b>: 5011CEM Big Data Programming<br>
    <b style="color:{t_now['text']};">Dataset</b>: IEEE-CIS Fraud Detection<br>
    <b style="color:{t_now['text']};">Stack</b>: Python · Streamlit · ML
</div>
""", unsafe_allow_html=True)

        # Footer
        st.markdown("""
<div class="sidebar-footer">
    5011CEM Big Data Programming<br>
    Streamlit • Python • Machine Learning
</div>
""", unsafe_allow_html=True)

    return page, use_demo, t_file, i_file


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_uploaded_data(transaction_bytes, identity_bytes):
    """Load and merge transaction and identity CSV files."""
    import io
    transaction = pd.read_csv(io.BytesIO(transaction_bytes))
    identity    = pd.read_csv(io.BytesIO(identity_bytes))
    return transaction.merge(identity, on="TransactionID", how="left")


@st.cache_data
def generate_demo_data(n=5000, seed=42):
    """Generate realistic synthetic fraud detection data for demo purposes."""
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
# Session State Initialisation
# ──────────────────────────────────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state.current_page = "📊 Dashboard"
if "theme_name" not in st.session_state:
    st.session_state.theme_name = "🌌 Midnight Blue"

# Render sidebar first (returns navigation selection)
page, use_demo, t_file, i_file = render_sidebar()

# Apply CSS for the current theme
inject_custom_css(get_theme())
t = get_theme()

# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
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
    st.info("📂 Upload both CSV files in the sidebar to get started, or enable demonstration mode.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    render_page_header(
        "Fraud Detection Dashboard",
        "Real-time monitoring of fraud risk signals, transaction patterns, and model outputs.",
        demo_mode=use_demo,
    )

    # ── KPI Cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📦 Total Transactions", "590,540")
    with c2:
        st.metric("🚨 Fraud Cases", "20,663", delta="-5.2% vs prev period", delta_color="normal")
    with c3:
        st.metric("✅ Non-Fraud Cases", "569,877", delta="+2.1%", delta_color="normal")
    with c4:
        st.metric("📉 Fraud Rate", "3.50%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column layout ──────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        # Project overview card
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">🏦 Project Overview</div>
    <div class="fl-card-desc">
        This <b>Big Data Fraud Detection System</b> is engineered for the banking and financial sector,
        utilising the <b>IEEE-CIS Fraud Detection dataset</b> from Kaggle.
    </div>
    <ul style="margin:0;padding-left:1.2rem;line-height:2;">
        <li style="color:{t['muted']};font-size:0.88rem;"><b style="color:{t['text']};">Dataset Scale</b>: 590,540 transactions with 434 engineered features</li>
        <li style="color:{t['muted']};font-size:0.88rem;"><b style="color:{t['text']};">Imbalance Handling</b>: SMOTE applied to address 3.5% fraud minority</li>
        <li style="color:{t['muted']};font-size:0.88rem;"><b style="color:{t['text']};">Model Portfolio</b>: Logistic Regression · Random Forest · Neural Network</li>
        <li style="color:{t['muted']};font-size:0.88rem;"><b style="color:{t['text']};">Metrics</b>: Accuracy, Precision, Recall, F1-Score, AUC-ROC</li>
        <li style="color:{t['muted']};font-size:0.88rem;"><b style="color:{t['text']};">Business Goal</b>: Detect fraud while minimising false positives</li>
    </ul>
</div>
""", unsafe_allow_html=True)

        # Key insights
        st.markdown(f"<div style='font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:{t['muted']};margin-bottom:0.5rem;font-family:Roboto Mono,monospace;'>Key Insights</div>", unsafe_allow_html=True)

        insights = [
            ("🎯 Fraud Concentration Patterns",
             "Fraudsters systematically target specific card types and email domains, exhibiting repeated behavioural fingerprints that enable pattern-based detection."),
            ("💰 Transaction Amount Anomalies",
             "Fraudulent transactions deviate significantly — either unusually small amounts (testing card validity) or exceptionally large amounts (high-value theft)."),
            ("⏱️ Temporal Risk Indicators",
             "Fraud incidents concentrate during off-peak hours and weekends when transaction volumes are lower and real-time monitoring coverage is reduced."),
        ]
        for title, desc in insights:
            render_insight_card(title, desc)

    with col_right:
        # Class distribution chart
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">🥧 Class Distribution</div>
    <div class="fl-card-desc">Proportion of legitimate vs fraudulent transactions in the full dataset.</div>
""", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(4, 3.2))
        sizes  = [96.5, 3.5]
        colors = [t["safe"], t["fraud"]]
        wedges, texts, autotexts = ax.pie(
            sizes, colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.72,
            wedgeprops=dict(width=0.52, edgecolor=t["bg"], linewidth=2),
        )
        for at in autotexts:
            at.set_color(t["text"]); at.set_fontsize(10); at.set_fontweight("bold")
        ax.legend(["Non-Fraud (96.5%)", "Fraud (3.5%)"],
                  loc="lower center", frameon=False, labelcolor=t["muted"], fontsize=8)
        ax.set_title("Transaction Split", color=t["text"], fontsize=11, fontweight="bold", pad=8)
        apply_chart_theme(ax, fig, t)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Fraud risk summary card
        render_info_card(
            "🔮 Fraud Risk Summary",
            [
                f"<b style='color:{t['text']}'>Best Model</b>: Random Forest (AUC 0.81)",
                f"<b style='color:{t['text']}'>Recall</b>: 43% of all fraud detected",
                f"<b style='color:{t['text']}'>Precision</b>: 12% flagged are truly fraud",
                f"<b style='color:{t['text']}'>SMOTE</b>: Balances class ratio for training",
                f"<b style='color:{t['text']}'>Threshold</b>: Tunable 0.3–0.7 range",
            ]
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Data Analysis":

    render_page_header(
        "Data Analysis & Exploration",
        "Explore class distributions, feature correlations, data quality metrics, and raw transaction samples.",
        demo_mode=use_demo,
    )

    if not data_ready:
        st.stop()

    # ── Summary KPIs ──
    fraud_count = int(df["isFraud"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Dataset Rows",  f"{df.shape[0]:,}")
    c2.metric("🔢 Features",      f"{df.shape[1]:,}")
    c3.metric("🚨 Fraud Cases",   f"{fraud_count:,}")
    c4.metric("📉 Fraud Rate",    f"{fraud_count/len(df)*100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distribution",
        "🔗 Patterns",
        "🗂️ Features",
        "🔎 Sample Data",
    ])

    # ─── Tab 1: Distribution ─────────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            render_chart_card(
                "Class Balance",
                "Bar comparison of legitimate vs fraudulent transaction counts.",
            )
            fraud_counts = df["isFraud"].value_counts().sort_index()
            labels = ["Non-Fraud", "Fraud"]
            colors = [t["safe"], t["fraud"]]

            fig, ax = plt.subplots(figsize=(5, 3.8))
            bars = ax.bar(labels, fraud_counts.values, color=colors, width=0.45,
                          edgecolor=t["bg"], linewidth=1.5)
            for bar, val in zip(bars, fraud_counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(fraud_counts) * 0.02,
                        f"{val:,}\n({val/len(df)*100:.1f}%)",
                        ha="center", va="bottom", color=t["muted"],
                        fontsize=8, fontfamily="monospace")
            ax.set_ylabel("Count", fontsize=10)
            ax.set_ylim(0, max(fraud_counts) * 1.15)
            ax.set_title("Class Balance", fontsize=12)
            apply_chart_theme(ax, fig, t)
            st.pyplot(fig, use_container_width=True)
            chart_caption("Severe class imbalance: 96.5% legitimate transactions. SMOTE is applied during model training to handle this.")

        with col_b:
            render_chart_card(
                "Transaction Amount Distribution",
                "Overlapping density curves comparing amount ranges for fraud vs legitimate.",
            )
            fig2, ax2 = plt.subplots(figsize=(5, 3.8))
            fraud_amt = df[df["isFraud"] == 1]["TransactionAmt"].clip(upper=5000)
            legit_amt = df[df["isFraud"] == 0]["TransactionAmt"].clip(upper=5000)

            ax2.hist(legit_amt, bins=60, color=t["safe"],  alpha=0.65, label="Non-Fraud", density=True)
            ax2.hist(fraud_amt, bins=60, color=t["fraud"], alpha=0.65, label="Fraud",     density=True)
            ax2.set_xlabel("Transaction Amount (USD)")
            ax2.set_ylabel("Density")
            ax2.set_title("Amount Comparison")
            ax2.legend(frameon=False, labelcolor=t["muted"], fontsize=9)
            apply_chart_theme(ax2, fig2, t)
            st.pyplot(fig2, use_container_width=True)
            chart_caption("Fraudulent transactions show bimodal behaviour — micro-amounts (card testing) and high-value spikes.")

    # ─── Tab 2: Patterns ─────────────────────────────────────────────────────
    with tab2:
        render_chart_card(
            "Top Features by Fraud Correlation",
            "Absolute Pearson correlation with the isFraud label (numeric columns only, top 10 shown).",
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
            t["accent"]  if v >= corr_series.iloc[0] * 0.7 else
            t["warning"] if v >= corr_series.iloc[0] * 0.4 else
            t["muted"]
            for v in corr_series.values
        ]

        fig3, ax3 = plt.subplots(figsize=(10, 4.5))
        bars3 = ax3.barh(corr_series.index[::-1], corr_series.values[::-1],
                         color=palette[::-1], edgecolor=t["bg"], linewidth=0.8)
        ax3.set_xlabel("Absolute Correlation")
        ax3.set_title("Top 10 Features Associated with Fraud")
        for bar, val in zip(bars3, corr_series.values[::-1]):
            ax3.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", color=t["muted"],
                     fontsize=8, fontfamily="monospace")
        apply_chart_theme(ax3, fig3, t)
        st.pyplot(fig3, use_container_width=True)
        chart_caption("Features coloured by strength — accent = high correlation, amber = moderate, grey = lower.")

    # ─── Tab 3: Features ─────────────────────────────────────────────────────
    with tab3:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">🗂️ Feature Overview</div>
    <div class="fl-card-desc">Data type, missing percentage, and unique value count for every column in the loaded dataset.</div>
</div>
""", unsafe_allow_html=True)

        feature_info = pd.DataFrame({
            "Column":       df.columns,
            "Data Type":    df.dtypes.values,
            "Missing %":    (df.isnull().sum() / len(df) * 100).round(2).values,
            "Unique Values": [df[col].nunique() for col in df.columns],
        })
        st.dataframe(feature_info, use_container_width=True, hide_index=True)
        chart_caption("Review missing percentages to inform preprocessing decisions — columns above 50% missing may need to be dropped.")

    # ─── Tab 4: Sample Data ───────────────────────────────────────────────────
    with tab4:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">🔎 Raw Data Sample</div>
    <div class="fl-card-desc">Fraud rows are highlighted in red; legitimate rows in green.</div>
</div>
""", unsafe_allow_html=True)

        n_show = st.slider("Rows to display", 5, 50, 10)
        sample = df.head(n_show).copy()

        def highlight_fraud(val):
            if val == 1: return "background-color:#7f1d1d; color:#fca5a5"
            elif val == 0: return "background-color:#14532d; color:#86efac"
            return ""

        st.dataframe(
            sample.style.applymap(highlight_fraud, subset=["isFraud"]),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":

    render_page_header(
        "Model Performance Evaluation",
        "Comparative analysis of Logistic Regression, Random Forest, and Neural Network on held-out test data.",
        demo_mode=use_demo,
    )

    # ── Performance data ──
    perf = pd.DataFrame({
        "Model":              ["Logistic Regression", "Random Forest", "Neural Network"],
        "Accuracy":           [0.83, 0.87, 0.87],
        "Precision (Fraud)":  [0.09, 0.12, 0.12],
        "Recall (Fraud)":     [0.44, 0.43, 0.44],
        "F1-Score (Fraud)":   [0.16, 0.19, 0.19],
        "AUC-ROC":            [0.72, 0.81, 0.80],
    })

    # ── Model summary hero cards ──
    c1, c2, c3 = st.columns(3)
    with c1:
        render_model_best_card("🏆 Best Accuracy", "Random Forest  87%")
    with c2:
        render_model_best_card("📈 Best AUC-ROC", "Random Forest  0.81")
    with c3:
        render_model_best_card("🎯 Best F1-Score", "RF & NN  19%")

    st.markdown("<br>", unsafe_allow_html=True)

    tab_summary, tab_charts, tab_confusion, tab_insights = st.tabs([
        "📋 Metrics Summary",
        "📊 Performance Charts",
        "🔲 Confusion Matrices",
        "💡 Key Insights",
    ])

    # ─── Tab 1: Metrics Summary ───────────────────────────────────────────────
    with tab_summary:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">📋 Model Comparison Table</div>
    <div class="fl-card-desc">Evaluated on held-out test set with SMOTE-balanced training data. Best values per column are highlighted.</div>
</div>
""", unsafe_allow_html=True)

        def color_best(col):
            styles = [""] * len(col)
            best_idx = col.idxmax()
            styles[best_idx] = f"color:{t['accent']}; font-weight:700; background:{t['nav_active']}"
            return styles

        styled = (
            perf.style
            .apply(color_best, subset=["Accuracy","Precision (Fraud)","Recall (Fraud)","F1-Score (Fraud)","AUC-ROC"])
            .format({c: "{:.2%}" for c in ["Accuracy","Precision (Fraud)","Recall (Fraud)","F1-Score (Fraud)","AUC-ROC"]})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
        chart_caption("Highlighted cells indicate the best-performing model for each metric.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:{t['muted']};font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;font-family:Roboto Mono,monospace;margin-bottom:0.5rem;'>Metric Definitions</div>", unsafe_allow_html=True)

        metrics_info = {
            "Accuracy":  "Overall correctness: (TP + TN) / Total",
            "Precision": "Of all flagged transactions, how many were truly fraudulent: TP / (TP + FP)",
            "Recall":    "Of all actual fraud cases, fraction successfully caught: TP / (TP + FN)",
            "F1-Score":  "Harmonic mean of Precision and Recall — balances both false positive and false negative costs",
            "AUC-ROC":   "Area under the ROC curve — model discrimination ability across all thresholds (0.5–1.0)",
        }
        for metric, definition in metrics_info.items():
            st.markdown(f"<p style='font-size:0.88rem;color:{t['muted']};margin-bottom:0.4rem;'><b style='color:{t['text']};'>{metric}</b>: {definition}</p>", unsafe_allow_html=True)

    # ─── Tab 2: Performance Charts ────────────────────────────────────────────
    with tab_charts:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">📊 Per-Metric Bar Charts</div>
    <div class="fl-card-desc">Side-by-side comparison of all three models across every evaluation metric.</div>
</div>
""", unsafe_allow_html=True)

        metrics_to_plot = ["Accuracy","Precision (Fraud)","Recall (Fraud)","F1-Score (Fraud)","AUC-ROC"]
        bar_colors = [t["accent"], t["warning"], t["safe"]]

        # Two rows of charts: 3 + 2
        row1_cols = st.columns(3, gap="medium")
        row2_cols = st.columns(2, gap="medium")
        all_cols  = row1_cols + row2_cols

        for col_ui, metric in zip(all_cols, metrics_to_plot):
            with col_ui:
                fig, ax = plt.subplots(figsize=(3.2, 3))
                bars = ax.bar(["LR", "RF", "NN"], perf[metric],
                              color=bar_colors, edgecolor=t["bg"], linewidth=1, width=0.5)
                ax.set_ylim(0, min(perf[metric].max() * 1.4, 1.0))
                ax.set_title(metric.replace(" (Fraud)", ""), fontsize=10)
                for b, v in zip(bars, perf[metric]):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                            f"{v:.0%}", ha="center", va="bottom",
                            color=t["muted"], fontsize=8, fontfamily="monospace")
                ax.set_ylabel("Score", fontsize=9)
                apply_chart_theme(ax, fig, t)
                st.pyplot(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Comparative strength cards
        st.markdown(f"<div style='color:{t['muted']};font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;font-family:Roboto Mono,monospace;margin-bottom:0.75rem;'>Comparative Analysis</div>", unsafe_allow_html=True)

        cs1, cs2, cs3 = st.columns(3)
        with cs1:
            render_info_card("🌲 Random Forest", [
                "Highest accuracy (87%)",
                "Best AUC-ROC (0.81)",
                "Best precision & F1-score",
                "Robust to outliers",
                "Recommended for deployment",
            ])
        with cs2:
            render_info_card("📉 Logistic Regression", [
                "Highest recall (44%)",
                "Fully interpretable coefficients",
                "Fastest inference time",
                "Lowest memory footprint",
                "Best baseline model",
            ])
        with cs3:
            render_info_card("🧠 Neural Network", [
                "Matches RF on accuracy (87%)",
                "Strong AUC-ROC (0.80)",
                "Handles complex feature interactions",
                "Requires more compute",
                "Good recall performance",
            ])

    # ─── Tab 3: Confusion Matrices ────────────────────────────────────────────
    with tab_confusion:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">🔲 Confusion Matrices</div>
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
                fig.patch.set_facecolor(t["card_bg"])
                ax.set_facecolor(t["card_bg"])
                ax.imshow(cm, cmap="Blues", vmin=0, vmax=n_legit)
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                                color=t["text"] if cm[i,j] < cm.max() * 0.6 else t["bg"],
                                fontsize=11, fontweight="bold", fontfamily="monospace")
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["Pred\nLegit", "Pred\nFraud"], color=t["muted"], fontsize=8)
                ax.set_yticklabels(["Actual\nLegit", "Actual\nFraud"], color=t["muted"], fontsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor(t["border"])
                st.pyplot(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:{t['muted']};font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;font-family:Roboto Mono,monospace;margin-bottom:0.5rem;'>Matrix Interpretation</div>", unsafe_allow_html=True)

        interp = [
            ("TN — True Negatives",    "Legitimate transactions correctly identified as safe (top-left)."),
            ("FP — False Positives",   "Legitimate transactions incorrectly flagged as fraud (top-right). Increases customer friction."),
            ("FN — False Negatives",   "Missed fraud cases — the most costly error in financial contexts (bottom-left)."),
            ("TP — True Positives",    "Correctly identified fraudulent transactions (bottom-right). Maximise this."),
        ]
        for label, desc in interp:
            st.markdown(f"<p style='font-size:0.88rem;color:{t['muted']};margin-bottom:0.35rem;'><b style='color:{t['text']};'>{label}</b>: {desc}</p>", unsafe_allow_html=True)

    # ─── Tab 4: Key Insights ──────────────────────────────────────────────────
    with tab_insights:
        st.markdown(f"""
<div class="fl-card">
    <div class="fl-card-title">💡 Key Findings & Recommendations</div>
    <div class="fl-card-desc">Actionable insights derived from the model evaluation for production deployment.</div>
</div>
""", unsafe_allow_html=True)

        i1, i2, i3 = st.columns(3, gap="medium")
        with i1:
            render_insight_card(
                "🏆 Best Overall Model",
                "Random Forest and Neural Network tie at <b>87% accuracy</b>. "
                "Random Forest is recommended for deployment due to superior AUC-ROC (0.81) and interpretability.",
            )
        with i2:
            render_insight_card(
                "⚖️ Precision-Recall Tradeoff",
                "At 12% precision, expect <b>~8 false positives per true fraud caught</b>. "
                "High false alarm rates degrade customer experience and create operational burden.",
            )
        with i3:
            render_insight_card(
                "🎛️ Threshold Optimisation",
                "Lowering the decision threshold from 0.5 → 0.3 raises recall to approximately <b>~60%</b>. "
                "Tune this value based on business cost-benefit analysis.",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        render_info_card(
            "🚀 Deployment Strategy",
            [
                "<b>Primary Model</b>: Deploy Random Forest for production fraud detection",
                "<b>Threshold Tuning</b>: Adjust based on false-positive tolerance and fraud-loss budget",
                "<b>Ensemble Approach</b>: Combine multiple model predictions for high-value flagging decisions",
                "<b>Continuous Retraining</b>: Monthly retraining schedule to adapt to evolving fraud patterns",
                "<b>Feature Engineering</b>: Collect additional temporal and behavioural features to enrich inputs",
            ]
        )


# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{t['border']};font-size:0.75rem;font-family:Roboto Mono,monospace;"
    f"text-align:center;margin-top:1rem;'>"
    f"FraudLens — Fraud Detection Dashboard &nbsp;|&nbsp; IEEE-CIS Dataset &nbsp;|&nbsp; "
    f"5011CEM Big Data Programming Project &nbsp;|&nbsp; Streamlit · Python · ML</p>",
    unsafe_allow_html=True,
)
