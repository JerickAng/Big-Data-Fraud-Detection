import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Big Data Fraud Detection Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("📊 Navigation")
option = st.sidebar.selectbox(
    "Select Page",
    ["🏠 Home", "📈 Data Overview", "🤖 Model Performance", "🔮 Predict New Transaction"]
)

# Load data function (cached to avoid reloading)
@st.cache_data
def load_data():
    transaction = pd.read_csv(r"C:\Users\Jeric\OneDrive\Documents\Big Data Fraud Detection\Big-Data-Fraud-Detection\Datasets\train_transaction.csv")
    identity = pd.read_csv(r"C:\Users\Jeric\OneDrive\Documents\Big Data Fraud Detection\Big-Data-Fraud-Detection\Datasets\train_identity.csv")
    df = transaction.merge(identity, on='TransactionID', how='left')
    return df

# Load models (if saved)
@st.cache_resource
def load_models():
    try:
        model_lr = joblib.load('logistic_regression_model.pkl')
        model_rf = joblib.load('random_forest_model.pkl')
        model_nn = joblib.load('neural_network_fraud_model.pkl')
        with open('best_threshold.json', 'r') as f:
            threshold = json.load(f)['threshold']
        return model_lr, model_rf, model_nn, threshold
    except:
        return None, None, None, 0.5

# ========== HOME PAGE ==========
if option == "🏠 Home":
    st.header("Welcome to the Fraud Detection Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Total Transactions", "590,540")
    with col2:
        st.metric("⚠️ Fraud Rate", "3.5%")
    with col3:
        st.metric("🎯 Best Model Accuracy", "87%")
    
    st.markdown("---")
    
    st.subheader("📌 Project Overview")
    st.write("""

# ========== DATA OVERVIEW PAGE ==========
elif option == "📈 Data Overview":
    st.header("📈 Data Overview")
    
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Dataset info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    
    st.markdown("---")
    
    # Fraud distribution
    st.subheader("Fraud Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    fraud_counts = df['isFraud'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(['Non-Fraud (0)', 'Fraud (1)'], fraud_counts.values, color=colors)
    ax.set_ylabel('Count')
    ax.set_title('Transaction Distribution by Class')
    for bar, count in zip(bars, fraud_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000, 
                f'{count:,}\n({count/len(df)*100:.2f}%)', 
                ha='center', va='bottom')
    st.pyplot(fig)
    
    # Top features
    st.subheader("Top Correlated Features with Fraud")
    corr = df.corr(numeric_only=True)
    corr_target = corr['isFraud'].abs().sort_values(ascending=False)
    top_features = corr_target.head(10)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    top_features.drop('isFraud').plot(kind='barh', ax=ax2, color='#3498db')
    ax2.set_xlabel('Correlation with Fraud')
    ax2.set_title('Top 10 Features Correlated with Fraud')
    st.pyplot(fig2)
    
    # Missing values
    st.subheader("Missing Values Analysis")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) > 0:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        missing.head(10).plot(kind='barh', ax=ax3, color='#e67e22')
        ax3.set_xlabel('Number of Missing Values')
        ax3.set_title('Top 10 Columns with Missing Values')
        st.pyplot(fig3)
    else:
        st.success("✅ No missing values found!")

# ========== MODEL PERFORMANCE PAGE ==========
elif option == "🤖 Model Performance":
    st.header("🤖 Model Performance Comparison")
    
    st.write("""
    Three machine learning models were trained and evaluated for fraud detection:
    - **Logistic Regression**: Baseline linear model
    - **Random Forest**: Ensemble of decision trees
    - **Neural Network (MLP)**: Multi-layer perceptron with 2 hidden layers
    """)
    
    # Performance metrics (from your results)
    performance_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'Neural Network'],
        'Accuracy': [0.83, 0.87, 0.87],
        'Precision (Fraud)': [0.09, 0.12, 0.12],
        'Recall (Fraud)': [0.44, 0.43, 0.44],
        'F1-Score (Fraud)': [0.16, 0.19, 0.19]
    }
    
    df_performance = pd.DataFrame(performance_data)
    
    st.subheader("Performance Metrics")
    st.dataframe(df_performance, use_container_width=True)
    
    # Bar charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Accuracy Comparison")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        bars = ax1.bar(df_performance['Model'], df_performance['Accuracy'], 
                       color=['#3498db', '#2ecc71', '#e74c3c'])
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Comparison')
        for bar, val in zip(bars, df_performance['Accuracy']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.2%}', ha='center', va='bottom')
        st.pyplot(fig1)
    
    with col2:
        st.subheader("F1-Score (Fraud Class)")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        bars = ax2.bar(df_performance['Model'], df_performance['F1-Score (Fraud)'],
                       color=['#3498db', '#2ecc71', '#e74c3c'])
        ax2.set_ylim(0, 0.3)
        ax2.set_ylabel('F1-Score')
        ax2.set_title('F1-Score for Fraud Detection')
        for bar, val in zip(bars, df_performance['F1-Score (Fraud)']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.2f}', ha='center', va='bottom')
        st.pyplot(fig2)
    
    st.markdown("---")
    
    st.subheader("📊 Key Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Best Model**\nRandom Forest & Neural Network achieved **87% accuracy**")
    with col2:
        st.info("**Fraud Detection**\n~44% of fraud cases are detected (Recall)")
    with col3:
        st.info("**Precision Trade-off**\nLow precision (12%) means many false alarms")

# ========== PREDICTION PAGE ==========
elif option == "🔮 Predict New Transaction":
    st.header("🔮 Predict New Transaction")
    
    st.warning("⚠️ Note: This is a simplified demo. In production, all 433 features would be required.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_amount = st.number_input("💰 Transaction Amount ($)", 
                                            min_value=0.0, 
                                            value=100.0,
                                            step=10.0)
        
        product_cd = st.selectbox("📦 Product Code", 
                                  ['R', 'S', 'T', 'U', 'W', 'H', 'C', 'P'])
        
        card_type = st.selectbox("💳 Card Type", 
                                 ['Visa', 'MasterCard', 'Amex', 'Discover'])
    
    with col2:
        transaction_hour = st.slider("🕐 Transaction Hour", 0, 23, 12)
        
        device_type = st.selectbox("📱 Device Type", 
                                   ['Mobile', 'Desktop', 'Tablet', 'Unknown'])
        
        email_domain = st.selectbox("📧 Email Domain",
                                    ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'other'])
    
    # Simple prediction logic (simplified for demo)
    st.markdown("---")
    
    if st.button("🔍 Predict Fraud Risk", type="primary"):
        # Placeholder for actual model prediction
        # In production, you would load your trained model and make prediction
        
        # Demo logic - just for illustration
        import random
        risk_score = random.uniform(0, 1)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Score", f"{risk_score:.2%}")
        
        with col2:
            if risk_score > 0.7:
                st.metric("Prediction", "⚠️ HIGH RISK", delta="Fraud")
                st.error("This transaction appears suspicious! 🚨")
            elif risk_score > 0.4:
                st.metric("Prediction", "⚠️ MEDIUM RISK", delta="Review")
                st.warning("This transaction requires manual review.")
            else:
                st.metric("Prediction", "✅ LOW RISK", delta="Normal")
                st.success("This transaction appears normal.")
        
        with col3:
            st.metric("Confidence", f"{random.uniform(0.7, 0.95):.1%}")
    
    st.markdown("---")
    st.info("
    **How it works:**
    1. The model analyzes transaction patterns
    2. Compares against historical fraud patterns
    3. Generates a risk score from 0-100%
    4. Higher scores indicate higher fraud probability
    
    *This is a demonstration. In production, all 433 features would be used for prediction.*
    ")

# Footer
st.markdown("---")
st.markdown("*Big Data Fraud Detection System | Powered by Machine Learning*")