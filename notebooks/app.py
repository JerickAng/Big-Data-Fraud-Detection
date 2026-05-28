import streamlit as st
import pandas as pd
import numpy as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    ["🏠 Home", "📈 Data Overview", "🤖 Model Performance"]
)

# Load data function (cached to avoid reloading)
@st.cache_data
def load_data():
    transaction = pd.read_csv(r"C:\Users\Jeric\OneDrive\Documents\Big Data Fraud Detection\Big-Data-Fraud-Detection\Datasets\train_transaction.csv")
    identity = pd.read_csv(r"C:\Users\Jeric\OneDrive\Documents\Big Data Fraud Detection\Big-Data-Fraud-Detection\Datasets\train_identity.csv")
    df = transaction.merge(identity, on='TransactionID', how='left')
    return df

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
    st.write("This dashboard demonstrates a Big Data Fraud Detection System for the banking and financial sector.")
    st.write("")
    st.write("**Key Features:**")
    st.write("- Data Processing: Handles large-scale transaction data (590,540 records, 434 features)")
    st.write("- Machine Learning Models: Logistic Regression, Random Forest, and Neural Network")
    st.write("- Class Balancing: SMOTE (Synthetic Minority Over-sampling Technique)")

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
    top_features = top_features.drop('isFraud')
    ax2.barh(top_features.index, top_features.values, color='#3498db')
    ax2.set_xlabel('Correlation with Fraud')
    ax2.set_title('Top 10 Features Correlated with Fraud')
    st.pyplot(fig2)

# ========== MODEL PERFORMANCE PAGE ==========
elif option == "🤖 Model Performance":
    st.header("🤖 Model Performance Comparison")
    
    st.write("Three machine learning models were trained and evaluated for fraud detection:")
    st.write("- **Logistic Regression**: Baseline linear model")
    st.write("- **Random Forest**: Ensemble of decision trees")
    st.write("- **Neural Network (MLP)**: Multi-layer perceptron with 2 hidden layers")
    
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

# Footer
st.markdown("---")
st.markdown("*Big Data Fraud Detection System | Powered by Machine Learning*")
