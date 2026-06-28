# Big-Data-Fraud-Detection



A machine learning project that detects fraudulent financial transactions using the IEEE-CIS Fraud Detection dataset, with an interactive Streamlit dashboard to explore the data and results.



\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

&#x20;Files



&#x20;`eda\_preprocessing.ipynb`

A Jupyter notebook that handles all the data work. It loads and merges the raw transaction and identity CSVs, cleans up missing values, selects the most important features, balances the dataset using SMOTE, trains a Logistic Regression and Random Forest model, and evaluates them with metrics like accuracy, recall, and AUC-ROC.



\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

&#x20;`Superapp.py`

The Streamlit dashboard. It has three pages — a Dashboard with key stats and an overview, a Data Analysis page to explore the dataset visually, and a Model Performance page showing how each model compares. It runs on either a built-in demo dataset or your own uploaded CSVs.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_



Setup



1\. Clone the repository

bash

git clone https://github.com/your-username/your-repo-name.git

cd your-repo-name





2\. Install dependencies

bash

pip install streamlit pandas numpy matplotlib scikit-learn imbalanced-learn jupyter





3\. Add your dataset (optional)

If you want to use the real IEEE-CIS data, download it from \[Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) and place the files in the project folder:



train\_transaction.csv

train\_identity.csv

```

If you skip this, the dashboard will run on demo data automatically.



\---



**Running the Dashboard**



Open Command Prompt, navigate to the project folder, and run:



streamlit run Superapp.py





Your browser will open automatically at `http://localhost:8501`.



\---



**Running the Notebook**



bash

jupyter notebook eda\_preprocessing.ipynb



> Make sure to update the file paths inside the notebook to point to where your CSVs are saved on your machine.

