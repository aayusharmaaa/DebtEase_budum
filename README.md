# 💰 DebtEase: Conversational AI for Personalized Debt Management

A Streamlit-based web app that analyzes user income, expenses, and financial goals to deliver personalized savings recommendations across different spending categories using XGBoost regression models. It also generates contextual financial advice using a Gemini based API.

---

## 🚀 Features

- 🧠 Predicts potential monthly savings in categories like groceries, transport, entertainment, etc.
- 💬 Generates personalized financial advice using Gemini API
- 📊 Displays a financial summary with savings rate, potential improvements, and breakdowns
- 🧾 Shows clear actionable insights via charts and savings tables
- 🧩 Clean and intuitive Streamlit dashboard interface

---

## 🛠️ Tech Stack

### 🖥️ Frontend
- **Streamlit** – Interactive dashboard and UI
- **Streamlit Components** – Metrics, forms, tables, and layouts
- **Pandas + Plotly** – For formatting and potential visualization
- **Markdown/HTML** – For styled UI elements

### 🧪 Backend
- **Python**
- **Pandas** – Data processing and feature extraction
- **Scikit-learn** – Label encoding and preprocessing
- **XGBoost** – Regression models trained to predict savings in 8 spending categories
- **Joblib** – For saving and loading models and encoders
- **Google Gemini API** – To generate personalized financial coaching via LLM

---

## 📂 Project Structure

📁 personal-finance-advisor/
│
├── 📄 app.py                  # Streamlit frontend (UI, user inputs, prediction, dashboard)
├── 📄 main.py                 # Model training script using XGBoost
├── 📄 gemini_backend.py       # Gemini API integration to generate financial advice
│
├── 📁 models/                 # Folder to store trained XGBoost models
│   ├── model_Groceries.pkl
│   ├── model_Transport.pkl
│   └── ...                   # One model per spending category
│
├── 📁 data/
│   └── data.csv              # Training dataset with income, expenses, targets
│
├── 📁 assets/
│   └── flowchart.png         # Project flow diagram
│
├── 📄 label_encoders.pkl     # Encoded label mappings for Occupation, City_Tier
├── 📄 README.md              # Project overview and instructions
│


