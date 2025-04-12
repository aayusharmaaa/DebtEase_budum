# main.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Load Dataset
file_path = r"C:\Users\Aayush Sharma\OneDrive\Documents\data.csv"
df = pd.read_csv(file_path)

# Drop missing values
df.dropna(inplace=True)

# Encode categorical columns
label_encoders = {}
categorical_cols = ['Occupation', 'City_Tier']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define input features
features = [
    'Desired_Savings', 'Desired_Savings_Percentage', 'Income', 'Age', 'Dependents', 'Rent',
    'Loan_Repayment', 'Insurance', 'Groceries', 'Transport', 'Eating_Out', 'Entertainment',
    'Utilities', 'Healthcare', 'Education', 'Miscellaneous', 'Disposable_Income',
    'City_Tier', 'Occupation'
]

# Define savings target columns
target_columns = {
    'Groceries': 'Potential_Savings_Groceries',
    'Transport': 'Potential_Savings_Transport',
    'Eating_Out': 'Potential_Savings_Eating_Out',
    'Entertainment': 'Potential_Savings_Entertainment',
    'Utilities': 'Potential_Savings_Utilities',
    'Healthcare': 'Potential_Savings_Healthcare',
    'Education': 'Potential_Savings_Education',
    'Miscellaneous': 'Potential_Savings_Miscellaneous'
}

# Train and save model for each category
for category, target in target_columns.items():
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, f'model_{category}.pkl')
    print(f"✅ Model for {category} saved as model_{category}.pkl")

# Save label encoders for use in app
joblib.dump(label_encoders, 'label_encoders.pkl')
print("✅ Label encoders saved as label_encoders.pkl")