import streamlit as st
import pandas as pd
import joblib
import numpy as np
from gemini_backend import get_financial_advice

# Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Personal Finance Advisor", page_icon="💰", layout="wide")


# Load models and encoders
@st.cache_resource
def load_models():
    models = {}
    categories = [
        'Groceries', 'Transport', 'Eating_Out', 'Entertainment',
        'Utilities', 'Healthcare', 'Education', 'Miscellaneous'
    ]
    for category in categories:
        models[category] = joblib.load(f'model_{category}.pkl')

    label_encoders = joblib.load('label_encoders.pkl')
    return models, label_encoders


models, label_encoders = load_models()

st.title("DebtEase")
st.subheader("Get personalized savings recommendations with AI")

# Create form for user input
with st.form("finance_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Income & Savings Goals")
        income = st.number_input("Monthly Income (₹)", min_value=0, value=50000)
        desired_savings = st.number_input("Desired Monthly Savings (₹)", min_value=0, value=10000)
        desired_savings_percentage = st.slider("Desired Savings Percentage", 0, 80, 20)

        st.subheader("Personal Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)

        occupation_list = list(label_encoders['Occupation'].classes_)
        occupation = st.selectbox("Occupation", occupation_list)

        city_tier_list = list(label_encoders['City_Tier'].classes_)
        city_tier = st.selectbox("City Tier", city_tier_list)

    with col2:
        st.subheader("Fixed Expenses")
        rent = st.number_input("Monthly Rent (₹)", min_value=0, value=15000)
        loan_repayment = st.number_input("Loan Repayments (₹)", min_value=0, value=5000)
        insurance = st.number_input("Insurance Premiums (₹)", min_value=0, value=2000)

        st.subheader("Variable Expenses")
        groceries = st.number_input("Groceries (₹)", min_value=0, value=8000)
        transport = st.number_input("Transport (₹)", min_value=0, value=3000)
        eating_out = st.number_input("Eating Out (₹)", min_value=0, value=4000)
        entertainment = st.number_input("Entertainment (₹)", min_value=0, value=3000)
        utilities = st.number_input("Utilities (Bills) (₹)", min_value=0, value=2000)
        healthcare = st.number_input("Healthcare (₹)", min_value=0, value=1500)
        education = st.number_input("Education (₹)", min_value=0, value=2000)
        miscellaneous = st.number_input("Miscellaneous (₹)", min_value=0, value=2000)

    submitted = st.form_submit_button("Analyze My Finances")

# Process and display results when submitted
if submitted:
    # Calculate disposable income
    total_expenses = (rent + loan_repayment + insurance + groceries + transport +
                      eating_out + entertainment + utilities + healthcare +
                      education + miscellaneous)
    disposable_income = income - total_expenses

    # Encode categorical features
    encoded_occupation = label_encoders['Occupation'].transform([occupation])[0]
    encoded_city_tier = label_encoders['City_Tier'].transform([city_tier])[0]

    # Create input dataframe for prediction
    input_data = {
        'Desired_Savings': desired_savings,
        'Desired_Savings_Percentage': desired_savings_percentage,
        'Income': income,
        'Age': age,
        'Dependents': dependents,
        'Rent': rent,
        'Loan_Repayment': loan_repayment,
        'Insurance': insurance,
        'Groceries': groceries,
        'Transport': transport,
        'Eating_Out': eating_out,
        'Entertainment': entertainment,
        'Utilities': utilities,
        'Healthcare': healthcare,
        'Education': education,
        'Miscellaneous': miscellaneous,
        'Disposable_Income': disposable_income,
        'City_Tier': encoded_city_tier,
        'Occupation': encoded_occupation
    }

    input_df = pd.DataFrame([input_data])

    # Make predictions for each category
    savings_predictions = {}
    for category, model in models.items():
        prediction = model.predict(input_df)[0]
        # Ensure predictions are reasonable (not negative and not more than current spending)
        prediction = max(0, prediction)
        prediction = min(prediction, input_data[category])
        savings_predictions[category] = round(prediction, 2)

    # Display results
    st.subheader("Your Financial Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Monthly Income", f"₹{income:,}")
        st.metric("Current Total Expenses", f"₹{total_expenses:,}")
        st.metric("Current Disposable Income", f"₹{disposable_income:,}")

        current_savings_rate = (disposable_income / income) * 100
        st.metric("Current Savings Rate", f"{current_savings_rate:.1f}%")

        if current_savings_rate < desired_savings_percentage:
            st.warning(f"Your current savings rate is below your goal of {desired_savings_percentage}%")
        else:
            st.success(f"You're meeting your savings goal of {desired_savings_percentage}%!")

    # Calculate total potential savings
    total_potential_savings = sum(savings_predictions.values())

    with col2:
        st.metric("Total Potential Monthly Savings", f"₹{total_potential_savings:,}")
        new_savings_rate = ((disposable_income + total_potential_savings) / income) * 100
        st.metric("Potential New Savings Rate", f"{new_savings_rate:.1f}%")

        if new_savings_rate >= desired_savings_percentage:
            st.success("By following these recommendations, you can achieve your savings goal!")
        else:
            st.info(
                "These recommendations will improve your savings, but additional adjustments may be needed to reach your goal.")

    # Display potential savings by category
    st.subheader("Potential Savings by Category")

    # Create data for the bar chart
    categories = list(savings_predictions.keys())
    values = list(savings_predictions.values())
    current_spending = [input_data[cat] for cat in categories]

    # Display as a table for clarity
    savings_data = {
        'Category': categories,
        'Current Spending (₹)': current_spending,
        'Potential Savings (₹)': values,
        'Remaining Spending (₹)': [c - s for c, s in zip(current_spending, values)],
        'Savings (%)': [round((s / c) * 100, 1) if c > 0 else 0 for s, c in zip(values, current_spending)]
    }

    df_savings = pd.DataFrame(savings_data)
    st.dataframe(df_savings.style.highlight_max(subset=['Potential Savings (₹)']), use_container_width=True)

    # Get personalized advice from Gemini
    user_data = {
        'income': income,
        'savings_goal': desired_savings,
        'total_expenses': total_expenses,
        'disposable_income': disposable_income,
        'current_savings_rate': current_savings_rate,
        'age': age,
        'dependents': dependents,
        'occupation': occupation,
        'city_tier': city_tier,
        'potential_savings': savings_predictions,
        'total_potential_savings': total_potential_savings
    }

    # Get advice from Gemini
    st.subheader("Personalized Financial Advice")
    with st.spinner("Generating personalized financial advice..."):
        advice = get_financial_advice(user_data)
        st.markdown(advice)