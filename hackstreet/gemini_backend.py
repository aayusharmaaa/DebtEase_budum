import requests
import json

# API key for Google Gemini
GEMINI_API_KEY = "AIzaSyCbwAj_jEB4q4TP2xF9eCt63TF7zYLIxSk"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def get_financial_advice(user_data):
    """
    Get personalized financial advice from Google Gemini based on user data and predictions.

    Args:
        user_data (dict): Dictionary containing user financial data and predictions

    Returns:
        str: Personalized financial advice
    """
    # Format the prompt for Gemini
    prompt = f"""
    As a financial advisor, please provide personalized financial advice based on the following information:

    Monthly Income: ₹{user_data['income']}
    Age: {user_data['age']}
    Dependents: {user_data['dependents']}
    Occupation: {user_data['occupation']}
    City Tier: {user_data['city_tier']}

    Current Financial Situation:
    - Total Monthly Expenses: ₹{user_data['total_expenses']}
    - Current Disposable Income: ₹{user_data['disposable_income']}
    - Current Savings Rate: {user_data['current_savings_rate']:.1f}%
    - Monthly Savings Goal: ₹{user_data['savings_goal']}

    Our AI model has analyzed their spending patterns and identified these potential monthly savings:
    """

    # Add potential savings by category
    for category, amount in user_data['potential_savings'].items():
        prompt += f"- {category}: ₹{amount:,.2f}\n"

    prompt += f"""
    Total Potential Monthly Savings: ₹{user_data['total_potential_savings']:,.2f}

    Please provide:
    1. A brief assessment of their current financial situation
    2. Specific, actionable advice for each spending category where significant savings potential exists
    3. Long-term financial planning tips based on their age, income, and dependents
    4. Investment suggestions appropriate for their profile
    5. Any additional financial recommendations

    Format your response in clear sections with markdown formatting and focus on practical, culturally relevant advice for an Indian context.
    """

    # Prepare request to Gemini API
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    # Make API request
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            data=json.dumps(data)
        )

        # Parse response
        if response.status_code == 200:
            response_data = response.json()
            # Extract the text from the response
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                candidate = response_data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    for part in candidate['content']['parts']:
                        if 'text' in part:
                            return part['text']

            return "I couldn't generate specific financial advice at this time."
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return "Sorry, I couldn't connect to the financial advice service. Please try again later."

    except Exception as e:
        print(f"Exception: {e}")
        return "There was an error generating financial advice. Please try again later."


# Test function if this file is run directly
if __name__ == "__main__":
    # Test data
    test_data = {
        'income': 50000,
        'savings_goal': 10000,
        'total_expenses': 42000,
        'disposable_income': 8000,
        'current_savings_rate': 16.0,
        'age': 30,
        'dependents': 1,
        'occupation': 'Software Engineer',
        'city_tier': 'Tier 1',
        'potential_savings': {
            'Groceries': 1500,
            'Transport': 800,
            'Eating_Out': 1200,
            'Entertainment': 900,
            'Utilities': 300,
            'Healthcare': 200,
            'Education': 400,
            'Miscellaneous': 600
        },
        'total_potential_savings': 5900
    }

    advice = get_financial_advice(test_data)
    print(advice)