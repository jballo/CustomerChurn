import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import utils as ut

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))


def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


xgboost_model = load_model('xgb_model.pkl')

naive_bayes_model = load_model('nb_model.pkl')

random_forest_model = load_model('rf_model.pkl')

decision_tree_model = load_model('dt_model.pkl')

svm_model = load_model('svm_model.pkl')

knn_model = load_model('knn_model.pkl')

voting_classifier_model = load_model('voting_clf.pkl')

xgboost_SMOTE_model = load_model('xgboost-SMOTE.pkl')

xgboost_featureEngneered_model = load_model('xgboost-featureEngineered.pkl')

gb_classifier_featureEngineered = load_model(
    'gb-classifier-featureEngineered.pkl')

stacking_model = load_model('stacking_boost_clf.pkl')


def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_products, has_credit_card, is_active_member,
                  estimated_salary):
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == "France" else 0,
        'Geography_Germany': 1 if location == "Germany" else 0,
        'Geography_Spain': 1 if location == "Spain" else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


def make_predictions(input_df, input_dict):
    probabilities = {
        'XGBoost':
        xgboost_model.predict_proba(input_df)[0][1],
        'Gradient Boosting':
        gb_classifier_featureEngineered.predict_proba(input_df)[0][1],
        'Stacking':
        stacking_model.predict_proba(input_df)[0][1],
    }

    avg_probability = np.mean(list(probabilities.values()))

    col1, col2 = st.columns(2)

    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            f"The customer has a {avg_probability:.2%} probability of churning."
        )

    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)

    return avg_probability


def explain_prediction(probability, input_dict, surname):
    # Predefine the feature importance (from greatest to least) in a human-readable way
    feature_importance = [
        ("Number of Products Used",
         "Customers who use more products are less likely to leave, while those using fewer products might be at risk."
         ),
        ("Account Activity",
         "Being active in managing the account is a sign of customer engagement."
         ),
        ("Age",
         "Older customers might feel more secure, while younger ones might explore alternatives."
         ),
        ("Location",
         "Certain regions show different patterns in customer loyalty."),
        ("Balance",
         "Customers with higher account balances are less likely to churn."),
        ("Gender",
         "We notice small differences in engagement based on gender, with certain groups showing more loyalty."
         ),
        ("Credit Score",
         "A higher credit score indicates better financial health, which may contribute to retention."
         ), ("Salary", "Higher salary customers tend to stay loyal."),
        ("Credit Card Ownership",
         "Having a credit card with us may increase customer loyalty."),
        ("Tenure",
         "The longer someone has been with us, the more likely they are to stay."
         )
    ]

    # Create a plain-language prompt with embedded feature importance
    prompt = f"""
    You are a customer relations expert analyzing customer engagement data. Based on the information provided below, the system has calculated that {surname} has a {round(probability * 100, 1)}% chance of leaving our services.

    Here is {surname}'s profile:
    {input_dict}

    Based on key factors we typically observe, the following are most relevant:
    - {feature_importance[0][0]}: {feature_importance[0][1]}
    - {feature_importance[1][0]}: {feature_importance[1][1]}
    - {feature_importance[2][0]}: {feature_importance[2][1]}
    - {feature_importance[3][0]}: {feature_importance[3][1]}
    - {feature_importance[4][0]}: {feature_importance[4][1]}

    Provide a clear and concise explanation of {surname}'s risk of leaving, but avoid using technical terms like 'model' or 'prediction.' Focus on how these factors reflect customer engagement and loyalty trends.
    """

    print("EXPLANATION PROMPT: ", prompt)

    raw_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )

    return raw_response.choices[0].message.content


def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""
    You are writing a professional, customer-facing email on behalf of the bank, addressed to a client named {surname}. 
    The goal of this email is to reassure {surname} of the bank's commitment to supporting them, while subtly addressing potential areas of concern based on their profile. 
    The tone should be warm, proactive, and solution-oriented, without directly referencing the machine learning prediction.

    Here is {surname}'s profile:
    {input_dict}

    Based on the client's current profile and engagement with our services, we believe there may be opportunities to strengthen the relationship. 
    Please draft a message that:
    - Reassures the client of our ongoing support.
    - Encourages them to explore additional financial products or services that could benefit them.
    - Offers to schedule a meeting or conversation to better understand their needs and how we can further assist them.
    - Emphasizes that their satisfaction and success are important to us.

    Do not mention any predictions or probabilities of leaving the bank. Instead, focus on expressing our desire to enhance the client's experience and provide value.

    """
    raw_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )

    print("\n\nEMAIL PROMPT: ", prompt)

    return raw_response.choices[0].message.content


st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

customers = [
    f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
]
selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:

    selection_customer_id = int(selected_customer_option.split(" - ")[0])

    print("Selected Customer ID: ", selection_customer_id)

    selected_surname = selected_customer_option.split(" - ")[1]

    print("Surname: ", selected_surname)

    selected_customer = df.loc[df['CustomerId'] ==
                               selection_customer_id].iloc[0]

    print("Selected Customer: ", selected_customer)

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score",
                                       min_value=300,
                                       max_value=850,
                                       value=int(
                                           selected_customer['CreditScore']))

        location = st.selectbox("Location", ["Spain", "France", "Germany"],
                                index=["Spain", "France", "Germany"
                                       ].index(selected_customer['Geography']))

        gender = st.radio(
            "Gender", ["Male", "Female"],
            index=0 if selected_customer['Gender'] == 'Male' else 1)

        age = st.number_input("Age",
                              min_value=18,
                              max_value=100,
                              value=int(selected_customer['Age']))

        tenure = st.number_input("Tenure (years)",
                                 min_value=0,
                                 max_value=50,
                                 value=int(selected_customer['Tenure']))

    with col2:

        balance = st.number_input("Balance",
                                  min_value=0.0,
                                  value=float(selected_customer['Balance']))

        num_products = st.number_input("Number of Products",
                                       min_value=1,
                                       max_value=10,
                                       value=int(
                                           selected_customer['NumOfProducts']))

        has_credit_card = st.checkbox("Has Credit Card",
                                      value=bool(
                                          selected_customer['HasCrCard']))

        is_active_member = st.checkbox(
            "Is Active Member",
            value=bool(selected_customer['IsActiveMember']))

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer['EstimatedSalary']))

    input_df, input_dict = prepare_input(credit_score, location, gender, age,
                                         tenure, balance, num_products,
                                         has_credit_card, is_active_member,
                                         estimated_salary)
    avg_probability = make_predictions(input_df, input_dict)

    explanation = explain_prediction(avg_probability, input_dict,
                                     selected_customer['Surname'])

    st.markdown("------")

    st.subheader("Explanation of Prediction")

    st.markdown(explanation)

    email = generate_email(avg_probability, input_dict, explanation,
                           selected_customer['Surname'])

    st.markdown("------")

    st.subheader("Personalized Email")

    st.markdown(email)
