import streamlit as st
import pandas as pd
import numpy as np
import joblib



st.set_page_config(
page_title="CrediSight",
page_icon="💳",
layout="wide"
)



st.markdown("""

<style>

.main {
    background-color: #0E1117;
}

.big-title {
    font-size: 3rem;
    font-weight: 800;
    color: white;
}

.subtitle {
    color: #A0A0A0;
    font-size: 1.1rem;
}

.metric-box {
    background-color: #1E222A;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

</style>

""", unsafe_allow_html=True)

model = joblib.load("creditiq_model.pkl")
month_encoder = joblib.load("month_encoder.pkl")
occupation_encoder = joblib.load("occupation_encoder.pkl")
credit_mix_encoder = joblib.load("credit_mix_encoder.pkl")
payment_encoder = joblib.load("payment_encoder.pkl")
behaviour_encoder = joblib.load("behaviour_encoder.pkl")
target_encoder = joblib.load("target_encoder.pkl")



st.markdown(
"<div class='big-title'>💳 CrediSight</div>",
unsafe_allow_html=True
)

st.markdown(
"<div class='subtitle'>AI-Powered Creditworthiness Assessment System</div>",
unsafe_allow_html=True
)

st.divider()

col1, col2 = st.columns(2)

with col1:

 month = st.selectbox(
    "📅 Month",
    month_encoder.classes_
)

age = st.number_input(
    "👤 Age",
    min_value=18,
    max_value=100,
    value=30
)

occupation = st.selectbox(
    "💼 Occupation",
    occupation_encoder.classes_
)

annual_income = st.number_input(
    "💰 Annual Income",
    min_value=0.0,
    value=50000.0
)

monthly_salary = st.number_input(
    "🏦 Monthly In-Hand Salary",
    min_value=0.0,
    value=4000.0
)

interest_rate = st.number_input(
    "📈 Interest Rate",
    min_value=0,
    value=8
)

outstanding_debt = st.number_input(
    "💸 Outstanding Debt",
    min_value=0.0,
    value=5000.0
)

credit_mix = st.selectbox(
    "🎯 Credit Mix",
    credit_mix_encoder.classes_
)

with col2:

 num_accounts = st.number_input(
    "🏛 Bank Accounts",
    min_value=0,
    value=2
)

num_cards = st.number_input(
    "💳 Credit Cards",
    min_value=0,
    value=1
)

num_loans = st.number_input(
    "📑 Number of Loans",
    min_value=0,
    value=1
)

delayed_days = st.number_input(
    "⏳ Delay From Due Date",
    min_value=0,
    value=5
)

delayed_payments = st.number_input(
    "⚠ Delayed Payments",
    min_value=0.0,
    value=1.0
)

credit_inquiries = st.number_input(
    "🔍 Credit Inquiries",
    min_value=0.0,
    value=1.0
)

utilization = st.slider(
    "📊 Credit Utilization Ratio",
    0.0,
    100.0,
    30.0
)

history_age = st.number_input(
    "📚 Credit History Age (Months)",
    min_value=0,
    value=120
)



if st.button("🚀 Assess Credit Profile"):

    input_df = pd.DataFrame([{
    "Month": month_encoder.transform([month])[0],
    "Age": age,
    "Occupation": occupation_encoder.transform([occupation])[0],
    "Annual_Income": annual_income,
    "Monthly_Inhand_Salary": monthly_salary,
    "Num_Bank_Accounts": num_accounts,
    "Num_Credit_Card": num_cards,
    "Interest_Rate": interest_rate,
    "Num_of_Loan": num_loans,
    "Delay_from_due_date": delayed_days,
    "Num_of_Delayed_Payment": delayed_payments,
    "Changed_Credit_Limit": 0,
    "Num_Credit_Inquiries": credit_inquiries,
    "Credit_Mix": credit_mix_encoder.transform([credit_mix])[0],
    "Outstanding_Debt": outstanding_debt,
    "Credit_Utilization_Ratio": utilization,
    "Payment_of_Min_Amount": 1,
    "Total_EMI_per_month": 0,
    "Amount_invested_monthly": 0,
    "Payment_Behaviour": 0,
    "Monthly_Balance": monthly_salary,
    "Credit_History_Age_Months": history_age,
    "Debt_to_Income": outstanding_debt / (annual_income + 1),
}])


    prediction = model.predict(input_df)[0]

    probs = model.predict_proba(input_df)[0]

    label = target_encoder.inverse_transform([prediction])[0]

    st.divider()

    if label.lower() == "good":
        st.success(f"🟢 Credit Score Category: {label}")

    elif label.lower() == "standard":
        st.warning(f"🟡 Credit Score Category: {label}")

    else:
        st.error(f"🔴 Credit Score Category: {label}")

    prob_df = pd.DataFrame({
        "Category": target_encoder.classes_,
        "Probability": probs
    })

    st.subheader("📊 Prediction Confidence")

    st.bar_chart(
        prob_df.set_index("Category")
    )

    st.dataframe(prob_df)
