import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- Title ---
st.title("Company Valuation Tool (EV/S)")  # Updated title

# --- Input Fields ---
st.header("Input Company Data")

# Initialize data dictionary
if 'data' not in st.session_state:
    st.session_state.data = {
        'revenue_growth': [],
        'gross_profit_margin': [],
        'ev_s': []  # Changed to 'ev_s'
    }

# Input data for multiple companies
with st.form("data_entry"):
    st.write("--- New Company Entry ---")
    revenue_growth = st.number_input("Enter Revenue Growth (%)", value=0.0) / 100
    gross_profit_margin = st.number_input("Enter Gross Profit Margin (%)", min_value=0.0, value=0.0) / 100
    ev_s = st.number_input("Enter EV/S Multiple", value=0.0)  # Changed label

    submit_company = st.form_submit_button("Add Company Data")
    if submit_company:
        st.session_state.data['revenue_growth'].append(revenue_growth)
        st.session_state.data['gross_profit_margin'].append(gross_profit_margin)
        st.session_state.data['ev_s'].append(ev_s)  # Changed to 'ev_s'

# Convert data to DataFrame
df = pd.DataFrame(st.session_state.data)
st.write(df)  # Display the entered data

# --- Regression Analysis ---
X = df[['revenue_growth', 'gross_profit_margin']]
y = df['ev_s']  # Changed to 'ev_s'

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# --- Prediction and Output ---
st.header("Prediction")
with st.form("prediction_input"):
    new_revenue_growth = st.number_input("Enter Revenue Growth for Target Company (%)", value=0.0) / 100
    new_gross_profit_margin = st.number_input("Enter Gross Profit Margin for Target Company (%)", min_value=0.0, value=0.0) / 100

    predict_button = st.form_submit_button("Calculate EV/S")  # Changed label
    if predict_button:
        new_company_data = pd.DataFrame([[new_revenue_growth, new_gross_profit_margin]],
                                        columns=['revenue_growth', 'gross_profit_margin'])
        predicted_multiple = model.predict(new_company_data)
        st.metric("Predicted EV/S Multiple", f"{predicted_multiple[0]:.2f}")  # Changed label

# --- Reset Button (at the bottom) ---
st.markdown("<br><br>", unsafe_allow_html=True)
if st.button("Reset Data"):
    st.session_state.data = {
        'revenue_growth': [],
        'gross_profit_margin': [],
        'ev_s': []  # Changed to 'ev_s'
    }
    st.experimental_rerun()
