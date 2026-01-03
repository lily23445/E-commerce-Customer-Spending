import streamlit as st
import pickle
import numpy as np

with open("lm.pkl", "rb") as f:
    lm = pickle.load(f)


st.title("E-commerce Customer Spending Predictor")

st.markdown("""
Enter the details of a customer to predict their spending.
""")

# Input features
session_length = st.slider("Avg. Session Length", 0.0, 60.0, 3.0)
time_on_app = st.slider("Time on App (minutes)", 0.0, 60.0, 3.0)
time_on_website = st.slider("Time on Website (minutes)", 0.0, 60.0, 3.0)
length_of_membership = st.slider("Length of Membership (Years)", 0.0, 10.0, 3.0)

# Make prediction
if st.button("Predict Spending"):
    features = np.array([[session_length, time_on_app, time_on_website, length_of_membership]])
    prediction = lm.predict(features)
    st.success(f"Predicted Spending: ${prediction[0]:.2f}")

st.sidebar.subheader("How to Use")
st.sidebar.write("""
- Adjust the sliders for each customer feature.
- Click 'Predict Spending' to see the expected spend.
- Values are approximate, based on a linear regression model.
""")

st.sidebar.subheader("Model Info")
st.sidebar.write("""
- Linear Regression
- Trained on synthetic customer data
- Predicts expected spending ($)
- Evaluated using RMSE and R²
""")


st.sidebar.markdown("---")
st.sidebar.markdown("**Author:** Kalash")
st.sidebar.markdown("[Portfolio/GitHub](https://github.com/lily23445)")

