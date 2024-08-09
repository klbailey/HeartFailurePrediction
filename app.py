import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
import sklearn
# Print scikit-learn version
# st.write(f"scikit-learn version: {sklearn.__version__}")
# print("pandas version:", pd.__version__)
# print("streamlit version:", st.__version__)
# Display joblib version
# st.write(f"joblib version: {joblib.__version__}")
# import sys
# print("Python version:", sys.version)

# Title and summary
st.markdown("<h1 style='text-align: center;'>Heart Failure Prediction</h1>", unsafe_allow_html=True)

st.markdown("""
<p style='text-align: center;'>
<b>Time (days):</b> Follow-up Period <br>
<b>Ejection Fraction (%):</b> Percentage of blood leaving the heart with each contraction <br>
<b>Age (years):</b> Patient Age <br>
<b>Platelets:</b> Platelet count in the blood <br>
<b>Creatinine Phosphokinase:</b> Level of the CPK enzyme in the blood
</p>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align: center;'>
<b>Using RandomForestClassifier (80:20 split ratio)</b> <br>
<b>Train:</b> Accuracy: 0.9300, Precision: 0.9273, Recall: 0.8361, F1 Score: 0.8793 <br>
<b>Test:</b> Accuracy: 0.9000, Precision: 0.8750, Recall: 0.8235, F1 Score: 0.8485
</p>
""", unsafe_allow_html=True)

# Load the model
# model_path = 'C:/Users/klbai/OneDrive/Desktop/Capstone/Final1/random_forest_model_best.joblib'
model_path = 'random_forest_model_best.joblib'
model = joblib.load(model_path)
print(type(model))
# Create and fit the scaler (Replace with the scaler fitted on your actual training data)
example_train_data = pd.DataFrame({
    'time': [10, 20, 30],
    'ejection_fraction': [30, 40, 50],
    'age': [40, 50, 60],
    'platelets': [200000, 250000, 300000],
    'creatinine_phosphokinase': [1000, 2000, 3000]
})

scaler = StandardScaler()
scaler.fit(example_train_data)

# Function to scale the input data
def scale_input_data(data, scaler):
    return scaler.transform(data)

# Manual input section for individual predictions
st.write("### Input New Data for Prediction")

# User Inputs
time = st.number_input("Time (days)", min_value=0)
ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100)
age = st.number_input("Age (years)", min_value=0)
platelets = st.number_input("Platelets (per microliter)", min_value=0)
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=0)

# Check if all input fields are provided
if st.button('Predict'):
    if time is not None and ejection_fraction is not None and age is not None and platelets is not None and creatinine_phosphokinase is not None:
        # Prepare the input data
        input_data = pd.DataFrame({
            'time': [time],
            'ejection_fraction': [ejection_fraction],
            'age': [age],
            'platelets': [platelets],
            'creatinine_phosphokinase': [creatinine_phosphokinase]
        })

        # Scale the input data
        input_data_scaled = scale_input_data(input_data, scaler)

        # Display input data
        st.write("### Input Data:")
        st.write(input_data)

        # Make prediction
        try:
            prediction = model.predict(input_data_scaled)
            prediction_proba = model.predict_proba(input_data_scaled)

            st.write("### Prediction Result:")
            st.write("Death" if prediction[0] == 1 else "No Death")
            st.write("Prediction Probability for Death:", prediction_proba[0][1])
            st.write("Prediction Probability for No Death:", prediction_proba[0][0])
            
        except Exception as e:
            st.write(f"Error: {e}")
    else:
        st.write("Please enter all required fields to make a prediction.")

# Debugging section (only shown after user input)
if st.button('Show Debugging Known Data Points'):
    st.write("### Debugging Known Data Points")

    known_inputs = [
        {'time': 10, 'ejection_fraction': 20, 'age': 85, 'platelets': 180000, 'creatinine_phosphokinase': 1500},
        {'time': 60, 'ejection_fraction': 65, 'age': 45, 'platelets': 250000, 'creatinine_phosphokinase': 1500}
    ]

    for idx, known_input in enumerate(known_inputs):
        df = pd.DataFrame([known_input])
        st.write(f"### Known Data Point {idx+1}:")
        st.write(df)

        try:
            known_input_scaled = scale_input_data(df, scaler)
            known_prediction = model.predict(known_input_scaled)
            known_prediction_proba = model.predict_proba(known_input_scaled)

            st.write(f"### Known Data Point {idx+1} Prediction Result:")
            st.write("Death" if known_prediction[0] == 1 else "No Death")
            st.write("Prediction Probability for Death:", known_prediction_proba[0][1])
            st.write("Prediction Probability for No Death:", known_prediction_proba[0][0])
            
        except Exception as e:
            st.write(f"Error with Known Data Point {idx+1}: {e}")
