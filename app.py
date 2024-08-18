import streamlit as st 
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pickle


model = load_model('model2.h5')
pipe = pickle.load(open('pipe.pkl','rb'))


## streamlit app
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', ['France','Spain','Germany'])
gender = st.selectbox('Gender', ['Male','Female'])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


# convert to dataFrame.
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Geography':[geography],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}, index=[0])

input_data = pipe.transform(input_data)

# Predict churn
prediction = model.predict(input_data)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')