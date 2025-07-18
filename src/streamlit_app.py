import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle

# Load the model and preprocessing objects
model = tf.keras.models.load_model('../models/salaryPredictor/regression_model.h5')

with open('../models/salaryPredictor/label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('../models/salaryPredictor/onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('../models/salaryPredictor/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title('Salary Prediction')

# Input fields
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
age = st.slider('Age', 18, 100, 30)
tenure = st.slider('Tenure', 0, 20, 5)
balance = st.number_input('Balance', value=0.0)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_credit_card = st.checkbox('Has Credit Card')
is_active_member = st.checkbox('Is Active Member')
exited = st.checkbox('Has Exited')  # Adding the Exited field

if st.button('Predict Salary'):
    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [1 if has_credit_card else 0],
        'IsActiveMember': [1 if is_active_member else 0],
        'Exited': [1 if exited else 0],  # Include Exited column
    })

    # Handle geography encoding
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded, 
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    # Combine all features
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Ensure columns are in the same order as during training
    expected_columns = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
                       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited',
                       'Geography_France', 'Geography_Germany', 'Geography_Spain']
    
    input_data = input_data[expected_columns]

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    
    st.write('### Predicted Salary')
    st.write(f'${prediction[0][0]:,.2f}')

st.markdown("""
### Feature Information:
- **Credit Score**: Customer's credit score (300-850)
- **Geography**: Customer's location
- **Gender**: Customer's gender
- **Age**: Customer's age
- **Tenure**: Number of years as a customer
- **Balance**: Current account balance
- **Number of Products**: Number of bank products the customer uses
- **Has Credit Card**: Whether the customer has a credit card
- **Is Active Member**: Whether the customer is an active member
- **Has Exited**: Whether the customer has churned
""")