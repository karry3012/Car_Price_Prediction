# import pickle
# import numpy as np
# import pandas as pd
# # Load the model
# with open('LinearRegressionModel.pkl', 'rb') as f:
#     model = pickle.load(f)

# x=model.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],
#                           data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))

# print(x)


# Streamlit UI

# import streamlit as st
# import pandas as pd
# import pickle

# # Load the model
# model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# # Load the dataset
# df = pd.read_csv('car_price_clean_data.csv')

# # Set up the Streamlit app
# st.title('Car Price Prediction')

# # Create input fields with options from the dataset
# name = st.selectbox('Car Name', df['name'].unique())
# company = st.selectbox('Company', df['company'].unique())
# year = st.selectbox('Year', sorted(df['year'].unique(), reverse=True))
# kms_driven = st.number_input('Kilometers Driven', min_value=0, value=10000)
# fuel_type = st.selectbox('Fuel Type', df['fuel_type'].unique())

# # Create a DataFrame with the input
# input_data = pd.DataFrame({
#     'name': [name],
#     'company': [company],
#     'year': [year],
#     'kms_driven': [kms_driven],
#     'fuel_type': [fuel_type]
# })

# # Make prediction
# if st.button('Predict Price'):
#     prediction = model.predict(input_data)
#     st.success(f'The predicted price of the car is ₹{prediction[0]:,.2f}')

# # Add some information about the model
# st.info('This model predicts car prices based on features like the car name, company, year, kilometers driven, and fuel type.')



import streamlit as st
import pandas as pd
import pickle

# Load the model
@st.cache_resource
def load_model():
    return pickle.load(open('LinearRegressionModel.pkl', 'rb'))

model = load_model()

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('car_price_clean_data.csv')
    df['name'] = df['name'].astype(str)
    df['company'] = df['company'].astype(str)
    df['year'] = df['year'].astype(int)
    df['fuel_type'] = df['fuel_type'].astype(str)
    return df

df = load_data()

# Set up the Streamlit app
st.title('Car Price Prediction')

# Create input fields with options from the dataset
company = st.selectbox('Company', [''] + sorted(df['company'].unique()))

# Filter car names based on selected company
if company:
    car_options = df[df['company'] == company]['name'].unique()
else:
    car_options = df['name'].unique()

name = st.selectbox('Car Name', [''] + sorted(car_options))

year = st.selectbox('Year', [''] + sorted(df['year'].unique(), reverse=True))
kms_driven = st.number_input('Kilometers Driven', min_value=0, value=10000)
fuel_type = st.selectbox('Fuel Type', [''] + sorted(df['fuel_type'].unique()))

# Create a DataFrame with the input
input_data = pd.DataFrame({
    'name': [name],
    'company': [company],
    'year': [year],
    'kms_driven': [kms_driven],
    'fuel_type': [fuel_type]
})

# Make prediction
if st.button('Predict Price'):
    if '' in input_data.values:
        st.warning('Please fill in all fields before predicting.')
    else:
        prediction = model.predict(input_data)
        st.success(f'The predicted price of the car is ₹{prediction[0]:,.2f}')



