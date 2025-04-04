import streamlit as st
import pandas as pd
import numpy as np
from pycaret.regression import load_model, predict_model
import base64

@st.cache_resource
def load_housing_model():
    return load_model('house')

@st.cache_data
def load_dataset():
    return pd.read_csv('housing.csv')

model = load_housing_model()
data = load_dataset()

def download_dataset():
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return href

def download_model():
    with open('house.pkl', 'rb') as f:
        bytes = f.read()
    b64 = base64.b64encode(bytes).decode()
    href = f'data:file/pkl;base64,{b64}'
    return href

def get_user_input():
    st.header("House Information Form")
    
    col1, col2 = st.columns(2)
    
    with col1:
        longitude = st.number_input('Longitude', min_value=-124.5, max_value=-114.0, value=-122.0, step=0.1, format="%.1f")
        latitude = st.number_input('Latitude', min_value=32.5, max_value=42.0, value=37.5, step=0.1, format="%.1f")
        housing_median_age = st.number_input('Median Age of Houses in Block', min_value=1, max_value=52, value=30)
        total_rooms = st.number_input('Total Rooms in Block', min_value=2, max_value=40000, value=2000)
        total_bedrooms = st.number_input('Total Bedrooms in Block', min_value=1, max_value=6500, value=500)
    
    with col2:
        population = st.number_input('Population in Block', min_value=3, max_value=15000, value=1000)
        households = st.number_input('Households in Block', min_value=1, max_value=6000, value=500)
        median_income = st.number_input('Median Income in Block ($10,000s)', min_value=0.5, max_value=15.0, value=3.0, step=0.1, format="%.1f")
        ocean_proximity = st.selectbox('Ocean Proximity', 
                                      ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'])
    
    user_data = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

def main():
    st.title('California Housing Price Prediction App')
    st.write("""
    This app predicts the median house value for California districts based on housing information.
    Please fill out the form below and click the 'Predict' button.
    """)
    
    st.sidebar.title("Options")
    
    if st.sidebar.button("View Dataset"):
        st.subheader("California Housing Dataset")
        st.write(data)
    
    dataset_download = download_dataset()
    st.sidebar.download_button(
        label="Download Dataset",
        data=data.to_csv(index=False),
        file_name='housing_dataset.csv',
        mime='text/csv'
    )
    
    with open('house.pkl', 'rb') as f:
        model_bytes = f.read()
    st.sidebar.download_button(
        label="Download Model",
        data=model_bytes,
        file_name='housing_model.pkl',
        mime='application/octet-stream'
    )
    
    user_input = get_user_input()
    
    st.subheader('House Input Summary')
    st.write(user_input)
    
    if st.button('Predict Median House Value'):
        prediction = predict_model(model, data=user_input)
        
        st.subheader('Prediction Result')
        predicted_value = prediction['prediction_label'][0]
        
        st.success(f'**Predicted Median House Value:** ${predicted_value:,.2f}')
        
       

if __name__ == '__main__':
    main()
