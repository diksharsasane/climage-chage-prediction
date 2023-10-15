# import streamlit as st
# import numpy as np
# import pickle

# # Load the pre-trained temperature prediction model
# with open('temp_major_city.pkl', 'rb') as file:
#     temp_major_city = pickle.load(file) 

# st.title("Climate Change Prediction")
# year = st.number_input('Year', min_value=1800, max_value=2050, value=2010)
# print(year)

# ok = st.button("Predict Temp")

# if ok:
#     bk_year = np.array([[year]], dtype=float)  # Ensure data type is set to float
#     try:
#         temp = temp_major_city.predict(bk_year)
#         bk_temp = np.round(temp[0],2)
#         st.subheader(f"The estimated temperature is {bk_temp} °C")
#     except Exception as exp:
#         st.error(f"An error occurred: {exp}")


import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the pre-trained temperature prediction model
with open('temp_major_city.pkl', 'rb') as file:
    temp_major_city = pickle.load(file) 

st.title("Climate Change Prediction")
year = st.number_input('Year', min_value=1800, max_value=2050, value=2010)

ok = st.button("Predict Temp")

if ok:
    bk_year = np.array([[year]], dtype=float)  # Ensure data type is set to float

    global_temp = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\climate\\GlobalTemperatures.csv')
    global_temp = global_temp[['dt', 'LandAndOceanAverageTemperature']]
    global_temp.dropna(inplace=True)
    global_temp['dt'] = pd.to_datetime(global_temp.dt).dt.strftime('%d/%m/%Y')
    global_temp['dt'] = global_temp['dt'].apply(lambda x:x[6:])
    global_temp = global_temp.groupby(['dt'])['LandAndOceanAverageTemperature'].mean().reset_index()

    try:
        temp = temp_major_city.predict(bk_year)
        bk_temp = np.round(temp[0], 2)
        st.subheader(f"The estimated temperature is {bk_temp} °C")

        # Generate a sample line chart (you can replace this with your actual data)
        # years = np.arange(1800, 2051)
        # temperatures = np.random.uniform(0, 10, len(years))  # Replace with actual temperature data
        # Replace with your actual data
        years = global_temp['dt'].astype(int)  # Assuming 'dt' column contains years as integers
        temperatures = global_temp['LandAndOceanAverageTemperature']  # Assuming this column contains temperatures


        # Create the line chart
        plt.figure(figsize=(8, 6))
        plt.plot(years, temperatures, label='Estimated Temperatures')
        plt.xlabel('Year')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature Trends Over Time')
        plt.legend()
        st.pyplot(plt)  # Display the chart in the Streamlit app

    except Exception as exp:
        st.error(f"An error occurred: {exp}")
