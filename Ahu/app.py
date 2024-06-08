import streamlit as st
import pickle

# Load the trained model from the pickle file
with open('Bagging_binary_tome.pkl', 'rb') as file:
    model = pickle.load(file)

# Create the Streamlit web app
st.title('AHU Commercial Project')

# Add input fields for the AHU features
mixed_air_temp = st.number_input('AHU: Mixed Air Temperature')
supply_air_fan_status = st.number_input('AHU: Supply Air Fan Status')
return_air_fan_status = st.number_input('AHU: Return Air Fan Status')
supply_air_fan_speed_control = st.number_input('AHU: Supply Air Fan Speed Control Signal')
return_air_fan_speed_control = st.number_input('AHU: Return Air Fan Speed Control Signal')
outdoor_air_damper_control = st.number_input('AHU: Outdoor Air Damper Control Signal')
return_air_damper_control = st.number_input('AHU: Return Air Damper Control Signal')
heating_coil_valve_control = st.number_input('AHU: Heating Coil Valve Control Signal')
supply_air_duct_static_pressure = st.number_input('AHU: Supply Air Duct Static Pressure')
occupancy_mode_indicator = st.number_input('Occupancy Mode Indicator')
supply_air_temp = st.number_input('AHU: Supply Air Temperature')
outdoor_air_temp = st.number_input('AHU: Outdoor Air Temperature')
return_air_temp = st.number_input('AHU: Return Air Temperature')
cooling_coil_valve_control = st.number_input('AHU: Cooling Coil Valve Control Signal')

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Sections",["EDA", "Predict"])


# Create a button to make predictions
if st.button('Predict'):
    # Prepare the input features for prediction
    input_data = [[mixed_air_temp, supply_air_fan_status, return_air_fan_status,
                   supply_air_fan_speed_control, return_air_fan_speed_control,
                   outdoor_air_damper_control, return_air_damper_control,
                   heating_coil_valve_control, supply_air_duct_static_pressure,
                   occupancy_mode_indicator, supply_air_temp, outdoor_air_temp,
                   return_air_temp, cooling_coil_valve_control]]

    # Make predictions using the trained model
    prediction = model.predict(input_data)

    # Display the prediction
    st.write('Prediction:', prediction)
