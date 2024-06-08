import os
import pickle
import streamlit as st
import numpy as np
from Eda1 import show_eda
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# Load pre-trained machine learning model
try:
    scaleModel = pickle.load(open("scale.sav", 'rb'))
    model_path = pickle.load(open("modelHeart.sav", 'rb'))
except Exception as e:
    st.error(f"Error loading models: {e}")

# Render heart disease prediction page
def render_heart_disease_prediction():
    st.title('Heart Disease Prediction using ML')

    # Collect user inputs for heart disease prediction
    col1, col2, col3 = st.columns(3)
    with col1:
        Gender = st.selectbox('Gender', ['Male', 'Female'])
    with col2:
        Age = st.number_input('Age', min_value=1, max_value=120)
    with col3:
        is_smoking = st.selectbox('Smoking Status', ['Yes', 'No'])
    
    with col1:
        Cigs_Per_Day = st.number_input('Cigarettes Per Day', min_value=0, max_value=30)
    with col2:
        BP_Meds = st.number_input('Blood Pressure Medication((0=No,1=Yes))', min_value=0.0, max_value=1.0)
    with col3:
       Prevalent_Stroke = st.number_input('Prevalent Stroke(0=No,1=Yes)', min_value=0, max_value=1)
    
    with col1:
        Prevalent_Hyp = st.number_input('Prevalent Hypertension((0=No,1=Yes))', min_value=0, max_value=1)
    with col2:
       Diabetes = st.number_input('Diabetes', min_value=0, max_value=1)
    with col3:
        Tot_Chol = st.number_input('Total Cholesterol (mg/dL)', min_value=50, max_value=500)
    
    with col1:
        Sys_BP = st.number_input('Systolic Blood Pressure (mmHg)', min_value=50.0, max_value=300.0)
    with col2:
       Dia_BP = st.number_input('Diastolic Blood Pressure (mmHg)', min_value=30.0, max_value=200.0)
    with col3:
         Bmi = st.number_input('BMI', min_value=10.0, max_value=70.0)
    
    with col1:
        Heart_Rate = st.number_input('Heart Rate (bpm)', min_value=30, max_value=200)
    with col2:
        Glucose = st.number_input('Glucose (mg/dL)', min_value=50, max_value=500)

    # Perform heart disease prediction
    if st.button('Heart Disease Test Result'):
        prediction = predict_heart_disease(Age, Gender, is_smoking, Cigs_Per_Day, BP_Meds,
                                           Prevalent_Stroke, Prevalent_Hyp, Diabetes, Tot_Chol,
                                           Sys_BP, Dia_BP, Bmi, Heart_Rate, Glucose)
        st.success(prediction)

# Function to predict heart disease based on user inputs
def predict_heart_disease(age, gender, smoking_status, cigs_per_day, bp_meds, prevalent_stroke,
                           prevalent_hyp, diabetes, tot_chol, sys_bp, dia_bp, bmi, heart_rate, glucose):
    # Prepare user input as a numpy array
    user_input = np.array([age,
                           0.0 if gender == 'Male' else 1.0,
                           1.0 if smoking_status == 'Yes' else 0.0,
                           cigs_per_day,
                           bp_meds,
                           prevalent_stroke,
                           prevalent_hyp,
                           diabetes,
                           tot_chol,
                           sys_bp,
                           dia_bp,
                           bmi,
                           heart_rate,
                           glucose]).reshape(1, -1)

    # Scale the input data using the loaded scaler
    scaled_input = scaleModel.transform(user_input)

    # Make prediction using the loaded model
    prediction = model_path.predict(scaled_input)

    # Return prediction result based on the model output
    if prediction[0] == 1:
        return 'The person is having heart disease'
    elif prediction[0] == 0:
        return 'The person does not have any heart disease'
    else:
        return 'Invalid prediction'

# Render EDA (Exploratory Data Analysis) section
def render_eda():
    st.title('Exploratory Data Analysis (EDA)')
    # Add your EDA content here
    show_eda()

# Render BMI (Body Mass Index) Calculator section
def render_bmi_calculator():
    st.title("BMI Calculator")
    st.markdown("Calculate your Body Mass Index (BMI)")

    weight = st.number_input("Enter weight (kg)", min_value=0.0)
    height = st.number_input("Enter height (m)", min_value=0.0)

    if st.button("Calculate BMI"):
        calculate_and_display_bmi(weight, height)

def calculate_and_display_bmi(weight, height):
    bmi_value = weight / (height ** 2)
    st.header("Your BMI Result")
    st.write(f"**BMI:** {bmi_value:.2f}")

    # Display BMI information
    st.header('BMI Categories and Meaning')

    st.write("**Underweight:** BMI < 18.5")
    st.write("**Normal weight:** 18.5 <= BMI < 25")
    st.write("**Overweight:** 25 <= BMI < 30")
    st.write("**Obesity:** BMI >= 30")

    st.markdown("""
        - **Underweight:** You may be at risk of health problems such as malnutrition.
        - **Normal weight:** Generally, this range is associated with good health.
        - **Overweight:** Excess body weight may lead to health issues like heart disease.
        - **Obesity:** Higher risk of serious health conditions such as diabetes and heart disease.
    """)

def render_Confusion_matrix():
    st.title('Comparison Between Algorithms')
    st.image('20.jpg')
    
    plt.show()

    





# Render About Us section
def render_about_us():
    st.title('About Us')
    st.markdown("""
        We are dedicated to providing innovative health solutions through our Streamlit app.
        Our goal is to empower users with valuable insights and predictions related to health and wellness.
        For any inquiries or feedback, please contact us at example@example.com.
    """)

# Main function to handle sidebar navigation
def main():
   # Sidebar with navigation options using option menu
    selected_page = st.sidebar.selectbox(
        'Navigation',
        ['Heart Disease Prediction', 'BMI Calculator', 'EDA', 'Confusion Matrix','About Us'],
        index=0
    )

    # Conditionally render selected page
    if selected_page == 'Heart Disease Prediction':
        render_heart_disease_prediction()
    elif selected_page == 'BMI Calculator':
        render_bmi_calculator()
    elif selected_page == 'EDA':
        render_eda()
    elif selected_page == 'Confusion Matrix':
        render_Confusion_matrix()


    elif selected_page == 'About Us':
        render_about_us()

if __name__ == "__main__":
    main()  # Run the main function to start the Streamlit app
