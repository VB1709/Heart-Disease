import os
import pickle
import streamlit as st
import joblib
from streamlit_option_menu import option_menu
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from Eda1 import show_eda
from bmi_calculator import calculate_bmi, bmi_category
from confusion_matrix_generator import ConfusionMatrixGenerator
# import bmi_calculator
# from login_module import login  # Import the login function from your login module






# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")





try:
    scaleModel = pickle.load(open("scale.sav",'rb'))
    model_path = pickle.load(open("modelHeart.sav",'rb'))

except Exception as e:
    print("Error:", e)










    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models


#heart_disease_model = pickle.load(open('C:\\Users\\adity\\Downloads\\multiple-disease-prediction-streamlit-app-main\\multiple-disease-prediction-streamlit-app-main\\colab_files_to_train_models\\model.sav', 'rb'))


# sidebar for navigation

def main():
    # Define your custom CSS style for the top bar
    top_bar_style = """
    <style>
    .top-bar {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """

    # Display the custom CSS style in the Streamlit app
   # st.markdown(top_bar_style, unsafe_allow_html=True)

    # Display the heading in the top bar
   # st.markdown('<div class="top-bar">Welcome</div>', unsafe_allow_html=True)

    # Rest of your Streamlit app content goes here
    
    # Apply the custom CSS using markdown
   











st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        width: 300px;  /* Adjust sidebar width */
        padding: 1rem; /* Add padding for spacing */
    }
    .sidebar .sidebar-content .stSelectbox {
        width: 100%;   /* Set the width of the option menu */
        padding: 0.5rem; /* Adjust padding inside the option menu */
        font-size: 18px; /* Set font size for menu items */
        border-radius: 8px; /* Add border radius for rounded corners */
        border: 2px solid #ccc; /* Set border width and color */
        box-sizing: border-box; /* Ensure border width is included in element size */
    }
    </style>
    """,
    unsafe_allow_html=True
)












# Sidebar with option menu
with st.sidebar:
    selected = option_menu(
        'Disease Prediction System',
        ['Heart Disease Prediction', 'EDA','conf','About Us', 'Logout'],
        menu_icon='hospital-fill',
        icons=['heart', 'activity','star', 'star', 'person'],
        default_index=0
    )


# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
         Gender = st.selectbox('Gender', ['Male', 'Female'])

    with col2:
         Age = st.number_input('Age', min_value=1, max_value=120)

    with col3:
        is_smoking = st.selectbox('Smoking Status', ['Yes', 'No'])
    
    with col1:
        Cigs_Per_Day = st.number_input('Cigarettes Per Day', min_value=0, max_value=100)

    with col2:
         BP_Meds = st.number_input('Blood Pressure Medication', min_value=0.0, max_value=1.0)

    with col3:
       Prevalent_Stroke = st.number_input('Prevalent Stroke', min_value=0, max_value=1)

    with col1:
        Prevalent_Hyp = st.number_input('Prevalent Hypertension', min_value=0, max_value=1)

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

    # code for Prediction
    heart_diagnosis = ''

# creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        # Get user input
       # Collecting User Input with Streamlit



# Create user_input list with collected values
        user_input = [
                Age,
                Gender,
                is_smoking,
                Cigs_Per_Day,
                BP_Meds,
                Prevalent_Stroke,
                Prevalent_Hyp,
                Diabetes,
                Tot_Chol,
                Sys_BP,
                Dia_BP,
                Bmi,
                Heart_Rate,
                Glucose
]

# Note: You can use these user_input values further for processing or model prediction


        
        if user_input[1].upper() == 'MALE':
            user_input.remove(user_input[1])
            user_input.insert(1, 0.0)
        else:
            user_input.remove(user_input[1])
            user_input.insert(1, 1.0)
        
        if user_input[2].upper() == "YES":
            user_input.remove(user_input[2])
            user_input.insert(2, 1.0)
        else:
            user_input.remove(user_input[2])
            user_input.insert(2, 0.0)

        # Convert any non-numeric values to numeric      
        user_input = [float(x) if isinstance(x, (int, float)) else 0.0 for x in user_input]
        
        print("user=", user_input)

        # Check if all fields are filled
        if len(user_input) == 14:
            try:
                # Specify the full path to the model file
                # scaleModel = pickle.load(open("C:/Users/adity/Downloads/PROJECT/scale.sav",'rb'))
                # model_path = pickle.load(open("C:/Users/adity/Downloads/PROJECT/model.sav",'rb'))

                # Load the model from the specified path
                # heart_pred_model = joblib.load(model_path)

                inputReshaped = np.asarray(user_input).reshape(1, -1)
                dataAfterScal = scaleModel.transform(inputReshaped)
                
                
                print(inputReshaped, dataAfterScal)
                print(model_path.predict(dataAfterScal))
                # Use the loaded model to make predictions
                heart_prediction = model_path.predict(dataAfterScal)
                print("Value= ", heart_prediction)
                
                

                
                if heart_prediction[0] == 1:
                    heart_diagnosis = 'The person is expected to have heart disease'
                elif heart_prediction[0] == 0:
                    heart_diagnosis = 'The person does not have any heart disease'
                else:
                    heart_diagnosis = 'Invalid'

            except ValueError as e:
                heart_diagnosis = 'Invalid: ' + str(e)
        else:
            heart_diagnosis = 'Please fill in all the fields'


    st.success(heart_diagnosis)


elif selected == 'EDA':
    # Display EDA content
    show_eda()

    



elif selected == 'Conf':
    main()















elif selected == 'About Us':
    st.title('About Us')
    st.markdown("""
        We are dedicated to providing innovative health solutions through our Streamlit app.
        Our goal is to empower users with valuable insights and predictions related to health and wellness.
        For any inquiries or feedback, please contact us at example@example.com.
    """)




# Heart Disease Prediction Page
elif selected == 'Logout':
       
       

           st.warning('Logged out successfully!')



def calculate_and_display_bmi(weight, height):
    bmi_value = calculate_bmi(weight, height)
    category = bmi_category(bmi_value)

    # Custom styling for BMI result text
    bmi_result_style = (
        "background-color: #222; color: #fff; padding: 12px; "
        "border-radius: 8px; font-size: 18px; text-align: center;"
    )

    st.markdown("---")
    st.header("Your BMI Result")
    result_text = f"**BMI:** {bmi_value:.2f}\n**Category:** {category}"
    st.markdown(f"<div style='{bmi_result_style}'>{result_text}</div>", unsafe_allow_html=True)


    st.title("BMI Calculator")
    st.markdown("Calculate your Body Mass Index (BMI)")

    weight = st.number_input("Enter weight (kg)", min_value=0.0)
    height = st.number_input("Enter height (m)", min_value=0.0)

    if st.button("Calculate BMI"):
        calculate_and_display_bmi(weight, height)

if __name__ == "__main__":
    # Sidebar navigation menu
    selected_page = st.sidebar.selectbox('Navigation', ['BMI Calculator', 'Confusion Matrix'])

    # Display selected page content based on menu selection
    #if selected_page == 'BMI Calculator':
    main()  # Call main function to display BMI Calculator

