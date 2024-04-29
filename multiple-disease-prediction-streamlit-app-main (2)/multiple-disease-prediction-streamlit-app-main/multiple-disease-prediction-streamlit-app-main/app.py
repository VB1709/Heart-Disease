import os
import pickle
import streamlit as st
import joblib
from streamlit_option_menu import option_menu
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt






# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

#diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_pred.sav', 'rb'))


#parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

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
    st.markdown(top_bar_style, unsafe_allow_html=True)

    # Display the heading in the top bar
    st.markdown('<div class="top-bar">Welcome</div>', unsafe_allow_html=True)

    # Rest of your Streamlit app content goes here

if __name__ == "__main__":
    main()


with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           [
                            'Heart Disease Prediction',
                            'Logout'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)


# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Gender = st.text_input('Gender')

    with col2:
        Age = st.text_input('Age')

    #with col3:
     #   Education = st.text_input('Education')

   # with col1:
      #  Behavioral = st.text_input('Behavioral')

    with col3:
        is_smoking = st.text_input('is_smoking')

    with col1:
        Cigs_Per_Day = st.text_input('Cigs Per Day')

   # with col1:
      #  Medical_history = st.text_input('Medical history')

    with col2:
        BP_Meds = st.text_input('BP Meds')

    with col3:
        Prevalent_Stroke = st.text_input('Prevalent Stroke')

    with col1:
        Prevalent_Hyp = st.text_input('Prevalent Hyp')

    with col2:
        Diabetes = st.text_input('Diabetes')

    #with col3:
        #Medical_current = st.text_input('Medical current')

    with col3:
        Tot_Chol = st.text_input('Tot Chol ')

    with col1:
        Dia_BP = st.text_input('Dia BP')

    with col2:
        Sys_BP = st.text_input('Sys BP')

    with col3:
        Bmi = st.text_input('BMI')

    with col1:
        Heart_Rate  = st.text_input('Heart Rate ')

    with col2:
        Glucose = st.text_input('Glucose ')

    # with col3:
    #     Predict_variable = st.text_input('Predict variable')

    # code for Prediction
    heart_diagnosis = ''

# creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        # Get user input
        user_input = [Gender, float(Age), is_smoking, float(Cigs_Per_Day), float(BP_Meds), float(Prevalent_Stroke), float(Prevalent_Hyp),
                    float(Diabetes), float(Tot_Chol),float(Sys_BP), float(Dia_BP), float(Bmi), float(Heart_Rate), float(Glucose) ]
        
        if user_input[0].upper() == 'M':
            user_input.remove(user_input[0])
            user_input.insert(0, 0.0)
        else:
            user_input.remove(user_input[0])
            user_input.insert(0, 1.0)
        
        if user_input[2].upper() == "YES":
            user_input.remove(user_input[2])
            user_input.insert(2, 1.0)
        else:
            user_input.remove(user_input[2])
            user_input.insert(2, 0.0)

        # Convert any non-numeric values to numeric      
        # user_input = [float(x) if isinstance(x, (int, float)) else 0.0 for x in user_input]
        
        print(user_input)

        # Check if all fields are filled
        if len(user_input) == 14:
            try:
                # Specify the full path to the model file
                model_path = 'C:\\Users\\adity\\Downloads\\multiple-disease-prediction-streamlit-app-main\\multiple-disease-prediction-streamlit-app-main\\colab_files_to_train_models\\trytry.sav'
                
                model_path1 = 'C:\\Users\\adity\\Downloads\\multiple-disease-prediction-streamlit-app-main\\multiple-disease-prediction-streamlit-app-main\\colab_files_to_train_models\\scalar.sav'

                user_input1 = joblib.load(model_path1)

                # Load the model from the specified path
                heart_pred_model = joblib.load(model_path)

                trial = np.array(user_input).reshape(1, -1)
                dataAfterScal = user_input1.transform(trial)

                print(trial, dataAfterScal)
                # Use the loaded model to make predictions
                heart_prediction = heart_pred_model.predict(dataAfterScal)

                if heart_prediction[0] == 1:
                    heart_diagnosis = 'The person is having heart disease'
                elif heart_prediction[0] == 0:
                    heart_diagnosis = 'The person does not have any heart disease'
                else:
                    heart_diagnosis = 'Invalid'

            except ValueError as e:
                heart_diagnosis = 'Invalid: ' + str(e)
        else:
            heart_diagnosis = 'Please fill in all the fields'

        # if all(user_input):
        # try:
        #         # Specify the full path to the model file
        #         model_path = 'C:\\Users\\adity\\Downloads\\multiple-disease-prediction-streamlit-app-main\\multiple-disease-prediction-streamlit-app-main\\colab_files_to_train_models\\trytry.sav'

        #         # Load the model from the specified path
        #         heart_pred_model = joblib.load(model_path)

        #         # Use the loaded model to make predictions
        #         heart_prediction = heart_pred_model.predict([user_input])

        #         if heart_prediction[0] == 1:
        #             heart_diagnosis = 'The person is having heart disease'
        #         elif heart_prediction[0] == 0:
        #             heart_diagnosis = 'The person does not have any heart disease'
        #         else:
        #             heart_diagnosis = 'Invalid'

        # except ValueError as e:
        #         heart_diagnosis = 'Invalid: ' + str(e)
        # else:
        #     heart_diagnosis = 'Please fill in all the fields'

    st.success(heart_diagnosis)

# Heart Disease Prediction Page
if selected == 'Logout':
       
       

        st.warning('Logged out successfully!')

# Parkinson's Prediction Page
#if selected == "Parkinsons Prediction":

    # page title
    #st.title("Parkinson's Disease Prediction using ML")

    #col1, col2, col3, col4, col5 = st.columns(5)

    #with col1:
     #   fo = st.text_input('MDVP:Fo(Hz)')

    #with col2:
     #   fhi = st.text_input('MDVP:Fhi(Hz)')

    #with col3:
       # flo = st.text_input('MDVP:Flo(Hz)')

    #with col4:
     #   Jitter_percent = st.text_input('MDVP:Jitter(%)')

    #with col5:
     #   Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    #with col1:
     #   RAP = st.text_input('MDVP:RAP')

    #with col2:
     #   PPQ = st.text_input('MDVP:PPQ')

    #with col3:
     #   DDP = st.text_input('Jitter:DDP')

    #with col4:
     #   Shimmer = st.text_input('MDVP:Shimmer')

    #with col5:
     #   Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    #with col1:
     #   APQ3 = st.text_input('Shimmer:APQ3')

    #with col2:
     #   APQ5 = st.text_input('Shimmer:APQ5')

    #with col3:
     #   APQ = st.text_input('MDVP:APQ')

    #with col4:
     #   DDA = st.text_input('Shimmer:DDA')

    #with col5:
     #   NHR = st.text_input('NHR')

    #with col1:
     #   HNR = st.text_input('HNR')

    #with col2:
     #   RPDE = st.text_input('RPDE')

    #with col3:
     #   DFA = st.text_input('DFA')

    #with col4:
     #   spread1 = st.text_input('spread1')

    #with col5:
     #   spread2 = st.text_input('spread2')

    #with col1:
     #   D2 = st.text_input('D2')

    #with col2:
     #   PPE = st.text_input('PPE')

    # code for Prediction
    #parkinsons_diagnosis = ''

    # creating a button for Prediction    
    #if st.button("Parkinson's Test Result"):

       # user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
        #              RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
         #             APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        #user_input = [float(x) for x in user_input]

        #parkinsons_prediction = parkinsons_model.predict([user_input])

        #if parkinsons_prediction[0] == 1:
         #   parkinsons_diagnosis = "The person has Parkinson's disease"
        #else:
          #  parkinsons_diagnosis = "The person does not have Parkinson's disease"

    #st.success(parkinsons_diagnosis)
