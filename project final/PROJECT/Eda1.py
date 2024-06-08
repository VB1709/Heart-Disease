import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_eda():
    # Load your dataset for EDA (replace this with your dataset loading logic)
    df = pd.read_csv('data_risk.csv')

    # Display summary statistics for the 'age' column
    st.subheader("Summary Statistics for 'age' column:")
    st.write(df['age'].describe())

    # Create the heatmap plot for missing values
    plt.figure(figsize=(6, 3))  # Set the figure size
    sns.heatmap((df.isna().sum()).to_frame(name='').T, cmap='summer', annot=True, fmt='0.0f')
    plt.title('Count of Missing Values (Test Data)', fontsize=18)

    # Display the heatmap plot
    st.pyplot()

    # Age Distribution of All Patients
    fig, ax0 = plt.subplots(figsize=(12, 5))
    sns.histplot(data=df, x='age', kde=True, ax=ax0)
    ax0.set_title('Distribution of Patients by Age')
    ax0.set_xlabel('Age')

    # Display the age distribution histogram
    st.pyplot(fig)

    # Create the pie chart for the proportion of CHD patients
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.pie(df['TenYearCHD'].value_counts(), labels=['Will not get heart disease', 'Might get heart disease'], startangle=90, shadow=True, autopct='%.2f%%')
    ax1.set_title('Proportion of CHD Patients')

    # Display the pie chart
    st.pyplot(fig)

    st.image('1.jpg')
    st.image('2.jpg')
    st.image('3.jpg')
    st.image('4.jpg')
    st.image('5.jpg')
    st.image('6.jpg')
    st.image('7.jpg')
    st.image('8.jpg')
    st.image('9.jpg')
    st.image('10.jpg')
    st.image('11.jpg')
    st.image('12.jpg')
    #st.image('C:\\Users\\adity\\Downloads\\PROJECT\\13.jpg')
    #st.image('C:\\Users\\adity\\Downloads\\PROJECT\\14.jpg')
    st.image('15.jpg')
    st.image('16.jpg')
    st.image('17.jpg')
    st.image('18.jpg')






    





if __name__ == '__main__':
    show_eda()
