import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(data):
    # Perform exploratory data analysis (replace this with your EDA logic)
    # Example: Generating and saving visualizations
    sns.pairplot(data)
    plt.savefig('pairplot.png')  # Save the pairplot as an image file

def display_images_from_eda():
    # Check if images from EDA exist
    if not os.path.exists('pairplot.png'):
        st.warning("No EDA images found. Please perform EDA first.")
        return
    
    # Display EDA images
    st.header("Exploratory Data Analysis (EDA) Images")
    st.markdown("Visualizations generated during EDA:")

    # Display pairplot image
    st.subheader("Pairplot")
    st.image('pairplot.png', caption='Pairplot')

def main():
    st.title("Exploratory Data Analysis (EDA) with Streamlit")

    # Sidebar navigation menu
    selected_option = st.sidebar.selectbox("Select an option", ["Perform EDA", "View EDA Images"])

    if selected_option == "Perform EDA":
        st.subheader("Perform Exploratory Data Analysis (EDA)")
        # Load your data (replace this with your data loading logic)
        data = pd.read_csv("try_data.csv")

        # Perform EDA
        perform_eda(data)
        st.success("Exploratory Data Analysis completed.")

    elif selected_option == "View EDA Images":
        display_images_from_eda()

if __name__ == "__main__":
    main()
