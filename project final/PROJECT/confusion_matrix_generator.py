import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import streamlit as st

class ConfusionMatrixGenerator:
    def __init__(self, true_labels, predicted_labels, algorithm_name):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.algorithm_name = algorithm_name

    def generate_confusion_matrix(self):
        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        return cm

    def plot_confusion_matrix(self):
        cm = self.generate_confusion_matrix()
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=.5, annot_kws={"size": 16})
        plt.title(f'Confusion Matrix - {self.algorithm_name}')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        return plt

def main():
        st.title('Algorithm Comparison with Confusion Matrices')

    # Sidebar selection for algorithm
        selected_algorithm = st.sidebar.selectbox('Select Algorithm', ['Algorithm A', 'Algorithm B', 'Algorithm C', 'Algorithm D'])

    # Placeholder for simulated dataset (replace with your data loading logic)
        true_labels = [0, 1, 1, 0, 2, 3, 2, 1, 3, 0]  # Example: True labels
        predicted_labels = [0, 1, 0, 0, 2, 3, 2, 1, 2, 0]  # Example: Predicted labels

    # Instantiate ConfusionMatrixGenerator for the selected algorithm
        cm_generator = ConfusionMatrixGenerator(true_labels, predicted_labels, selected_algorithm)

    # Display confusion matrix plot for the selected algorithm
        if  st.button('Generate Confusion Matrix'):
            st.write(f'Confusion Matrix for {selected_algorithm}:')
            st.pyplot(cm_generator.plot_confusion_matrix())

if __name__ == '__main__':
    main()