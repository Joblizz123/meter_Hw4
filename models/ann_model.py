import sys
import os
os.chdir("../")

os.getcwd()
import sys
sys.path.append('C:/Users/joeso/Documents/Homework4/meter_Hw4')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from base_ann import BaseANN   # Import the BaseANN class from base_ann.py


class ANNModel(BaseANN):
    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.001, max_iter=200):
        super().__init__(hidden_layer_sizes, learning_rate, max_iter)
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter
        )


    def train(self):
        """Train the ANN model"""
        self.model.fit(self.X_train, self.y_train)

    def test(self):
        """Evaluate the model"""
        y_pred = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)

 

    def plot_confusion_matrix(y_true, y_pred, class_labels):
        """
        Plots the confusion matrix for model predictions.

        :param y_true: True labels
        :param y_pred: Predicted labels
        :param class_labels: List of class names
        """
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        
        plt.figure(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title("Confusion Matrix")
        plt.show()

    
