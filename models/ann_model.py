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


     def plot_training_results(history):
        """
        Plots training vs validation loss and accuracy to check for overfitting.
        
        :param history: The history object returned from model.fit()
        """
        # Extract values
        loss = history.history['loss']
        val_loss = history.history.get('val_loss', None)  # Validation loss (if available)
        accuracy = history.history.get('accuracy', None)  # Training accuracy
        val_accuracy = history.history.get('val_accuracy', None)  # Validation accuracy (if available)

        epochs = range(1, len(loss) + 1)

        # Plot Loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, 'b-', label="Training Loss")
        if val_loss:
            plt.plot(epochs, val_loss, 'r-', label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()

        # Plot Accuracy (if available)
        if accuracy:
            plt.subplot(1, 2, 2)
            plt.plot(epochs, accuracy, 'b-', label="Training Accuracy")
            if val_accuracy:
                plt.plot(epochs, val_accuracy, 'r-', label="Validation Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title("Accuracy Curve")
            plt.legend()
        plt.show()


    def train_ann(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Trains the ANN model and plots results.
        
        :param model: Compiled ANN model
        :param X_train: Training data
        :param y_train: Training labels
        :param X_val: Validation data
        :param y_val: Validation labels
        :param epochs: Number of training epochs
        :param batch_size: Batch size for training
        :return: Trained model
        """
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                            epochs=epochs, batch_size=batch_size, verbose=1)

        # Plot training results
        plot_training_results(history)

        return model

    # def train(self):
    #     """Train the ANN model"""
    #     self.model.fit(self.X_train, self.y_train)

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

    
