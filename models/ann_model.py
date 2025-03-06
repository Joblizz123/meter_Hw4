from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from base_model import BaseANN  # Importing the BaseANN class

class ANNModel(BaseANN):
    def __init__(self, file_path, hidden_layer_sizes=(100,), learning_rate_init=0.01, max_iter=200):
        super().__init__(file_path)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.model = None

    def create_model(self):
        """Initializes the ANN model."""
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=42
        )
        print("ANN Model Created!")

    def train_model(self):
        """Trains the ANN model."""
        if self.model is None:
            raise ValueError("Model has not been created. Call create_model() first.")
        
        self.model.fit(self.X_train, self.y_train)
        print("Model Training Completed!")

    def evaluate_model(self):
        """Evaluates the model and prints the accuracy."""
        if self.model is None:
            raise ValueError("Model has not been trained. Call train_model() first.")

        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        return accuracy