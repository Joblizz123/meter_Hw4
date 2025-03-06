import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BaseANN:
    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.001, max_iter=200):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
    
    def load_data(self, file_path):
        """Load dataset from .txt file"""
        data = pd.read_csv('../data/Meter_A.txt', sep='\t', header=None).dropna()
        self.X = data.iloc[:, :-1].values  # Features
        self.y = data.iloc[:, -1].values   # Target
    
    def preprocess_data(self):
        """Split data and scale features"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X, self.y, test_size=0.3, random_state=42)
        
        #Scaling the data using StandardScaler
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
