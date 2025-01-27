import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    #convertir la salida lineal del modelo en una probabilidad
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    #entrenar el modelo con el conjunto de datos
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.array(y).reshape(-1, 1) 
        
        # inicializa con los pesos y el sesgo
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
        # GRADIENTE DESCENDIENTE
        for i in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)
            
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


    #predecir la salida del modelo para un conjunto de datos
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        
        return (predictions >= 0.5).astype(int) #.astype(int) asegura que sea entero
        