# NumPy library import
import numpy as np

# Logistic Regression Python class object
class LogisticRegression():

    """LogisticRegression (LR) model in Python from scratch.
       The code comments throughout the LR model explain the code.
       Constructor Input Parameters:
       Learning Rate (lr), default setting 0.001
       Number of Iterations (n_iters), default setting 1000
       lr and n_iters can be set as hyperparameters to train the 
       Logistic Regression model.
    """

    # Logistic Regression class constructor
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr            # Learning rate
        self.n_iters = n_iters  # Number of iterations (n_iters)
        self.weights = None     # Weight parameters
        self.bias = None        # Bias parameter

    # Python fit method to fit the data to the model during training
    def fit(self, X, y):

        """Python fit method to fit the training data to the LR model.
           The training data is passed through the LR model in the forward
           and backward pass directions because gradient descent adjusts the
           weight parameters during training to train the model.
           Python Method Input Parameters:
           x: features from the dataset
           y: prediction targets from the dataset
       """

        # Stores the shape of the dataset features in the n_samples and n_features Python variables
        n_samples, n_features = X.shape

        # Initialises the weight parameters and the bias during training
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training loop to control the number of training iterations
        for _ in range(self.n_iters):

            # Stores the training predictions in the linear_pred Python variable
            linear_pred = np.dot(X, self.weights) + self.bias

            # Stores the predictions after being passes to the Sigmoid activation function
            # in the predictions Python variable
            predictions = self.sigmoid(linear_pred)

            # Calcualtes the partial derivatives for the weights (dw) and the bias (db)
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # Updates the weights and bias parameters during training
            self.weights = self.weights - self.lr * dw
            self.bias - self.lr * db

    # Predict Python method to make predictions in inference mode at test time
    def predict(self, X):

        """Python predict method to run the test data through the model in the 
           forward pass direction only during inference mode.
           Python Method Input Parameter:
           X: features from the dataset
        """

        # Stores the inference mode predictions in the linear_pred Python variable
        linear_pred = np.dot(X, self.weights) + self.bias

        # Stores the predictions after being passes to the Sigmoid activation function
        # in the y_pred Python variable
        y_pred = self.sigmoid(linear_pred)

        # List comprehension to add all the class predictions to a Python list
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred

    # Sigmoid activation function Pyhton method with NumPy
    def sigmoid(self, x):

        """Sigmoid activation function to classify examples from the dataset.
            If the probalistic value is from 0.0 to 0.49 the example bleongs to class 0.
            If the probalistic value is from 0.5 to 1 the example bleongs to class 1.
            Python Method Input Parameters:
            x: features from the dataset
        """

        # Computes and returns the result of the Sigmoid activation function
        return 1 / (1 + np.exp(-x))