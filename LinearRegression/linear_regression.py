# Imports the NumPy library module
import numpy as np

# Linear Regression Python class object
class LinearRegression:
    """Linear Regression (LR) model in Python from scratch.
       The code comments throughout the LR model explain the code.
       Object Parameters:
       x = features of the dataset
       y = target values of the dataset
       Learning Rate (lr)
       Number of Iterations (n_iters)
       x, y, LR and n_iters can be set as hyperparameters to train the
       model on different datasets.
    """

    # Linear Regression Python object constructor
    def __init__(self, x, y, lr=0.01, n_iters=10000):
        self.features = x           # Stores the features from the dataset
        self.target = y             # Stores the target values y from the dataset
        self.lr = lr                # Learning rate
        self.n_iters = n_iters      # Number of iterations
        self.weights = None         # Initialises the weights to 0 and stores their values
        self.bias = None            # Initialises the bias to 0 and stores their values
        self.n_samples = len(x)     # Sets the number of samples to the length of the features x

    # Python method to fit the data to the Multiple Linear Regression model
    def fit(self):
        '''
        Fit method to train the linear regression model.
        '''

        # Initialize weights and bias to zero values
        self.weights = np.zeros(self.features.shape[1])
        self.bias = 0

        # Gradient Descent implementation
        for i in range(self.n_iters):
            # Line equation that stores the predictions of the LR model during training
            y_pred = np.dot(self.features, self.weights) + self.bias

            # Calculate derivatives of the weights and bias
            dw = (1 / self.n_samples) * (2 * np.dot(self.features.T, (y_pred - self.target)))
            db = (1 / self.n_samples) * (2 * np.sum(y_pred - self.target))

            # Updates the weights and the bias of the LR during training
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    # Python prediction method for the LR algorithm
    def predict(self, X):
        ''' Makes predictions using the line equation.
            X: features from the dataset (X_test)
        '''

        # Returns the prediction made from the LR model
        return np.dot(X, self.weights) + self.bias