# Linear Regression Python class object
class SimpleLinearRegression:
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
    def __init__(self, x, y, lr=0.01, epochs=1000):
        self.features = x           # Stores the features from the dataset
        self.target = y             # Stores the target values y from the dataset
        self.lr = lr                # Learning rate
        self.epochs = epochs        # Number of iterations
        self.weights = 0            # Initialises the weights to 0 and stores their values
        self.bias = 0               # Initialises the bias to 0 and stores their values
        self.n_samples = len(x)     # Sets the number of samples to the length of the features x

    # Python method to fit the data to the Simple Linear Regression model
    def fit(self):
        ''' Fit method to train the linear regression model'''

        # Gradient Descent implementation
        for i in range(self.epochs):

            # Line equation that stores the predictions of the LR model during training
            y_pred = self.weights * self.features + self.bias

            # Calculate derivatives of the wieghts and bias
            dw = (-2 / self.n_samples) * sum(self.features * (self.target - y_pred))
            db = (-1 / self.n_samples) * sum(self.target - y_pred)

            # Updates the weights and the bias of the LR during training
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    # Python prediction method for the LR algorithm
    def predict(self, X):
        ''' Makes predictions using the line equation.
            X = features from the dataset
        '''

        # Returns the prediction made from the LR model
        return self.weights * X + self.bias