import numpy as np

# Regression class object to prcess continious data
class Regression:
    
    # Regression class constructor
    def __init__(self, regularization, lr, n_iters):
        self.n_samples = None # Dataset samples
        self.features = None # Features from the dataset
        self.weights = None # weight parameters
        self.bias = None # bias parameter
        self.regularization = regularization # rgularisation
        self.lr = lr # learning rate
        self.n_iters = n_iters # iterations
        
    # Regression class cost function to calculate the loss of the model to determine the performance
    def __calculate_cost(self, y, y_pred):
        
        # Calculates the cost of the regression model during training and returns the cost value
        return (1 / (2*self.n_samples)) * np.sum(np.square(y_pred-y)) + self.regularization(self.weights)
        
    # Python fit method to fit the regression model during training
    def fit(self, X, y):

        # Retrieves the y_train target values dataset length in preperation for reshaping the dataset to 1 dimension
        dim_1 = len(y)

        # Reshapes the y_train target values for the Lasso Regression algorithm and converts it to a NumPy array
        y = y.values.reshape(dim_1, 1)
        
        # Initialises the wieght parameters before training
        X = np.insert(X, 0, 1, axis=1)
        
        # stores the number of samples and features from the dataset
        self.n_samples, self.features = X.shape
        
        # Updates the weight parameters after initilization
        self.weights = np.zeros((self.features,1))
        
        # Training loop for the regression model
        for e in range(1, self.n_iters+1):
            
            # Stores the training predictions made by the regression model
            y_pred = np.dot(X, self.weights)
            
            # Calculats the cost of the regression model during training
            cost = self.__calculate_cost(y, y_pred)
            
            # Partial differentiation for the weigh parameters
            dw = (1/self.n_samples) * np.dot(X.T, (y_pred - y)) + self.regularization.derivation(self.weights)
            
            # Updates the weight parameter after each training iteration
            self.weights = self.weights - self.lr * dw
            
            # Prints the cost of hte regression model after every 100 iterations
            if e % 100 == 0:
                
                # Print statement to print the cost to the screen with an F string
                print(f"The Cost in iteration {e}:\t {cost}")

        # Prints out training complete to the screen
        print("Training Complete")

    # Python predict method to make predictions on the trained regression model at test time
    def predict(self, X_test):
        
        # Reshapes the test dataset X_test for the regression model
        X_test = np.insert(X_test, 0 , 1, axis= 1)
        
        # Stores the predictions made from the regression model at test time
        y_pred = np.dot(X_test, self.weights)
        
        # Returns the predictions made from the regression model at test time
        return y_pred
    
# LassoPenalty class object
class LassoPenalty:
    
    # LassoPenalty Class constructor
    def __init__(self, l):
        self.l = l # lambda regularization parameter
        
    # Calculats the regularization lambda parameter's absolute value
    def __call__(self,w):
        
        # Returns the regularization lambda parameter's absolute value
        return self.l * np.sum(np.abs(w))
        
    # Calculates the derivation of the Lasso regression model
    def derivation(self, w):
        
        # Returns the derivation of the Lasso regression model
        return self.l * np.sign(w)

# Lasso regression Python object that inherits the regression class object
class Lasso(Regression):
    
    # Lasso regression class constructor
    def __init__(self, l, lr, n_iters):
        
        # Stores the regularization parameter lambda calculated by the LassoPenalty class object
        self.regularization = LassoPenalty(l)
        
        # Calls the Regression class object to input the lambda regularisation poramter, 
        # learning rate and number of iterations (n_iters)
        super().__init__(self.regularization, lr, n_iters)
    


    
