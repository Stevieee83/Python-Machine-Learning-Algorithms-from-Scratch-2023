import numpy as np
import scipy.stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pathlib
import joblib
import pickle

# Regression metrics Python class
class RegressionMetrics:

    """ RegrressionMetrics Python class to evaluate and save trained models with
        the Pickle and Joblib libraries available to Python.
        X: dataset features
        y: dataset targets
    """

    # RegressionMetrics class constructor
    def __init__(self, X, y):
        self.predictions = X    # Stores the features from the dataset
        self.targets = y        # Stores the target values y from the dataset
        
    # Python method to calculate correlation coefficient metrics
    def correlation(self):
    
        """ Calculates the Pearson Linear Correlation Coefficient (PLCC), Spearman Rank
            Correlation Coefficient (SRCC) and the Kendal Rank Correlation Coefficients
            (KRCC) with the Scipy module available to Python.
        """

        # Calculate Pearson's Linear Correlation Coefficient (PLCC) with Scipy
        PLCC = scipy.stats.pearsonr(self.predictions, self.targets)[0]    # Pearson's r

        # Calculate Spearman Rank Correlation Coefficient (SRCC) with Scipy
        SRCC = scipy.stats.spearmanr(self.predictions, self.targets)[0]   # Spearman's rho

        # Calculate Kendalls Rank Correlation Coefficient (KRCC) with Scipy
        KRCC = scipy.stats.kendalltau(self.predictions, self.targets)[0]  # Kendall's tau

        # Optional return statement for PLCC, SRCC and KRCC
        #return PLCC, SRCC, KRCC
    
        # Prints out the correlation performance metric results to the screen
        print("PLCC: ", PLCC)
        print("SRCC: ", SRCC)
        print("KRCC: ", KRCC)

    # Mean Squared Error regression metric from ScikitLearn
    def mse_metric(self):
        
        """Calculates the Mean Squared Error (MSE) evaluation metric with ScikitLearn"""
            
        # Calls the Mean Squared Error from ScikitLearn
        mse = mean_squared_error(self.predictions, self.targets)
            
        # Returns the Mean Squared Error
        return mse
        
    # Mean Absolute Error regression metric from ScikitLearn
    def mae_metric(self):
        
        """Calculates the Mean Absolute Error (MAE) evaluation metric with ScikitLearn"""
        
        # Calls the Mean Absolute Error from ScikitLearn
        mae = mean_absolute_error(self.predictions, self.targets)
            
        # Returns the Mean Absolute Error
        return mae
            
        
    # Root Mean Squared Error regression metric from ScikitLearn
    def rmse_metric(self):
        
        """Calculates the Root Mean Squared Error (RMSE) evaluation metric with ScikitLearn"""
            
        # Calls the Mean Squared Error from ScikitLearn
        rmse = mean_squared_error(self.predictions, self.targets)
            
        # Returns the Root Mean Squared Error (RMSE)
        return np.sqrt(rmse)

    # R2_Score regression metric from ScikitLearn
    def r2_score(self):

        """Calculates the R2 score regression evaluation metric using ScikitLearn"""

        # Stores the r2_score metric in the score Python variable
        score = r2_score(self.predictions, self.targets)

        # Returns the r2_score regression evaluation metric
        return score

    # Python method to display the predictions made by the LR model on the test dataset
    def regression_plot(self, X_test, y_pred, file_path_check, file_path):

        """Python method to output a regression line plot of the LR model's predictions.
           X_test: test dataset features
           y_pred: regression model predicted values
           file_path_check: file path directory to check if it has been created previously or not
           file_path: file path to save the regression plot to a designated folder.
        """

        # Sets the style of the line plot to whitegrid colour
        sns.set_style('whitegrid')

        # Stores the plot values in the axes variable and plots the line graph
        axes = sns.regplot(x=X_test, y=y_pred)

        # Sets the title, x-axis label and the y-axis labels of the plot
        plt.title('Linear Regression Predictions')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary (£)')

        # Class method to check if there is a file directory to output the plot to
        self.create_file_path(file_path_check)

        # Outputs the LR plot to a png image file
        plt.savefig(file_path)

        # Displays the LR predictions plot to the screen
        plt.show()

    # Dump saved model with Joblib to a designated file directory
    def save_model_joblib(self, model, file_path_check, file_path):
        
        """Dumps the trained model to a Joblib file.
           model: passes in the trained model Python variable to be saved.
           file_path_check: file path directory to check if it has been created previously or not.
           file_path: file path to save the regression plot to a designated folder.
        """
            
        # Creates a file path directory for the saved model to be stored in
        self.create_file_path(file_path_check)
        
        # Dumps the Joblib file with the saved model to the file_path directory
        joblib.dump(model, file_path)

    # Load saved model with Joblib
    def load_model_joblib(self, file_path):

        """ Load a saved model with the Joblib module to the Python environment.
            A Python variable is required i.e. model, to save the loaded model
            to the environment into.
            file_path: file path directory to load a trained model file from.
        """

        # Returns the model file after loading it to the Python environment
        return joblib.load(file_path)
            
    # Dump the saved model with Pickle to a designated file directory
    def save_model_pickle(self, model, file_path_check, file_path):
        
        """Dumps the trained model to a Pickle file.
           model: passes in the trained model Python variable to be saved.
           file_path_check: file path directory to check if it has been created previously or not.
           file_path: file path to save the regression plot to a designated folder.
        """
            
        # Creates a file path directory for the saved model to be stored in
        self.create_file_path(file_path_check)
        
        # Dumps the Pickle file with the saved model to the file_path directory
        pickle.dump(model, open(file_path, "wb"))

    # Load the saved model with Pickle
    def load_model_pickle(self, file_path):

        """ Load a saved model with the Pickle module to the Python environment.
            A Python variable is required i.e. model, to save the loaded model
            to the environment into.
            file_path: file path directory to load a trained model file from.
        """
            
        # Returns the model file after loading it to the Python environment
        return pickle.load(open(file_path, "rb"))
            
    # Creates a file directory to store the saved model if required
    def create_file_path(self, file_path_check):

        """Creates a file path to cave any plots to if the file path does not exist.
           file_path_check: file path directory to check if it has been created previously or not.
        """

        # If the output directory does not exist, the OS modules makes the directory
        # Creates the file path variable
        file = pathlib.Path(file_path_check)

        # Conditional if statement to control the CSV data dump is required or not
        if file.exists():
            # Passes the if statement as the file directory for the saved model exists
            pass
        else:
            # Outputs the folder contents to CSV files for each column
            os.makedirs(file_path_check)