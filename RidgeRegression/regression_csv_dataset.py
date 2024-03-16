import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pathlib
from mlxtend.plotting import heatmap
from mlxtend.plotting import scatterplotmatrix
import matplotlib.pyplot as plt
import numpy as np

# Regression dataset class constructor
class RegressionDatasetCSV:

    """Imports and preprocesses a regression csv dataset
       Args:
            path: input file path where the CSV data is stored
            target: name of the target feature from the overall CSV dataset
            test_size: sets the test size of the dataset split, default 0.2 or 20% test data
            random_state: random state number to repeat the randomness, default = 42
        """

    # RegressionDatasetCSV class constructor
    def __init__(self, path, target, test_size=0.2, random_state=42):

        # RegressionDatasetCSV class attributes
        self.csv_data = pd.read_csv(path)       # Reads in the CSV data and stores it in the csv_data class attribute
        self.target = target                    # Stores the target values from the CSV dataset
        self.test_size = test_size              # Stores the test size for splitting the dataset
        self.random_state = random_state        # Stores the random state to repeat randomness

    # Python method to normalise the data with ScikitLearn's StandardScaler
    def standard_scale(self):

        """Normalises the CSV data with Scikitlearn's Standard Scaler"""

        # Stores the CSV data in the X Python variable
        X = self.csv_data

        # Pops the targets y from the CSV DataFrame X
        y = X.pop(self.target)

        # Creates a new Pandas dataframe of the dataset to automatically create a Python list of the numeric features
        df_numerical = self.csv_data

        # Drops the non-numeric features from the overall dataset
        df_numerical = df_numerical.select_dtypes(include='number')

        # Stores the numerical feature names in a Python list
        list_numerical = list(df_numerical.columns)

        # Creates the StandardScaler object from ScikitLearn
        scaler = StandardScaler().fit(X[list_numerical])

        # Fits the normalisation for the model
        X[list_numerical] = scaler.transform(X[list_numerical])

        # Performs a train test split using ScikitLearn
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # Returns the X_train, X_test, y_train, y_test normalised dataset splits
        return X_train, X_test, y_train, y_test


    # Python method to split the training and testing datasets
    def train_test_split(self, X, y):

        """Splits the training and testing datasets with Scikitlearn's train_test_split"""

        # Performs a train test split using ScikitLearn
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # Returns the X_train, X_test, y_train, y_test dataset splits
        return X_train, X_test, y_train, y_test

    # Python method to create a scatter plot matrix of the CSV dataset
    def sctterplot_matrix(self, file_path_dir, file_name):

        """Creates a scatterplot matrix of the CSV dataset"""

        # Stores the CSV data in the df Python variable
        df = self.csv_data

        # Stores the feature names from the dataset in a Python list
        columns = list(df.columns)

        # Creates a scatterplot matrix using MlXtend
        scatterplotmatrix(df[columns].values, figsize=(10, 8), names=columns, alpha=0.5)

        # Reduces the space between the sublots in the subplot layout
        plt.tight_layout()

        # Checks if the file path directory has been created or not
        # If the file path directory is not created the class method creates the directory
        self.create_file_path(file_path_dir)

        # Outputs the line plot to an image
        plt.savefig(file_path_dir + file_name)

        # Displays the scatterplot matrix to the screen
        plt.show()

    # Python method to create a correlation heatmap plot
    def correlation(self, title, file_path_dir, file_name):

        """Creates a correlation heatmap plot
           title: Adds a title to the plot
           file_path_dir: enters the file path where the project plotting file is ot be created
           file_name: enters the file_name of the plot to be saved"""

        # Stores the CSV data in the df Python variable
        df = self.csv_data

        # Stores the feature names from the dataset in a Python list
        columns = list(df.columns)

        # Stores the correlation plot data in the cm local DataFrame variable
        cm = np.corrcoef(df[columns].values.T)

        # Cretaes a heatmap correlation plot using MlXtend
        hm = heatmap(cm,
                     row_names=columns,
                     column_names=columns)

        # Gives the line graph plot a title
        plt.title(title)

        # Checks if the file path directory has been created or not
        # If the file path directory is not created the class method creates the directory
        self.create_file_path(file_path_dir)

        # Outputs the line plot to an image
        plt.savefig(file_path_dir + file_name)

        # Displays the heatmap correlation plot to the screen using matplotlib.pyplot
        plt.show()


    # Creates a file directory to store the saved model if required
    def create_file_path(self, file_path_check):

        """Creates a file path to save any plots to if the file path does not exist"""

        # If the output directory does not exist, the OS modules makes the directory
        # Creates the file path variable
        file = pathlib.Path(file_path_check)

        # Conditional if statement to make a file path directory required or not
        if file.exists():
            # Passes the if statement as the file directory exists
            pass
        else:
            # Makes a file path directory with the path variable stirng text
            os.makedirs(file_path_check)