#Library imports
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ClassificationDatasetCSV dataset class constructor
class ClassificationDatasetCSV:

    """Imports and preprocesses a classification csv dataset
       path: input file path where the CSV data is stored
       target: name of the target feature from the overall CSV dataset
       test_size: sets the test size of the dataset split, default 0.2 or 20% test data
       random_state: random state number to repeat the randomness, default = 42"""
       
    # ClassificationDatasetCSV class constructor
    def __init__(self, path, label, labels=None, test_size=0.2, random_state=42):

        # ClassificationDatasetCSV class attributes
        self.csv_data = pd.read_csv(path)       # Reads in the CSV data and stores it in the csv_data class attribute
        self.label = label                      # Stores the label column name of the CSV dataset
        self.labels = labels                    # Stores the label values from the CSV dataset
        self.test_size = test_size              # Stores the test size for splitting the dataset
        self.random_state = random_state        # Stores the random state to repeat randomness

    # Plots the class balance
    def class_ballance(self, figure):

        """"Plots the class balance for binary classification problems.
            Python method input parameters:
            figure: The Matplotlib figure number
        """

        # Stores the class labels for plotting
        y1 = self.csv_data[self.label]

        # Checking for imbalance dataset (with half of the dataset)
        plt.figure(figure, figsize=(2.5, 5))

        # Adds a title to the classification plot
        plt.title("Classification Result")

        # Stores the Seaborn count plot in the p1 variable
        p1 = sns.countplot(x=y1, palette = 'plasma')

        # For loop to iterate through the targets DataFrame and plot the class balance
        for p in p1.patches:

            # Gets the height of the plot from the loop counter
            height = p.get_height()

            # Creats the Fstring text for the Seaborn countplo
            p1.text(p.get_x() + p.get_width() / 2.,
                    height,
                    f'{height / self.csv_data.shape[0] * 100:.2f}%',
                    ha='center', fontsize=12)

        # Stores the class labels in the labels class attribute
        self.label_graph = y1

    # Defines a balancing function using the SMOTE library from SkLearn
    def smote_balance(self):
    
        """Carry out SMOTE balancing over the data"""

        # Stores the complete CSV dataset
        X1 = self.csv_data

        # Stores the targets in the y1 local variable
        y1 = self.csv_data.pop(self.label)

        # Constructs the SMOTE balanced dataset from Imblearn
        sm = SMOTE(random_state=self.random_state)

        # Stores the SMOTE balanced data in the X_sm and y_sm variables
        X_sm, y_sm = sm.fit_resample(X1, y1)

        # Prints out the shape of the dataset features X before and after SMOTE balancing
        print(f'''Shape of X before SMOTE: {X1.shape}
            Shape of X after SMOTE: {X_sm.shape}''')

        # Prints out the shape of the dataset labels y before and after SMOTE balancing
        print(f'''Shape of y before SMOTE: {y1.shape}
            Shape of y after SMOTE: {y_sm.shape}''')

        # Stores the SMOTE balanced data in the class attributes
        self.csv_data = X_sm
        self.labels = y_sm

        # Prints out the SMOTE balanced data, targets and the features
        print("CSV_data Class Attribute Reset", X_sm)
        print("Targets Class Attribute Reset", y_sm)

    # Plots the class balance
    def after_class_ballance(self, figure):

        """"Plots the class balance for binary classification problems
            Python method input parameters:
            figure: The Matplotlib figure number
        """

        # Stores the targets in the y1 local variable
        y1 = self.labels

        # Checking for imbalance dataset (with half of the dataset)
        plt.figure(figure, figsize=(2.5, 5))

        # Adds a title to the classification plot
        plt.title("Classification Result")

        # Stores the Seaborn count plot in the p1 variable
        p1 = sns.countplot(x=y1, palette = 'plasma')

        # For loop to iterate through the targets DataFrame and plot the class balance
        for p in p1.patches:

            # Gets the height of the plot from the loop counter
            height = p.get_height()

            # Creats the F string text for the Seaborn countplot
            p1.text(p.get_x() + p.get_width() / 2.,
                    height,
                    f'{height / self.csv_data.shape[0] * 100:.2f}%',
                    ha='center', fontsize=12)

    # Python method to normalise the data with ScikitLearn's StandardScaler
    def standard_scale(self):
        
        """Normalises the CSV data with Scikitlearn's Standard Scaler"""

        # Stores the CSV data in the df Python variable
        X = self.csv_data

        # Pops the targets y from the CSV DataFrame df
        y = self.labels

        # Creates a new Pandas DataFrame of the dataset to create a Python list of the numeric features automatically
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