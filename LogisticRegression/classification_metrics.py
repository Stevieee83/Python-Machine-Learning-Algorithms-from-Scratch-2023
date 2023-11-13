# Library imports
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn import metrics
import os
import pathlib
import joblib
import pickle

# Classification metrics class object
class ClassificationMetrics():

    """ ClassificationMetrics Python class to evaluate and save trained models with
        the Pickle and Joblib libraries available to Python.
        X: dataset features
        y: dataset targets
    """

    # ClassificationMetrics class constructor
    def __init__(self, X, y):
        self.predictions = X    # Stores the dataset features X
        self.targets = y        # Stores the dataset labels y
        
    # Cross Entropy classification metric from ScikitLearn
    def cross_entropy_loss(self):
        
        """Calculates the Cross-Entropy (CE) loss evaluation metric with ScikitLearn.
           No input arguments required."""
            
        # Calls the Cross Entropy Loss from ScikitLearn
        ce = log_loss(self.predictions, self.targets)
            
        # Returns the Cross-Entropy Loss
        return ce
        
    # Classification accuracy method from ScikitLearn
    def accuracy(self):

        """Calculates the Accuracy evaluation metric with ScikitLearn.
           No input arguments required."""
        
        # Calls the accuracy method from ScikitLearn
        acc = accuracy_score(self.predictions, self.targets)
        
        # Returns the Accuracy
        return acc * 100
        
       
    # Classification precision method from ScikitLearn
    def precision(self):

        """Calculates the Precision evaluation metric with ScikitLearn.
           No input arguments required."""
        
        # Calls the precision method from ScikitLearn
        pre = precision_score(self.predictions, self.targets)
        
        # Returns the Precision
        return pre * 100
        
    # Classification recall method from ScikitLearn
    def recall(self):

        """Calculates the Recall evaluation metric with ScikitLearn.
           No input arguments required."""
        
        # Calls the accuracy method from ScikitLearn
        rec = recall_score(self.predictions, self.targets)
        
        # Returns the Precision
        return rec * 100
        
    # Classification f1-Score method from ScikitLearn
    def f1_score(self):

        """Calculates the F1-Score evaluation metric with ScikitLearn.
           No input arguments required."""

        # Calls the accuracy method from ScikitLearn
        f_one = f1_score(self.predictions, self.targets)
        
        # Returns the Precision
        return f_one * 100

    # Classification Report from ScikitLearn
    def classification_report(self, y_test, y_pred, names):

        """Displays the Classification Report from ScikitLearn.
           Input Arguments: y_test - test dataset targets y
                            y_pred - Model predictions
                            names - class labels
        """

        # Displays the classification report for the deep neural network to the screen
        cr = classification_report(y_test, y_pred, target_names=names)
        print(cr)
        
    # Displays the training loss plot to the screen
    def train_loss_plot(self, X_axis, test_CE, figure, title, parameter, loss, file_path_dir, file_name):

        """Displays the Training Loss plot from Matplotlib.
           Input Arguments: X_axis - metric plotted samples
                            test_CE - Cross Entropy Loss per metric plotted sample
                            figure - sets the figure number of hte Confusion Matrix plot
                            title - plot title
                            parameter - model parameter for x-axis
                            loss - loss type plotted, for the y-axis name
                            file_path_dir - file directory to save plot to
                            file_name - file name to save plot to"""

        # Sets the confusion matrix plot to a figure number
        plt.figure(figure)
        
        # Creates the matplotlib.pyplot line graph plot for figure 1
        plt.plot(X_axis, test_CE)

        # Gives the line graph plot a title
        plt.title(title)

        # Gives the x axis of the line graph a title
        plt.xlabel('x - ' + parameter)

        # Gives the y axis of the line graph a title
        plt.ylabel('y - ' + loss)

        # Checks if the file path directory has been created or not
        # If the file path directory is not created, the class method creates the directory
        self.create_file_path(file_path_dir)

        # Outputs the line plot to an image
        plt.savefig(file_path_dir + file_name)

        # Displays the line graph plot figure
        plt.show()
        
        
    # Displays the training accuracy plot to the screen
    def train_accuracy_plot(self, X_axis, test_CE, figure, title, parameter, file_path_dir, file_name):

        """Displays the Training Accuracy plot from Matplotlib.
           Input Arguments: X_axis - metric plotted samples
                            test_CE - Cross Entropy Loss per metric plotted sample
                            figure - sets the figure number of the Confusion Matrix plot
                            title - plot title
                            parameter - model parameter for x-axis
                            file_path_dir - file directory to save plot to
                            file_name - file name to save plot to"""
        
        # Creates the matplotlib.pyplot line graph plot for the figure
        plt.plot(X_axis, test_CE)

        # Sets the training accuracy plot to figure
        plt.figure(figure)

        # Gives the line graph plot a title
        plt.title(title)

        # Gives the x axis of the line graph a title
        plt.xlabel('x - ' + parameter)

        # Gives the y axis of the line graph a title
        plt.ylabel('y - Accuracy')

        # Checks if the file path directory has been created or not
        # If the file path directory is not created, the class method creates the directory
        self.create_file_path(file_path_dir)

        # Outputs the line plot to an image
        plt.savefig(file_path_dir + file_name)

        # Displays the line graph plot figure
        plt.show()
        
    # Confusion matrix method from ScikitLearn and mlxtend
    def confusion_matrix(self, y_test, y_pred, class_names, figure, title, file_path_dir, file_path):

        """Displays the Confusion Matrix plot from ScikitLearn and MLXtend.
           Input Arguments: y_test - test dataset targets y
                            y_pred - Model predictions
                            class_names - class labels
                            figure - figure number for plot
                            title - plot title
                            parameter - loss parameter for x-axis
                            file_path_dir - file directory to save plot to
                            file_name - file name to save plot to"""
        
        # Setup confusion matrix instance and compare predictions to targets
        confmat = confusion_matrix(y_test, y_pred)
        
        # Plot the confusion matrix
        fig, ax = plot_confusion_matrix(
                 conf_mat=confmat,          # inputs the confusion matrix plotting values to the MlXtend method
                 class_names=class_names,   # turn the row and column labels into class names
                 figsize=(10, 7),           # sets the confusion matrix figure size to 10 x 7
                 show_normed=True,          # displays the proportion of training examples per class on the confusion matrix plot
                 colorbar=True              # adds the colour bar to the confusion matrix plot
        );

        # Sets the confusion matrix plot to figure 1
        plt.figure(figure)
        
        # Confusion matrix title
        plt.title(title)

        # Checks if the file path directory has been created or not
        # If the file path directory is not created, the class method creates the directory
        self.create_file_path(file_path_dir)
        
        # Saves the Confusion Matrix to a png file
        fig.savefig(file_path_dir + file_path)
        
        # Displays the Confusion Matrix heat map plot
        plt.show()

    # Confusion matrix method from ScikitLearn and mlxtend
    def confusion_matrix_scikitlearn(self, y_test, y_pred, file_path_dir, file_path):

        """Displays the Confusion Matrix plot from ScikitLearn and Seaborn.
           Input Arguments: y_test - test dataset targets y
                            y_pred - Model predictions
                            file_path_dir - file directory to save plot to
                            file_name - file name to save plot to"""

        # Confusion matrix plot from ScikitLearn
        cm = confusion_matrix(y_test, y_pred)
 
        # Plots the confusion matrix heatmap
        sns.heatmap(cm, 
            annot=True,
            fmt='g', 
            xticklabels=['True','False'],
            yticklabels=['True','False'])

        # Labels the y axis
        plt.ylabel('Prediction',fontsize=13)

        # Labels the X axis
        plt.xlabel('Actual',fontsize=13)

        # Creates a confusing matrix plot
        plt.title('Confusion Matrix',fontsize=17)

        # Checks if the file path directory has been created or not
        # If the file path directory is not created, the class method creates the directory
        self.create_file_path(file_path_dir)

        # Outputs the line plot to an image
        plt.savefig(file_path_dir + file_path)

        # Displays the confusion matrix plot to the screen
        plt.show()


    # AUROC curve plot method from ScikitLearn
    def roc_curve(self, y_test, y_preds, figure, title, file_path_dir, file_path):

        """Displays the Confusion Matrix plot from ScikitLearn and MLXtend.
           Input Arguments: y_test - test dataset targets y
                            y_pred - Model predictions
                            class_names - class labels
                            figure - figure number for plot
                            title - plot title
                            parameter - loss parameter for x-axis
                            file_path_dir - file directory to save plot to
                            file_name - file name to save plot to"""

        # Stores the false predictions in the for Python variable
        # Stores the true predictions in the tpr Python variable
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_preds)

        # Stores the AUROC curve false and positive prediction values
        # in the roc_auc Python variable
        roc_auc = metrics.auc(fpr, tpr)

        # Sets the AUROC curve plot as figure 2
        plt.figure(figure)

        # Adds the AUROC plot title
        plt.title(title)

        # Plots the AUROC curves to a Matplotlib plot
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')

        # Sets the limit for the X and Y axes
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        # Lavels the X and Y axes
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        # Checks if the file path directory has been created or not
        # If the file path directory is not created, the class method creates the directory
        self.create_file_path(file_path_dir)

        # saves the Confusion Matrix to a png file
        plt.savefig(file_path_dir + file_path)

        # Displays the AROC curve plot to the screen
        plt.show()
        
    # Dump saved model with Joblib to a designated file directory
    def save_model_joblib(self, model, file_path):
        
        """Dumps the trained model to a Joblib file
           Input Arguments: model - the trained machine learning classification model
                            file_path - the file path to save the trained machine learning classification model to"""
            
        # Creates a file path directory for the saved model to be stored in
        self.create_file_path(file_path)
        
        # Dumps the Joblib file with the saved model to the file_path directory
        joblib.dump(model, file_path)
            
        
    # Load saved model with Joblib
    def load_model_joblib(self, file_path):

        """Load a saved model with the Joblib module to the Python environment.
           A Python variable is required i.e. model to save the loaded model
           to the environment into.
           Input Arguments: file_path - the file path to save the trained machine learning classification model to"""
            
        # Returns the model file after loading it to the Python environment
        return joblib.load(file_path)
            
    # Dump saved model with Pickle to a designated file directory
    def save_model_pickle(self, model, file_path):
        
        """Dumps the trained model to a Pickle file.
           Input Arguments: model - the trained machine learning classification model
                            file_path - the file path to save the trained machine learning classification model to"""
            
        # Creates a file path directory for the saved model to be stored in
        self.create_file_path(file_path)
        
        # Dumps the Pickle file with the saved model to the file_path directory
        pickle.dump(model, open(file_path, "wb"))
            
        
    # Load saved model with Pickle
    def load_model_pickle(self, file_path):

        """Load a saved model with the Pickle module to the Python environment.
           A Python variable is required i.e. model to save the loaded model
           to the environment into.
           Input Arguments: file_path - the file path to save the trained machine learning classification model to"""
            
        # Returns the model file after loading it to the Python environment
        return pickle.load(open(file_path, "rb"))
            
    # Creates a file directory to store the saved model into if required
    def create_file_path(self, file_path):

        """Class method to check if there is a file path directory to save plots or models to. If
           there is no file directory, it is created to save the model to.
           Input Arguments: file_path - the file path to save the trained machine learning classification model to"""
            
        # If the output directory does not exist the OS modules makes the directory
        # Creates the file path variable
        file = pathlib.Path(file_path)

        # Conditional if statement to control the CSV data dump is required or not
        if file.exists():
            # Passes the if statement as the file directory for the saved model exists
            pass
        else:
            # Outputs the folder contents to csv files for each column
            os.makedirs(file_path)