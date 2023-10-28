import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from simple_linear_regression import SimpleLinearRegression
from regression_metrics import RegressionMetrics

# Load the salary dataset to the Jupyter Notebook with the Pandas library
# https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression
df = pd.read_csv('./dataset/Salary_dataset.csv')

# Seperate the X features from the salary dataset
X = df.iloc[:,1:2]

# Seperate the target values from the salary dataset
y  = df.iloc[:,2:3]

# Normalises the data features with ScikitLearn StandardScaler object
# Creates the StandardScaler object from ScikitLearn
scaler = StandardScaler().fit(X)

# Fits the normalisation fo rth model
X = scaler.transform(X)

# Carry out a train, test split using the ScikitLearn library
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Converts the y_test traget values from a DataFrame to a NumPy array
y_train = y_train.values

# Converts the y_test target values from a DataFrame to a NumPy array
y_test = y_test.values

# Creats, the class object for the simple Linear Regression model
regressor = SimpleLinearRegression(X_train, y_train, lr=0.01, epochs=1000)
 
# Trains the LR model with the fit method
regressor.fit()

# Predicts the salary amount with the trained LR model on the test dataset features X_test
y_pred = regressor.predict(X_test)

# Flattens the 2D pred array to 1D from the model and stores it in the flat Python variable
flat = y_pred.flatten()

# Flattens the y_test 2D array to 1D and stores it in the y variable
y = y_test.flatten()

# Calls the correlation coefficient Python method to calculate PLCC, SRCC, and KRCC with SciPy
test_results = RegressionMetrics(flat, y)

# Calculates and prints out the correlation metrics to the screen
test_results.correlation()

# Call the Root Mean Squared Error (RMSE) from the RegressionMetrics object to display test results
error = test_results.rmse_metric()

# Prints out the Root Mean Squared Error (RMSE) to the screen
print("Root Mean Squared Error (RMSE): ", error)

# Call the Root Mean Squared Error (RMSE) from the RegressionMetrics object to display test results
r2 = test_results.r2_score()

# Prints out the Root Mean Squared Error (RMSE) to the screen
print("R2 Score: ", r2)

# File paths to check for and save LR predictions plot to
file_path = "./plots/LR_predictions.png"
file_path_check = "./plots/"

# Plots the predictions from the LR model to the screen
test_results.regression_plot(X_test, y_pred, file_path_check, file_path)