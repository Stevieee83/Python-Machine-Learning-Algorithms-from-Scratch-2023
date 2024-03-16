from regression_metrics import RegressionMetrics
from regression_csv_dataset import RegressionDatasetCSV
from ridge_regression import Ridge

# Input parameters to load the regression CSV dataset
path = "./dataset/multiple_linear_regression_dataset.csv"
target = "income"
test_size = 0.3
random_state = 33

# Creates the data object of the regression CSV data
data = RegressionDatasetCSV(path, target, test_size, random_state)

# Loads the train and testing datasets from the Lasso regression algorithm
X_train, X_test, y_train, y_test = data.standard_scale()

# Stores the input the parameters to the Ridge regression algorithm
parameters = {
    "l" : 0.1,
    "lr" : 0.1,
    "n_iters" : 300
}

# Creates the Ridge Regression algorithm and stores it in the model object
model = Ridge(**parameters)

# Fits the data to the Ridge regression algorithm during training
model.fit(X_train, y_train)

# Runs the Ridge regression model in inference mode and stores the predictions in the y_pred Python variable
y_pred = model.predict(X_test.values)

# Flattens the 2D pred array to 1D from the model and stores it in the flat Python variable
flat = y_pred.flatten()

# Flattens the y_test 2D array to 1D and stores it in the y variable
y = y_test.values.flatten()

# Calls the correlation coefficient Python method to calculate PLCC, SRCC, and KRCC with SciPy
test_results = RegressionMetrics(flat, y)

# Calculates and prints out the correlation metrics to the screen
test_results.correlation()

# Call the Root Mean Squared Error (RMSE) from the RegressionMetrics object to display test results
error = test_results.rmse_metric()

# Prints out the Root Mean Squared Error (RMSE) to the screen
print("Root Mean Squared Error (RMSE): ", error)

# Calculates the R2 Score regression evaluation metric
r2 = test_results.r2_score()

# Prints out the R2 Score to the screen
print("r2 score: ", r2)