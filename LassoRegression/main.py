from regression_metrics import RegressionMetrics
from regression_csv_dataset import RegressionDatasetCSV
from lasso_regression import Lasso

# Input parameters to load the regression CSV dataset
path = "./dataset/multiple_linear_regression_dataset.csv"
target = "income"
test_size = 0.3
random_state = 33

# Creates the data object of the regression CSV data
data = RegressionDatasetCSV(path, target, test_size, random_state)

# Creates the Python variables for the exploritory data analysis plots
file_path_dir = "./plots/"
file_name = "scatterplot_matrix.png"

# Checks the correlation of the dataset with a scatterplot matrix plot
data.sctterplot_matrix(file_path_dir, file_name)

# Creates the title variable for the correlation heatmap plot
title = "Corrolation Heatmap Plot"
file_name = "correlation_heat_map.png"

# Checks the correlation of the dataset with a corrolation heatmap plot
data.correlation(title, file_path_dir, file_name)

# Loads the train and testing datasets from the Lasso regression algorithm
X_train, X_test, y_train, y_test = data.standard_scale()

# Stores the input the parameters to the Lasso regression algorithm
parameters = {
    "l" : 0.1,
    "lr" : 0.1,
    "n_iters" : 700
}

# Creates the Lasso Regression algorithm and stores it in the model object
model = Lasso(**parameters)

# Fits the data to the Lasso regression algorithm during training
model.fit(X_train, y_train)

# Runs the Lasso regression model in inference mode and stores the predictions in the y_pred Python variable
y_pred = model.predict(X_test.values)

# Flattens the 2D y_pred array to 1D from the model and stores it in the flat Python variable
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