from logistic_regression import LogisticRegression
from classification_metrics import ClassificationMetrics
from classification_csv_dataset import ClassificationDatasetCSV
from timeit import default_timer as timer

# Input parameters to load the regression CSV dataset
path = "./dataset/heart.csv"
label = "target"
test_size = 0.3
random_state = 42

# Creates the data object of the classification CSV data
data = ClassificationDatasetCSV(path, label, test_size, random_state)

# Creates the figure number for the class_balance method
figure = 1

# Plots the class balance in a bar plot
data.class_ballance(figure)

# Calls the imbleran SMOTE Python method and balances the dataset
data.smote_balance()

# Creates the figure number for the class_balance method
figure = 2

# Plots the class balance in a bar plot
data.after_class_ballance(figure)

# Calls the standard_scale method from the ClassificationDatasetCSV object
X_train, X_test, y_train, y_test = data.standard_scale()

# Prints out the X_train Pandas DataFrame to the screen
print("Pandas DataFrame X_train Normalised")
print(X_train)
print("\n")

# Prints out the X_test Pandas DataFrame to the screen
print("Pandas DataFrame X_test Normalised")
print(X_test)
print("\n")

# Stores the start time in the end variable
start = timer()

# Logistic Regression variable that stores the constructed regression model 0.0076
clf = LogisticRegression(lr=0.0076, n_iters=1000)

# Logistic Regression fit function to fit the training data to
clf.fit(X_train, y_train)

# Stores the end time in the end variable
end = timer()

# Calculates the training time by subtracting the end time from the start time in seconds
train_time = end - start

# Prints out the training time to the screen
print('Train Time')
print(train_time, 'seconds')
print("\n")

# Stores the start time in the end variable
start = timer()

# Stores the classification predictions from the test dataset in the y_pred variable
y_pred = clf.predict(X_test)

# Stores the end time in the end variable
end = timer()

# Calculates the inference time in the same way as the training time
inference_time = end - start

# Prints out the inference time to the screen
print('Inference Time')
print(inference_time, 'seconds')
print("\n")

# Prints out the model predictions to the screen
print('Model Predictions')
print(y_pred)
print("\n")

# Calls the ClassificationMetrics object
test_results = ClassificationMetrics(y_pred, y_test)

# Calculates and prints out the accuracy test metric to the screen
acc = test_results.accuracy()

# Prints out the test accuracy to the screen
print(f'Test Accuracy: {acc:.2f}%')

# Calculates and prints out the precision test metric to the screen
pre = test_results.precision()

# Prints out the test precision to the screen
print(f'Test Precision: {pre:.2f}%')

# Calculates and prints out the recall test metric to the screen
rec = test_results.recall()

# Prints out the test recall to the screen
print(f'Test Recall: {rec:.2f}%')

# Calculates and prints out the f1_score test metric to the screen
f_one = test_results.f1_score()

# Prints out the test F1 Score to the screen
print(f'Test F1 Score: {f_one:.2f}%')

# Creates the target names list for the classification report
names = ['Heart Disease (1)', 'Disease Free (0)']

# Displays the classification report to the screen
test_results.classification_report(y_test, y_pred, names)

# Confusion Matrix plot function parameters
class_names = ['Heart Disease', 'Disease Free']
file_path_dir = './plots/'
title = 'Logistic Regression Classification Confusion Matrix'
file_path = 'confusion_matrix.png'
figure = 3

# Displays the Confusion Matrix plot to the screen
test_results.confusion_matrix(y_test, y_pred, class_names, figure, title, file_path_dir, file_path)

# AUROC plot function parameters
file_path_dir = './plots/'
file_path = 'AUROC.png'
title = 'Logistic Regression AUROC Curve Plot'
figure = 4

# Displays the Confusion Matrix plot to the screen
test_results.roc_curve(y_test, y_pred, figure, title, file_path_dir, file_path)