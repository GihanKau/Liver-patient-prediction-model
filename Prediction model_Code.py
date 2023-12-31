"""
Name        : B.M.G.G.K. Rajapaksha
Faculty     : Faculty of Science, Bioinformatics
Index NO.   : S14210
Date        : 25/07/2022

Task        : Assignment 2; A supervised machine learning algorithm (KNN) to train and test the
            SCS4204_IS4103_CS4104 _dataset.xlsx dataset

Input       : 1. SCS4204_IS4103_CS4104 _dataset.xlsx dataset
                (It contains training and testing data)

Output      : 1. Accuracy
              2. Precision
              3. Sensitivity
              4. Specificity
              5. Error Rate

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, accuracy_score, precision_score
import matplotlib.pyplot as plt

# load the datasets

datasets = pd.ExcelFile('SCS4204_IS4103_CS4104 _dataset.xlsx')
training = pd.read_excel(datasets, 'Training Dataset')
testing = pd.read_excel(datasets, 'Testing Dataset')

print('Training dataset\n', training, '\n')
print('Testing dataset\n', testing, '\n')

print('Length of the testing dataset - ', len(testing))
print('Length of the training dataset - ', len(training), '\n')

# Pre processing the dataset
# convert categorical values into numerical values
# both 'Gender' and 'Class' attributes are categorical variable. Hence need to convert them into numerical values before
# preprocessing

# give numerical labels for 'Gender' and 'Class' attribute values in testing dataset

training['Gender'].replace(['Male', 'Female'], [0, 1], inplace=True)
training['Class'].replace(['Yes', 'No'], [1, 0], inplace=True)

testing['Gender'].replace(['Male', 'Female'], [0, 1], inplace=True)
testing['Class'].replace(['Yes', 'No'], [1, 0], inplace=True)

# Replace '?' values from the mean value of the respective columns (training dataset)

empty_not_accepted = ['Age', 'Gender', 'TB', 'DB', 'ALK', 'SGPT', 'SGOT', 'TP', 'ALB', 'AG_Ratio', 'Class']

accepted = empty_not_accepted

for column in accepted:
    if column not in ('Gender', 'Class'):
        training[column] = training[column].replace('?', np.NaN)
        mean = int(training[column].mean(skipna=True))
        training[column] = training[column].replace(np.NaN, mean)

    else:
        training[column] = training[column].replace('?', np.NaN)
        training = training.dropna(how='any', axis=0)

# Replace '?' values from the mean value of the respective columns (testing dataset)

empty_not_accepted = ['Age', 'Gender', 'TB', 'DB', 'ALK', 'SGPT', 'SGOT', 'TP', 'ALB', 'AG_Ratio', 'Class']

accepted = empty_not_accepted

for column in accepted:
    if column not in ('Gender', 'Class'):
        testing[column] = testing[column].replace('?', np.NaN)
        mean = int(testing[column].mean(skipna=True))
        testing[column] = testing[column].replace(np.NaN, mean)
    else:
        testing[column] = testing[column].replace('?', np.NaN)
        testing = testing.dropna(how='any', axis=0)

# Split the training dataset into x and y

train_x = training.iloc[:, 1:11]  # [all rows,column 1-11]
train_y = training.iloc[:, 11]  # [all rows,column 11]

print('First 10 values of train_x\n', train_x.head(10), '\n')
print('First 10 values of train_y\n', train_y.head(10), '\n')

# Split the testing dataset into x and y

test_x = testing.iloc[:, 1:11]  # [all rows,column 1-11]
test_y = testing.iloc[:, 11]  # [all rows,column 11]

print('First 10 values of test_x\n', test_x.head(10), '\n')
print('First 10 values of test_y\n', test_y.head(10), '\n')

# Feature Scaling

sc_X = StandardScaler()
train_x = sc_X.fit_transform(train_x)
test_x = sc_X.transform(test_x)

print('Standardized Training X data\n', train_x, '\n')

# hyperparameter tuning

# List of Hyperparameters that we want to tune

leaf_size = list(range(1, 30))
n_neighbors = list(range(1, 30))
p = [1, 2]

# get hyperparameters into a dictionary

hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

# Define the model
knn = KNeighborsClassifier()

# Use GridSearch
clf = GridSearchCV(knn, hyperparameters, cv=10)  # cv = class validation

# here cv = 10 means we have to divide the dataset into 5 sets/folds

# Fit the model

best_model = clf.fit(train_x, train_y)

print("Estimated best hyperparameters \n")

print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'], '\n')

# predict the test set results
pred_y = clf.predict(test_x)
print('Predicted y values for test data \n', pred_y, '\n')

# Evaluate model
cm = confusion_matrix(test_y, pred_y)
print('Confusion matrix \n', cm, '\n')

# Confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
disp.plot()
plt.show()

tn, fp, fn, tp = cm.ravel()

# tn = true negative
# tp = true positive
# fn = false negative
# fp = false positive

# Manual calculations

# print('Accuracy : ', round(((tp+tn)/(tp+fn+fp+tn))*100, 2), '%')
# print('Precision (Positive Predictive Value) : ', round((tp/(tp+fp))*100, 2), '%')
# print('Sensitivity (Hit rate/Recall) : ', round((tp/(tp+fn))*100, 2), '%')
# print('Specificity : ', round((tn/(tn+fp))*100, 2), '%')
# print('FN Rate (Miss rate) : ', round((fn/(tp+fn))*100, 2), '%')
# print('TP Rate (False Alarm Rate) : ', round((fp/(fp+tn))*100, 2), '%')
# print('Error rate : ', round(((fp+fn)/(tp+fn+fp+tn))*100, 2), '%')

# same calculations using sklearn

print('Accuracy : ', round((accuracy_score(test_y, pred_y)*100), 2), '%')
print('Precision (Positive Predictive Value) : ', round((precision_score(test_y, pred_y)*100), 2), '%')
print('Sensitivity (Hit rate/Recall) : ', round((tp/(tp+fn))*100, 2), '%')
print('Specificity : ', round((recall_score(test_y, pred_y)*100), 2), '%')
print('Error rate : ', round(((fp+fn)/(tp+fn+fp+tn))*100, 2), '%')