# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:18:34 2017

@author: Chase
"""
import pandas
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, recall_score

results = pandas.read_csv('results-knn-11.csv')

y_true = label_binarize(results['Actual'], classes=[' <=50K', ' >50K'])
y_score = results['Posterior Probability']

y_pred_class = label_binarize(results['Predicted'], classes=[' <=50K', ' >50K'])


# first argument is true values, second argument is predicted values
confusion = confusion_matrix(y_true, y_pred_class)
print('Confusion\n', confusion)

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

mse = (((y_pred_class - y_true) ** 2).sum()) / len(y_pred_class)
print('Mean Square Error', mse)

accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Accuracy', accuracy)
print('Accuracy builtin', accuracy_score(y_true, y_pred_class))

specificity = TN / (TN + FP)
print('Specificity', specificity)

sensitivity = TP / float(FN + TP)
print('Sensitivity', sensitivity)
print('Recall builtin', recall_score(y_true, y_pred_class))

# Should be equivalent to MSE above
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Classification error', classification_error)
print('1-accuracy builtin', 1 - accuracy_score(y_true, y_pred_class))



# might be able to simplify by doing:
fpr, tpr, thresholds = roc_curve(
    y_true,
    y_score
)

roc_auc = roc_auc_score(y_true, y_score)

# Generate actual plot
plt.figure()
lw = 2

# Actual ROC curve
plt.plot(
    fpr, 
    tpr, 
    color='darkorange',
    lw=lw, 
    label='ROC Curve (area = %0.2f)' % roc_auc
)

# Diagonal
plt.plot(
    [0, 1], 
    [0, 1], 
    color='navy', 
    lw=lw,
    linestyle='--'
)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver operator curve')
plt.legend(loc='lower right')
plt.show()
