
# Off the shelf kNN classifier example

import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, binarize

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, recall_score

training_df = pandas.read_csv('samples/income_tr.csv')
test_df = pandas.read_csv('samples/income_te.csv')

all_df = pandas.concat([training_df, test_df])

# nbrs = NearestNeighbors(n_neighbors=2).fit(training_df)

def transform_category(attr):
    le = LabelEncoder()

    # Fit against labels across both datasets, as test may contain some not in training
    le.fit(all_df[attr].unique())
    training_df[attr] = le.transform(training_df[attr])
    test_df[attr] = le.transform(test_df[attr])

transform_category('workclass')
transform_category('marital_status')
transform_category('occupation')
transform_category('relationship')
transform_category('race')
transform_category('gender')
transform_category('native_country')

transform_category('class')


# Specify a subset of attributes to use for training and testing
attributes = [
    'age',
    'education_cat',
    'capital_gain',
    'capital_loss',
    'hour_per_week',
    'workclass',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'native_country'
]

data = training_df[attributes]

class_labels = training_df['class']
test = test_df[attributes]

# Vendor kNN classifier
knn = KNeighborsClassifier(n_neighbors = 40)

# train classifier
knn.fit(data, class_labels)

# Predict classes for the test set
predictions = knn.predict(test)

y_true = test_df['class']
y_score = knn.predict_proba(test)[:, 1]

# Changing the threshold to improve ROC curve
# y_pred_class = binarize(y_score, 0.2)[0]
# confusion = confusion_matrix(y_true, y_pred_class)
# print('0.1 confusion\n', confusion)

# predictions = y_pred_class

# first argument is true values, second argument is predicted values
confusion = confusion_matrix(y_true, predictions)
print('Confusion\n', confusion)

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

mse = (((predictions - y_true) ** 2).sum()) / len(predictions)
print('Mean Square Error', mse)

accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Accuracy', accuracy)
print('Accuracy builtin', accuracy_score(y_true, predictions))

specificity = TN / (TN + FP)
print('Specificity', specificity)

sensitivity = TP / float(FN + TP)
print('Sensitivity', sensitivity)
print('Recall builtin', recall_score(y_true, predictions))

# Should be equivalent to MSE above
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Classification error', classification_error)
print('1-accuracy builtin', 1 - accuracy_score(y_true, predictions))



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

# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])
    
evaluate_threshold(0.3)


