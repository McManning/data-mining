
# Off the shelf kNN classifier example

import pandas
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

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
    'fnlwgt',
    'education_cat',
    'capital_gain',
    'capital_loss',
    'hour_per_week'
]

data = training_df[attributes]

class_labels = training_df['class']
test = test_df[attributes]

# Vendor kNN classifier
knn = KNeighborsClassifier(n_neighbors = 11)

# train classifier
knn.fit(data, class_labels)

# Predict classes for the test set
predictions = knn.predict(test)

# Report correct predictions
correct = 0
for i, x in test_df.iterrows():
    if x['class'] == predictions[i]:
        correct += 1

actual = test_df['class']

mse = (((predictions - actual) ** 2).sum()) / len(predictions)

print(predictions)
print(correct)
print(correct/len(predictions))
print(mse)

# actual = test['class']
# mean square error is mse = (((predictions - actual) ** 2).sum()) / len(predictions)

