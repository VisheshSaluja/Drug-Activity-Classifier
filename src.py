#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install -U scikit-learn


# In[8]:


from imblearn.over_sampling import SMOTE
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.pipeline import Pipeline

# Function placeholders: load_dataset() and transform_to_sparse_matrix()
# Assume these functions are defined as per your specifications

#########################

def load_dataset(filename, is_train=True):
    data = []
    labels = [] if is_train else None
    with open(filename) as file:
        for line in file:
            parts = line.strip().split()
            if is_train:
                labels.append(int(parts[0]))
                features = [(int(feat), 1) for feat in parts[1:]]
            else:
                features = [(int(feat), 1) for feat in parts]
            data.append(features)
    return (data, labels) if is_train else data

# Function to transform data into a sparse matrix
def transform_to_sparse_matrix(data, num_features):
    row_ind, col_ind, values = [], [], []
    for i, row in enumerate(data):
        for feat, val in row:
            row_ind.append(i)
            col_ind.append(feat)
            values.append(val)
    return csr_matrix((values, (row_ind, col_ind)), shape=(len(data), num_features))

#########################

train_data, train_labels = load_dataset('Train_data.txt', is_train=True)
test_data = load_dataset('Test_data.txt', is_train=False)

# Finding the maximum feature index
train_features = [feat for row in train_data for feat, _ in row]
test_features = [feat for row in test_data for feat, _ in row]
num_features = max(train_features + test_features) + 1

X_train = transform_to_sparse_matrix(train_data, num_features)
y_train = np.array(train_labels)
X_test = transform_to_sparse_matrix(test_data, num_features)

# Set up the F1 scorer for model evaluation
f1_scorer = make_scorer(f1_score)

# Define models and pipelines
model_params = [
    {
        'name': 'DecisionTree',
        'pipeline': Pipeline([
            ('maxabsscaler', MaxAbsScaler()),
            ('feature_selection', SelectKBest(chi2)),
            ('smote', SMOTE(random_state=42)),
            ('model', DecisionTreeClassifier(random_state=42))
        ]),
        'params': {
            'feature_selection__k': [100, 200, 300],  # Example values; adjust based on dataset size and characteristics
            'model__max_depth': [10, 20, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__criterion': ['gini', 'entropy']
        }
    },
    {
        'name': 'NaiveBayes',
        'pipeline': Pipeline([
            ('maxabsscaler', MaxAbsScaler()),
            ('feature_selection', SelectKBest(chi2)),
            ('smote', SMOTE(random_state=42)),
            ('model', BernoulliNB())
        ]),
        'params': {
            'feature_selection__k': [100, 200, 300],
            'model__alpha': np.logspace(-3, 3, 7)
        }
    }
]

results = []

# Loop through each model configuration and perform grid search
for config in model_params:
    gscv = GridSearchCV(config['pipeline'], config['params'], cv=5, scoring=f1_scorer, n_jobs=-1, verbose=1)
    gscv.fit(X_train, y_train)
    score = gscv.best_score_
    print(f"{config['name']} best score: {score}")
    results.append((score, config['name'], gscv))

# Select the best model based on F1 score
best_score, best_model_name, best_model_gscv = max(results, key=lambda item: item[0])
print(f"Best Model: {best_model_name} with F1 score: {best_score}")

# Predict on the test data using the best found model
predictions = best_model_gscv.predict(X_test)

# Write predictions to a file
with open('best_model_improved_predictions_final_28.txt', 'w') as f:
    for pred in predictions:
        f.write(f"{pred}\n")

print("Predictions have been written to 'best_model_improved_predictions_final_28.txt'")


# In[ ]:


test_f1_score = f1_score(y_test, predictions)  # Make sure y_test is your actual test labels

