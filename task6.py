import pickle
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

def load_batch(batch_path):
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        X = batch[b'data']
        y = np.array(batch[b'labels'])
        return X, y

def load_cifar10(data_dir):
    X_all = []
    y_all = []
    # 5 training batches
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        X, y = load_batch(batch_path)
        X_all.append(X)
        y_all.append(y)
    X_train = np.concatenate(X_all)
    y_train = np.concatenate(y_all)
    # test batch
    X_test, y_test = load_batch(os.path.join(data_dir, "test_batch"))
    return X_train/255, y_train, X_test/255, y_test

# Load the data
# You may need to specify ‘data_dir’ the directory of dataset folder
data_dir = "cifar-10-batches-py"
X_train, y_train, X_test, y_test = load_cifar10(data_dir)

pca = PCA(n_components=200)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

# RANDOM FOREST
# Define Parameter Grid
rf_params = {
    'n_estimators': [50, 100],       # Number of trees
    'max_depth': [None, 20],         # Maximum depth of tree
    'min_samples_split': [2, 5]
}

# Initialize Classifier
rf = RandomForestClassifier()

# Grid Search with Cross Validation (cv=3)
grid_rf = GridSearchCV(
    estimator=rf, 
    param_grid=rf_params, 
    cv=3, 
    scoring='accuracy', 
    verbose=2, 
    n_jobs=-1
)
grid_rf.fit(X_train_reduced, y_train)

# Evaluation
best_rf = grid_rf.best_estimator_
best_rf.fit(X_train_reduced, y_train)
y_pred_rf = best_rf.predict(X_test_reduced)
rf_acc = accuracy_score(y_test, y_pred_rf)

print(f"Best RF Params: {grid_rf.best_params_}")
print(f"Random Forest Test Accuracy: {rf_acc:.4f}%")

# ADABOOST
# Define Parameter Grid
# AdaBoost can be slower to train, so we keep the grid smaller
ada_params = {
    'n_estimators': [50, 100],
    'learning_rate': [0.1, 1.0]
}

# Initialize Classifier
ada = AdaBoostClassifier(algorithm='SAMME')

# Grid Search with Cross Validation
grid_ada = GridSearchCV(
    estimator=ada, 
    param_grid=ada_params, 
    cv=3, 
    scoring='accuracy', 
    verbose=2, 
    n_jobs=-1
)
grid_ada.fit(X_train_reduced, y_train)

# Evaluation
best_ada = grid_ada.best_estimator_
best_ada.fit(X_train_reduced, y_train)
y_pred_ada = best_ada.predict(X_test_reduced)
ada_acc = accuracy_score(y_test, y_pred_ada)

print(f"Best AdaBoost Params: {grid_ada.best_params_}")
print(f"AdaBoost Test Accuracy: {ada_acc:.4f}%")

print("\n--- Final Results ---")
print(f"Random Forest: {rf_acc:.4f}")
print(f"AdaBoost:      {ada_acc:.4f}")