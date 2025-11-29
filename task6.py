import pickle
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import randint, uniform

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
# Define Parameter Distribution
rf_dist = {
        'n_estimators': randint(100, 500),        # Integer between 100 and 500
        'max_depth': [None, 10, 20, 30, 40, 50],  # Specific options
        'min_samples_split': randint(2, 50),      # Integer between 2 and 11
        'min_samples_leaf': randint(1, 20),        # Integer between 1 and 5
        'max_features': ['sqrt', 'log2']          # Feature selection method
    }

# Initialize Classifier
rf = RandomForestClassifier()

# Grid Search with Cross Validation (cv=3)
rs_rf = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_dist,
    n_iter=20,       
    cv=3,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1,
)
rs_rf.fit(X_train_reduced, y_train)

# Evaluation
best_rf = rs_rf.best_estimator_
# best_rf.fit(X_train_reduced, y_train)
y_pred_rf = best_rf.predict(X_test_reduced)
rf_acc = accuracy_score(y_test, y_pred_rf)

print(f"Best RF Params: {rs_rf.best_params_}")
print(f"Best Cross-Validation Accuracy: {rs_rf.best_score_:.4f}")
print(f"Random Forest Test Accuracy: {rf_acc:.4f}%")

# ADABOOST
# Define Parameter Distribution
ada_dist = {
        'n_estimators': randint(50, 300),         # Number of estimators
        'learning_rate': uniform(0.01, 1.5)       # Float between 0.01 and 1.51
    }

# Initialize Classifier
ada = AdaBoostClassifier(algorithm='SAMME')

# random Search with Cross Validation
rs_ada = RandomizedSearchCV(
        estimator=ada,
        param_distributions=ada_dist,
        n_iter=20,
        cv=3,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1,
)
rs_ada.fit(X_train_reduced, y_train)

# Evaluation
best_ada = rs_ada.best_estimator_
# best_ada.fit(X_train_reduced, y_train)
y_pred_ada = best_ada.predict(X_test_reduced)
ada_acc = accuracy_score(y_test, y_pred_ada)

print(f"Best AdaBoost Params: {rs_ada.best_params_}")
print(f"Best Cross-Validation Accuracy: {rs_ada.best_score_:.4f}")
print(f"AdaBoost Test Accuracy: {ada_acc:.4f}%")

print("\n--- Final Results ---")
print(f"Random Forest: {rf_acc:.4f}")
print(f"AdaBoost:      {ada_acc:.4f}")